"""
Piecewise 実験結果に対して、SGD のみを再計算して診断・差し替えするユーティリティ。

想定する用途
------------
- SGD の平均誤差系列が「一部 trial だけ爆発」して平均が歪んでいないか確認したい
- 既存の *_meta.json の `results.metrics.sgd` を SGD 再計算結果で置き換えたい
- 既存の figure PNG を（他手法の曲線は meta の値のまま）SGD だけ差し替えて上書き保存したい

使い方例
--------
SGD の outlier seed を調べる（差し替えはしない）:
    python -m code.diagnose_rerun_sgd_piecewise --meta_json path/to/*_meta.json

SGD の平均系列を meta に差し替え（別ファイルに保存）:
    python -m code.diagnose_rerun_sgd_piecewise --meta_json ... --write_updated_meta

SGD の平均系列を meta に差し替え、PNG も上書き:
    python -m code.diagnose_rerun_sgd_piecewise --meta_json ... --write_updated_meta --overwrite_figure
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from code.data_gen import generate_piecewise_X_with_exog
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.io.plotting import apply_style
from utils.offline_solver import solve_offline_sem_lasso_batch


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / pp).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _compute_error(
    S_hat: np.ndarray,
    S_true: np.ndarray,
    error_normalization: str,
    S_offline: Optional[np.ndarray],
    eps: float = 1e-12,
) -> float:
    if error_normalization == "offline_solution" and S_offline is not None:
        num = np.linalg.norm(S_hat - S_offline) ** 2
        den = np.linalg.norm(S_offline) ** 2 + eps
        return float(num / den)
    num = np.linalg.norm(S_hat - S_true) ** 2
    den = np.linalg.norm(S_true) ** 2 + eps
    return float(num / den)


def _extract_hyperparam_json_path_from_meta(meta: Dict[str, Any]) -> Optional[Path]:
    # 1) 明示的に meta["hyperparam_json"] があれば使う
    hp = meta.get("hyperparam_json")
    if isinstance(hp, str) and hp.strip():
        return _resolve_path(hp)

    # 2) meta["command"] に "--hyperparam_json" が含まれる想定
    cmd = meta.get("command")
    if isinstance(cmd, list):
        for i in range(len(cmd) - 1):
            if str(cmd[i]).endswith("--hyperparam_json"):
                return _resolve_path(str(cmd[i + 1]))
    return None


@dataclass(frozen=True)
class SGDRunConfig:
    # data / seeds
    trial_seeds: List[int]
    N: int
    T: int
    sparsity: float
    max_weight: float
    std_e: float
    K: int
    # generator kwargs
    s_type: str
    t_min: float
    t_max: float
    z_dist: str
    # metric
    error_normalization: str
    offline_lambda_l1: Optional[float]
    # sgd hyperparams
    sgd_lambda_reg: float
    sgd_alpha: float
    beta_sgd: float
    C: int


def _build_sgd_run_config(meta: Dict[str, Any], hyperparam_json_path: Optional[Path]) -> SGDRunConfig:
    cfg = meta["config"]
    gen = meta.get("generator", {}).get("kwargs", {})
    metric = meta.get("metric", {})

    trial_seeds = list(cfg["trial_seeds"])
    N = int(cfg["N"])
    T = int(cfg["T"])
    sparsity = float(cfg["sparsity"])
    max_weight = float(cfg["max_weight"])
    std_e = float(cfg["std_e"])
    K = int(cfg["K"])

    s_type = str(gen.get("s_type", "random"))
    t_min = float(gen.get("t_min", 0.5))
    t_max = float(gen.get("t_max", 1.0))
    z_dist = str(gen.get("z_dist", "uniform01"))

    error_normalization = str(metric.get("error_normalization", "true_value"))
    offline_lambda_l1 = metric.get("offline_lambda_l1", None)
    offline_lambda_l1_f = float(offline_lambda_l1) if offline_lambda_l1 is not None else None

    # run_piecewise.py のロジックに合わせる:
    # - sgd は sgd_cfg に lambda_reg/alpha が無ければ pc の値を使う
    # - C は pc の C を使う（sgd_cfg からは読まない実装）
    sgd_beta_sgd = None
    pc_lambda_reg = None
    pc_alpha = None
    pc_C = None
    sgd_lambda_reg_override = None
    sgd_alpha_override = None

    if hyperparam_json_path is not None and hyperparam_json_path.is_file():
        hp = _load_json(hyperparam_json_path)
        pc = hp.get("pc", {}) if isinstance(hp, dict) else {}
        sgd = hp.get("sgd", {}) if isinstance(hp, dict) else {}

        if isinstance(pc, dict):
            pc_lambda_reg = pc.get("lambda_reg", None)
            pc_alpha = pc.get("alpha", None)
            pc_C = pc.get("C", None)

        if isinstance(sgd, dict):
            sgd_beta_sgd = sgd.get("beta_sgd", None)
            sgd_lambda_reg_override = sgd.get("lambda_reg", None)
            sgd_alpha_override = sgd.get("alpha", None)

    # fallback: meta に書かれている値を使う（古い meta でも動くように）
    meta_methods = meta.get("methods", {})
    meta_pc = meta_methods.get("pc", {}).get("hyperparams", {}) if isinstance(meta_methods, dict) else {}
    meta_sgd = meta_methods.get("sgd", {}).get("hyperparams", {}) if isinstance(meta_methods, dict) else {}

    if pc_lambda_reg is None:
        pc_lambda_reg = meta_pc.get("lambda_reg", None)
    if pc_alpha is None:
        pc_alpha = meta_pc.get("alpha", None)
    if pc_C is None:
        pc_C = meta_pc.get("C", None)

    if sgd_beta_sgd is None:
        sgd_beta_sgd = meta_sgd.get("beta_sgd", None)

    if pc_lambda_reg is None or pc_alpha is None or pc_C is None or sgd_beta_sgd is None:
        raise ValueError("必要なハイパラが不足しています（pc: lambda_reg/alpha/C, sgd: beta_sgd）")

    sgd_lambda_reg = float(sgd_lambda_reg_override) if sgd_lambda_reg_override is not None else float(pc_lambda_reg)
    sgd_alpha = float(sgd_alpha_override) if sgd_alpha_override is not None else float(pc_alpha)
    beta_sgd = float(sgd_beta_sgd)
    C = int(pc_C)

    return SGDRunConfig(
        trial_seeds=trial_seeds,
        N=N,
        T=T,
        sparsity=sparsity,
        max_weight=max_weight,
        std_e=std_e,
        K=K,
        s_type=s_type,
        t_min=t_min,
        t_max=t_max,
        z_dist=z_dist,
        error_normalization=error_normalization,
        offline_lambda_l1=offline_lambda_l1_f,
        sgd_lambda_reg=sgd_lambda_reg,
        sgd_alpha=sgd_alpha,
        beta_sgd=beta_sgd,
        C=C,
    )


def _run_one_trial_sgd(run_cfg: SGDRunConfig, trial_seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(int(trial_seed))

    S_series, B_true, U, Y = generate_piecewise_X_with_exog(
        N=run_cfg.N,
        T=run_cfg.T,
        sparsity=run_cfg.sparsity,
        max_weight=run_cfg.max_weight,
        std_e=run_cfg.std_e,
        K=run_cfg.K,
        s_type=run_cfg.s_type,
        t_min=run_cfg.t_min,
        t_max=run_cfg.t_max,
        z_dist=run_cfg.z_dist,
        rng=rng,
    )

    S_offline = None
    if run_cfg.error_normalization == "offline_solution":
        if run_cfg.offline_lambda_l1 is None:
            raise ValueError("error_normalization=offline_solution ですが offline_lambda_l1 がありません")
        S_offline = solve_offline_sem_lasso_batch(Y, U, float(run_cfg.offline_lambda_l1))

    sgd = PCSEM(
        run_cfg.N,
        np.zeros((run_cfg.N, run_cfg.N)),
        run_cfg.sgd_lambda_reg,
        run_cfg.sgd_alpha,
        run_cfg.beta_sgd,
        0.0,  # gamma=0
        0,  # P=0
        run_cfg.C,
        show_progress=False,
        name="sgd_baseline",
        T_init=B_true,
    )
    estimates_sgd, _ = sgd.run(Y, U)

    errors = [
        _compute_error(estimates_sgd[t], S_series[t], run_cfg.error_normalization, S_offline)
        for t in range(run_cfg.T)
    ]
    arr = np.asarray(errors, dtype=float)

    return {
        "seed": int(trial_seed),
        "error": arr,
        "error_final": float(arr[-1]),
        "error_max": float(np.nanmax(arr)),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
    }


def _plot_with_replaced_sgd(meta: Dict[str, Any], new_sgd_mean: np.ndarray, save_path: Path, show: bool) -> None:
    metrics = meta.get("results", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("meta['results']['metrics'] が不正です")

    def _to_arr(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        return np.asarray(x, dtype=float)

    # 他手法は meta の値を使用し、SGDだけ新しい平均で差し替え
    error_co = _to_arr(metrics.get("co"))
    error_pc = _to_arr(metrics.get("pc"))
    error_pg = _to_arr(metrics.get("pg"))
    error_pp = _to_arr(metrics.get("pp"))
    error_sgd = np.asarray(new_sgd_mean, dtype=float)

    T = int(meta["config"]["T"])

    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)
    plt.figure(figsize=(10, 6))
    if error_co is not None:
        plt.plot(error_co, color="blue", label="Correction Only")
    if error_pc is not None:
        plt.plot(error_pc, color="limegreen", label="Prediction Correction")
    if error_sgd is not None:
        plt.plot(error_sgd, color="cyan", label="SGD")
    if error_pg is not None:
        plt.plot(error_pg, color="magenta", label="ProxGrad")
    if error_pp is not None:
        plt.plot(error_pp, color="red", label="Proposed (PP)")

    plt.yscale("log")
    plt.xlim(left=0, right=T)
    plt.xlabel("t")

    error_normalization = str(meta.get("metric", {}).get("error_normalization", "true_value"))
    if error_normalization == "offline_solution":
        plt.ylabel(r"Average $\frac{\|\hat{S} - S_{\mathrm{offline}}\|_F^2}{\|S_{\mathrm{offline}}\|_F^2}$")
    else:
        plt.ylabel(r"Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^*\|_F^2}$")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGD を再計算して outlier 診断/差し替え")
    p.add_argument("--meta_json", type=Path, required=True, help="対象の *_meta.json")
    p.add_argument("--hyperparam_json", type=Path, default=None, help="ハイパラJSON（省略時は meta から推定）")
    p.add_argument("--n_jobs", type=int, default=-1, help="joblib 並列数（-1: 全コア）")
    p.add_argument("--topk", type=int, default=10, help="outlier として表示する上位件数")
    p.add_argument(
        "--exclude_top_by_max",
        type=int,
        default=0,
        help="最大誤差が大きい trial を上位K件除外して平均を計算（0で無効）",
    )
    p.add_argument(
        "--exclude_seeds",
        type=int,
        nargs="*",
        default=None,
        help="除外する seed を明示指定（例: --exclude_seeds 5 84）",
    )
    p.add_argument("--write_updated_meta", action="store_true", help="SGD平均系列を meta に反映したJSONを保存")
    p.add_argument("--inplace", action="store_true", help="meta_json を上書き（write_updated_meta と併用推奨）")
    p.add_argument("--overwrite_figure", action="store_true", help="metaにある figure_path の PNG を上書き")
    p.add_argument("--figure_out", type=Path, default=None, help="figure の保存先（省略時は meta の figure_path）")
    p.add_argument("--no_show", action="store_true", help="plt.show() しない")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    meta_path = _resolve_path(args.meta_json)
    meta = _load_json(meta_path)

    hp_path: Optional[Path]
    if args.hyperparam_json is not None:
        hp_path = _resolve_path(args.hyperparam_json)
    else:
        hp_path = _extract_hyperparam_json_path_from_meta(meta)

    run_cfg = _build_sgd_run_config(meta, hp_path)

    # 並列で SGD を再計算
    seeds = run_cfg.trial_seeds
    with tqdm_joblib(tqdm(desc="SGD rerun", total=len(seeds))):
        results = Parallel(n_jobs=args.n_jobs, batch_size=1, prefer="threads")(
            delayed(_run_one_trial_sgd)(run_cfg, s) for s in seeds
        )

    # 集計
    per_trial_errors = np.stack([r["error"] for r in results], axis=0)  # (num_trials, T)
    sgd_mean_raw = np.nanmean(per_trial_errors, axis=0)

    # outlier 診断
    error_max = np.asarray([r["error_max"] for r in results], dtype=float)
    error_final = np.asarray([r["error_final"] for r in results], dtype=float)
    has_nan = any(bool(r["has_nan"]) for r in results)
    has_inf = any(bool(r["has_inf"]) for r in results)

    topk = int(args.topk)
    idx_max = np.argsort(-error_max)[:topk]
    idx_final = np.argsort(-error_final)[:topk]

    # 除外指定がある場合は、平均を「フィルタ後」で計算する
    excluded_seeds: List[int] = []
    if args.exclude_seeds:
        excluded_seeds.extend([int(s) for s in args.exclude_seeds])
    if int(args.exclude_top_by_max) > 0:
        k = int(args.exclude_top_by_max)
        excluded_seeds.extend([int(results[int(i)]["seed"]) for i in np.argsort(-error_max)[:k]])
    excluded_seeds = sorted(set(excluded_seeds))

    if excluded_seeds:
        keep_mask = np.array([int(r["seed"]) not in excluded_seeds for r in results], dtype=bool)
        if not bool(keep_mask.any()):
            raise ValueError("除外が多すぎて残りの trial がありません")
        sgd_mean = np.nanmean(per_trial_errors[keep_mask], axis=0)
    else:
        sgd_mean = sgd_mean_raw

    def _pick(idx: np.ndarray) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in idx:
            rr = results[int(i)]
            out.append(
                {
                    "seed": rr["seed"],
                    "error_max": rr["error_max"],
                    "error_final": rr["error_final"],
                    "has_nan": rr["has_nan"],
                    "has_inf": rr["has_inf"],
                }
            )
        return out

    report = {
        "created_at": datetime.now().isoformat(),
        "meta_json": str(meta_path),
        "hyperparam_json": str(hp_path) if hp_path is not None else None,
        "num_trials": len(seeds),
        "T": run_cfg.T,
        "sgd_hyperparams_used": {
            "lambda_reg": run_cfg.sgd_lambda_reg,
            "alpha": run_cfg.sgd_alpha,
            "beta_sgd": run_cfg.beta_sgd,
            "C": run_cfg.C,
        },
        "data_gen_used": {
            "N": run_cfg.N,
            "T": run_cfg.T,
            "sparsity": run_cfg.sparsity,
            "max_weight": run_cfg.max_weight,
            "std_e": run_cfg.std_e,
            "K": run_cfg.K,
            "s_type": run_cfg.s_type,
            "t_min": run_cfg.t_min,
            "t_max": run_cfg.t_max,
            "z_dist": run_cfg.z_dist,
        },
        "sanity": {"has_any_nan": has_nan, "has_any_inf": has_inf},
        "outliers_by_max_error": _pick(idx_max),
        "outliers_by_final_error": _pick(idx_final),
        "excluded_seeds_for_mean": excluded_seeds,
        "sgd_mean_first10": sgd_mean[:10].tolist(),
        "sgd_mean_last10": sgd_mean[-10:].tolist(),
        "sgd_mean_raw_first10": sgd_mean_raw[:10].tolist(),
        "sgd_mean_raw_last10": sgd_mean_raw[-10:].tolist(),
        "sgd_final_error_quantiles": {
            "q50": float(np.nanquantile(error_final, 0.50)),
            "q90": float(np.nanquantile(error_final, 0.90)),
            "q95": float(np.nanquantile(error_final, 0.95)),
            "q99": float(np.nanquantile(error_final, 0.99)),
            "max": float(np.nanmax(error_final)),
        },
        "sgd_max_error_quantiles": {
            "q50": float(np.nanquantile(error_max, 0.50)),
            "q90": float(np.nanquantile(error_max, 0.90)),
            "q95": float(np.nanquantile(error_max, 0.95)),
            "q99": float(np.nanquantile(error_max, 0.99)),
            "max": float(np.nanmax(error_max)),
        },
    }

    out_dir = meta_path.parent
    report_path = out_dir / f"{meta_path.stem}_sgd_rerun_report.json"
    _save_json(report_path, report)
    print(f"[saved] {report_path}")

    # meta 更新（必要なら）
    if args.write_updated_meta:
        updated = dict(meta)
        updated.setdefault("results", {}).setdefault("metrics", {})
        updated["results"]["metrics"]["sgd"] = sgd_mean.tolist()
        updated.setdefault("reruns", {})
        updated["reruns"]["sgd"] = {
            "created_at": datetime.now().isoformat(),
            "report": str(report_path),
        }

        if args.inplace:
            updated_meta_path = meta_path
        else:
            updated_meta_path = out_dir / f"{meta_path.stem}_UPDATED_SGD_meta.json"
        _save_json(updated_meta_path, updated)
        print(f"[saved] {updated_meta_path}")

        # figure 上書き（必要なら）
        if args.overwrite_figure:
            if args.figure_out is not None:
                fig_path = _resolve_path(args.figure_out)
            else:
                fig_rel = meta.get("results", {}).get("figure_path", None)
                if not isinstance(fig_rel, str) or not fig_rel.strip():
                    raise ValueError("meta['results']['figure_path'] が見つかりません")
                fig_path = _resolve_path(fig_rel)
            _plot_with_replaced_sgd(updated, sgd_mean, fig_path, show=(not args.no_show))
            print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()

