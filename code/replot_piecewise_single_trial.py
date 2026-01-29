"""
Meta JSON から piecewise 実験を 1 trial だけ再実行してプロットする。

用途:
- num_trials=100 の平均プロットではなく、1回（単一 seed）の軌跡を見たい
- 縦軸ラベルの "Average" を外したい

例:
    python -m code.replot_piecewise_single_trial --meta path/to/*_meta.json --trial_index 0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from code.data_gen import generate_piecewise_X_with_exog
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from models.tvgti_pc.prediction_correction_sem_noexog import PredictionCorrectionSEMNoExog as PCSEMNoExog
from utils.io.plotting import apply_style
from utils.metrics import compute_error_series, compute_normalized_error
from utils.offline_solver import solve_offline_sem_lasso_batch


@dataclass(frozen=True)
class Variant:
    key: str
    normalization: str
    divide_by_n2: bool


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _lookahead_effective(r: int, q: int, cfg_value: int) -> int:
    if int(cfg_value) == -1:
        return int(r) + int(q) - 2
    return max(0, int(cfg_value))


def _pc_T_init(N: int, T_true: np.ndarray, use_true: bool, identity_scale: float) -> np.ndarray:
    if bool(use_true):
        return T_true
    return np.eye(N) * float(identity_scale)


def _pp_b0(T_true: np.ndarray, mode: str) -> np.ndarray:
    mode_norm = str(mode).strip()
    if mode_norm == "true_T_diag":
        return np.diag(T_true)
    return np.ones(T_true.shape[0])


def _variants_from_meta(meta: Dict[str, Any]) -> List[Variant]:
    metric = meta.get("metric", {}) or {}
    variants = metric.get("plot_variants")
    if isinstance(variants, list) and variants:
        out: List[Variant] = []
        for v in variants:
            out.append(
                Variant(
                    key=str(v["key"]),
                    normalization=str(v["normalization"]),
                    divide_by_n2=bool(v["divide_by_n2"]),
                )
            )
        return out

    normalization = str(metric.get("error_normalization", "true_value"))
    divide_by_n2 = bool(metric.get("divide_by_n2", False))
    key = ("true" if normalization == "true_value" else "offline") + f"_n2={int(divide_by_n2)}"
    return [Variant(key=key, normalization=normalization, divide_by_n2=divide_by_n2)]


def _figure_name_for_variant(meta: Dict[str, Any], variant: Variant) -> Optional[str]:
    results = meta.get("results", {}) or {}
    figures = results.get("figures")
    if isinstance(figures, list):
        for rec in figures:
            if (
                str(rec.get("normalization")) == variant.normalization
                and bool(rec.get("divide_by_n2")) == bool(variant.divide_by_n2)
            ):
                return str(rec.get("figure"))
    fig = results.get("figure")
    if fig is not None:
        return str(fig)
    return None


def _make_output_name(base_name: str, trial_seed: int) -> str:
    name = base_name
    if "num_trials=" in name:
        name = name.replace("num_trials=100", "num_trials=1")
        name = name.replace("num_trials=10", "num_trials=1")
        name = name.replace("num_trials=2", "num_trials=1")
    name = re.sub(r"seed=\d+", f"seed={int(trial_seed)}", name)
    stem = Path(name).stem
    suffix = Path(name).suffix or ".png"
    return f"{stem}_trial_seed={trial_seed}{suffix}"


def _plot_one(
    out_path: Path,
    T: int,
    series_by_method: Dict[str, np.ndarray],
    normalization: str,
    divide_by_n2: bool,
) -> None:
    plt.figure(figsize=(10, 6))
    if "co" in series_by_method:
        plt.plot(series_by_method["co"], color="blue", label="Correction Only")
    if "pc" in series_by_method:
        plt.plot(series_by_method["pc"], color="limegreen", label="Prediction Correction")
    if "sgd" in series_by_method:
        plt.plot(series_by_method["sgd"], color="cyan", label="SGD")
    if "pg" in series_by_method:
        plt.plot(series_by_method["pg"], color="magenta", label="ProxGrad")
    if "pp_sgd" in series_by_method:
        plt.plot(series_by_method["pp_sgd"], color="orange", label="PP-SGD (q=1,r=1)")
    if "pp" in series_by_method:
        plt.plot(series_by_method["pp"], color="red", label="Proposed (PP)")
    plt.yscale("log")
    plt.xlim(left=0, right=T)
    plt.xlabel("t")
    if normalization == "offline_solution":
        ylabel = r"$\frac{\|\hat{S} - S^*\|_F^2}{\|S^* - S_{\mathrm{offline}}\|_F^2}$"
    else:
        ylabel = r"$\frac{\|\hat{S} - S^*\|_F^2}{\|S^*\|_F^2}$"
    if divide_by_n2:
        ylabel = ylabel + r"\,$/\,N^2$"
    plt.ylabel(ylabel)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()


def run_single_trial_from_meta(meta_path: Path, trial_index: int, variant_key: Optional[str], out_dir: Optional[Path]) -> List[Path]:
    meta = _load_json(meta_path)

    cfg = meta.get("config", {}) or {}
    N = int(cfg["N"])
    T = int(cfg["T"])
    sparsity = float(cfg["sparsity"])
    max_weight = float(cfg["max_weight"])
    std_e = float(cfg["std_e"])
    K = int(cfg["K"])
    trial_seeds = cfg.get("trial_seeds") or []
    if not isinstance(trial_seeds, list) or not trial_seeds:
        trial_seeds = [int(cfg.get("seed_base", 0))]
    if trial_index < 0 or trial_index >= len(trial_seeds):
        raise ValueError(f"trial_index out of range: {trial_index} (available: 0..{len(trial_seeds)-1})")
    trial_seed = int(trial_seeds[trial_index])

    gen = meta.get("generator", {}) or {}
    gen_kwargs = gen.get("kwargs", {}) or {}
    s_type = str(gen_kwargs.get("s_type", "random"))
    t_min = float(gen_kwargs.get("t_min", 0.5))
    t_max = float(gen_kwargs.get("t_max", 1.0))
    z_dist = str(gen_kwargs.get("z_dist", "uniform01"))

    methods = meta.get("methods", {}) or {}
    flags = {k: bool(v.get("enabled", False)) for k, v in methods.items() if isinstance(v, dict)}

    comp = meta.get("comparison", {}) or {}
    pc_model = str(comp.get("pc_model", "exog")).strip()
    pc_use_true_T_init = bool(comp.get("pc_use_true_T_init", True))
    pc_T_init_identity_scale = float(comp.get("pc_T_init_identity_scale", 1.0))
    pp_init_b0 = str(comp.get("pp_init_b0", "ones")).strip()
    pp_lookahead_cfg = int(comp.get("pp_lookahead", 0))

    metric = meta.get("metric", {}) or {}
    burn_in_effective = int(metric.get("burn_in_effective", 0))
    offline_lambda_l1 = metric.get("offline_lambda_l1", None)
    if offline_lambda_l1 is not None:
        offline_lambda_l1 = float(offline_lambda_l1)

    variants_all = _variants_from_meta(meta)
    variants = [v for v in variants_all if (variant_key is None or v.key == variant_key)]
    if not variants:
        raise ValueError(f"variant not found: {variant_key}")

    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    rng = np.random.default_rng(trial_seed)
    S_series, B_true, U, Y = generate_piecewise_X_with_exog(
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
        rng=rng,
    )

    # offline solution (only when needed)
    S_offline_cache: Dict[str, Optional[np.ndarray]] = {}
    for v in variants:
        if v.normalization != "offline_solution":
            S_offline_cache[v.key] = None
            continue
        if offline_lambda_l1 is None:
            raise ValueError("offline_solution requested but meta.metric.offline_lambda_l1 is None")
        S_offline_cache[v.key] = solve_offline_sem_lasso_batch(Y, U, offline_lambda_l1)

    # run methods once
    method_series: Dict[str, List[np.ndarray]] = {}

    if flags.get("pp", False):
        hp = methods["pp"]["hyperparams"]
        r = int(hp["r"])
        q = int(hp["q"])
        rho = float(hp["rho"])
        mu_lambda = float(hp["mu_lambda"])
        lambda_S = float(hp.get("lambda_S", hp.get("lambdaS", 0.0)))
        lookahead = _lookahead_effective(r, q, pp_lookahead_cfg)
        model = PPExogenousSEM(
            N,
            np.zeros((N, N)),
            _pp_b0(B_true, pp_init_b0),
            r=r,
            q=q,
            rho=rho,
            mu_lambda=mu_lambda,
            lambda_S=lambda_S,
            lookahead=lookahead,
        )
        S_hat_list, _ = model.run(Y, U)
        method_series["pp"] = S_hat_list

    if flags.get("pp_sgd", False):
        hp = methods["pp_sgd"]["hyperparams"]
        r = int(hp.get("r", 1))
        q = int(hp.get("q", 1))
        rho = float(hp["rho"])
        mu_lambda = float(hp["mu_lambda"])
        lambda_S = float(hp.get("lambda_S", hp.get("lambdaS", 0.0)))
        lookahead = _lookahead_effective(r, q, pp_lookahead_cfg)
        model = PPExogenousSEM(
            N,
            np.zeros((N, N)),
            _pp_b0(B_true, pp_init_b0),
            r=r,
            q=q,
            rho=rho,
            mu_lambda=mu_lambda,
            lambda_S=lambda_S,
            lookahead=lookahead,
        )
        S_hat_list, _ = model.run(Y, U)
        method_series["pp_sgd"] = S_hat_list

    if flags.get("pc", False):
        hp = methods["pc"]["hyperparams"]
        lambda_reg = float(hp["lambda_reg"])
        alpha = float(hp["alpha"])
        beta = float(hp["beta"])
        gamma = float(hp["gamma"])
        P = int(hp["P"])
        C = int(hp["C"])
        X = Y
        if pc_model == "noexog":
            pc = PCSEMNoExog(
                N, np.zeros((N, N)), lambda_reg, alpha, beta, gamma, P, C,
                show_progress=False, name="pc_noexog",
            )
            estimates, _ = pc.run(X, Z=None)
        else:
            pc = PCSEM(
                N,
                np.zeros((N, N)),
                lambda_reg,
                alpha,
                beta,
                gamma,
                P,
                C,
                show_progress=False,
                name="pc_baseline",
                T_init=_pc_T_init(N, B_true, pc_use_true_T_init, pc_T_init_identity_scale),
            )
            estimates, _ = pc.run(X, U)
        method_series["pc"] = estimates

    if flags.get("co", False):
        hp = methods["co"]["hyperparams"]
        lambda_reg = float(hp["lambda_reg"])
        alpha = float(hp["alpha"])
        beta_co = float(hp["beta_co"])
        gamma = float(hp["gamma"])
        C = int(hp["C"])
        X = Y
        if pc_model == "noexog":
            co = PCSEMNoExog(
                N, np.zeros((N, N)), lambda_reg, alpha, beta_co, gamma, 0, C,
                show_progress=False, name="co_noexog",
            )
            estimates, _ = co.run(X, Z=None)
        else:
            co = PCSEM(
                N,
                np.zeros((N, N)),
                lambda_reg,
                alpha,
                beta_co,
                gamma,
                0,
                C,
                show_progress=False,
                name="co_baseline",
                T_init=_pc_T_init(N, B_true, pc_use_true_T_init, pc_T_init_identity_scale),
            )
            estimates, _ = co.run(X, U)
        method_series["co"] = estimates

    if flags.get("sgd", False):
        hp = methods["sgd"]["hyperparams"]
        beta_sgd = float(hp["beta_sgd"])
        lambda_reg = float(hp.get("lambda_reg", methods.get("pc", {}).get("hyperparams", {}).get("lambda_reg", 1e-3)))
        alpha = float(hp.get("alpha", methods.get("pc", {}).get("hyperparams", {}).get("alpha", 1e-2)))
        C = int(hp.get("C", methods.get("pc", {}).get("hyperparams", {}).get("C", 1)))
        X = Y
        if pc_model == "noexog":
            sgd = PCSEMNoExog(
                N, np.zeros((N, N)), lambda_reg, alpha, beta_sgd, 0.0, 0, C,
                show_progress=False, name="sgd_noexog",
            )
            estimates, _ = sgd.run(X, Z=None)
        else:
            sgd = PCSEM(
                N,
                np.zeros((N, N)),
                lambda_reg,
                alpha,
                beta_sgd,
                0.0,
                0,
                C,
                show_progress=False,
                name="sgd_baseline",
                T_init=_pc_T_init(N, B_true, pc_use_true_T_init, pc_T_init_identity_scale),
            )
            estimates, _ = sgd.run(X, U)
        method_series["sgd"] = estimates

    if flags.get("pg", False):
        hp = methods["pg"]["hyperparams"]
        lambda_reg = float(hp["lambda_reg"])
        step_scale = float(hp.get("step_scale", 1.0))
        step_size = hp.get("step_size")
        step_size_f = float(step_size) if step_size is not None else None
        max_iter = int(hp.get("max_iter", 500))
        tol = float(hp.get("tol", 1e-4))
        use_fista = bool(hp.get("use_fista", True))
        use_backtracking = bool(hp.get("use_backtracking", False))
        X = Y
        pg_config = ProximalGradientConfig(
            lambda_reg=lambda_reg,
            step_size=step_size_f,
            step_scale=step_scale,
            max_iter=max_iter,
            tol=tol,
            use_fista=use_fista,
            use_backtracking=use_backtracking,
            show_progress=False,
            name="pg_baseline",
        )
        pg_model = ProximalGradientBatchSEM(N, pg_config)
        estimates, _ = pg_model.run(X, U)
        method_series["pg"] = estimates

    # build per-variant error series and plot
    saved: List[Path] = []
    base_out_dir = out_dir if out_dir is not None else Path(meta.get("result_dir") or meta_path.parent)

    for v in variants:
        S_offline = S_offline_cache[v.key]
        series_by_method: Dict[str, np.ndarray] = {}
        for method_name, estimates in method_series.items():
            series = compute_error_series(
                estimates,
                S_series,
                S_offline,
                v.normalization,
                v.divide_by_n2,
            )
            series_by_method[method_name] = np.array(series, dtype=float)

        baseline0 = compute_normalized_error(
            np.zeros((N, N)),
            S_series[0],
            S_offline,
            normalization=v.normalization,
            divide_by_n2=v.divide_by_n2,
        )
        for method_name, arr in series_by_method.items():
            if len(arr) > 0:
                arr[0] = float(baseline0)
                series_by_method[method_name] = arr

        base_fig = _figure_name_for_variant(meta, v)
        if base_fig is None:
            base_fig = f"piecewise_N={N}_T={T}_{v.key}.png"
        out_name = _make_output_name(base_fig, trial_seed)
        out_path = base_out_dir / out_name
        _plot_one(out_path, T=T, series_by_method=series_by_method, normalization=v.normalization, divide_by_n2=v.divide_by_n2)
        saved.append(out_path)

        if burn_in_effective > 0:
            burn_path = out_path.with_name(out_path.stem + f"_burnin{burn_in_effective}.png")
            plt.figure(figsize=(10, 6))
            for method_name, arr in series_by_method.items():
                if method_name == "co":
                    plt.plot(arr, color="blue", label="Correction Only")
                elif method_name == "pc":
                    plt.plot(arr, color="limegreen", label="Prediction Correction")
                elif method_name == "sgd":
                    plt.plot(arr, color="cyan", label="SGD")
                elif method_name == "pg":
                    plt.plot(arr, color="magenta", label="ProxGrad")
                elif method_name == "pp_sgd":
                    plt.plot(arr, color="orange", label="PP-SGD (q=1,r=1)")
                elif method_name == "pp":
                    plt.plot(arr, color="red", label="Proposed (PP)")
            plt.yscale("log")
            plt.xlim(left=burn_in_effective, right=T)
            plt.xlabel("t")
            if v.normalization == "offline_solution":
                ylabel = r"$\frac{\|\hat{S} - S^*\|_F^2}{\|S^* - S_{\mathrm{offline}}\|_F^2}$"
            else:
                ylabel = r"$\frac{\|\hat{S} - S^*\|_F^2}{\|S^*\|_F^2}$"
            if v.divide_by_n2:
                ylabel = ylabel + r"\,$/\,N^2$"
            plt.ylabel(ylabel)
            plt.grid(True, which="both")
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(burn_path), bbox_inches="tight")
            plt.close()
            saved.append(burn_path)

    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run piecewise meta.json with a single trial and re-plot.")
    p.add_argument("--meta", type=Path, required=True, help="*_meta.json path")
    p.add_argument("--trial_index", type=int, default=0, help="which trial (index in meta.config.trial_seeds)")
    p.add_argument("--trial_seed", type=int, default=None, help="override trial seed (ignores trial_index)")
    p.add_argument("--variant", type=str, default=None, help="variant key (e.g., true_n2=0). If omitted, plots all variants.")
    p.add_argument("--out_dir", type=Path, default=None, help="output directory (default: meta.result_dir or meta parent)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.trial_seed is None:
        saved = run_single_trial_from_meta(args.meta, trial_index=args.trial_index, variant_key=args.variant, out_dir=args.out_dir)
    else:
        meta = _load_json(args.meta)
        cfg = meta.get("config", {}) or {}
        cfg["trial_seeds"] = [int(args.trial_seed)]
        tmp_path = args.meta.with_suffix(".tmp.single_seed.meta.json")
        tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            saved = run_single_trial_from_meta(tmp_path, trial_index=0, variant_key=args.variant, out_dir=args.out_dir)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except TypeError:
                # py<3.8 fallback
                if tmp_path.exists():
                    tmp_path.unlink()
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
