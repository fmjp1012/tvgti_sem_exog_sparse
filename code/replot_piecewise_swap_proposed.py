"""
指定した meta.json（ベース）に対して、Proposed (PP) だけ別 meta.json のハイパラに差し替えて再プロットする。

要件:
- trial100回平均（ベース meta の trial_seeds で Proposed を再実行→平均）を出す
- 1回（単一 seed）の結果も出す（ベース meta の trial_seeds[trial_index]）

例:
    python -m code.replot_piecewise_swap_proposed \\
      --base_meta result/260127/exog_sparse_piecewise/images/*_meta.json \\
      --pp_meta   result/260129/exog_sparse_piecewise/images/*_meta.json \\
      --variant true_n2=0 \\
      --trial_index 0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from code.data_gen import generate_piecewise_X_with_exog
from code.replot_piecewise_single_trial import (
    Variant,
    _lookahead_effective,
    _pp_b0,
    run_single_trial_from_meta,
)
from models.pp_exog import PPExogenousSEM
from utils.io.plotting import apply_style
from utils.metrics import compute_error_series, compute_normalized_error
from utils.offline_solver import solve_offline_sem_lasso_batch


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _variant_from_meta(meta: Dict[str, Any], variant_key: str) -> Variant:
    metric = meta.get("metric", {}) or {}
    variants = metric.get("plot_variants") or []
    for v in variants:
        if str(v.get("key")) == str(variant_key):
            return Variant(
                key=str(v["key"]),
                normalization=str(v["normalization"]),
                divide_by_n2=bool(v["divide_by_n2"]),
            )
    raise KeyError(f"variant {variant_key!r} not found in meta.metric.plot_variants")


def _figure_name_for_variant(meta: Dict[str, Any], variant_key: str) -> Optional[str]:
    figures = (meta.get("results", {}) or {}).get("figures") or []
    for rec in figures:
        if str(rec.get("key")) == str(variant_key):
            fig = rec.get("figure")
            return str(fig) if fig else None
    return None


def _format_float_for_name(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("+", "").replace(".", "p").replace("-", "m")


def _build_ppswap_suffix(pp_hp: Dict[str, Any]) -> str:
    r = int(pp_hp["r"])
    q = int(pp_hp["q"])
    rho = float(pp_hp["rho"])
    mu_lambda = float(pp_hp["mu_lambda"])
    lambda_S = float(pp_hp.get("lambda_S", pp_hp.get("lambdaS", 0.0)))
    return (
        f"_ppswap_q={q}_r={r}"
        f"_rho={_format_float_for_name(rho)}"
        f"_mulambda={_format_float_for_name(mu_lambda)}"
        f"_lambdaS={_format_float_for_name(lambda_S)}"
    )


def _plot_avg(
    out_path: Path,
    *,
    T: int,
    normalization: str,
    divide_by_n2: bool,
    series_by_method: Dict[str, np.ndarray],
) -> None:
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
    plt.xlim(left=0, right=T)
    plt.xlabel("t")
    if normalization == "offline_solution":
        ylabel = r"Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^* - S_{\mathrm{offline}}\|_F^2}$"
    else:
        ylabel = r"Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^*\|_F^2}$"
    if divide_by_n2:
        ylabel = ylabel + r"\,$/\,N^2$"
    plt.ylabel(ylabel)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()


@dataclass(frozen=True)
class _BaseConfig:
    N: int
    T: int
    sparsity: float
    max_weight: float
    std_e: float
    K: int
    trial_seeds: List[int]
    generator_kwargs: Dict[str, Any]
    offline_lambda_l1: Optional[float]
    pp_lookahead_cfg: int
    pp_init_b0: str


def _extract_base_config(meta: Dict[str, Any]) -> _BaseConfig:
    cfg = meta.get("config", {}) or {}
    gen = meta.get("generator", {}) or {}
    gen_kwargs = (gen.get("kwargs", {}) or {}).copy()
    metric = meta.get("metric", {}) or {}
    comparison = meta.get("comparison", {}) or {}
    return _BaseConfig(
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        sparsity=float(cfg["sparsity"]),
        max_weight=float(cfg["max_weight"]),
        std_e=float(cfg["std_e"]),
        K=int(cfg["K"]),
        trial_seeds=[int(x) for x in (cfg.get("trial_seeds") or [])],
        generator_kwargs=gen_kwargs,
        offline_lambda_l1=float(metric["offline_lambda_l1"]) if metric.get("offline_lambda_l1") is not None else None,
        pp_lookahead_cfg=int(comparison.get("pp_lookahead", 0)),
        pp_init_b0=str(comparison.get("pp_init_b0", "true_T_diag")),
    )


def _run_pp_one_trial(
    *,
    base: _BaseConfig,
    trial_seed: int,
    variant: Variant,
    pp_hp: Dict[str, Any],
) -> np.ndarray:
    rng = np.random.default_rng(int(trial_seed))
    S_series, B_true, U, Y = generate_piecewise_X_with_exog(
        N=base.N,
        T=base.T,
        sparsity=base.sparsity,
        max_weight=base.max_weight,
        std_e=base.std_e,
        K=base.K,
        rng=rng,
        **base.generator_kwargs,
    )

    S_offline: Optional[np.ndarray] = None
    if variant.normalization == "offline_solution":
        if base.offline_lambda_l1 is None:
            raise ValueError("offline_solution requested but meta.metric.offline_lambda_l1 is None")
        S_offline = solve_offline_sem_lasso_batch(Y, U, base.offline_lambda_l1)

    r = int(pp_hp["r"])
    q = int(pp_hp["q"])
    rho = float(pp_hp["rho"])
    mu_lambda = float(pp_hp["mu_lambda"])
    lambda_S = float(pp_hp.get("lambda_S", pp_hp.get("lambdaS", 0.0)))
    lookahead = _lookahead_effective(r, q, base.pp_lookahead_cfg)
    model = PPExogenousSEM(
        base.N,
        np.zeros((base.N, base.N)),
        _pp_b0(B_true, base.pp_init_b0),
        r=r,
        q=q,
        rho=rho,
        mu_lambda=mu_lambda,
        lambda_S=lambda_S,
        lookahead=lookahead,
    )
    S_hat_list, _ = model.run(Y, U)

    series = np.array(
        compute_error_series(
            S_hat_list,
            S_series,
            S_offline,
            variant.normalization,
            variant.divide_by_n2,
        ),
        dtype=float,
    )

    baseline0 = compute_normalized_error(
        np.zeros((base.N, base.N)),
        S_series[0],
        S_offline,
        normalization=variant.normalization,
        divide_by_n2=variant.divide_by_n2,
    )
    if len(series) > 0:
        series[0] = float(baseline0)
    return series


def _update_meta_for_ppswap(
    base_meta: Dict[str, Any],
    *,
    out_dir: Path,
    variant_key: str,
    pp_hp: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    meta = copy.deepcopy(base_meta)
    suffix = _build_ppswap_suffix(pp_hp)

    methods = meta.get("methods", {}) or {}
    if "pp" not in methods:
        raise KeyError("base meta has no methods.pp")
    methods["pp"]["hyperparams"] = dict(pp_hp)
    meta["methods"] = methods

    # default out_dir for replot_piecewise_single_trial
    meta["result_dir"] = str(out_dir)

    # Keep filenames under common filesystem limits (<=255 bytes).
    # Single-trial replot will append `_trial_seed=...` and may also rewrite `num_trials=...`.
    cfg = meta.get("config", {}) or {}
    v = _variant_from_meta(meta, variant_key)
    norm_tag = "offline" if v.normalization == "offline_solution" else "true"
    n2 = int(bool(v.divide_by_n2))
    new_base_fig = (
        f"piecewise_K={int(cfg['K'])}_N={int(cfg['N'])}_T={int(cfg['T'])}"
        f"_maxweight={float(cfg['max_weight'])}_stde={float(cfg['std_e'])}"
        f"_n2={n2}_norm={norm_tag}_num_trials={int(cfg.get('num_trials', 0))}"
        f"_seed={int(cfg.get('seed_base', cfg.get('trial_seeds', [0])[0] if cfg.get('trial_seeds') else 0))}"
        f"{suffix}.png"
    )

    # results.figures の該当 figure を差し替え（single-trial の出力名にも使われる）
    figures = (meta.get("results", {}) or {}).get("figures") or []
    for rec in figures:
        if str(rec.get("key")) == str(variant_key):
            rec["figure"] = new_base_fig
            rec["figure_path"] = str(out_dir / new_base_fig)

    return meta, new_base_fig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swap Proposed (PP) hyperparams and replot avg + single-trial.")
    p.add_argument("--base_meta", type=Path, required=True, help="base *_meta.json (baselines taken from here)")
    p.add_argument("--pp_meta", type=Path, required=True, help="pp *_meta.json (pp hyperparams taken from here)")
    p.add_argument("--variant", type=str, default="true_n2=0", help="variant key (default: true_n2=0)")
    p.add_argument("--trial_index", type=int, default=0, help="single trial index in base_meta.config.trial_seeds")
    p.add_argument("--out_dir", type=Path, default=None, help="output directory (default: base_meta.result_dir)")
    p.add_argument("--n_jobs", type=int, default=-1, help="joblib n_jobs for PP rerun")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_meta = _load_json(args.base_meta)
    pp_meta = _load_json(args.pp_meta)

    pp_hp = ((pp_meta.get("methods", {}) or {}).get("pp", {}) or {}).get("hyperparams")
    if not isinstance(pp_hp, dict) or not pp_hp:
        raise ValueError("pp_meta.methods.pp.hyperparams is missing/empty")

    base_cfg = _extract_base_config(base_meta)
    variant = _variant_from_meta(base_meta, args.variant)

    out_dir = args.out_dir if args.out_dir is not None else Path(base_meta.get("result_dir") or args.base_meta.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare updated meta (used for single-trial rerun + figure naming)
    swapped_meta, base_fig_name = _update_meta_for_ppswap(
        base_meta,
        out_dir=out_dir,
        variant_key=args.variant,
        pp_hp=pp_hp,
    )
    meta_id = hashlib.sha1(
        f"{args.base_meta.resolve()}|{args.pp_meta.resolve()}|{args.variant}|{_build_ppswap_suffix(pp_hp)}".encode("utf-8")
    ).hexdigest()[:10]
    swapped_meta_path = out_dir / f"ppswap_{meta_id}_{args.variant}{_build_ppswap_suffix(pp_hp)}.meta.json"
    swapped_meta_path.write_text(json.dumps(swapped_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    # rerun PP over base trial seeds and average
    if not base_cfg.trial_seeds:
        raise ValueError("base_meta.config.trial_seeds is empty")
    pp_series = Parallel(n_jobs=int(args.n_jobs), prefer="processes")(
        delayed(_run_pp_one_trial)(base=base_cfg, trial_seed=s, variant=variant, pp_hp=pp_hp) for s in base_cfg.trial_seeds
    )
    pp_mean = np.mean(np.stack(pp_series, axis=0), axis=0)

    # baselines from base meta (already averaged)
    metrics_by_variant = ((base_meta.get("results", {}) or {}).get("metrics_by_variant") or {})
    base_variant_metrics = metrics_by_variant.get(args.variant) or {}
    base_methods = base_meta.get("methods", {}) or {}

    series_by_method: Dict[str, np.ndarray] = {}
    for key in ["co", "pc", "sgd", "pg", "pp_sgd"]:
        if (base_methods.get(key, {}) or {}).get("enabled") and key in base_variant_metrics:
            series_by_method[key] = np.array(base_variant_metrics[key], dtype=float)
    series_by_method["pp"] = np.array(pp_mean, dtype=float)

    out_avg = out_dir / base_fig_name
    _plot_avg(
        out_avg,
        T=base_cfg.T,
        normalization=variant.normalization,
        divide_by_n2=variant.divide_by_n2,
        series_by_method=series_by_method,
    )

    # single trial rerun (all enabled methods, with swapped pp hyperparams)
    saved = run_single_trial_from_meta(swapped_meta_path, trial_index=int(args.trial_index), variant_key=args.variant, out_dir=out_dir)

    print(out_avg)
    for p in saved:
        print(p)
    print(swapped_meta_path)


if __name__ == "__main__":
    main()
