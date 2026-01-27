"""
Piecewise シナリオのシミュレーション実行スクリプト

設定は code/config.py で一元管理されています。
このスクリプトを実行する前に config.py を編集して設定を変更してください。

使用方法:
    python -m code.run_piecewise
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from code.config import get_config, get_default_hyperparams_dict, print_config_summary, config_to_dict
from code.data_gen import generate_piecewise_X_with_exog
from models.pp_exog import PPExogenousSEM
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from models.tvgti_pc.prediction_correction_sem_noexog import PredictionCorrectionSEMNoExog as PCSEMNoExog
from utils.io.plotting import apply_style, plot_heatmaps, plot_heatmaps_suite
from utils.io.results import backup_script, create_result_dir, make_result_filename, save_json
from utils.offline_solver import solve_offline_sem_lasso_batch
from utils.metrics import compute_error_series
from utils.metrics import compute_normalized_error
from utils.repro import collect_environment_info


def _get_git_info(repo_root: Path) -> Dict[str, object]:
    """
    再現性のために git の状態をメタに埋める。
    （gitが無い/失敗しても実行は継続する）
    """
    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "head": head,
        "branch": branch,
        "is_dirty": bool(dirty),
    }

def _fmt(value: object) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _print_block(title: str, items: Dict[str, object]) -> None:
    if not items:
        return
    print(f"--- {title} ---")
    for key, value in items.items():
        print(f"{key:>24}: {_fmt(value)}")


def print_piecewise_summary(
    num_trials: int,
    run_flags: Dict[str, bool],
    hyperparams: Dict[str, Dict[str, Any]],
    hyperparam_path: Optional[Path],
) -> None:
    """実行設定のサマリーを表示"""
    cfg = get_config()
    
    print("=== Experiment Configuration (from config.py) ===")
    common_items: Dict[str, object] = {
        "Scenario": "piecewise",
        "Hyperparam JSON": str(hyperparam_path) if hyperparam_path else "<default>",
        "Num Trials": num_trials,
        "Seed (base)": cfg.common.seed,
        "N": cfg.common.N,
        "T": cfg.common.T,
        "sparsity": cfg.common.sparsity,
        "max_weight": cfg.common.max_weight,
        "std_e": cfg.common.std_e,
    }
    _print_block("Common Parameters", common_items)
    _print_block("Scenario Parameters", {"K": cfg.piecewise.K})
    
    flag_items = {name.upper(): "ON" if v else "OFF" for name, v in run_flags.items()}
    _print_block("Run Flags", flag_items)
    
    metric_items: Dict[str, object] = {
        "error_normalization": cfg.metric.error_normalization,
        "burn_in": getattr(cfg.metric, "burn_in", 0),
    }
    if cfg.metric.error_normalization == "offline_solution":
        # offline_lambda_l1 はハイパラJSONまたは探索範囲から取得
        offline_space = cfg.search_spaces.offline.offline_lambda_l1
        metric_items["offline_lambda_l1 (range)"] = f"[{offline_space.low}, {offline_space.high}]"
    _print_block("Metric Settings", metric_items)

    # 比較条件
    comp = getattr(cfg, "comparison", None)
    if comp is not None:
        _print_block("Comparison Settings", {
            "pc_use_true_T_init": getattr(comp, "pc_use_true_T_init", True),
            "pc_T_init_identity_scale": getattr(comp, "pc_T_init_identity_scale", 1.0),
            "pp_init_b0": getattr(comp, "pp_init_b0", "ones"),
            "pp_lookahead": getattr(comp, "pp_lookahead", 0),
        })
    
    for method_key, params in hyperparams.items():
        label = f"{method_key.upper()} Hyperparams"
        _print_block(label, params)
    
    _print_block("Data Generation", {
        "s_type": cfg.data_gen.s_type,
        "t_min": cfg.data_gen.t_min,
        "t_max": cfg.data_gen.t_max,
        "z_dist": cfg.data_gen.z_dist,
    })
    print("------------------------------")


def load_hyperparams(json_path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    """ハイパーパラメータJSONを読み込む"""
    if json_path is None:
        return None
    if not json_path.is_file():
        raise FileNotFoundError(f"ハイパラJSONが見つかりません: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(hyperparam_path: Optional[Path] = None) -> None:
    """メイン処理"""
    if len(sys.argv) > 1:
        raise SystemExit("CLI 引数は使用できません。code/config.py を編集してください。")
    # config.py 側の hyperparam_json を使う
    # config.py から設定を取得
    cfg = get_config()
    if hyperparam_path is None and getattr(cfg, "hyperparam_json", None) is not None:
        hyperparam_path = cfg.hyperparam_json
    
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)
    run_wall_start = time.perf_counter()
    
    # 手法フラグ
    run_pp_flag = cfg.methods.pp
    run_pp_sgd_flag = getattr(cfg.methods, "pp_sgd", False)
    run_pc_flag = cfg.methods.pc
    run_co_flag = cfg.methods.co
    run_sgd_flag = cfg.methods.sgd
    run_pg_flag = cfg.methods.pg
    
    num_trials = cfg.run.num_trials
    
    # シミュレーションパラメータ
    N = cfg.common.N
    T = cfg.common.T
    sparsity = cfg.common.sparsity
    max_weight = cfg.common.max_weight
    std_e = cfg.common.std_e
    K = cfg.piecewise.K
    seed = cfg.common.seed
    
    # ハイパーパラメータ読み込み
    loaded_hyperparams = load_hyperparams(hyperparam_path)
    default_hyperparams = get_default_hyperparams_dict()
    hyperparams = loaded_hyperparams if loaded_hyperparams else default_hyperparams
    
    pp_cfg = hyperparams.get("pp", {})
    pp_sgd_cfg = hyperparams.get("pp_sgd", {})
    pc_cfg = hyperparams.get("pc", {})
    co_cfg = hyperparams.get("co", {})
    sgd_cfg = hyperparams.get("sgd", {})
    pg_cfg = hyperparams.get("pg", {})
    
    # PP法のハイパーパラメータ
    r = int(pp_cfg.get("r", cfg.hyperparams.pp.r))
    q = int(pp_cfg.get("q", cfg.hyperparams.pp.q))
    rho = float(pp_cfg.get("rho", cfg.hyperparams.pp.rho))
    mu_lambda = float(pp_cfg.get("mu_lambda", cfg.hyperparams.pp.mu_lambda))
    lambda_S = float(pp_cfg.get("lambda_S", cfg.hyperparams.pp.lambda_S))

    # PP-SGD（q=1,r=1固定）のハイパーパラメータ
    r_pp_sgd = 1
    q_pp_sgd = 1
    rho_pp_sgd = float(pp_sgd_cfg.get("rho", cfg.hyperparams.pp_sgd.rho))
    mu_lambda_pp_sgd = float(pp_sgd_cfg.get("mu_lambda", cfg.hyperparams.pp_sgd.mu_lambda))
    lambda_S_pp_sgd = float(pp_sgd_cfg.get("lambda_S", cfg.hyperparams.pp_sgd.lambda_S))
    
    # PC法のハイパーパラメータ
    lambda_reg = float(pc_cfg.get("lambda_reg", cfg.hyperparams.pc.lambda_reg))
    alpha = float(pc_cfg.get("alpha", cfg.hyperparams.pc.alpha))
    beta = float(pc_cfg.get("beta", cfg.hyperparams.pc.beta))
    gamma = float(pc_cfg.get("gamma", cfg.hyperparams.pc.gamma))
    P = int(pc_cfg.get("P", cfg.hyperparams.pc.P))
    C = int(pc_cfg.get("C", cfg.hyperparams.pc.C))
    
    # CO/SGD法のハイパーパラメータ
    beta_co = float(co_cfg.get("beta_co", cfg.hyperparams.co.beta_co))
    beta_sgd = float(sgd_cfg.get("beta_sgd", cfg.hyperparams.sgd.beta_sgd))

    # SGD specific params (if provided in sgd config, otherwise use shared)
    sgd_lambda_reg = float(sgd_cfg.get("lambda_reg", lambda_reg))
    sgd_alpha = float(sgd_cfg.get("alpha", alpha))
    
    # PG法のハイパーパラメータ
    lambda_pg = float(pg_cfg.get("lambda_reg", cfg.hyperparams.pg.lambda_reg))
    step_scale_pg = float(pg_cfg.get("step_scale", cfg.hyperparams.pg.step_scale))
    step_size_pg_raw = pg_cfg.get("step_size", cfg.hyperparams.pg.step_size)
    step_size_pg = float(step_size_pg_raw) if step_size_pg_raw is not None else None
    max_iter_pg = int(pg_cfg.get("max_iter", cfg.hyperparams.pg.max_iter))
    tol_pg = float(pg_cfg.get("tol", cfg.hyperparams.pg.tol))
    use_fista_pg = _coerce_bool(pg_cfg.get("use_fista", cfg.hyperparams.pg.use_fista), default=True)
    use_backtracking_pg = _coerce_bool(pg_cfg.get("use_backtracking", cfg.hyperparams.pg.use_backtracking), default=False)
    
    print_piecewise_summary(
        num_trials=num_trials,
        run_flags={
            "pp": run_pp_flag,
            "pp_sgd": run_pp_sgd_flag,
            "pc": run_pc_flag,
            "co": run_co_flag,
            "sgd": run_sgd_flag,
            "pg": run_pg_flag,
        },
        hyperparams={
            "pp": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda, "lambda_S": lambda_S},
            "pp_sgd": {"r": r_pp_sgd, "q": q_pp_sgd, "rho": rho_pp_sgd, "mu_lambda": mu_lambda_pp_sgd, "lambda_S": lambda_S_pp_sgd},
            "pc": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C},
            "co": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C},
            "sgd": {"lambda_reg": sgd_lambda_reg, "alpha": sgd_alpha, "beta_sgd": beta_sgd, "C": C},
            "pg": {
                "lambda_reg": lambda_pg,
                "step_scale": step_scale_pg,
                "step_size": step_size_pg,
                "use_fista": use_fista_pg,
                "use_backtracking": use_backtracking_pg,
                "max_iter": max_iter_pg,
                "tol": tol_pg,
            },
        },
        hyperparam_path=hyperparam_path,
    )
    
    S0_pc = np.zeros((N, N))
    
    # 評価指標の設定
    error_normalization = cfg.metric.error_normalization
    burn_in_cfg = int(getattr(cfg.metric, "burn_in", 0))
    divide_by_n2 = bool(getattr(cfg.metric, "divide_by_n2", False))

    def _variant_key(normalization: str, div_n2: bool) -> str:
        norm_tag = "true" if normalization == "true_value" else "offline"
        return f"{norm_tag}_n2={int(div_n2)}"

    plot_variants = [
        {"normalization": "true_value", "divide_by_n2": False},
        {"normalization": "true_value", "divide_by_n2": True},
        {"normalization": "offline_solution", "divide_by_n2": False},
        {"normalization": "offline_solution", "divide_by_n2": True},
    ]
    for variant in plot_variants:
        variant["key"] = _variant_key(variant["normalization"], variant["divide_by_n2"])

    primary_variant_key = _variant_key(error_normalization, divide_by_n2)

    # offline_lambda_l1 の取得（ハイパラJSONから読み込むか、探索範囲の幾何平均を使用）
    offline_lambda_l1 = None
    if any(v["normalization"] == "offline_solution" for v in plot_variants):
        if hyperparams.get("offline_lambda_l1") is not None:
            offline_lambda_l1 = float(hyperparams["offline_lambda_l1"])
        else:
            # 探索範囲の幾何平均をデフォルトとして使用（対数スケール）
            offline_space = cfg.search_spaces.offline.offline_lambda_l1
            import math
            offline_lambda_l1 = math.sqrt(offline_space.low * offline_space.high)
    
    # burn-in（序盤の“データ不足”区間を除外して平均を見る）
    if burn_in_cfg == -1:
        burn_in = int(r) + int(q) - 2
    else:
        burn_in = max(0, int(burn_in_cfg))

    comp = getattr(cfg, "comparison", None)
    pc_model = "exog" if comp is None else str(getattr(comp, "pc_model", "exog")).strip()
    pc_use_true_T = True if comp is None else bool(getattr(comp, "pc_use_true_T_init", True))
    pc_T_scale = 1.0 if comp is None else float(getattr(comp, "pc_T_init_identity_scale", 1.0))
    pp_init_b0_mode = "ones" if comp is None else str(getattr(comp, "pp_init_b0", "ones")).strip()
    pp_lookahead_cfg = 0 if comp is None else int(getattr(comp, "pp_lookahead", 0))
    pp_lookahead = (int(r) + int(q) - 2) if pp_lookahead_cfg == -1 else max(0, int(pp_lookahead_cfg))
    pp_sgd_lookahead = (int(r_pp_sgd) + int(q_pp_sgd) - 2) if pp_lookahead_cfg == -1 else max(0, int(pp_lookahead_cfg))

    # 結果ディレクトリ作成
    result_dir = create_result_dir(cfg.output.result_root, cfg.output.subdir_piecewise, extra_tag='images')

    def _pc_T_init(T_true: np.ndarray) -> np.ndarray:
        return T_true if pc_use_true_T else (np.eye(N) * pc_T_scale)

    def _pp_b0(T_true: np.ndarray) -> np.ndarray:
        if pp_init_b0_mode == "true_T_diag":
            return np.diag(T_true)
        return np.ones(N)
    
    def run_trial(trial_seed: int):
        t0 = time.perf_counter()
        rng = np.random.default_rng(trial_seed)
        S_series, B_true, U, Y = generate_piecewise_X_with_exog(
            N=N,
            T=T,
            sparsity=sparsity,
            max_weight=max_weight,
            std_e=std_e,
            K=K,
            s_type=cfg.data_gen.s_type,
            t_min=cfg.data_gen.t_min,
            t_max=cfg.data_gen.t_max,
            z_dist=cfg.data_gen.z_dist,
            rng=rng,
        )
        t1 = time.perf_counter()
        errors = {v["key"]: {} for v in plot_variants}
        estimates_final = {"True": S_series[-1]}
        
        # オフライン解を計算（必要な場合）
        S_offline = None
        if any(v["normalization"] == "offline_solution" for v in plot_variants):
            S_offline = solve_offline_sem_lasso_batch(Y, U, offline_lambda_l1)
            estimates_final["Offline"] = S_offline
        t2 = time.perf_counter()
        
        if run_pp_flag:
            S0 = np.zeros((N, N))
            b0 = _pp_b0(B_true)
            model = PPExogenousSEM(
                N, S0, b0,
                r=r, q=q, rho=rho, mu_lambda=mu_lambda, lambda_S=lambda_S,
                lookahead=pp_lookahead,
            )
            S_hat_list, _ = model.run(Y, U)
            for variant in plot_variants:
                errors[variant["key"]]["pp"] = compute_error_series(
                    S_hat_list,
                    S_series,
                    S_offline,
                    variant["normalization"],
                    variant["divide_by_n2"],
                )
            estimates_final['PP'] = S_hat_list[-1]

        if run_pp_sgd_flag:
            S0 = np.zeros((N, N))
            b0 = _pp_b0(B_true)
            model = PPExogenousSEM(
                N, S0, b0,
                r=r_pp_sgd, q=q_pp_sgd,
                rho=rho_pp_sgd, mu_lambda=mu_lambda_pp_sgd, lambda_S=lambda_S_pp_sgd,
                lookahead=pp_sgd_lookahead,
            )
            S_hat_list, _ = model.run(Y, U)
            for variant in plot_variants:
                errors[variant["key"]]["pp_sgd"] = compute_error_series(
                    S_hat_list,
                    S_series,
                    S_offline,
                    variant["normalization"],
                    variant["divide_by_n2"],
                )
            estimates_final["PP-SGD"] = S_hat_list[-1]
        
        if run_pc_flag:
            X = Y
            if pc_model == "noexog":
                pc = PCSEMNoExog(
                    N, S0_pc, lambda_reg, alpha, beta, gamma, P, C,
                    show_progress=False, name="pc_noexog",
                )
                estimates_pc, _ = pc.run(X, Z=None)
            else:
                pc = PCSEM(
                    N, S0_pc, lambda_reg, alpha, beta, gamma, P, C,
                    show_progress=False, name="pc_baseline", T_init=_pc_T_init(B_true),
                )
                estimates_pc, _ = pc.run(X, U)
            for variant in plot_variants:
                errors[variant["key"]]["pc"] = compute_error_series(
                    estimates_pc,
                    S_series,
                    S_offline,
                    variant["normalization"],
                    variant["divide_by_n2"],
                )
            estimates_final['PC'] = estimates_pc[-1]
        
        if run_co_flag:
            X = Y
            if pc_model == "noexog":
                co = PCSEMNoExog(
                    N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C,
                    show_progress=False, name="co_noexog",
                )
                estimates_co, _ = co.run(X, Z=None)
            else:
                co = PCSEM(
                    N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C,
                    show_progress=False, name="co_baseline", T_init=_pc_T_init(B_true),
                )
                estimates_co, _ = co.run(X, U)
            for variant in plot_variants:
                errors[variant["key"]]["co"] = compute_error_series(
                    estimates_co,
                    S_series,
                    S_offline,
                    variant["normalization"],
                    variant["divide_by_n2"],
                )
            estimates_final['CO'] = estimates_co[-1]
        
        if run_sgd_flag:
            X = Y
            if pc_model == "noexog":
                sgd = PCSEMNoExog(
                    N, S0_pc, sgd_lambda_reg, sgd_alpha, beta_sgd, 0.0, 0, C,
                    show_progress=False, name="sgd_noexog",
                )
                estimates_sgd, _ = sgd.run(X, Z=None)
            else:
                sgd = PCSEM(
                    N, S0_pc, sgd_lambda_reg, sgd_alpha, beta_sgd, 0.0, 0, C,
                    show_progress=False, name="sgd_baseline", T_init=_pc_T_init(B_true),
                )
                estimates_sgd, _ = sgd.run(X, U)
            for variant in plot_variants:
                errors[variant["key"]]["sgd"] = compute_error_series(
                    estimates_sgd,
                    S_series,
                    S_offline,
                    variant["normalization"],
                    variant["divide_by_n2"],
                )
            estimates_final['SGD'] = estimates_sgd[-1]
        
        if run_pg_flag:
            X = Y
            pg_config = ProximalGradientConfig(
                lambda_reg=lambda_pg,
                step_size=step_size_pg,
                step_scale=step_scale_pg,
                max_iter=max_iter_pg,
                tol=tol_pg,
                use_fista=use_fista_pg,
                use_backtracking=use_backtracking_pg,
                show_progress=False,
                name="pg_baseline",
            )
            pg_model = ProximalGradientBatchSEM(N, pg_config)
            estimates_pg, _ = pg_model.run(X, U)
            for variant in plot_variants:
                errors[variant["key"]]["pg"] = compute_error_series(
                    estimates_pg,
                    S_series,
                    S_offline,
                    variant["normalization"],
                    variant["divide_by_n2"],
                )
            estimates_final['PG'] = estimates_pg[-1]

        # 全手法で t=0 の誤差を同じ初期値（S0=0）に揃える
        for variant in plot_variants:
            baseline0 = compute_normalized_error(
                np.zeros((N, N)),
                S_series[0],
                S_offline,
                normalization=variant["normalization"],
                divide_by_n2=variant["divide_by_n2"],
            )
            v_errors = errors[variant["key"]]
            for method_name in list(v_errors.keys()):
                if v_errors[method_name]:
                    v_errors[method_name][0] = float(baseline0)
        t3 = time.perf_counter()
        timing = {
            "total_sec": t3 - t0,
            "data_gen_sec": t1 - t0,
            "offline_sec": t2 - t1,
            "methods_sec": t3 - t2,
        }
        return errors, estimates_final, timing
    
    trial_seeds = [seed + i for i in range(num_trials)]
    method_flags = {
        "pp": run_pp_flag,
        "pp_sgd": run_pp_sgd_flag,
        "pc": run_pc_flag,
        "co": run_co_flag,
        "sgd": run_sgd_flag,
        "pg": run_pg_flag,
    }
    error_totals: Dict[str, Dict[str, np.ndarray]] = {}
    for variant in plot_variants:
        totals: Dict[str, np.ndarray] = {}
        for method_name, enabled in method_flags.items():
            if enabled:
                totals[method_name] = np.zeros(T)
        error_totals[variant["key"]] = totals
    
    with tqdm_joblib(tqdm(desc="Progress", total=num_trials)):
        results = Parallel(n_jobs=-1, batch_size=1, prefer="threads")(
            delayed(run_trial)(ts) for ts in trial_seeds
        )
    
    last_estimates = None
    trial_timings: Dict[str, Dict[str, float]] = {}
    for (errs, estimates_final, timing), trial_seed in zip(results, trial_seeds):
        trial_timings[str(trial_seed)] = timing
        for variant in plot_variants:
            v_key = variant["key"]
            for method_name, enabled in method_flags.items():
                if not enabled:
                    continue
                error_totals[v_key][method_name] += np.array(errs[v_key][method_name])
        last_estimates = estimates_final
    
    error_means: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for variant in plot_variants:
        v_key = variant["key"]
        v_means: Dict[str, Optional[np.ndarray]] = {}
        for method_name, enabled in method_flags.items():
            if enabled:
                v_means[method_name] = error_totals[v_key][method_name] / num_trials
            else:
                v_means[method_name] = None
        error_means[v_key] = v_means

    def _mean_after_burnin(arr: Optional[np.ndarray]) -> Optional[float]:
        if arr is None:
            return None
        if burn_in <= 0:
            return float(np.mean(arr))
        if burn_in >= len(arr):
            return None
        return float(np.mean(arr[burn_in:]))

    summary_means_by_variant: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for variant in plot_variants:
        v_key = variant["key"]
        v_means = error_means[v_key]
        summary_means_by_variant[v_key] = {
            "full_mean": {
                "pp": float(np.mean(v_means["pp"])) if run_pp_flag else None,
                "pp_sgd": float(np.mean(v_means["pp_sgd"])) if run_pp_sgd_flag else None,
                "pc": float(np.mean(v_means["pc"])) if run_pc_flag else None,
                "co": float(np.mean(v_means["co"])) if run_co_flag else None,
                "sgd": float(np.mean(v_means["sgd"])) if run_sgd_flag else None,
                "pg": float(np.mean(v_means["pg"])) if run_pg_flag else None,
            },
            "post_burnin_mean": {
                "pp": _mean_after_burnin(v_means["pp"]),
                "pp_sgd": _mean_after_burnin(v_means["pp_sgd"]),
                "pc": _mean_after_burnin(v_means["pc"]),
                "co": _mean_after_burnin(v_means["co"]),
                "sgd": _mean_after_burnin(v_means["sgd"]),
                "pg": _mean_after_burnin(v_means["pg"]),
            },
        }

    summary_means = summary_means_by_variant.get(primary_variant_key, {})

    timing_summary: Dict[str, Optional[float]] = {}
    if trial_timings:
        totals = [v.get("total_sec", 0.0) for v in trial_timings.values()]
        timing_summary = {
            "total_sec_sum": float(np.sum(totals)) if totals else None,
            "total_sec_mean": float(np.mean(totals)) if totals else None,
            "total_sec_median": float(np.median(totals)) if totals else None,
            "total_sec_min": float(np.min(totals)) if totals else None,
            "total_sec_max": float(np.max(totals)) if totals else None,
        }
    
    run_started_at = datetime.now()
    figure_records = []
    for variant in plot_variants:
        v_key = variant["key"]
        v_means = error_means[v_key]
        plt.figure(figsize=(10, 6))
        if run_co_flag:
            plt.plot(v_means["co"], color='blue', label='Correction Only')
        if run_pc_flag:
            plt.plot(v_means["pc"], color='limegreen', label='Prediction Correction')
        if run_sgd_flag:
            plt.plot(v_means["sgd"], color='cyan', label='SGD')
        if run_pg_flag:
            plt.plot(v_means["pg"], color='magenta', label='ProxGrad')
        if run_pp_sgd_flag:
            plt.plot(v_means["pp_sgd"], color='orange', label='PP-SGD (q=1,r=1)')
        if run_pp_flag:
            plt.plot(v_means["pp"], color='red', label='Proposed (PP)')
        plt.yscale('log')
        plt.xlim(left=0, right=T)
        plt.xlabel('t')
        if variant["normalization"] == "offline_solution":
            ylabel = r'Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^* - S_{\mathrm{offline}}\|_F^2}$'
        else:
            ylabel = r'Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^*\|_F^2}$'
        if variant["divide_by_n2"]:
            ylabel = ylabel + r'\,$/\,N^2$'
        plt.ylabel(ylabel)
        plt.grid(True, which='both')
        plt.legend()

        filename = make_result_filename(
            prefix="piecewise",
            params={
                "N": N, "T": T, "num_trials": num_trials,
                "maxweight": max_weight, "stde": std_e, "K": K,
                "seed": seed, "r": r, "q": q, "rho": rho, "mulambda": mu_lambda, "lambdaS": lambda_S,
                "norm": "true" if variant["normalization"] == "true_value" else "offline",
                "n2": int(variant["divide_by_n2"]),
            },
            suffix=".png",
        )
        print(filename)
        plt.tight_layout()
        plt.savefig(str(Path(result_dir) / filename), bbox_inches='tight')
        plt.show()
        figure_records.append({
            "key": v_key,
            "normalization": variant["normalization"],
            "divide_by_n2": variant["divide_by_n2"],
            "figure": filename,
            "figure_path": str(Path(result_dir) / filename),
        })
    
    # ヒートマップ表示（最後の試行の最終時刻）
    # 3種類のヒートマップを生成：全体、推定のみ、差分
    cfg = get_config()
    if cfg.output.save_heatmap and last_estimates is not None:
        heatmap_filename = filename.replace(".png", "_heatmap.png")
        use_offline_ref = error_normalization == "offline_solution"
        plot_heatmaps_suite(
            matrices=last_estimates,
            base_save_path=Path(result_dir) / heatmap_filename,
            title_suffix=f"at t={T-1} (last trial)",
            show=True,
            use_offline_as_reference=use_offline_ref,
        )
    
    # メタデータ保存
    scripts_dir = Path(result_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copies: Dict[str, str] = {}
    
    run_script_copy = backup_script(Path(__file__), scripts_dir)
    script_copies["run_piecewise"] = str(run_script_copy)
    
    config_path = Path(__file__).resolve().parent / "config.py"
    if config_path.exists():
        config_copy = backup_script(config_path, scripts_dir)
        script_copies["config"] = str(config_copy)
    
    data_gen_path = Path(__file__).resolve().parent / "data_gen.py"
    if data_gen_path.exists():
        data_gen_copy = backup_script(data_gen_path, scripts_dir)
        script_copies["data_gen"] = str(data_gen_copy)
    
    if hyperparam_path is not None and hyperparam_path.is_file():
        hyper_copy = backup_script(hyperparam_path, scripts_dir)
        script_copies["hyperparams_json"] = str(hyper_copy)

    env_info = collect_environment_info(
        package_names=[
            "numpy",
            "scipy",
            "cvxpy",
            "optuna",
            "matplotlib",
            "joblib",
            "tqdm",
            "tqdm-joblib",
            "networkx",
        ]
    )
    
    metadata = {
        "created_at": run_started_at.isoformat(),
        "command": sys.argv,
        "repro": {
            # 実験の再現に必要な“環境側”情報
            "git": _get_git_info(Path(__file__).resolve().parents[1]),
            # 与えたハイパラJSONの中身（パスが移動しても再現できるように）
            "hyperparam_json_content": loaded_hyperparams,
            "environment": env_info,
        },
        "config": {
            "num_trials": num_trials,
            "seed_base": seed,
            "trial_seeds": trial_seeds,
            "N": N, "T": T, "sparsity": sparsity,
            "max_weight": max_weight, "std_e": std_e, "K": K,
        },
        # config.py の全設定スナップショット（CONFIG_MAIN/TEST の全フィールド）
        "config_full": config_to_dict(cfg),
        "metric": {
            "error_normalization": error_normalization,
            "offline_lambda_l1": offline_lambda_l1,
            "burn_in": burn_in_cfg,
            "burn_in_effective": burn_in,
            "divide_by_n2": divide_by_n2,
            "plot_variants": [
                {
                    "key": v["key"],
                    "normalization": v["normalization"],
                    "divide_by_n2": v["divide_by_n2"],
                }
                for v in plot_variants
            ],
        },
        "comparison": {
            "pc_model": pc_model,
            "pc_use_true_T_init": pc_use_true_T,
            "pc_T_init_identity_scale": pc_T_scale,
            "pp_init_b0": pp_init_b0_mode,
            "pp_lookahead": pp_lookahead_cfg,
            "pp_lookahead_effective": pp_lookahead,
        },
        "methods": {
            "pp": {"enabled": run_pp_flag, "hyperparams": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda, "lambda_S": lambda_S}},
            "pp_sgd": {"enabled": run_pp_sgd_flag, "hyperparams": {"r": r_pp_sgd, "q": q_pp_sgd, "rho": rho_pp_sgd, "mu_lambda": mu_lambda_pp_sgd, "lambda_S": lambda_S_pp_sgd}},
            "pc": {"enabled": run_pc_flag, "hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C}},
            "co": {"enabled": run_co_flag, "hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C}},
            "sgd": {"enabled": run_sgd_flag, "hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_sgd": beta_sgd, "C": C}},
            "pg": {
                "enabled": run_pg_flag,
                "hyperparams": {
                    "lambda_reg": lambda_pg, "step_scale": step_scale_pg, "step_size": step_size_pg,
                    "use_fista": use_fista_pg, "use_backtracking": use_backtracking_pg,
                    "max_iter": max_iter_pg, "tol": tol_pg,
                },
            },
        },
        "generator": {
            "function": "code.data_gen.generate_piecewise_X_with_exog",
            "kwargs": {
                "s_type": cfg.data_gen.s_type,
                "t_min": cfg.data_gen.t_min,
                "t_max": cfg.data_gen.t_max,
                "z_dist": cfg.data_gen.z_dist,
            },
        },
        "results": {
            "figures": figure_records,
            "summary_means": summary_means,
            "summary_means_by_variant": summary_means_by_variant,
            "timings": {
                "trial_sec": trial_timings,
                "summary": timing_summary,
                "overall_sec": float(time.perf_counter() - run_wall_start),
            },
            "metrics": {
                "pp": error_means[primary_variant_key]["pp"].tolist() if run_pp_flag else None,
                "pp_sgd": error_means[primary_variant_key]["pp_sgd"].tolist() if run_pp_sgd_flag else None,
                "pc": error_means[primary_variant_key]["pc"].tolist() if run_pc_flag else None,
                "co": error_means[primary_variant_key]["co"].tolist() if run_co_flag else None,
                "sgd": error_means[primary_variant_key]["sgd"].tolist() if run_sgd_flag else None,
                "pg": error_means[primary_variant_key]["pg"].tolist() if run_pg_flag else None,
            },
            "metrics_by_variant": {
                v["key"]: {
                    "pp": error_means[v["key"]]["pp"].tolist() if run_pp_flag else None,
                    "pp_sgd": error_means[v["key"]]["pp_sgd"].tolist() if run_pp_sgd_flag else None,
                    "pc": error_means[v["key"]]["pc"].tolist() if run_pc_flag else None,
                    "co": error_means[v["key"]]["co"].tolist() if run_co_flag else None,
                    "sgd": error_means[v["key"]]["sgd"].tolist() if run_sgd_flag else None,
                    "pg": error_means[v["key"]]["pg"].tolist() if run_pg_flag else None,
                }
                for v in plot_variants
            },
        },
        "snapshots": script_copies,
        "hyperparam_json": str(hyperparam_path) if hyperparam_path is not None else None,
        "result_dir": str(result_dir),
    }
    meta_name = f"{Path(filename).stem}_meta.json"
    save_json(metadata, Path(result_dir), name=meta_name)

    used_hyperparams_payload = {
        "hyperparam_json": str(hyperparam_path) if hyperparam_path is not None else None,
        "hyperparam_json_content": loaded_hyperparams,
        "resolved_hyperparams": {
            "pp": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda, "lambda_S": lambda_S},
            "pp_sgd": {"r": r_pp_sgd, "q": q_pp_sgd, "rho": rho_pp_sgd, "mu_lambda": mu_lambda_pp_sgd, "lambda_S": lambda_S_pp_sgd},
            "pc": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C},
            "co": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C},
            "sgd": {"lambda_reg": sgd_lambda_reg, "alpha": sgd_alpha, "beta_sgd": beta_sgd, "C": C},
            "pg": {
                "lambda_reg": lambda_pg,
                "step_scale": step_scale_pg,
                "step_size": step_size_pg,
                "use_fista": use_fista_pg,
                "use_backtracking": use_backtracking_pg,
                "max_iter": max_iter_pg,
                "tol": tol_pg,
            },
        },
        "offline_lambda_l1": offline_lambda_l1,
        "note": "Resolved from hyperparam_json if provided; otherwise config defaults.",
    }
    save_json(used_hyperparams_payload, Path(result_dir), name="used_hyperparams.json")


if __name__ == "__main__":
    main()
