"""
Piecewise シナリオのシミュレーション実行スクリプト

設定は code/config.py で一元管理されています。
このスクリプトを実行する前に config.py を編集して設定を変更してください。

使用方法:
    python -m code.run_piecewise
    python -m code.run_piecewise --hyperparam_json path/to/hyperparams.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from code.config import get_config, get_default_hyperparams_dict, print_config_summary
from code.data_gen import generate_piecewise_X_with_exog
from models.pp_exog import PPExogenousSEM
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.io.plotting import apply_style, plot_heatmaps
from utils.io.results import backup_script, create_result_dir, make_result_filename, save_json
from utils.offline_solver import solve_offline_sem_lasso_batch


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
    }
    if cfg.metric.error_normalization == "offline_solution":
        # offline_lambda_l1 はハイパラJSONまたは探索範囲から取得
        offline_space = cfg.search_spaces.offline.offline_lambda_l1
        metric_items["offline_lambda_l1 (range)"] = f"[{offline_space.low}, {offline_space.high}]"
    _print_block("Metric Settings", metric_items)
    
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


def parse_args() -> Optional[Path]:
    """コマンドライン引数をパース（ハイパラJSONパスのみ）"""
    parser = argparse.ArgumentParser(
        description="Piecewiseシナリオの実験実行（設定は config.py で変更）"
    )
    parser.add_argument(
        "--hyperparam_json",
        type=Path,
        default=None,
        help="ハイパーパラメータJSONのパス（省略時はconfig.pyのデフォルト値を使用）"
    )
    args = parser.parse_args()
    return args.hyperparam_json


def main() -> None:
    """メイン処理"""
    hyperparam_path = parse_args()
    
    # config.py から設定を取得
    cfg = get_config()
    
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)
    
    # 手法フラグ
    run_pp_flag = cfg.methods.pp
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
    pc_cfg = hyperparams.get("pc", {})
    co_cfg = hyperparams.get("co", {})
    sgd_cfg = hyperparams.get("sgd", {})
    pg_cfg = hyperparams.get("pg", {})
    
    # PP法のハイパーパラメータ
    r = int(pp_cfg.get("r", cfg.hyperparams.pp.r))
    q = int(pp_cfg.get("q", cfg.hyperparams.pp.q))
    rho = float(pp_cfg.get("rho", cfg.hyperparams.pp.rho))
    mu_lambda = float(pp_cfg.get("mu_lambda", cfg.hyperparams.pp.mu_lambda))
    
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
            "pc": run_pc_flag,
            "co": run_co_flag,
            "sgd": run_sgd_flag,
            "pg": run_pg_flag,
        },
        hyperparams={
            "pp": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda},
            "pc": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C},
            "co": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C},
            "sgd": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_sgd": beta_sgd, "C": C},
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
    
    # offline_lambda_l1 の取得（ハイパラJSONから読み込むか、探索範囲の幾何平均を使用）
    offline_lambda_l1 = None
    if error_normalization == "offline_solution":
        if hyperparams.get("offline_lambda_l1") is not None:
            offline_lambda_l1 = float(hyperparams["offline_lambda_l1"])
        else:
            # 探索範囲の幾何平均をデフォルトとして使用（対数スケール）
            offline_space = cfg.search_spaces.offline.offline_lambda_l1
            import math
            offline_lambda_l1 = math.sqrt(offline_space.low * offline_space.high)
    
    def compute_error(S_hat: np.ndarray, S_true: np.ndarray, S_offline: Optional[np.ndarray], eps: float = 1e-12) -> float:
        """誤差を計算（正規化方法に応じて分母を変更）"""
        numerator = np.linalg.norm(S_hat - S_true) ** 2
        if error_normalization == "offline_solution" and S_offline is not None:
            denominator = np.linalg.norm(S_true - S_offline) ** 2 + eps
        else:
            denominator = np.linalg.norm(S_true) ** 2 + eps
        return numerator / denominator
    
    def run_trial(trial_seed: int):
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
        errors = {}
        estimates_final = {"True": S_series[-1]}
        
        # オフライン解を計算（必要な場合）
        S_offline = None
        if error_normalization == "offline_solution":
            S_offline = solve_offline_sem_lasso_batch(Y, U, offline_lambda_l1)
            estimates_final['Offline'] = S_offline
        
        if run_pp_flag:
            S0 = np.zeros((N, N))
            b0 = np.ones(N)
            model = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
            S_hat_list, _ = model.run(Y, U)
            error_pp = [
                compute_error(S_hat_list[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['pp'] = error_pp
            estimates_final['PP'] = S_hat_list[-1]
        
        if run_pc_flag:
            X = Y
            pc = PCSEM(
                N, S0_pc, lambda_reg, alpha, beta, gamma, P, C,
                show_progress=False, name="pc_baseline", T_init=B_true,
            )
            estimates_pc, _ = pc.run(X, U)
            error_pc = [
                compute_error(estimates_pc[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['pc'] = error_pc
            estimates_final['PC'] = estimates_pc[-1]
        
        if run_co_flag:
            X = Y
            co = PCSEM(
                N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C,
                show_progress=False, name="co_baseline", T_init=B_true,
            )
            estimates_co, _ = co.run(X, U)
            error_co = [
                compute_error(estimates_co[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['co'] = error_co
            estimates_final['CO'] = estimates_co[-1]
        
        if run_sgd_flag:
            X = Y
            sgd = PCSEM(
                N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C,
                show_progress=False, name="sgd_baseline", T_init=B_true,
            )
            estimates_sgd, _ = sgd.run(X, U)
            error_sgd = [
                compute_error(estimates_sgd[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['sgd'] = error_sgd
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
            error_pg = [
                compute_error(estimates_pg[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['pg'] = error_pg
            estimates_final['PG'] = estimates_pg[-1]
        
        return errors, estimates_final
    
    trial_seeds = [seed + i for i in range(num_trials)]
    error_pp_total = np.zeros(T) if run_pp_flag else None
    error_pc_total = np.zeros(T) if run_pc_flag else None
    error_co_total = np.zeros(T) if run_co_flag else None
    error_sgd_total = np.zeros(T) if run_sgd_flag else None
    error_pg_total = np.zeros(T) if run_pg_flag else None
    
    with tqdm_joblib(tqdm(desc="Progress", total=num_trials)):
        results = Parallel(n_jobs=-1, batch_size=1, prefer="threads")(
            delayed(run_trial)(ts) for ts in trial_seeds
        )
    
    last_estimates = None
    for errs, estimates_final in results:
        if run_pp_flag:
            error_pp_total += np.array(errs['pp'])
        if run_pc_flag:
            error_pc_total += np.array(errs['pc'])
        if run_co_flag:
            error_co_total += np.array(errs['co'])
        if run_sgd_flag:
            error_sgd_total += np.array(errs['sgd'])
        if run_pg_flag:
            error_pg_total += np.array(errs['pg'])
        last_estimates = estimates_final
    
    error_pp_mean = error_pp_total / num_trials if run_pp_flag else None
    error_pc_mean = error_pc_total / num_trials if run_pc_flag else None
    error_co_mean = error_co_total / num_trials if run_co_flag else None
    error_sgd_mean = error_sgd_total / num_trials if run_sgd_flag else None
    error_pg_mean = error_pg_total / num_trials if run_pg_flag else None
    
    plt.figure(figsize=(10, 6))
    if run_co_flag:
        plt.plot(error_co_mean, color='blue', label='Correction Only')
    if run_pc_flag:
        plt.plot(error_pc_mean, color='limegreen', label='Prediction Correction')
    if run_sgd_flag:
        plt.plot(error_sgd_mean, color='cyan', label='SGD')
    if run_pg_flag:
        plt.plot(error_pg_mean, color='magenta', label='ProxGrad')
    if run_pp_flag:
        plt.plot(error_pp_mean, color='red', label='Proposed (PP)')
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    if error_normalization == "offline_solution":
        plt.ylabel('Average Error Ratio (vs Offline)')
    else:
        plt.ylabel('Average NSE')
    plt.grid(True, which='both')
    plt.legend()
    
    run_started_at = datetime.now()
    filename = make_result_filename(
        prefix="piecewise",
        params={
            "N": N, "T": T, "num_trials": num_trials,
            "maxweight": max_weight, "stde": std_e, "K": K,
            "seed": seed, "r": r, "q": q, "rho": rho, "mulambda": mu_lambda,
        },
        suffix=".png",
    )
    print(filename)
    result_dir = create_result_dir(cfg.output.result_root, cfg.output.subdir_piecewise, extra_tag='images')
    plt.savefig(str(Path(result_dir) / filename))
    plt.show()
    
    # ヒートマップ表示（最後の試行の最終時刻）
    if last_estimates is not None:
        heatmap_filename = filename.replace(".png", "_heatmap.png")
        plot_heatmaps(
            matrices=last_estimates,
            save_path=Path(result_dir) / heatmap_filename,
            title=f"Estimated vs True at t={T-1} (last trial)",
            show=True,
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
    
    metadata = {
        "created_at": run_started_at.isoformat(),
        "command": sys.argv,
        "config": {
            "num_trials": num_trials,
            "seed_base": seed,
            "trial_seeds": trial_seeds,
            "N": N, "T": T, "sparsity": sparsity,
            "max_weight": max_weight, "std_e": std_e, "K": K,
        },
        "metric": {
            "error_normalization": error_normalization,
            "offline_lambda_l1": offline_lambda_l1,
        },
        "methods": {
            "pp": {"enabled": run_pp_flag, "hyperparams": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda}},
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
            "figure": filename,
            "figure_path": str(Path(result_dir) / filename),
            "metrics": {
                "pp": error_pp_mean.tolist() if run_pp_flag else None,
                "pc": error_pc_mean.tolist() if run_pc_flag else None,
                "co": error_co_mean.tolist() if run_co_flag else None,
                "sgd": error_sgd_mean.tolist() if run_sgd_flag else None,
                "pg": error_pg_mean.tolist() if run_pg_flag else None,
            },
        },
        "snapshots": script_copies,
        "hyperparam_json": str(hyperparam_path) if hyperparam_path is not None else None,
        "result_dir": str(result_dir),
    }
    meta_name = f"{Path(filename).stem}_meta.json"
    save_json(metadata, Path(result_dir), name=meta_name)


if __name__ == "__main__":
    main()
