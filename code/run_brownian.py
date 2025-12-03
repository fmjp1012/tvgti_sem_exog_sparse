import os
import sys
import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM

from code.config import get_config
from code.data_gen import generate_brownian_piecewise_X_with_exog
from models.pp_exog import PPExogenousSEM
from utils.io.plotting import apply_style, plot_heatmaps
from utils.io.results import create_result_dir, backup_script, make_result_filename, save_json
from utils.offline_solver import solve_offline_sem_lasso_batch


def main():
    # プロット設定
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    # config.py から評価指標設定を取得
    cfg = get_config()
    error_normalization = cfg.metric.error_normalization
    
    # offline_lambda_l1 の取得（探索範囲の幾何平均を使用）
    offline_lambda_l1 = None
    if error_normalization == "offline_solution":
        # 探索範囲の幾何平均をデフォルトとして使用（対数スケール）
        offline_space = cfg.search_spaces.offline.offline_lambda_l1
        import math
        offline_lambda_l1 = math.sqrt(offline_space.low * offline_space.high)

    run_pc_flag = True
    run_co_flag = True
    run_sgd_flag = True
    run_pp_flag = True
    num_trials = 5

    N = 20
    T = 1000
    sparsity = 0.7
    max_weight = 0.5
    std_e = 0.05
    K = 5
    std_S = 0.05
    seed = 3

    r = 50
    q = 5
    rho = 1e-3
    mu_lambda = 0.05

    S0_pc = np.zeros((N, N))
    lambda_reg = 1e-3
    alpha = 1e-2
    beta = 1e-2
    gamma = 0.9
    P = 1
    C = 1
    beta_co = 0.02
    beta_sgd = 0.0269

    def compute_error(S_hat: np.ndarray, S_true: np.ndarray, S_offline: Optional[np.ndarray], eps: float = 1e-12) -> float:
        """誤差を計算（正規化方法に応じて分母を変更）"""
        numerator = np.linalg.norm(S_hat - S_true) ** 2
        if error_normalization == "offline_solution" and S_offline is not None:
            denominator = np.linalg.norm(S_true - S_offline) ** 2 + eps
        else:
            denominator = np.linalg.norm(S_true) ** 2 + eps
        return float(numerator / denominator)

    def run_trial(trial_seed: int):
        rng = np.random.default_rng(trial_seed)
        S_series, T_mat, Z, Y = generate_brownian_piecewise_X_with_exog(
            N=N,
            T=T,
            sparsity=sparsity,
            max_weight=max_weight,
            std_e=std_e,
            K=K,
            std_S=std_S,
            s_type="random",
            t_min=0.5,
            t_max=1.0,
            z_dist="uniform01",
            rng=rng,
        )
        errors = {}
        estimates_final = {"True": S_series[-1]}
        
        # オフライン解を計算（必要な場合）
        S_offline = None
        if error_normalization == "offline_solution":
            S_offline = solve_offline_sem_lasso_batch(Y, Z, offline_lambda_l1)
            estimates_final['Offline'] = S_offline
        
        if run_pp_flag:
            S0 = np.zeros((N, N))
            b0 = np.ones(N)
            model = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
            S_hat_list, _ = model.run(Y, Z)
            error_pp = [
                compute_error(S_hat_list[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['pp'] = error_pp
            estimates_final['PP'] = S_hat_list[-1]
        if run_pc_flag:
            X = Y
            pc = PCSEM(
                N,
                S0_pc,
                lambda_reg,
                alpha,
                beta,
                gamma,
                P,
                C,
                show_progress=False,
                name="pc_baseline",
                T_init=T_mat,
            )
            estimates_pc, _ = pc.run(X, Z)
            error_pc = [
                compute_error(estimates_pc[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['pc'] = error_pc
            estimates_final['PC'] = estimates_pc[-1]
        if run_co_flag:
            X = Y
            co = PCSEM(
                N,
                S0_pc,
                lambda_reg,
                alpha,
                beta_co,
                gamma,
                0,
                C,
                show_progress=False,
                name="co_baseline",
                T_init=T_mat,
            )
            estimates_co, _ = co.run(X, Z)
            error_co = [
                compute_error(estimates_co[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['co'] = error_co
            estimates_final['CO'] = estimates_co[-1]
        if run_sgd_flag:
            X = Y
            sgd = PCSEM(
                N,
                S0_pc,
                lambda_reg,
                alpha,
                beta_sgd,
                0.0,
                0,
                C,
                show_progress=False,
                name="sgd_baseline",
                T_init=T_mat,
            )
            estimates_sgd, _ = sgd.run(X, Z)
            error_sgd = [
                compute_error(estimates_sgd[t], S_series[t], S_offline)
                for t in range(T)
            ]
            errors['sgd'] = error_sgd
            estimates_final['SGD'] = estimates_sgd[-1]
        return errors, estimates_final

    trial_seeds = [seed + i for i in range(num_trials)]
    error_pp_total = np.zeros(T) if run_pp_flag else None
    error_pc_total = np.zeros(T) if run_pc_flag else None
    error_co_total = np.zeros(T) if run_co_flag else None
    error_sgd_total = np.zeros(T) if run_sgd_flag else None

    with tqdm_joblib(tqdm(desc="Progress", total=num_trials)):
        results = Parallel(n_jobs=-1, batch_size=1, prefer="threads")(delayed(run_trial)(ts) for ts in trial_seeds)

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
        last_estimates = estimates_final

    if run_pp_flag:
        error_pp_mean = error_pp_total / num_trials
    if run_pc_flag:
        error_pc_mean = error_pc_total / num_trials
    if run_co_flag:
        error_co_mean = error_co_total / num_trials
    if run_sgd_flag:
        error_sgd_mean = error_sgd_total / num_trials

    plt.figure(figsize=(10, 6))
    if run_co_flag:
        plt.plot(error_co_mean, color='blue', label='Correction Only')
    if run_pc_flag:
        plt.plot(error_pc_mean, color='limegreen', label='Prediction Correction')
    if run_sgd_flag:
        plt.plot(error_sgd_mean, color='cyan', label='SGD')
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

    filename = make_result_filename(
        prefix="brownian",
        params={
            "N": N,
            "T": T,
            "num_trials": num_trials,
            "maxweight": max_weight,
            "stde": std_e,
            "K": K,
            "seed": seed,
            "r": r,
            "q": q,
            "rho": rho,
            "mulambda": mu_lambda,
        },
        suffix=".png",
    )
    print(filename)
    result_dir = create_result_dir(Path('./result'), 'exog_sparse_brownian', extra_tag='images')
    figure_path = Path(result_dir) / filename
    plt.savefig(str(figure_path))
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

    scripts_dir = Path(result_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copies = {"run_brownian": str(backup_script(Path(__file__), scripts_dir))}
    data_gen_path = Path(__file__).resolve().parent / "data_gen.py"
    if data_gen_path.exists():
        script_copies["data_gen"] = str(backup_script(data_gen_path, scripts_dir))

    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "command": sys.argv,
        "config": {
            "num_trials": num_trials,
            "trial_seeds": trial_seeds,
            "N": N,
            "T": T,
            "K": K,
            "sparsity": sparsity,
            "max_weight": max_weight,
            "std_e": std_e,
            "std_S": std_S,
            "seed_base": seed,
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
        },
        "generator": {
            "function": "code.data_gen.generate_brownian_piecewise_X_with_exog",
            "kwargs": {
                "t_min": 0.5,
                "t_max": 1.0,
                "z_dist": "uniform01",
            },
        },
        "results": {
            "figure": filename,
            "figure_path": str(figure_path),
            "metrics": {
                "pp": error_pp_mean.tolist() if run_pp_flag else None,
                "pc": error_pc_mean.tolist() if run_pc_flag else None,
                "co": error_co_mean.tolist() if run_co_flag else None,
                "sgd": error_sgd_mean.tolist() if run_sgd_flag else None,
            },
        },
        "snapshots": script_copies,
    }
    save_json(metadata, Path(result_dir), name=f"{figure_path.stem}_meta.json")


if __name__ == "__main__":
    main()
