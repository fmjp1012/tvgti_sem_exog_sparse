import os
import sys
import datetime

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from code.data_gen import generate_brownian_piecewise_X_with_exog
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import create_result_dir, backup_script, make_result_filename, save_json


def main():
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    N = 20
    T = 1000
    sparsity = 0.7
    max_weight = 0.5
    std_e = 0.05
    K = 5
    std_S = 0.05
    seed = 3
    rng = np.random.default_rng(seed)

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

    S0 = np.zeros((N, N)); b0 = np.ones(N)
    r = 50; q = 5; rho = 1e-3; mu_lambda = 0.05
    pp = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = pp.run(Y, Z)

    # PC/CO/SGD baselines
    X = Y
    S0_pc = np.zeros((N, N))
    lambda_reg = 1e-3; alpha = 1e-2; beta = 1e-2; gamma = 0.9; P = 1; C = 1
    pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline", T_init=T_mat)
    estimates_pc, _ = pc.run(X, Z)
    beta_co = 0.02
    co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline", T_init=T_mat)
    estimates_co, _ = co.run(X, Z)
    beta_sgd = 0.0269
    sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline", T_init=T_mat)
    estimates_sgd, _ = sgd.run(X, Z)

    err_pp = [
        float((np.linalg.norm(S_hat_list[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]
    err_pc = [
        float((np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]
    err_co = [
        float((np.linalg.norm(estimates_co[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]
    err_sgd = [
        float((np.linalg.norm(estimates_sgd[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(err_co, color='blue', label='Correction Only')
    plt.plot(err_pc, color='limegreen', label='Prediction Correction')
    plt.plot(err_sgd, color='cyan', label='SGD')
    plt.plot(err_pp, color='red', label='Proposed (PP)')
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.grid(True, which='both')
    plt.legend()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = make_result_filename(
        prefix="brownian_once",
        params={
            "N": N,
            "T": T,
            "K": K,
            "stdS": std_S,
            "seed": seed,
            "r": r,
            "q": q,
            "rho": rho,
            "mulambda": mu_lambda,
        },
        suffix=".png",
    )
    print(filename)
    result_dir = create_result_dir(Path('./result'), 'exog_sparse_brownian_once', extra_tag='images')
    figure_path = Path(result_dir) / filename
    plt.savefig(str(figure_path))
    plt.show()

    scripts_dir = Path(result_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copies = {"run_brownian_once": str(backup_script(Path(__file__), scripts_dir))}
    data_gen_path = Path(__file__).resolve().parent / "data_gen.py"
    if data_gen_path.exists():
        script_copies["data_gen"] = str(backup_script(data_gen_path, scripts_dir))

    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "command": sys.argv,
        "config": {
            "N": N,
            "T": T,
            "K": K,
            "sparsity": sparsity,
            "max_weight": max_weight,
            "std_e": std_e,
            "std_S": std_S,
            "seed": seed,
        },
        "methods": {
            "pp": {"hyperparams": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda}},
            "pc": {"hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C}},
            "co": {"hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C}},
            "sgd": {"hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_sgd": beta_sgd, "C": C}},
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
                "pp": err_pp,
                "pc": err_pc,
                "co": err_co,
                "sgd": err_sgd,
            },
        },
        "snapshots": script_copies,
    }
    save_json(metadata, Path(result_dir), name=f"{figure_path.stem}_meta.json")


if __name__ == "__main__":
    main()
