import os
import shutil
import datetime

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys

from code.data_gen import generate_brownian_piecewise_Y_with_exog
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varying_sem import TimeVaryingSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import create_result_dir, backup_script


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
    np.random.seed(seed)

    S_series, B_true, U, Y = generate_brownian_piecewise_Y_with_exog(
        N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
        std_S=std_S, s_type="random", b_min=0.5, b_max=1.0, u_dist="uniform01"
    )

    S0 = np.zeros((N, N)); b0 = np.ones(N)
    r = 50; q = 5; rho = 1e-3; mu_lambda = 0.05
    pp = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = pp.run(Y, U)

    # PC/CO/SGD baselines
    X = Y
    S0_pc = np.zeros((N, N))
    lambda_reg = 1e-3; alpha = 1e-2; beta = 1e-2; gamma = 0.9; P = 1; C = 1
    pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline")
    estimates_pc, _ = pc.run(X)
    beta_co = 0.02
    co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline")
    estimates_co, _ = co.run(X)
    beta_sgd = 0.0269
    sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline")
    estimates_sgd, _ = sgd.run(X)

    err_pp = [
        (np.linalg.norm(S_hat_list[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]
    err_pc = [
        (np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]
    err_co = [
        (np.linalg.norm(estimates_co[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(T)
    ]
    err_sgd = [
        (np.linalg.norm(estimates_sgd[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
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
    notebook_filename = os.path.basename(__file__)
    filename = (f'timestamp{timestamp}_result_N{N}_notebook_filename{notebook_filename}_'
                f'T{T}_K{K}_stdS{std_S}_seed{seed}_r{r}_q{q}_rho{rho}_mulambda{mu_lambda}.png')
    print(filename)
    result_dir = create_result_dir(Path('./result'), 'exog_sparse_brownian_once', extra_tag='images')
    plt.savefig(str(Path(result_dir) / filename))
    plt.show()

    backup_script(Path(__file__), Path(result_dir))


if __name__ == "__main__":
    main()


