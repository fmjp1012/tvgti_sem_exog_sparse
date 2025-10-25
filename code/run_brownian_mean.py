import os
import shutil
import datetime

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

import sys

from code.data_gen import generate_brownian_piecewise_Y_with_exog
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varying_sem import TimeVaryingSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import create_result_dir, backup_script, make_result_filename


def main():
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    run_pc_flag = True
    run_co_flag = True
    run_sgd_flag = True
    run_pp_flag = True
    num_trials = 10

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

    def run_trial(trial_seed: int):
        np.random.seed(trial_seed)
        S_series, B_true, U, Y = generate_brownian_piecewise_Y_with_exog(
            N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
            std_S=std_S, s_type="random", b_min=0.5, b_max=1.0, u_dist="uniform01"
        )
        errors = {}
        if run_pp_flag:
            S0 = np.zeros((N, N)); b0 = np.ones(N)
            model = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
            S_hat_list, _ = model.run(Y, U)
            error_pp = [
                (np.linalg.norm(S_hat_list[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['pp'] = error_pp
        if run_pc_flag:
            X = Y
            pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline")
            estimates_pc, _ = pc.run(X)
            error_pc = [
                (np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['pc'] = error_pc
        if run_co_flag:
            X = Y
            co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline")
            estimates_co, _ = co.run(X)
            error_co = [
                (np.linalg.norm(estimates_co[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['co'] = error_co
        if run_sgd_flag:
            X = Y
            sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline")
            estimates_sgd, _ = sgd.run(X)
            error_sgd = [
                (np.linalg.norm(estimates_sgd[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['sgd'] = error_sgd
        return errors

    trial_seeds = [seed + i for i in range(num_trials)]
    error_pp_total = np.zeros(T) if run_pp_flag else None
    error_pc_total = np.zeros(T) if run_pc_flag else None
    error_co_total = np.zeros(T) if run_co_flag else None
    error_sgd_total = np.zeros(T) if run_sgd_flag else None

    with tqdm_joblib(tqdm(desc="Progress", total=num_trials)):
        results = Parallel(n_jobs=-1, batch_size=1)(delayed(run_trial)(ts) for ts in trial_seeds)

    for errs in results:
        if run_pp_flag:
            error_pp_total += np.array(errs['pp'])
        if run_pc_flag:
            error_pc_total += np.array(errs['pc'])
        if run_co_flag:
            error_co_total += np.array(errs['co'])
        if run_sgd_flag:
            error_sgd_total += np.array(errs['sgd'])

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
    plt.ylabel('Average NSE')
    plt.grid(True, which='both')
    plt.legend()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename = os.path.basename(__file__)
    filename = make_result_filename(
        prefix="brownian_mean",
        params={
            "N": N,
            "T": T,
            "num_trials": num_trials,
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
    result_dir = create_result_dir(Path('./result'), 'exog_sparse_brownian_mean', extra_tag='images')
    plt.savefig(str(Path(result_dir) / filename))
    plt.show()

    backup_script(Path(__file__), Path(result_dir))


if __name__ == "__main__":
    main()


