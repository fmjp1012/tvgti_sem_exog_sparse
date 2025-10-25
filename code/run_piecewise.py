from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from code.data_gen import generate_piecewise_Y_with_exog
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varying_sem import TimeVaryingSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import backup_script, create_result_dir, make_result_filename


@dataclass
class ExperimentConfig:
    run_pc: bool = True
    run_co: bool = True
    run_sgd: bool = True
    run_pp: bool = True
    num_trials: int = 100
    N: int = 20
    T: int = 1000
    sparsity: float = 0.7
    max_weight: float = 0.5
    std_e: float = 0.05
    K: int = 4
    seed: int = 3
    hyperparams: Dict[str, Dict[str, float]] | None = None


def load_hyperparams(json_path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    if json_path is None:
        return None
    if not json_path.is_file():
        raise FileNotFoundError(f"ハイパラJSONが見つかりません: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="ピースワイズシナリオの実験実行")
    parser.add_argument("--hyperparam_json", type=Path, default=None, help="ハイパーパラメータJSONのパス")
    parser.add_argument("--num_trials", type=int, default=100, help="試行回数")
    parser.add_argument("--seed", type=int, default=3, help="乱数シードの基点")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--max_weight", type=float, default=0.5)
    parser.add_argument("--std_e", type=float, default=0.05)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--run_pc", action="store_true", help="PC法を実行")
    parser.add_argument("--no_pc", action="store_true", help="PC法をスキップ")
    parser.add_argument("--run_co", action="store_true", help="CO法を実行")
    parser.add_argument("--no_co", action="store_true", help="CO法をスキップ")
    parser.add_argument("--run_sgd", action="store_true", help="SGD法を実行")
    parser.add_argument("--no_sgd", action="store_true", help="SGD法をスキップ")
    parser.add_argument("--run_pp", action="store_true", help="PP法を実行")
    parser.add_argument("--no_pp", action="store_true", help="PP法をスキップ")

    args = parser.parse_args()

    hyperparams = load_hyperparams(args.hyperparam_json)

    config = ExperimentConfig(
        num_trials=args.num_trials,
        seed=args.seed,
        N=args.N,
        T=args.T,
        sparsity=args.sparsity,
        max_weight=args.max_weight,
        std_e=args.std_e,
        K=args.K,
        hyperparams=hyperparams,
    )

    if args.no_pc:
        config.run_pc = False
    elif args.run_pc:
        config.run_pc = True

    if args.no_co:
        config.run_co = False
    elif args.run_co:
        config.run_co = True

    if args.no_sgd:
        config.run_sgd = False
    elif args.run_sgd:
        config.run_sgd = True

    if args.no_pp:
        config.run_pp = False
    elif args.run_pp:
        config.run_pp = True

    return config


def main() -> None:
    config = parse_args()

    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    run_pc_flag = config.run_pc
    run_co_flag = config.run_co
    run_sgd_flag = config.run_sgd
    run_pp_flag = config.run_pp
    num_trials = config.num_trials

    hyperparams = config.hyperparams or {}

    N = config.N
    T = config.T
    sparsity = config.sparsity
    max_weight = config.max_weight
    std_e = config.std_e
    K = config.K
    seed = config.seed

    pp_cfg = hyperparams.get("pp", {})
    pc_cfg = hyperparams.get("pc", {})
    co_cfg = hyperparams.get("co", {})
    sgd_cfg = hyperparams.get("sgd", {})

    r = int(pp_cfg.get("r", 50))
    q = int(pp_cfg.get("q", 5))
    rho = float(pp_cfg.get("rho", 1e-3))
    mu_lambda = float(pp_cfg.get("mu_lambda", 0.05))

    lambda_reg = float(pc_cfg.get("lambda_reg", 1e-3))
    alpha = float(pc_cfg.get("alpha", 1e-2))
    beta = float(pc_cfg.get("beta", 1e-2))
    gamma = float(pc_cfg.get("gamma", 0.9))
    P = int(pc_cfg.get("P", 1))
    C = int(pc_cfg.get("C", 1))

    beta_co = float(co_cfg.get("beta_co", 0.02))
    beta_sgd = float(sgd_cfg.get("beta_sgd", 0.0269))

    S0_pc = np.zeros((N, N))

    def run_trial(trial_seed: int):
        np.random.seed(trial_seed)
        S_series, B_true, U, Y = generate_piecewise_Y_with_exog(
            N=N,
            T=T,
            sparsity=sparsity,
            max_weight=max_weight,
            std_e=std_e,
            K=K,
            s_type="random",
            b_min=0.5,
            b_max=1.0,
            u_dist="uniform01",
        )
        errors = {}
        if run_pp_flag:
            S0 = np.zeros((N, N))
            b0 = np.ones(N)
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename = os.path.basename(__file__)
    filename = make_result_filename(
        prefix="piecewise",
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
    result_dir = create_result_dir(Path('./result'), 'exog_sparse_piecewise', extra_tag='images')
    plt.savefig(str(Path(result_dir) / filename))
    plt.show()

    backup_script(Path(__file__), Path(result_dir))


if __name__ == "__main__":
    main()


