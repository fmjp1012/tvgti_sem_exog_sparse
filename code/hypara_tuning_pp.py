from typing import Dict
from pathlib import Path

import numpy as np
import optuna
from scipy.linalg import norm

from code.data_gen import generate_piecewise_Y_with_exog
from models.pp_exog import PPExogenousSEM
from utils.io.results import create_result_dir


def run_once(r: int, q: int, rho: float, mu_lambda: float, N: int, T: int, sparsity: float) -> float:
    max_weight = 0.5
    std_e = 0.05
    K = 3
    S_series, B_true, Y = generate_piecewise_Y_with_exog(
        N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
        s_type="random", b_min=0.5, b_max=1.0, u_dist="uniform01"
    )
    U = np.random.uniform(0.0, 1.0, size=(N, T))

    S0 = np.zeros((N, N))
    b0 = np.ones(N)
    model = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = model.run(Y, U)

    S_true_T = S_series[-1]
    S_hat_T = S_hat_list[-1]
    frob_err = norm(S_hat_T - S_true_T)
    return frob_err


def main():
    N = 20
    T = 2000
    sparsity = 0.7

    def objective(trial: optuna.trial.Trial) -> float:
        r = trial.suggest_int("r", 10, 200, step=10)
        q = trial.suggest_int("q", 2, 10)
        rho = trial.suggest_float("rho", 1e-5, 1e-1, log=True)
        mu_lambda = trial.suggest_float("mu_lambda", 1e-4, 0.2, log=True)
        return run_once(r, q, rho, mu_lambda, N, T, sparsity)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best:", study.best_trial.params, "value=", study.best_value)


if __name__ == "__main__":
    main()


