from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna

from code.data_gen import (
    generate_linear_Y_with_exog,
    generate_piecewise_Y_with_exog,
)
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varying_sem import TimeVaryingSEM as PCSEM
from utils.io.results import create_result_dir


DEFAULT_HYPERPARAM_FALLBACK: Dict[str, Dict[str, float]] = {
    "pp": {"r": 50, "q": 5, "rho": 1e-3, "mu_lambda": 0.05},
    "pc": {"lambda_reg": 1e-3, "alpha": 1e-2, "beta": 1e-2, "gamma": 0.9, "P": 1, "C": 1},
    "co": {"beta_co": 0.02},
    "sgd": {"beta_sgd": 0.0269},
}


def tune_methods_for_scenario(
    generator: Callable[..., Any],
    generator_kwargs: Dict[str, Any],
    tuning_trials: int = 30,
    tuning_runs_per_trial: int = 5,
    seed: int = 3,
    fallback: Optional[Dict[str, Dict[str, float]]] = None,
    truncation_horizon: int = 400,
) -> Dict[str, Dict[str, float]]:
    """汎用ハイパラ調整ルーチン。"""

    if "N" not in generator_kwargs or "T" not in generator_kwargs:
        raise ValueError("generator_kwargs には少なくとも 'N' と 'T' を含める必要があります")

    np.random.seed(seed)
    best = copy.deepcopy(fallback or DEFAULT_HYPERPARAM_FALLBACK)

    N = int(generator_kwargs["N"])
    T = int(generator_kwargs["T"])
    T_tune = min(T, truncation_horizon)
    penalty_value = 1e6

    def objective_pp(trial: optuna.trial.Trial) -> float:
        r_suggested = int(best["pp"].get("r", DEFAULT_HYPERPARAM_FALLBACK["pp"]["r"]))
        q_suggested = int(best["pp"].get("q", DEFAULT_HYPERPARAM_FALLBACK["pp"]["q"]))
        rho_suggested = trial.suggest_float("rho", 1e-6, 1e-1, log=True)
        mu_lambda_suggested = trial.suggest_float("mu_lambda", 1e-4, 1.0, log=True)
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, _, U_gen, Y_gen = generator(**generator_kwargs)
            S0 = np.zeros((N, N))
            b0 = np.ones(N)
            model = PPExogenousSEM(N, S0, b0, r=r_suggested, q=q_suggested, rho=rho_suggested, mu_lambda=mu_lambda_suggested)
            S_hat_list, _ = model.run(Y_gen, U_gen)
            err = np.linalg.norm(S_hat_list[-1] - S_ser[-1], ord="fro")
            if not np.isfinite(err):
                err = penalty_value
            errs.append(err)
        return float(np.mean(errs))

    def objective_pc(trial: optuna.trial.Trial) -> float:
        lambda_reg_suggested = trial.suggest_float("lambda_reg", 1e-5, 1e-2, log=True)
        alpha_suggested = trial.suggest_float("alpha", 1e-4, 2e-1, log=True)
        beta_suggested = trial.suggest_float("beta_pc", 1e-4, 3e-1, log=True)
        gamma_suggested = trial.suggest_float("gamma", 0.85, 0.999)
        P_suggested = trial.suggest_int("P", 0, 2)
        C_suggested = trial.suggest_categorical("C", [1, 2, 5])
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, _, _, Y_gen = generator(**generator_kwargs)
            X = Y_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            try:
                pc = PCSEM(
                    N,
                    S0_pc,
                    lambda_reg_suggested,
                    alpha_suggested,
                    beta_suggested,
                    gamma_suggested,
                    P_suggested,
                    C_suggested,
                    show_progress=False,
                    name="pc_baseline",
                )
                estimates_pc, _ = pc.run(X)
                err_ts = [np.linalg.norm(estimates_pc[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def objective_co(trial: optuna.trial.Trial) -> float:
        beta_co_suggested = trial.suggest_float("beta_co", 1e-5, 1e0, log=True)
        gamma_suggested = trial.suggest_float("gamma", 0.85, 0.999)
        C_suggested = trial.suggest_categorical("C", [1, 2, 5])
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, _, _, Y_gen = generator(**generator_kwargs)
            X = Y_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            lambda_reg = best["pc"].get("lambda_reg", DEFAULT_HYPERPARAM_FALLBACK["pc"]["lambda_reg"])
            alpha = best["pc"].get("alpha", DEFAULT_HYPERPARAM_FALLBACK["pc"]["alpha"])
            P = 0
            try:
                co = PCSEM(
                    N,
                    S0_pc,
                    lambda_reg,
                    alpha,
                    beta_co_suggested,
                    gamma_suggested,
                    P,
                    C_suggested,
                    show_progress=False,
                    name="co_baseline",
                )
                estimates_co, _ = co.run(X)
                err_ts = [np.linalg.norm(estimates_co[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def objective_sgd(trial: optuna.trial.Trial) -> float:
        beta_sgd_suggested = trial.suggest_float("beta_sgd", 1e-5, 1e0, log=True)
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, _, _, Y_gen = generator(**generator_kwargs)
            X = Y_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            lambda_reg = best["pc"].get("lambda_reg", DEFAULT_HYPERPARAM_FALLBACK["pc"]["lambda_reg"])
            alpha = best["pc"].get("alpha", DEFAULT_HYPERPARAM_FALLBACK["pc"]["alpha"])
            gamma = 0.0
            P = 0
            C = best["pc"].get("C", DEFAULT_HYPERPARAM_FALLBACK["pc"]["C"])
            try:
                sgd = PCSEM(
                    N,
                    S0_pc,
                    lambda_reg,
                    alpha,
                    beta_sgd_suggested,
                    gamma,
                    P,
                    C,
                    show_progress=False,
                    name="sgd_baseline",
                )
                estimates_sgd, _ = sgd.run(X)
                err_ts = [np.linalg.norm(estimates_sgd[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_pp, n_trials=tuning_trials)
    best["pp"]["rho"] = study.best_params.get("rho", best["pp"]["rho"])
    best["pp"]["mu_lambda"] = study.best_params.get("mu_lambda", best["pp"]["mu_lambda"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_pc, n_trials=tuning_trials)
    best["pc"]["lambda_reg"] = study.best_params.get("lambda_reg", best["pc"]["lambda_reg"])
    best["pc"]["alpha"] = study.best_params.get("alpha", best["pc"]["alpha"])
    best["pc"]["beta"] = study.best_params.get("beta_pc", best["pc"].get("beta", DEFAULT_HYPERPARAM_FALLBACK["pc"]["beta"]))
    best["pc"]["gamma"] = study.best_params.get("gamma", best["pc"]["gamma"])
    best["pc"]["P"] = study.best_params.get("P", best["pc"].get("P", DEFAULT_HYPERPARAM_FALLBACK["pc"]["P"]))
    best["pc"]["C"] = study.best_params.get("C", best["pc"]["C"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_co, n_trials=tuning_trials)
    best["co"]["beta_co"] = study.best_params.get("beta_co", best["co"]["beta_co"])
    best["pc"]["gamma"] = study.best_params.get("gamma", best["pc"]["gamma"])
    best["pc"]["C"] = study.best_params.get("C", best["pc"]["C"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_sgd, n_trials=tuning_trials)
    best["sgd"]["beta_sgd"] = study.best_params.get("beta_sgd", best["sgd"]["beta_sgd"])

    return best


def tune_piecewise_all_methods(
    N: int = 20,
    T: int = 1000,
    sparsity: float = 0.7,
    max_weight: float = 0.5,
    std_e: float = 0.05,
    K: int = 4,
    tuning_trials: int = 30,
    tuning_runs_per_trial: int = 5,
    seed: int = 3,
) -> Dict[str, Dict[str, float]]:
    generator_kwargs = {
        "N": N,
        "T": T,
        "sparsity": sparsity,
        "max_weight": max_weight,
        "std_e": std_e,
        "K": K,
        "s_type": "random",
        "b_min": 0.5,
        "b_max": 1.0,
        "u_dist": "uniform01",
    }
    return tune_methods_for_scenario(
        generator=generate_piecewise_Y_with_exog,
        generator_kwargs=generator_kwargs,
        tuning_trials=tuning_trials,
        tuning_runs_per_trial=tuning_runs_per_trial,
        seed=seed,
    )


def tune_linear_all_methods(
    N: int = 20,
    T: int = 1000,
    sparsity: float = 0.6,
    max_weight: float = 0.5,
    std_e: float = 0.05,
    tuning_trials: int = 30,
    tuning_runs_per_trial: int = 5,
    seed: int = 3,
) -> Dict[str, Dict[str, float]]:
    generator_kwargs = {
        "N": N,
        "T": T,
        "sparsity": sparsity,
        "max_weight": max_weight,
        "std_e": std_e,
        "s_type": "random",
        "b_min": 0.5,
        "b_max": 1.0,
        "u_dist": "uniform01",
    }
    return tune_methods_for_scenario(
        generator=generate_linear_Y_with_exog,
        generator_kwargs=generator_kwargs,
        tuning_trials=tuning_trials,
        tuning_runs_per_trial=tuning_runs_per_trial,
        seed=seed,
    )


def save_best_hyperparams(
    best: Dict[str, Dict[str, float]],
    scenario: str,
    result_root: Path | str = Path("./result"),
    subdir: str = "exog_sparse_tuning",
    timestamp: Optional[str] = None,
    indent: int = 2,
) -> Path:
    """チューニング結果をJSONとして保存するヘルパー。"""

    ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = create_result_dir(Path(result_root), subdir)
    out_path = Path(result_dir) / f"{scenario}_best_hyperparams_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=indent)
    return out_path


