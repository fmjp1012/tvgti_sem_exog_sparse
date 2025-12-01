from __future__ import annotations

import copy
import datetime as dt
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import optuna

from code.data_gen import (
    generate_linear_X_with_exog,
    generate_piecewise_X_with_exog,
)
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from utils.io.results import create_result_dir, backup_script, save_json
from numpy.random import SeedSequence, Generator


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_list_of_ints(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _to_list_of_bools(value: Any) -> list[bool]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [_to_bool(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [_to_bool(v) for v in value]
    return [_to_bool(value)]


def _to_python_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, np.int_)):
        return int(value)
    return value


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _clean(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(val) for val in obj]
    return _to_python_value(obj)


def _clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _clean(val) for key, val in data.items()}


SUPPORTED_METHODS: Tuple[str, ...] = ("pp", "pc", "co", "sgd", "pg")


def _normalize_methods(methods: Optional[Iterable[Any]]) -> list[str]:
    if methods is None:
        return list(SUPPORTED_METHODS)

    if isinstance(methods, str):
        raw_items = methods.split(",")
    else:
        raw_items = list(methods)

    normalized: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, str):
            name = item.strip().lower()
        else:
            name = str(item).strip().lower()
        if name:
            normalized.append(name)

    deduped: list[str] = []
    for name in normalized:
        if name not in deduped:
            deduped.append(name)

    if not deduped:
        raise ValueError("少なくとも1つの手法を指定してください。")

    invalid = [name for name in deduped if name not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            f"未知の手法指定です: {', '.join(invalid)}. 利用可能: {', '.join(SUPPORTED_METHODS)}"
        )

    return deduped


DEFAULT_HYPERPARAM_FALLBACK: Dict[str, Dict[str, Any]] = {
    "pp": {"r": 50, "q": 5, "rho": 1e-3, "mu_lambda": 0.05},
    "pc": {"lambda_reg": 1e-3, "alpha": 1e-2, "beta": 1e-2, "gamma": 0.9, "P": 1, "C": 1},
    "co": {"beta_co": 0.02},
    "sgd": {"beta_sgd": 0.0269},
    "pg": {"lambda_reg": 1e-3, "step_scale": 1.0, "use_fista": True, "use_backtracking": False, "max_iter": 500, "tol": 1e-4},
}

DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "pp": {
        "rho": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "mu_lambda": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
    },
    "pc": {
        "lambda_reg": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "alpha": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "beta_pc": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "gamma": {"type": "float", "low": 0.85, "high": 0.999, "log": False},
        "P": {"type": "int", "low": 0, "high": 2, "step": 1},
        "C": {"type": "categorical", "choices": [1, 2, 5]},
    },
    "co": {
        "alpha": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "beta_co": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "gamma": {"type": "float", "low": 0.85, "high": 0.999, "log": False},
        "C": {"type": "categorical", "choices": [1, 2, 5]},
    },
    "sgd": {
        "alpha": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "beta_sgd": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
    },
    "pg": {
        "lambda_reg": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "step_scale": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "use_fista": {"type": "categorical", "choices": [True, False]},
    },
}

SEARCH_OVERRIDE_SPECS: Dict[str, Tuple[str, str, str, Callable[[Any], Any]]] = {
    "pp_rho_low": ("pp", "rho", "low", float),
    "pp_rho_high": ("pp", "rho", "high", float),
    "pp_rho_log": ("pp", "rho", "log", _to_bool),
    "pp_mu_lambda_low": ("pp", "mu_lambda", "low", float),
    "pp_mu_lambda_high": ("pp", "mu_lambda", "high", float),
    "pp_mu_lambda_log": ("pp", "mu_lambda", "log", _to_bool),
    "pc_lambda_reg_low": ("pc", "lambda_reg", "low", float),
    "pc_lambda_reg_high": ("pc", "lambda_reg", "high", float),
    "pc_lambda_reg_log": ("pc", "lambda_reg", "log", _to_bool),
    "pc_alpha_low": ("pc", "alpha", "low", float),
    "pc_alpha_high": ("pc", "alpha", "high", float),
    "pc_alpha_log": ("pc", "alpha", "log", _to_bool),
    "pc_beta_pc_low": ("pc", "beta_pc", "low", float),
    "pc_beta_pc_high": ("pc", "beta_pc", "high", float),
    "pc_beta_pc_log": ("pc", "beta_pc", "log", _to_bool),
    "pc_gamma_low": ("pc", "gamma", "low", float),
    "pc_gamma_high": ("pc", "gamma", "high", float),
    "pc_gamma_log": ("pc", "gamma", "log", _to_bool),
    "pc_P_min": ("pc", "P", "low", int),
    "pc_P_max": ("pc", "P", "high", int),
    "pc_P_step": ("pc", "P", "step", int),
    "pc_C_choices": ("pc", "C", "choices", _to_list_of_ints),
    "co_alpha_low": ("co", "alpha", "low", float),
    "co_alpha_high": ("co", "alpha", "high", float),
    "co_alpha_log": ("co", "alpha", "log", _to_bool),
    "co_beta_co_low": ("co", "beta_co", "low", float),
    "co_beta_co_high": ("co", "beta_co", "high", float),
    "co_beta_co_log": ("co", "beta_co", "log", _to_bool),
    "co_gamma_low": ("co", "gamma", "low", float),
    "co_gamma_high": ("co", "gamma", "high", float),
    "co_gamma_log": ("co", "gamma", "log", _to_bool),
    "co_C_choices": ("co", "C", "choices", _to_list_of_ints),
    "sgd_alpha_low": ("sgd", "alpha", "low", float),
    "sgd_alpha_high": ("sgd", "alpha", "high", float),
    "sgd_alpha_log": ("sgd", "alpha", "log", _to_bool),
    "sgd_beta_sgd_low": ("sgd", "beta_sgd", "low", float),
    "sgd_beta_sgd_high": ("sgd", "beta_sgd", "high", float),
    "sgd_beta_sgd_log": ("sgd", "beta_sgd", "log", _to_bool),
    "pg_lambda_reg_low": ("pg", "lambda_reg", "low", float),
    "pg_lambda_reg_high": ("pg", "lambda_reg", "high", float),
    "pg_lambda_reg_log": ("pg", "lambda_reg", "log", _to_bool),
    "pg_step_scale_low": ("pg", "step_scale", "low", float),
    "pg_step_scale_high": ("pg", "step_scale", "high", float),
    "pg_step_scale_log": ("pg", "step_scale", "log", _to_bool),
    "pg_use_fista_choices": ("pg", "use_fista", "choices", _to_list_of_bools),
}


def _resolve_param_space(
    search_spaces: Dict[str, Dict[str, Dict[str, Any]]], method: str, param: str
) -> Dict[str, Any]:
    """Return a merged copy of the search spec for the given method/param."""
    method_space = search_spaces.get(method, {})
    if param in method_space:
        return dict(method_space[param])
    default_space = DEFAULT_SEARCH_SPACES.get(method, {}).get(param)
    if default_space is not None:
        return dict(default_space)
    raise KeyError(f"{method}.{param} の探索空間が定義されていません。")


def _suggest_from_space(
    trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]
) -> Any:
    """Suggest a value from Optuna trial using the provided search spec."""
    param_type = spec.get("type")
    if param_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if param_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
        )
    if param_type == "categorical":
        choices = spec.get("choices")
        if choices is None:
            raise ValueError(f"{name} の choices が定義されていません。")
        return trial.suggest_categorical(name, list(choices))
    raise ValueError(f"{name} の探索空間 type={param_type} は未サポートです。")


def tune_methods_for_scenario(
    generator: Callable[..., Any],
    generator_kwargs: Dict[str, Any],
    tuning_trials: int = 30,
    tuning_runs_per_trial: int = 5,
    seed: int = 3,
    fallback: Optional[Dict[str, Dict[str, Any]]] = None,
    truncation_horizon: int = 400,
    search_space_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    methods: Optional[Iterable[Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """汎用ハイパラ調整ルーチン。"""

    if "N" not in generator_kwargs or "T" not in generator_kwargs:
        raise ValueError("generator_kwargs には少なくとも 'N' と 'T' を含める必要があります")

    selected_methods = _normalize_methods(methods)

    best = copy.deepcopy(fallback or DEFAULT_HYPERPARAM_FALLBACK)
    initial_best = copy.deepcopy(best)
    search_spaces = copy.deepcopy(DEFAULT_SEARCH_SPACES)
    if search_space_overrides:
        for method, params in search_space_overrides.items():
            method_space = search_spaces.setdefault(method, {})
            for param, fields in params.items():
                param_space = method_space.setdefault(param, {})
                param_space.update(fields)

    N = int(generator_kwargs["N"])
    T = int(generator_kwargs["T"])
    T_tune = min(T, truncation_horizon)
    penalty_value = 1e6
    method_ids = {"pp": 0, "pc": 1, "co": 2, "sgd": 3, "pg": 4}

    def make_rng(method: str, trial_number: int, run_index: int) -> Generator:
        entropy = [seed, method_ids[method], trial_number, run_index]
        return np.random.default_rng(SeedSequence(entropy))

    progress_queue: Queue[Dict[str, Any]] = Queue()
    pc_params_lock = threading.Lock()
    pc_shared_params: Dict[str, float | int] = {
        "lambda_reg": float(best["pc"].get("lambda_reg", DEFAULT_HYPERPARAM_FALLBACK["pc"]["lambda_reg"])),
        "alpha": float(best["pc"].get("alpha", DEFAULT_HYPERPARAM_FALLBACK["pc"]["alpha"])),
        "beta": float(best["pc"].get("beta", DEFAULT_HYPERPARAM_FALLBACK["pc"]["beta"])),
        "gamma": float(best["pc"].get("gamma", DEFAULT_HYPERPARAM_FALLBACK["pc"]["gamma"])),
        "P": int(best["pc"].get("P", DEFAULT_HYPERPARAM_FALLBACK["pc"]["P"])),
        "C": int(best["pc"].get("C", DEFAULT_HYPERPARAM_FALLBACK["pc"]["C"])),
    }

    def progress_worker() -> None:
        while True:
            item = progress_queue.get()
            if item is None:
                break
            method = item.get("method", "?")
            status = item.get("status")
            if status == "start":
                total = item.get("total")
                print(f"[{method}] tuning start (trials={total})", flush=True)
            elif status == "trial":
                trial_idx = item.get("trial")
                total = item.get("total")
                value = item.get("value")
                best_value = item.get("best_value")
                value_str = "nan" if value is None else f"{value:.6f}"
                best_str = "nan" if best_value is None else f"{best_value:.6f}"
                print(f"[{method}] trial {trial_idx}/{total} value={value_str} best={best_str}", flush=True)
            elif status == "completed":
                best_value = item.get("best_value")
                params = item.get("best_params", {})
                best_str = "nan" if best_value is None else f"{best_value:.6f}"
                print(f"[{method}] tuning completed best={best_str} params={params}", flush=True)

    progress_thread = threading.Thread(target=progress_worker, daemon=True)
    progress_thread.start()

    def snapshot_pc_params() -> Dict[str, float | int]:
        with pc_params_lock:
            return dict(pc_shared_params)

    def objective_pp(trial: optuna.trial.Trial) -> float:
        r_suggested = int(best["pp"].get("r", DEFAULT_HYPERPARAM_FALLBACK["pp"]["r"]))
        q_suggested = int(best["pp"].get("q", DEFAULT_HYPERPARAM_FALLBACK["pp"]["q"]))
        rho_suggested = _suggest_from_space(trial, "rho", _resolve_param_space(search_spaces, "pp", "rho"))
        mu_lambda_suggested = _suggest_from_space(
            trial, "mu_lambda", _resolve_param_space(search_spaces, "pp", "mu_lambda")
        )
        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pp", trial.number, run_idx)
            S_ser, _, U_gen, Y_gen = generator(rng=rng, **generator_kwargs)
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
        lambda_reg_suggested = _suggest_from_space(
            trial, "lambda_reg", _resolve_param_space(search_spaces, "pc", "lambda_reg")
        )
        alpha_suggested = _suggest_from_space(trial, "alpha", _resolve_param_space(search_spaces, "pc", "alpha"))
        beta_suggested = _suggest_from_space(trial, "beta_pc", _resolve_param_space(search_spaces, "pc", "beta_pc"))
        gamma_suggested = _suggest_from_space(trial, "gamma", _resolve_param_space(search_spaces, "pc", "gamma"))
        P_suggested = _suggest_from_space(trial, "P", _resolve_param_space(search_spaces, "pc", "P"))
        C_suggested = _suggest_from_space(trial, "C", _resolve_param_space(search_spaces, "pc", "C"))
        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pc", trial.number, run_idx)
            S_ser, T_mat, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
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
                    T_init=T_mat,
                )
                estimates_pc, _ = pc.run(X, Z)
                err_ts = [np.linalg.norm(estimates_pc[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def objective_co(trial: optuna.trial.Trial) -> float:
        try:
            alpha_suggested = _suggest_from_space(
                trial, "alpha", _resolve_param_space(search_spaces, "co", "alpha")
            )
        except KeyError:
            alpha_suggested = None
        beta_co_suggested = _suggest_from_space(
            trial, "beta_co", _resolve_param_space(search_spaces, "co", "beta_co")
        )
        gamma_suggested = _suggest_from_space(trial, "gamma", _resolve_param_space(search_spaces, "co", "gamma"))
        C_suggested = _suggest_from_space(trial, "C", _resolve_param_space(search_spaces, "co", "C"))
        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("co", trial.number, run_idx)
            S_ser, T_mat, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            snapshot = snapshot_pc_params()
            lambda_reg = float(snapshot.get("lambda_reg", DEFAULT_HYPERPARAM_FALLBACK["pc"]["lambda_reg"]))
            if alpha_suggested is not None:
                alpha = float(alpha_suggested)
            else:
                alpha = float(snapshot.get("alpha", DEFAULT_HYPERPARAM_FALLBACK["pc"]["alpha"]))
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
                    T_init=T_mat,
                )
                estimates_co, _ = co.run(X, Z)
                err_ts = [np.linalg.norm(estimates_co[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def objective_sgd(trial: optuna.trial.Trial) -> float:
        try:
            alpha_suggested = _suggest_from_space(
                trial, "alpha", _resolve_param_space(search_spaces, "sgd", "alpha")
            )
        except KeyError:
            alpha_suggested = None
        beta_sgd_suggested = _suggest_from_space(
            trial, "beta_sgd", _resolve_param_space(search_spaces, "sgd", "beta_sgd")
        )
        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("sgd", trial.number, run_idx)
            S_ser, T_mat, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            snapshot = snapshot_pc_params()
            lambda_reg = float(snapshot.get("lambda_reg", DEFAULT_HYPERPARAM_FALLBACK["pc"]["lambda_reg"]))
            if alpha_suggested is not None:
                alpha = float(alpha_suggested)
            else:
                alpha = float(snapshot.get("alpha", DEFAULT_HYPERPARAM_FALLBACK["pc"]["alpha"]))
            gamma = 0.0
            P = 0
            C = int(snapshot.get("C", DEFAULT_HYPERPARAM_FALLBACK["pc"]["C"]))
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
                    T_init=T_mat,
                )
                estimates_sgd, _ = sgd.run(X, Z)
                err_ts = [np.linalg.norm(estimates_sgd[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def objective_pg(trial: optuna.trial.Trial) -> float:
        lambda_reg_suggested = _suggest_from_space(
            trial, "lambda_reg", _resolve_param_space(search_spaces, "pg", "lambda_reg")
        )
        try:
            step_scale_suggested = _suggest_from_space(
                trial, "step_scale", _resolve_param_space(search_spaces, "pg", "step_scale")
            )
        except KeyError:
            step_scale_suggested = float(best["pg"].get("step_scale", DEFAULT_HYPERPARAM_FALLBACK["pg"]["step_scale"]))
        try:
            use_fista_suggested = _suggest_from_space(
                trial, "use_fista", _resolve_param_space(search_spaces, "pg", "use_fista")
            )
        except KeyError:
            use_fista_suggested = bool(best["pg"].get("use_fista", DEFAULT_HYPERPARAM_FALLBACK["pg"]["use_fista"]))

        max_iter_pg = int(best["pg"].get("max_iter", DEFAULT_HYPERPARAM_FALLBACK["pg"]["max_iter"]))
        tol_pg = float(best["pg"].get("tol", DEFAULT_HYPERPARAM_FALLBACK["pg"]["tol"]))
        use_backtracking_pg = bool(best["pg"].get("use_backtracking", DEFAULT_HYPERPARAM_FALLBACK["pg"]["use_backtracking"]))

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pg", trial.number, run_idx)
            S_ser, _, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            config_pg = ProximalGradientConfig(
                lambda_reg=float(lambda_reg_suggested),
                step_size=None,
                step_scale=float(step_scale_suggested),
                max_iter=max_iter_pg,
                tol=tol_pg,
                use_fista=bool(use_fista_suggested),
                use_backtracking=use_backtracking_pg,
                show_progress=False,
                name="pg_baseline",
            )
            model_pg = ProximalGradientBatchSEM(N, config_pg)
            try:
                estimates_pg, _ = model_pg.run(X, Z)
                err_ts = [np.linalg.norm(estimates_pg[t] - S_trunc[t], ord="fro") for t in range(len(S_trunc))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def update_pc_shared_from_pc(study: optuna.study.Study, _: optuna.trial.FrozenTrial) -> None:
        params = study.best_params
        if not params:
            return
        with pc_params_lock:
            if "lambda_reg" in params:
                pc_shared_params["lambda_reg"] = float(params["lambda_reg"])
            if "alpha" in params:
                pc_shared_params["alpha"] = float(params["alpha"])
            if "beta_pc" in params:
                pc_shared_params["beta"] = float(params["beta_pc"])
            if "gamma" in params:
                pc_shared_params["gamma"] = float(params["gamma"])
            if "P" in params:
                pc_shared_params["P"] = int(params["P"])
            if "C" in params:
                pc_shared_params["C"] = int(params["C"])

    def update_pc_shared_from_co(study: optuna.study.Study, _: optuna.trial.FrozenTrial) -> None:
        params = study.best_params
        if not params:
            return
        with pc_params_lock:
            if "alpha" in params:
                pc_shared_params["alpha"] = float(params["alpha"])
            if "gamma" in params:
                pc_shared_params["gamma"] = float(params["gamma"])
            if "C" in params:
                pc_shared_params["C"] = int(params["C"])

    def update_pc_shared_from_sgd(study: optuna.study.Study, _: optuna.trial.FrozenTrial) -> None:
        params = study.best_params
        if not params:
            return
        with pc_params_lock:
            if "alpha" in params:
                pc_shared_params["alpha"] = float(params["alpha"])

    def run_study(
        method_name: str,
        objective_fn: Callable[[optuna.trial.Trial], float],
        extra_callback: Optional[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] = None,
    ) -> tuple[str, Dict[str, Any], Optional[float]]:
        progress_queue.put({"method": method_name, "status": "start", "total": tuning_trials})
        study = optuna.create_study(direction="minimize")

        def wrapped_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            if extra_callback is not None:
                extra_callback(study, trial)
            progress_queue.put(
                {
                    "method": method_name,
                    "status": "trial",
                    "trial": trial.number + 1,
                    "total": tuning_trials,
                    "value": trial.value,
                    "best_value": study.best_value,
                }
            )

        study.optimize(objective_fn, n_trials=tuning_trials, callbacks=[wrapped_callback])
        progress_queue.put(
            {
                "method": method_name,
                "status": "completed",
                "best_value": study.best_value,
                "best_params": dict(study.best_params),
            }
        )
        return method_name, dict(study.best_params), study.best_value

    all_method_specs = {
        "pp": {"objective": objective_pp, "callback": None},
        "pc": {"objective": objective_pc, "callback": update_pc_shared_from_pc},
        "co": {"objective": objective_co, "callback": update_pc_shared_from_co},
        "sgd": {"objective": objective_sgd, "callback": update_pc_shared_from_sgd},
        "pg": {"objective": objective_pg, "callback": None},
    }

    method_specs = {name: all_method_specs[name] for name in selected_methods}

    method_results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(method_specs)) as executor:
        future_map = {
            executor.submit(run_study, name, spec["objective"], spec["callback"]): name for name, spec in method_specs.items()
        }
        for future in as_completed(future_map):
            method_name = future_map[future]
            try:
                result_name, result_params, result_value = future.result()
                method_results[result_name] = {"params": result_params, "best_value": result_value}
            except Exception as exc:  # pragma: no cover
                progress_queue.put(
                    {
                        "method": method_name,
                        "status": "completed",
                        "best_value": None,
                        "best_params": {},
                    }
                )
                progress_queue.put(None)
                progress_thread.join()
                raise exc

    progress_queue.put(None)
    progress_thread.join()

    if "pp" in method_results:
        params = method_results["pp"]["params"]
        best["pp"]["rho"] = float(params.get("rho", best["pp"]["rho"]))
        best["pp"]["mu_lambda"] = float(params.get("mu_lambda", best["pp"]["mu_lambda"]))

    if "pc" in method_results:
        params = method_results["pc"]["params"]
        best["pc"]["lambda_reg"] = float(params.get("lambda_reg", best["pc"]["lambda_reg"]))
        best["pc"]["alpha"] = float(params.get("alpha", best["pc"]["alpha"]))
        best["pc"]["beta"] = float(params.get("beta_pc", best["pc"].get("beta", DEFAULT_HYPERPARAM_FALLBACK["pc"]["beta"])))
        best["pc"]["gamma"] = float(params.get("gamma", best["pc"]["gamma"]))
        best["pc"]["P"] = int(params.get("P", best["pc"].get("P", DEFAULT_HYPERPARAM_FALLBACK["pc"]["P"])))
        best["pc"]["C"] = int(params.get("C", best["pc"]["C"]))

    if "co" in method_results:
        params = method_results["co"]["params"]
        best["co"]["beta_co"] = float(params.get("beta_co", best["co"]["beta_co"]))
        if "alpha" in params:
            best["pc"]["alpha"] = float(params["alpha"])
        if "gamma" in params:
            best["pc"]["gamma"] = float(params["gamma"])
        if "C" in params:
            best["pc"]["C"] = int(params["C"])

    if "sgd" in method_results:
        params = method_results["sgd"]["params"]
        best["sgd"]["beta_sgd"] = float(params.get("beta_sgd", best["sgd"]["beta_sgd"]))
        if "alpha" in params:
            best["pc"]["alpha"] = float(params["alpha"])

    if "pg" in method_results:
        params = method_results["pg"]["params"]
        best["pg"]["lambda_reg"] = float(params.get("lambda_reg", best["pg"]["lambda_reg"]))
        if "step_scale" in params:
            best["pg"]["step_scale"] = float(params["step_scale"])
        if "use_fista" in params:
            best["pg"]["use_fista"] = bool(params["use_fista"])

    best = {method: _clean_dict(params) for method, params in best.items()}
    initial_best_clean = {method: _clean_dict(params) for method, params in initial_best.items()}

    summary = {
        "seed": seed,
        "tuning_trials": tuning_trials,
        "tuning_runs_per_trial": tuning_runs_per_trial,
        "truncation_horizon": truncation_horizon,
        "generator_kwargs": {key: _to_python_value(val) for key, val in generator_kwargs.items()},
        "initial_fallback": initial_best_clean,
        "search_spaces": _clean(search_spaces),
        "search_overrides": _clean(search_space_overrides) if search_space_overrides else {},
        "method_results": {
            name: {
                "best_params": _clean_dict(result.get("params", {})),
                "best_value": _to_python_value(result.get("best_value")),
            }
            for name, result in method_results.items()
        },
    }

    return best, summary


def tune_piecewise_all_methods(
    N: int = 20,
    T: int = 1000,
    sparsity: float = 0.7,
    max_weight: float = 0.5,
    std_e: float = 0.05,
    K: int = 4,
    tuning_trials: int = 300,
    tuning_runs_per_trial: int = 1,
    seed: int = 3,
    search_space_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    methods: Optional[Iterable[Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    generator_kwargs = {
        "N": N,
        "T": T,
        "sparsity": sparsity,
        "max_weight": max_weight,
        "std_e": std_e,
        "K": K,
        "s_type": "regular",
        "t_min": 0.5,
        "t_max": 1.0,
        "z_dist": "uniform01",
    }
    best, summary = tune_methods_for_scenario(
        generator=generate_piecewise_X_with_exog,
        generator_kwargs=generator_kwargs,
        tuning_trials=tuning_trials,
        tuning_runs_per_trial=tuning_runs_per_trial,
        seed=seed,
        search_space_overrides=search_space_overrides,
        methods=methods,
    )
    summary["generator_name"] = "code.data_gen.generate_piecewise_X_with_exog"
    return best, summary


def tune_linear_all_methods(
    N: int = 20,
    T: int = 1000,
    sparsity: float = 0.6,
    max_weight: float = 0.5,
    std_e: float = 0.05,
    tuning_trials: int = 30,
    tuning_runs_per_trial: int = 5,
    seed: int = 3,
    search_space_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    methods: Optional[Iterable[Any]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    generator_kwargs = {
        "N": N,
        "T": T,
        "sparsity": sparsity,
        "max_weight": max_weight,
        "std_e": std_e,
        "s_type": "random",
        "t_min": 0.5,
        "t_max": 1.0,
        "z_dist": "uniform01",
    }
    best, summary = tune_methods_for_scenario(
        generator=generate_linear_X_with_exog,
        generator_kwargs=generator_kwargs,
        tuning_trials=tuning_trials,
        tuning_runs_per_trial=tuning_runs_per_trial,
        seed=seed,
        search_space_overrides=search_space_overrides,
        methods=methods,
    )
    summary["generator_name"] = "code.data_gen.generate_linear_X_with_exog"
    return best, summary


def save_best_hyperparams(
    best: Dict[str, Dict[str, Any]],
    scenario: str,
    result_root: Path | str = Path("./result"),
    subdir: str = "exog_sparse_tuning",
    timestamp: Optional[str] = None,
    indent: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
    script_paths: Optional[Dict[str, Path | str]] = None,
) -> Path:
    """チューニング結果をJSONとして保存するヘルパー。"""

    ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = create_result_dir(Path(result_root), subdir)
    out_path = Path(result_dir) / f"{scenario}_best_hyperparams_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=indent)

    script_copies: Dict[str, str] = {}
    if script_paths:
        scripts_dir = out_path.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for label, script_path in script_paths.items():
            if script_path is None:
                continue
            src = Path(script_path)
            if src.is_file():
                copied = backup_script(src, scripts_dir)
                script_copies[label] = str(copied)

    metadata_payload: Optional[Dict[str, Any]] = None
    if metadata is not None:
        metadata_payload = copy.deepcopy(metadata)
    if metadata_payload is not None or script_copies:
        if metadata_payload is None:
            metadata_payload = {}
        snapshots = metadata_payload.setdefault("snapshots", {})
        for key, value in script_copies.items():
            snapshots[key] = value
        metadata_payload["hyperparam_file"] = str(out_path)
        meta_name = f"{out_path.stem}_meta.json"
        save_json(metadata_payload, out_path.parent, name=meta_name)

    return out_path
