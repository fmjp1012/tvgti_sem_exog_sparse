"""
ハイパーパラメータチューニングモジュール

Optuna を使用した各手法のハイパーパラメータ最適化を提供します。
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import optuna

from code.config import (
    get_config,
    get_default_hyperparams_dict,
    get_enabled_methods,
    get_search_spaces_dict,
)
from code.data_gen import (
    generate_linear_X_with_exog,
    generate_piecewise_X_with_exog,
)
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from models.tvgti_pc.prediction_correction_sem_noexog import PredictionCorrectionSEMNoExog as PCSEMNoExog
from numpy.random import Generator, SeedSequence
from utils.formatting import clean_dict, coerce_bool, to_list_of_bools, to_list_of_ints, to_python_value
from utils.io.results import backup_script, create_result_dir, save_json
from utils.metrics import compute_frobenius_error
from utils.offline_solver import solve_offline_sem_lasso_batch

# Ctrl+C で中断するためのグローバルフラグ
_shutdown_event = threading.Event()


def _signal_handler(signum: int, frame: Any) -> None:
    """SIGINT (Ctrl+C) をキャッチしてシャットダウンフラグをセット"""
    print("\n[INFO] Ctrl+C detected. Shutting down gracefully...", flush=True)
    _shutdown_event.set()


SUPPORTED_METHODS: Tuple[str, ...] = ("pp", "pp_sgd", "pc", "co", "sgd", "pg")


def _normalize_methods(methods: Optional[Iterable[Any]]) -> list[str]:
    """
    手法リストを正規化する。
    常に config.py で有効化されている手法のみを返す。
    """
    enabled_in_config = get_enabled_methods()

    if methods is None:
        result = enabled_in_config
    else:
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

        invalid = [name for name in deduped if name not in SUPPORTED_METHODS]
        if invalid:
            raise ValueError(
                f"未知の手法指定です: {', '.join(invalid)}. 利用可能: {', '.join(SUPPORTED_METHODS)}"
            )

        result = [m for m in deduped if m in enabled_in_config]

    if not result:
        raise ValueError(
            f"実行する手法がありません。config.py の MethodFlags を確認してください。"
            f"\n現在有効: {enabled_in_config}"
        )

    return result


def _resolve_param_space(
    search_spaces: Dict[str, Dict[str, Dict[str, Any]]], method: str, param: str
) -> Dict[str, Any]:
    """指定されたmethod/paramの探索空間を返す。"""
    default_spaces = get_search_spaces_dict()
    method_space = search_spaces.get(method, {})
    if param in method_space:
        return dict(method_space[param])
    default_space = default_spaces.get(method, {}).get(param)
    if default_space is not None:
        return dict(default_space)
    raise KeyError(f"{method}.{param} の探索空間が定義されていません。")


def _suggest_from_space(
    trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]
) -> Any:
    """Optunaのtrialから探索空間に基づいて値を提案する。"""
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


# 探索範囲オーバーライド仕様
SEARCH_OVERRIDE_SPECS: Dict[str, Tuple[str, str, str, Callable[[Any], Any]]] = {
    "pp_r_choices": ("pp", "r", "choices", to_list_of_ints),
    "pp_q_choices": ("pp", "q", "choices", to_list_of_ints),
    "pp_rho_low": ("pp", "rho", "low", float),
    "pp_rho_high": ("pp", "rho", "high", float),
    "pp_rho_log": ("pp", "rho", "log", coerce_bool),
    "pp_mu_lambda_low": ("pp", "mu_lambda", "low", float),
    "pp_mu_lambda_high": ("pp", "mu_lambda", "high", float),
    "pp_mu_lambda_log": ("pp", "mu_lambda", "log", coerce_bool),
    "pp_lambda_S_low": ("pp", "lambda_S", "low", float),
    "pp_lambda_S_high": ("pp", "lambda_S", "high", float),
    "pp_lambda_S_log": ("pp", "lambda_S", "log", coerce_bool),
    "pc_lambda_reg_low": ("pc", "lambda_reg", "low", float),
    "pc_lambda_reg_high": ("pc", "lambda_reg", "high", float),
    "pc_lambda_reg_log": ("pc", "lambda_reg", "log", coerce_bool),
    "pc_alpha_low": ("pc", "alpha", "low", float),
    "pc_alpha_high": ("pc", "alpha", "high", float),
    "pc_alpha_log": ("pc", "alpha", "log", coerce_bool),
    "pc_beta_pc_low": ("pc", "beta_pc", "low", float),
    "pc_beta_pc_high": ("pc", "beta_pc", "high", float),
    "pc_beta_pc_log": ("pc", "beta_pc", "log", coerce_bool),
    "pc_gamma_low": ("pc", "gamma", "low", float),
    "pc_gamma_high": ("pc", "gamma", "high", float),
    "pc_gamma_log": ("pc", "gamma", "log", coerce_bool),
    "pc_P_min": ("pc", "P", "low", int),
    "pc_P_max": ("pc", "P", "high", int),
    "pc_P_step": ("pc", "P", "step", int),
    "pc_C_choices": ("pc", "C", "choices", to_list_of_ints),
    "co_alpha_low": ("co", "alpha", "low", float),
    "co_alpha_high": ("co", "alpha", "high", float),
    "co_alpha_log": ("co", "alpha", "log", coerce_bool),
    "co_beta_co_low": ("co", "beta_co", "low", float),
    "co_beta_co_high": ("co", "beta_co", "high", float),
    "co_beta_co_log": ("co", "beta_co", "log", coerce_bool),
    "co_gamma_low": ("co", "gamma", "low", float),
    "co_gamma_high": ("co", "gamma", "high", float),
    "co_gamma_log": ("co", "gamma", "log", coerce_bool),
    "co_C_choices": ("co", "C", "choices", to_list_of_ints),
    "sgd_alpha_low": ("sgd", "alpha", "low", float),
    "sgd_alpha_high": ("sgd", "alpha", "high", float),
    "sgd_alpha_log": ("sgd", "alpha", "log", coerce_bool),
    "sgd_beta_sgd_low": ("sgd", "beta_sgd", "low", float),
    "sgd_beta_sgd_high": ("sgd", "beta_sgd", "high", float),
    "sgd_beta_sgd_log": ("sgd", "beta_sgd", "log", coerce_bool),
    "pg_lambda_reg_low": ("pg", "lambda_reg", "low", float),
    "pg_lambda_reg_high": ("pg", "lambda_reg", "high", float),
    "pg_lambda_reg_log": ("pg", "lambda_reg", "log", coerce_bool),
    "pg_step_scale_low": ("pg", "step_scale", "low", float),
    "pg_step_scale_high": ("pg", "step_scale", "high", float),
    "pg_step_scale_log": ("pg", "step_scale", "log", coerce_bool),
    "pg_use_fista_choices": ("pg", "use_fista", "choices", to_list_of_bools),
}


def tune_methods_for_scenario(
    generator: Callable[..., Any],
    generator_kwargs: Dict[str, Any],
    tuning_trials: Optional[int] = None,
    tuning_runs_per_trial: Optional[int] = None,
    seed: Optional[int] = None,
    fallback: Optional[Dict[str, Dict[str, Any]]] = None,
    truncation_horizon: Optional[int] = None,
    search_space_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    methods: Optional[Iterable[Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """汎用ハイパラ調整ルーチン。config.py から設定を取得する。"""

    if "N" not in generator_kwargs or "T" not in generator_kwargs:
        raise ValueError("generator_kwargs には少なくとも 'N' と 'T' を含める必要があります")

    cfg = get_config()
    if tuning_trials is None:
        tuning_trials = cfg.tuning.tuning_trials
    if tuning_runs_per_trial is None:
        tuning_runs_per_trial = cfg.tuning.tuning_runs_per_trial
    if seed is None:
        seed = cfg.tuning.tuning_seed
    if truncation_horizon is None:
        truncation_horizon = cfg.tuning.truncation_horizon

    selected_methods = _normalize_methods(methods)

    default_fallback = get_default_hyperparams_dict()
    best = copy.deepcopy(fallback or default_fallback)
    initial_best = copy.deepcopy(best)
    search_spaces = copy.deepcopy(get_search_spaces_dict())
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
    # 乱数系列の固定用（既存手法のIDは維持し、追加手法は末尾に付与）
    method_ids = {"pp": 0, "pc": 1, "co": 2, "sgd": 3, "pg": 4, "pp_sgd": 5}

    error_normalization = cfg.metric.error_normalization

    def make_rng(method: str, trial_number: int, run_index: int) -> Generator:
        entropy = [seed, method_ids[method], trial_number, run_index]
        return np.random.default_rng(SeedSequence(entropy))

    def suggest_offline_lambda(trial: optuna.trial.Trial) -> float:
        offline_space = _resolve_param_space(search_spaces, "offline", "offline_lambda_l1")
        return _suggest_from_space(trial, "offline_lambda_l1", offline_space)

    progress_queue: Queue[Dict[str, Any]] = Queue()
    pc_params_lock = threading.Lock()
    pc_shared_params: Dict[str, float | int] = {
        "lambda_reg": float(best["pc"].get("lambda_reg", default_fallback["pc"]["lambda_reg"])),
        "alpha": float(best["pc"].get("alpha", default_fallback["pc"]["alpha"])),
        "beta": float(best["pc"].get("beta", default_fallback["pc"]["beta"])),
        "gamma": float(best["pc"].get("gamma", default_fallback["pc"]["gamma"])),
        "P": int(best["pc"].get("P", default_fallback["pc"]["P"])),
        "C": int(best["pc"].get("C", default_fallback["pc"]["C"])),
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

    def _compute_error_for_tuning(
        S_hat: np.ndarray,
        S_true: np.ndarray,
        S_offline: Optional[np.ndarray],
    ) -> float:
        divide_by_n2 = bool(getattr(cfg.metric, "divide_by_n2", False))
        return compute_frobenius_error(S_hat, S_true, S_offline, error_normalization, divide_by_n2)

    def objective_pp(trial: optuna.trial.Trial) -> float:
        # PP は r,q も最適化対象（未定義ならフォールバック）
        try:
            r_suggested = int(_suggest_from_space(trial, "r", _resolve_param_space(search_spaces, "pp", "r")))
        except KeyError:
            r_suggested = int(best["pp"].get("r", default_fallback["pp"]["r"]))
        try:
            q_suggested = int(_suggest_from_space(trial, "q", _resolve_param_space(search_spaces, "pp", "q")))
        except KeyError:
            q_suggested = int(best["pp"].get("q", default_fallback["pp"]["q"]))
        rho_suggested = _suggest_from_space(trial, "rho", _resolve_param_space(search_spaces, "pp", "rho"))
        mu_lambda_suggested = _suggest_from_space(
            trial, "mu_lambda", _resolve_param_space(search_spaces, "pp", "mu_lambda")
        )
        lambda_S_suggested = _suggest_from_space(
            trial, "lambda_S", _resolve_param_space(search_spaces, "pp", "lambda_S")
        )

        offline_lambda_l1 = suggest_offline_lambda(trial) if error_normalization == "offline_solution" else None
        burn_in_cfg = int(getattr(cfg.metric, "burn_in", 0))
        burn_in = (r_suggested + q_suggested - 2) if burn_in_cfg == -1 else max(0, burn_in_cfg)
        comp = getattr(cfg, "comparison", None)
        pp_lookahead_cfg = 0 if comp is None else int(getattr(comp, "pp_lookahead", 0))
        pp_lookahead = (r_suggested + q_suggested - 2) if pp_lookahead_cfg == -1 else max(0, pp_lookahead_cfg)

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pp", trial.number, run_idx)
            S_ser, T_mat, U_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            # チューニングは全手法で同じ打ち切り horizon と評価区間を使う
            X = Y_gen[:, :T_tune]
            U = U_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0 = np.zeros((N, N))
            # 比較条件（初期化）を cfg に合わせる
            pp_b0_mode = str(getattr(getattr(cfg, "comparison", object()), "pp_init_b0", "ones")).strip()
            b0 = np.diag(T_mat) if pp_b0_mode == "true_T_diag" else np.ones(N)
            model = PPExogenousSEM(
                N, S0, b0,
                r=r_suggested, q=q_suggested, rho=rho_suggested,
                mu_lambda=mu_lambda_suggested, lambda_S=lambda_S_suggested,
                lookahead=int(pp_lookahead),
            )
            S_hat_list, _ = model.run(X, U)

            S_offline = None
            if error_normalization == "offline_solution":
                S_offline = solve_offline_sem_lasso_batch(X, U, offline_lambda_l1)

            # PPだけ最終時刻評価になっていたため、PC/CO/SGDと同様に
            # 「時系列平均（burn-in以降）」で最適化する
            try:
                err_ts = [
                    _compute_error_for_tuning(S_hat_list[t], S_trunc[t], S_offline)
                    for t in range(len(S_trunc))
                ]
                err_ts_eval = err_ts[burn_in:] if burn_in < len(err_ts) else err_ts
                mean_err = float(np.mean(err_ts_eval)) if err_ts_eval else penalty_value
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    def objective_pp_sgd(trial: optuna.trial.Trial) -> float:
        # PP-SGD: r=q=1固定（更新に1データのみ使用）
        r_suggested = 1
        q_suggested = 1
        rho_suggested = _suggest_from_space(trial, "rho", _resolve_param_space(search_spaces, "pp_sgd", "rho"))
        mu_lambda_suggested = _suggest_from_space(
            trial, "mu_lambda", _resolve_param_space(search_spaces, "pp_sgd", "mu_lambda")
        )
        lambda_S_suggested = _suggest_from_space(
            trial, "lambda_S", _resolve_param_space(search_spaces, "pp_sgd", "lambda_S")
        )

        offline_lambda_l1 = suggest_offline_lambda(trial) if error_normalization == "offline_solution" else None
        burn_in_cfg = int(getattr(cfg.metric, "burn_in", 0))
        burn_in = (r_suggested + q_suggested - 2) if burn_in_cfg == -1 else max(0, burn_in_cfg)
        comp = getattr(cfg, "comparison", None)
        pp_lookahead_cfg = 0 if comp is None else int(getattr(comp, "pp_lookahead", 0))
        pp_lookahead = (r_suggested + q_suggested - 2) if pp_lookahead_cfg == -1 else max(0, pp_lookahead_cfg)

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pp_sgd", trial.number, run_idx)
            S_ser, T_mat, U_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            U = U_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0 = np.zeros((N, N))
            pp_b0_mode = str(getattr(getattr(cfg, "comparison", object()), "pp_init_b0", "ones")).strip()
            b0 = np.diag(T_mat) if pp_b0_mode == "true_T_diag" else np.ones(N)
            model = PPExogenousSEM(
                N,
                S0,
                b0,
                r=r_suggested,
                q=q_suggested,
                rho=rho_suggested,
                mu_lambda=mu_lambda_suggested,
                lambda_S=lambda_S_suggested,
                lookahead=int(pp_lookahead),
            )
            S_hat_list, _ = model.run(X, U)

            S_offline = None
            if error_normalization == "offline_solution":
                S_offline = solve_offline_sem_lasso_batch(X, U, offline_lambda_l1)

            try:
                err_ts = [
                    _compute_error_for_tuning(S_hat_list[t], S_trunc[t], S_offline)
                    for t in range(len(S_trunc))
                ]
                err_ts_eval = err_ts[burn_in:] if burn_in < len(err_ts) else err_ts
                mean_err = float(np.mean(err_ts_eval)) if err_ts_eval else penalty_value
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
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

        offline_lambda_l1 = suggest_offline_lambda(trial) if error_normalization == "offline_solution" else None
        burn_in_cfg = int(getattr(cfg.metric, "burn_in", 0))
        # burn-in は PP の特性に合わせる（cfgで統一）
        if burn_in_cfg == -1:
            burn_in = int(best["pp"].get("r", default_fallback["pp"]["r"])) + int(best["pp"].get("q", default_fallback["pp"]["q"])) - 2
        else:
            burn_in = max(0, burn_in_cfg)

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pc", trial.number, run_idx)
            S_ser, T_mat, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            comp = getattr(cfg, "comparison", None)
            pc_model = "exog" if comp is None else str(getattr(comp, "pc_model", "exog")).strip()
            if comp is not None and not bool(getattr(comp, "pc_use_true_T_init", True)):
                scale = float(getattr(comp, "pc_T_init_identity_scale", 1.0))
                T_init = np.eye(N) * scale
            else:
                T_init = T_mat

            S_offline = None
            if error_normalization == "offline_solution":
                S_offline = solve_offline_sem_lasso_batch(X, Z, offline_lambda_l1)

            try:
                if pc_model == "noexog":
                    pc = PCSEMNoExog(
                        N,
                        S0_pc,
                        lambda_reg_suggested,
                        alpha_suggested,
                        beta_suggested,
                        gamma_suggested,
                        P_suggested,
                        C_suggested,
                        show_progress=False,
                        name="pc_noexog",
                    )
                    estimates_pc, _ = pc.run(X, Z=None)
                else:
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
                        T_init=T_init,
                    )
                    estimates_pc, _ = pc.run(X, Z)
                err_ts = [_compute_error_for_tuning(estimates_pc[t], S_trunc[t], S_offline) for t in range(len(S_trunc))]
                err_ts_eval = err_ts[burn_in:] if burn_in < len(err_ts) else err_ts
                mean_err = float(np.mean(err_ts_eval)) if err_ts_eval else penalty_value
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

        offline_lambda_l1 = suggest_offline_lambda(trial) if error_normalization == "offline_solution" else None
        burn_in_cfg = int(getattr(cfg.metric, "burn_in", 0))
        if burn_in_cfg == -1:
            burn_in = int(best["pp"].get("r", default_fallback["pp"]["r"])) + int(best["pp"].get("q", default_fallback["pp"]["q"])) - 2
        else:
            burn_in = max(0, burn_in_cfg)

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("co", trial.number, run_idx)
            S_ser, T_mat, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            comp = getattr(cfg, "comparison", None)
            pc_model = "exog" if comp is None else str(getattr(comp, "pc_model", "exog")).strip()
            if comp is not None and not bool(getattr(comp, "pc_use_true_T_init", True)):
                scale = float(getattr(comp, "pc_T_init_identity_scale", 1.0))
                T_init = np.eye(N) * scale
            else:
                T_init = T_mat
            snapshot = snapshot_pc_params()
            lambda_reg = float(snapshot.get("lambda_reg", default_fallback["pc"]["lambda_reg"]))
            if alpha_suggested is not None:
                alpha = float(alpha_suggested)
            else:
                alpha = float(snapshot.get("alpha", default_fallback["pc"]["alpha"]))
            P = 0

            S_offline = None
            if error_normalization == "offline_solution":
                S_offline = solve_offline_sem_lasso_batch(X, Z, offline_lambda_l1)

            try:
                if pc_model == "noexog":
                    co = PCSEMNoExog(
                        N,
                        S0_pc,
                        lambda_reg,
                        alpha,
                        beta_co_suggested,
                        gamma_suggested,
                        P,
                        C_suggested,
                        show_progress=False,
                        name="co_noexog",
                    )
                    estimates_co, _ = co.run(X, Z=None)
                else:
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
                        T_init=T_init,
                    )
                    estimates_co, _ = co.run(X, Z)
                err_ts = [_compute_error_for_tuning(estimates_co[t], S_trunc[t], S_offline) for t in range(len(S_trunc))]
                err_ts_eval = err_ts[burn_in:] if burn_in < len(err_ts) else err_ts
                mean_err = float(np.mean(err_ts_eval)) if err_ts_eval else penalty_value
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

        offline_lambda_l1 = suggest_offline_lambda(trial) if error_normalization == "offline_solution" else None
        burn_in_cfg = int(getattr(cfg.metric, "burn_in", 0))
        if burn_in_cfg == -1:
            burn_in = int(best["pp"].get("r", default_fallback["pp"]["r"])) + int(best["pp"].get("q", default_fallback["pp"]["q"])) - 2
        else:
            burn_in = max(0, burn_in_cfg)

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("sgd", trial.number, run_idx)
            S_ser, T_mat, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            comp = getattr(cfg, "comparison", None)
            pc_model = "exog" if comp is None else str(getattr(comp, "pc_model", "exog")).strip()
            if comp is not None and not bool(getattr(comp, "pc_use_true_T_init", True)):
                scale = float(getattr(comp, "pc_T_init_identity_scale", 1.0))
                T_init = np.eye(N) * scale
            else:
                T_init = T_mat
            snapshot = snapshot_pc_params()
            lambda_reg = float(snapshot.get("lambda_reg", default_fallback["pc"]["lambda_reg"]))
            if alpha_suggested is not None:
                alpha = float(alpha_suggested)
            else:
                alpha = float(snapshot.get("alpha", default_fallback["pc"]["alpha"]))
            gamma = 0.0
            P = 0
            C = int(snapshot.get("C", default_fallback["pc"]["C"]))

            S_offline = None
            if error_normalization == "offline_solution":
                S_offline = solve_offline_sem_lasso_batch(X, Z, offline_lambda_l1)

            try:
                if pc_model == "noexog":
                    sgd = PCSEMNoExog(
                        N,
                        S0_pc,
                        lambda_reg,
                        alpha,
                        beta_sgd_suggested,
                        gamma,
                        P,
                        C,
                        show_progress=False,
                        name="sgd_noexog",
                    )
                    estimates_sgd, _ = sgd.run(X, Z=None)
                else:
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
                        T_init=T_init,
                    )
                    estimates_sgd, _ = sgd.run(X, Z)
                err_ts = [_compute_error_for_tuning(estimates_sgd[t], S_trunc[t], S_offline) for t in range(len(S_trunc))]
                err_ts_eval = err_ts[burn_in:] if burn_in < len(err_ts) else err_ts
                mean_err = float(np.mean(err_ts_eval)) if err_ts_eval else penalty_value
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
            step_scale_suggested = float(best["pg"].get("step_scale", default_fallback["pg"]["step_scale"]))
        try:
            use_fista_suggested = _suggest_from_space(
                trial, "use_fista", _resolve_param_space(search_spaces, "pg", "use_fista")
            )
        except KeyError:
            use_fista_suggested = bool(best["pg"].get("use_fista", default_fallback["pg"]["use_fista"]))

        max_iter_pg = int(best["pg"].get("max_iter", default_fallback["pg"]["max_iter"]))
        tol_pg = float(best["pg"].get("tol", default_fallback["pg"]["tol"]))
        use_backtracking_pg = bool(best["pg"].get("use_backtracking", default_fallback["pg"]["use_backtracking"]))

        offline_lambda_l1 = suggest_offline_lambda(trial) if error_normalization == "offline_solution" else None

        errs = []
        for run_idx in range(tuning_runs_per_trial):
            rng = make_rng("pg", trial.number, run_idx)
            S_ser, _, Z_gen, Y_gen = generator(rng=rng, **generator_kwargs)
            X = Y_gen[:, :T_tune]
            Z = Z_gen[:, :T_tune]
            S_trunc = S_ser[:T_tune]

            S_offline = None
            if error_normalization == "offline_solution":
                S_offline = solve_offline_sem_lasso_batch(X, Z, offline_lambda_l1)

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
                err_ts = [_compute_error_for_tuning(estimates_pg[t], S_trunc[t], S_offline) for t in range(len(S_trunc))]
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
            if _shutdown_event.is_set():
                study.stop()
                return
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
        "pp_sgd": {"objective": objective_pp_sgd, "callback": None},
        "pc": {"objective": objective_pc, "callback": update_pc_shared_from_pc},
        "co": {"objective": objective_co, "callback": update_pc_shared_from_co},
        "sgd": {"objective": objective_sgd, "callback": update_pc_shared_from_sgd},
        "pg": {"objective": objective_pg, "callback": None},
    }

    method_specs = {name: all_method_specs[name] for name in selected_methods}

    method_results: Dict[str, Dict[str, Any]] = {}

    _shutdown_event.clear()
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    try:
        with ThreadPoolExecutor(max_workers=len(method_specs)) as executor:
            future_map = {
                executor.submit(run_study, name, spec["objective"], spec["callback"]): name
                for name, spec in method_specs.items()
            }
            for future in as_completed(future_map):
                if _shutdown_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                method_name = future_map[future]
                try:
                    result_name, result_params, result_value = future.result()
                    method_results[result_name] = {"params": result_params, "best_value": result_value}
                except Exception as exc:
                    progress_queue.put(
                        {
                            "method": method_name,
                            "status": "completed",
                            "best_value": None,
                            "best_params": {},
                        }
                    )
                    if not _shutdown_event.is_set():
                        progress_queue.put(None)
                        progress_thread.join()
                        raise exc
    finally:
        signal.signal(signal.SIGINT, original_handler)

    progress_queue.put(None)
    progress_thread.join()

    if _shutdown_event.is_set():
        print("[INFO] Tuning was interrupted. Returning partial results.", flush=True)

    if "pp" in method_results:
        params = method_results["pp"]["params"]
        if "r" in params:
            best["pp"]["r"] = int(params.get("r", best["pp"]["r"]))
        if "q" in params:
            best["pp"]["q"] = int(params.get("q", best["pp"]["q"]))
        best["pp"]["rho"] = float(params.get("rho", best["pp"]["rho"]))
        best["pp"]["mu_lambda"] = float(params.get("mu_lambda", best["pp"]["mu_lambda"]))
        best["pp"]["lambda_S"] = float(params.get("lambda_S", best["pp"].get("lambda_S", default_fallback["pp"]["lambda_S"])))

    if "pc" in method_results:
        params = method_results["pc"]["params"]
        best["pc"]["lambda_reg"] = float(params.get("lambda_reg", best["pc"]["lambda_reg"]))
        best["pc"]["alpha"] = float(params.get("alpha", best["pc"]["alpha"]))
        best["pc"]["beta"] = float(params.get("beta_pc", best["pc"].get("beta", default_fallback["pc"]["beta"])))
        best["pc"]["gamma"] = float(params.get("gamma", best["pc"]["gamma"]))
        best["pc"]["P"] = int(params.get("P", best["pc"].get("P", default_fallback["pc"]["P"])))
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

    tuned_offline_lambda_l1 = None
    if error_normalization == "offline_solution":
        for method_name in ["pc", "pp", "co", "sgd", "pg"]:
            if method_name in method_results:
                params = method_results[method_name].get("params", {})
                if "offline_lambda_l1" in params:
                    tuned_offline_lambda_l1 = float(params["offline_lambda_l1"])
                    break
        if tuned_offline_lambda_l1 is None:
            import math

            offline_space = cfg.search_spaces.offline.offline_lambda_l1
            tuned_offline_lambda_l1 = math.sqrt(offline_space.low * offline_space.high)

    best = {method: clean_dict(params) for method, params in best.items()}
    initial_best_clean = {method: clean_dict(params) for method, params in initial_best.items()}

    if error_normalization == "offline_solution" and tuned_offline_lambda_l1 is not None:
        best["offline_lambda_l1"] = tuned_offline_lambda_l1

    def _clean_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: _clean_obj(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean_obj(val) for val in obj]
        return to_python_value(obj)

    summary = {
        "seed": seed,
        "tuning_trials": tuning_trials,
        "tuning_runs_per_trial": tuning_runs_per_trial,
        "truncation_horizon": truncation_horizon,
        "generator_kwargs": {key: to_python_value(val) for key, val in generator_kwargs.items()},
        "metric": {
            "error_normalization": error_normalization,
            "offline_lambda_l1": tuned_offline_lambda_l1 if error_normalization == "offline_solution" else None,
        },
        "initial_fallback": initial_best_clean,
        "search_spaces": _clean_obj(search_spaces),
        "search_overrides": _clean_obj(search_space_overrides) if search_space_overrides else {},
        "method_results": {
            name: {
                "best_params": clean_dict(result.get("params", {})),
                "best_value": to_python_value(result.get("best_value")),
            }
            for name, result in method_results.items()
        },
    }

    return best, summary


def tune_piecewise_all_methods(
    N: Optional[int] = None,
    T: Optional[int] = None,
    sparsity: Optional[float] = None,
    max_weight: Optional[float] = None,
    std_e: Optional[float] = None,
    K: Optional[int] = None,
    tuning_trials: Optional[int] = None,
    tuning_runs_per_trial: Optional[int] = None,
    seed: Optional[int] = None,
    search_space_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    methods: Optional[Iterable[Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Piecewiseシナリオのハイパーパラメータチューニング。
    config.py から設定を取得する。引数で明示的に指定した場合は上書き。
    """
    cfg = get_config()

    _N = N if N is not None else cfg.common.N
    _T = T if T is not None else cfg.common.T
    _sparsity = sparsity if sparsity is not None else cfg.common.sparsity
    _max_weight = max_weight if max_weight is not None else cfg.common.max_weight
    _std_e = std_e if std_e is not None else cfg.common.std_e
    _K = K if K is not None else cfg.piecewise.K

    generator_kwargs = {
        "N": _N,
        "T": _T,
        "sparsity": _sparsity,
        "max_weight": _max_weight,
        "std_e": _std_e,
        "K": _K,
        "s_type": cfg.data_gen.s_type,
        "t_min": cfg.data_gen.t_min,
        "t_max": cfg.data_gen.t_max,
        "z_dist": cfg.data_gen.z_dist,
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
    N: Optional[int] = None,
    T: Optional[int] = None,
    sparsity: Optional[float] = None,
    max_weight: Optional[float] = None,
    std_e: Optional[float] = None,
    tuning_trials: Optional[int] = None,
    tuning_runs_per_trial: Optional[int] = None,
    seed: Optional[int] = None,
    search_space_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    methods: Optional[Iterable[Any]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Linearシナリオのハイパーパラメータチューニング。
    config.py から設定を取得する。引数で明示的に指定した場合は上書き。
    """
    cfg = get_config()

    _N = N if N is not None else cfg.common.N
    _T = T if T is not None else cfg.common.T
    _sparsity = sparsity if sparsity is not None else cfg.common.sparsity
    _max_weight = max_weight if max_weight is not None else cfg.common.max_weight
    _std_e = std_e if std_e is not None else cfg.common.std_e

    generator_kwargs = {
        "N": _N,
        "T": _T,
        "sparsity": _sparsity,
        "max_weight": _max_weight,
        "std_e": _std_e,
        "s_type": cfg.data_gen.s_type,
        "t_min": cfg.data_gen.t_min,
        "t_max": cfg.data_gen.t_max,
        "z_dist": cfg.data_gen.z_dist,
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
