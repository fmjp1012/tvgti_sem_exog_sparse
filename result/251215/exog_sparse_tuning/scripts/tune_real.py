"""
実データ（real/）に対するハイパーパラメータチューニング（Optuna）。

目的関数（ユーザ指定）:
  system mismatch (relative) の平均を最小化
    mean_t  ||x_t - S_t x_t - T_t z_t||_2^2 / (||x_t||_2^2 + eps)
  ※ 現状 Z=0 なので ||x_t - S_t x_t|| が主。

設定:
  - code/real_config.py を編集して設定する（N/T、手法ON/OFF、tuning_trials など）

実行:
  /Users/fmjp/venv/default/bin/python -m code.tune_real
  make tune_real
"""

from __future__ import annotations

import datetime as dt
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna

from code.config import get_config, get_search_spaces_dict
from code.hyperparam_tuning import save_best_hyperparams
from code.real_config import get_real_config
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.offline_solver import solve_offline_sem_lasso_batch

# 実データの読み込み/前処理は run_real_mismatch_recon と同じものを使う
from code.run_real_mismatch_recon import _load_real_csv, _preprocess


def _suggest_float(trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]) -> float:
    return float(
        trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    )


def _suggest_int(trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]) -> int:
    step = int(spec.get("step", 1))
    return int(trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step))


def _suggest_categorical(trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]) -> Any:
    choices = spec.get("choices")
    if choices is None:
        raise ValueError(f"{name} の choices が未定義です")
    return trial.suggest_categorical(name, list(choices))


def _suggest(trial: optuna.trial.Trial, name: str, spec: Dict[str, Any]) -> Any:
    t = spec.get("type", "float")
    if t == "float":
        return _suggest_float(trial, name, spec)
    if t == "int":
        return _suggest_int(trial, name, spec)
    if t == "categorical":
        return _suggest_categorical(trial, name, spec)
    raise ValueError(f"{name} の type={t} は未サポートです")


def _system_mismatch_rel_mean(S_list: List[np.ndarray], X: np.ndarray, burn_in: int = 0, eps: float = 1e-12) -> float:
    """Z=0 前提で system mismatch relative の平均を返す。"""
    T = X.shape[1]
    start = max(0, min(int(burn_in), T - 1))
    vals = []
    for t in range(start, T):
        x = X[:, t]
        r = x - S_list[t] @ x
        num = float(np.dot(r, r))
        den = float(np.dot(x, x)) + eps
        vals.append(num / den)
    return float(np.mean(vals)) if vals else float("inf")


def _slice_for_tuning(X: np.ndarray, trunc_T: int, use_last: bool) -> np.ndarray:
    if trunc_T <= 0:
        return X
    T = X.shape[1]
    T_use = min(T, int(trunc_T))
    return X[:, -T_use:] if use_last else X[:, :T_use]


def tune_real() -> Tuple[Optional[Path], Dict[str, Any]]:
    cfg = get_config()
    real_cfg = get_real_config()
    spaces = get_search_spaces_dict()

    # データ準備
    X_full, _ = _load_real_csv(real_cfg.data.csv_path, N=int(real_cfg.data.N))
    use_last = str(real_cfg.data.slice_mode).lower() != "head"
    X_slice = X_full[:, -min(int(real_cfg.data.T), X_full.shape[1]) :] if use_last else X_full[:, : min(int(real_cfg.data.T), X_full.shape[1])]
    X_proc = _preprocess(X_slice, log1p=bool(real_cfg.data.log1p), standardize=bool(real_cfg.data.standardize))
    X_tune = _slice_for_tuning(X_proc, trunc_T=int(real_cfg.tuning.truncation_T), use_last=use_last)

    N = X_tune.shape[0]
    T = X_tune.shape[1]
    Z = np.zeros_like(X_tune)

    tuning_seed = int(real_cfg.tuning.tuning_seed)
    n_trials = int(real_cfg.tuning.tuning_trials)
    burn_in = int(real_cfg.tuning.burn_in)

    best: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "objective": "system_mismatch_rel_mean",
        "data": {
            "csv_path": str(real_cfg.data.csv_path),
            "slice_mode": real_cfg.data.slice_mode,
            "N": int(real_cfg.data.N),
            "T": int(real_cfg.data.T),
            "tuning_truncation_T": int(real_cfg.tuning.truncation_T),
            "log1p": bool(real_cfg.data.log1p),
            "standardize": bool(real_cfg.data.standardize),
        },
        "tuning": {
            "tuning_trials": n_trials,
            "tuning_seed": tuning_seed,
            "burn_in": burn_in,
        },
        "enabled_methods": {
            "pp": bool(real_cfg.methods.pp),
            "pc": bool(real_cfg.methods.pc),
            "co": bool(real_cfg.methods.co),
            "sgd": bool(real_cfg.methods.sgd),
            "pg": bool(real_cfg.methods.pg),
            "offline": True,
        },
        "results": {},
    }

    # ------------------------------------------------------------
    # offline_lambda_l1（定常）をチューニング
    # ------------------------------------------------------------
    offline_space = spaces["offline"]["offline_lambda_l1"]

    def objective_offline(trial: optuna.trial.Trial) -> float:
        lam = float(_suggest(trial, "offline_lambda_l1", offline_space))
        S = solve_offline_sem_lasso_batch(X_tune, Z, lam)
        S_list = [S for _ in range(T)]
        return _system_mismatch_rel_mean(S_list, X_tune, burn_in=burn_in)

    study_off = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tuning_seed))
    study_off.optimize(objective_offline, n_trials=n_trials)
    best_off = float(study_off.best_params["offline_lambda_l1"])
    best["offline_lambda_l1"] = best_off  # 既存のJSON慣例に合わせる
    summary["results"]["offline"] = {
        "best_value": float(study_off.best_value),
        "best_params": dict(study_off.best_params),
    }

    # ------------------------------------------------------------
    # PP
    # ------------------------------------------------------------
    if real_cfg.methods.pp:
        pp_r = int(cfg.hyperparams.pp.r)
        pp_q = int(cfg.hyperparams.pp.q)

        def objective_pp(trial: optuna.trial.Trial) -> float:
            rho = float(_suggest(trial, "rho", spaces["pp"]["rho"]))
            mu_lambda = float(_suggest(trial, "mu_lambda", spaces["pp"]["mu_lambda"]))
            lambda_S = float(_suggest(trial, "lambda_S", spaces["pp"]["lambda_S"]))
            S0 = np.zeros((N, N))
            b0 = np.ones(N)
            model = PPExogenousSEM(
                N,
                S0,
                b0,
                r=pp_r,
                q=pp_q,
                rho=rho,
                mu_lambda=mu_lambda,
                lambda_S=lambda_S,
            )
            S_list, _ = model.run(X_tune, Z)
            return _system_mismatch_rel_mean(S_list, X_tune, burn_in=burn_in)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tuning_seed))
        study.optimize(objective_pp, n_trials=n_trials)
        best["pp"] = {
            "r": pp_r,
            "q": pp_q,
            **{k: float(v) for k, v in study.best_params.items()},
        }
        summary["results"]["pp"] = {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    # ------------------------------------------------------------
    # PC（フル）
    # ------------------------------------------------------------
    pc_best: Optional[Dict[str, Any]] = None
    if real_cfg.methods.pc:
        def objective_pc(trial: optuna.trial.Trial) -> float:
            lambda_reg = float(_suggest(trial, "lambda_reg", spaces["pc"]["lambda_reg"]))
            alpha = float(_suggest(trial, "alpha", spaces["pc"]["alpha"]))
            beta = float(_suggest(trial, "beta", spaces["pc"]["beta_pc"]))
            gamma = float(_suggest(trial, "gamma", spaces["pc"]["gamma"]))
            P = int(_suggest(trial, "P", spaces["pc"]["P"]))
            C = int(_suggest(trial, "C", spaces["pc"]["C"]))
            S0 = np.zeros((N, N))
            model = PCSEM(N, S0, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_tune", T_init=np.eye(N))
            S_list, _ = model.run(X_tune, Z)
            return _system_mismatch_rel_mean(S_list, X_tune, burn_in=burn_in)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tuning_seed))
        study.optimize(objective_pc, n_trials=n_trials)
        pc_best = dict(study.best_params)
        best["pc"] = {
            "lambda_reg": float(pc_best["lambda_reg"]),
            "alpha": float(pc_best["alpha"]),
            "beta": float(pc_best["beta"]),
            "gamma": float(pc_best["gamma"]),
            "P": int(pc_best["P"]),
            "C": int(pc_best["C"]),
        }
        summary["results"]["pc"] = {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    # ------------------------------------------------------------
    # CO / SGD は、PCで得た alpha/gamma/C を固定して beta だけチューニング
    # ------------------------------------------------------------
    if real_cfg.methods.co:
        fixed_alpha = float(pc_best["alpha"]) if pc_best is not None else float(cfg.hyperparams.pc.alpha)
        fixed_gamma = float(pc_best["gamma"]) if pc_best is not None else float(cfg.hyperparams.pc.gamma)
        fixed_C = int(pc_best["C"]) if pc_best is not None else int(cfg.hyperparams.pc.C)
        fixed_lambda_reg = float(pc_best["lambda_reg"]) if pc_best is not None else float(cfg.hyperparams.pc.lambda_reg)

        def objective_co(trial: optuna.trial.Trial) -> float:
            beta_co = float(_suggest(trial, "beta_co", spaces["co"]["beta_co"]))
            S0 = np.zeros((N, N))
            model = PCSEM(N, S0, fixed_lambda_reg, fixed_alpha, beta_co, fixed_gamma, 0, fixed_C, show_progress=False, name="co_tune", T_init=np.eye(N))
            S_list, _ = model.run(X_tune, Z)
            return _system_mismatch_rel_mean(S_list, X_tune, burn_in=burn_in)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tuning_seed))
        study.optimize(objective_co, n_trials=n_trials)
        best["co"] = {"beta_co": float(study.best_params["beta_co"])}
        summary["results"]["co"] = {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    if real_cfg.methods.sgd:
        fixed_alpha = float(pc_best["alpha"]) if pc_best is not None else float(cfg.hyperparams.pc.alpha)
        fixed_C = int(pc_best["C"]) if pc_best is not None else int(cfg.hyperparams.pc.C)
        fixed_lambda_reg = float(pc_best["lambda_reg"]) if pc_best is not None else float(cfg.hyperparams.pc.lambda_reg)

        def objective_sgd(trial: optuna.trial.Trial) -> float:
            beta_sgd = float(_suggest(trial, "beta_sgd", spaces["sgd"]["beta_sgd"]))
            S0 = np.zeros((N, N))
            model = PCSEM(N, S0, fixed_lambda_reg, fixed_alpha, beta_sgd, 0.0, 0, fixed_C, show_progress=False, name="sgd_tune", T_init=np.eye(N))
            S_list, _ = model.run(X_tune, Z)
            return _system_mismatch_rel_mean(S_list, X_tune, burn_in=burn_in)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tuning_seed))
        study.optimize(objective_sgd, n_trials=n_trials)
        best["sgd"] = {"beta_sgd": float(study.best_params["beta_sgd"])}
        summary["results"]["sgd"] = {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    # ------------------------------------------------------------
    # PG（バッチ）
    # ------------------------------------------------------------
    if real_cfg.methods.pg:
        def objective_pg(trial: optuna.trial.Trial) -> float:
            lambda_reg = float(_suggest(trial, "lambda_reg", spaces["pg"]["lambda_reg"]))
            step_scale = float(_suggest(trial, "step_scale", spaces["pg"]["step_scale"]))
            use_fista = bool(_suggest(trial, "use_fista", spaces["pg"]["use_fista"]))
            pg_config = ProximalGradientConfig(
                lambda_reg=lambda_reg,
                step_size=None,
                step_scale=step_scale,
                max_iter=int(cfg.hyperparams.pg.max_iter),
                tol=float(cfg.hyperparams.pg.tol),
                use_fista=use_fista,
                use_backtracking=bool(cfg.hyperparams.pg.use_backtracking),
                show_progress=False,
                name="pg_tune",
            )
            model = ProximalGradientBatchSEM(N, pg_config)
            S_list, _ = model.run(X_tune, Z)
            return _system_mismatch_rel_mean(S_list, X_tune, burn_in=burn_in)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tuning_seed))
        study.optimize(objective_pg, n_trials=n_trials)
        params = dict(study.best_params)
        best["pg"] = {
            "lambda_reg": float(params["lambda_reg"]),
            "step_scale": float(params["step_scale"]),
            "use_fista": bool(params["use_fista"]),
        }
        summary["results"]["pg"] = {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    # 保存
    script_paths = {
        "tune_real": Path(__file__),
        "real_config": Path(__file__).resolve().parent / "real_config.py",
        "run_real_mismatch_recon": Path(__file__).resolve().parent / "run_real_mismatch_recon.py",
    }
    out_path = save_best_hyperparams(
        best=best,
        scenario="real",
        result_root=real_cfg.output.result_root,
        subdir="exog_sparse_tuning",
        metadata=summary,
        script_paths=script_paths,
    )
    return out_path, summary


def main() -> None:
    real_cfg = get_real_config()

    if real_cfg.skip_tuning:
        print("[tune_real] skip_tuning=True のためチューニングをスキップします。")
        return

    out_path, _ = tune_real()
    print(f"[tune_real] best hyperparams saved: {out_path}")

    if real_cfg.skip_run:
        print("[tune_real] skip_run=True のため実行をスキップします。")
        return

    # そのまま実データ実行まで回す（同じpythonで -m 実行）
    cmd = [sys.executable, "-m", "code.run_real_mismatch_recon", "--hyperparam_json", str(out_path)]
    print(f"[tune_real] run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

