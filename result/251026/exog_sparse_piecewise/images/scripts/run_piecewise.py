from __future__ import annotations

import argparse
import os
import sys
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from code.data_gen import generate_piecewise_X_with_exog
from models.pp_exog import PPExogenousSEM
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import backup_script, create_result_dir, make_result_filename, save_json


@dataclass
class ExperimentConfig:
    run_pc: bool = True
    run_co: bool = True
    run_sgd: bool = True
    run_pg: bool = True
    run_pp: bool = True
    num_trials: int = 100
    N: int = 20
    T: int = 1000
    sparsity: float = 0.7
    max_weight: float = 0.5
    std_e: float = 0.05
    K: int = 4
    seed: int = 3
    hyperparams: Dict[str, Dict[str, Any]] | None = None
    hyperparam_path: Optional[Path] = None


DATA_GEN_DEFAULTS = {
    "s_type": "random",
    "t_min": 0.5,
    "t_max": 1.0,
    "z_dist": "uniform01",
}


def _fmt(value: object) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _print_block(title: str, items: Dict[str, object]) -> None:
    if not items:
        return
    print(f"--- {title} ---")
    for key, value in items.items():
        print(f"{key:>24}: {_fmt(value)}")


def print_piecewise_summary(
    config: ExperimentConfig,
    run_flags: Dict[str, bool],
    hyperparams: Dict[str, Dict[str, Any]],
) -> None:
    print("=== Experiment Configuration ===")
    common_items: Dict[str, object] = {
        "Scenario": "piecewise",
        "Hyperparam JSON": config.hyperparam_path,
        "Num Trials": config.num_trials,
        "Seed (base)": config.seed,
        "N": config.N,
        "T": config.T,
        "sparsity": config.sparsity,
        "max_weight": config.max_weight,
        "std_e": config.std_e,
    }
    _print_block("Common Parameters", common_items)
    _print_block("Scenario Parameters", {"K": config.K})
    flag_items = {name.upper(): state for name, state in ((k, "ON" if v else "OFF") for k, v in run_flags.items())}
    _print_block("Run Flags", flag_items)
    for method_key, params in hyperparams.items():
        label = f"{method_key.upper()} Hyperparams"
        _print_block(label, params)
    _print_block("Data Generation", DATA_GEN_DEFAULTS)
    output_meta = {
        "maxweight": common_items["max_weight"],
        "stde": common_items["std_e"],
        "mulambda": hyperparams.get("pp", {}).get("mu_lambda"),
    }
    _print_block("Output Meta Keys", output_meta)
    _print_block("Output Root", {"result_dir": "./result/exog_sparse_piecewise"})
    print("------------------------------")


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
    parser.add_argument("--run_pg", action="store_true", help="PG法を実行")
    parser.add_argument("--no_pg", action="store_true", help="PG法をスキップ")
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
        hyperparam_path=args.hyperparam_json,
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

    if args.no_pg:
        config.run_pg = False
    elif args.run_pg:
        config.run_pg = True

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
    run_pg_flag = config.run_pg
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
    pg_cfg = hyperparams.get("pg", {})

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
    lambda_pg = float(pg_cfg.get("lambda_reg", 1e-3))
    step_scale_pg = float(pg_cfg.get("step_scale", 1.0))
    step_size_pg_raw = pg_cfg.get("step_size", None)
    step_size_pg = float(step_size_pg_raw) if step_size_pg_raw is not None else None
    max_iter_pg = int(pg_cfg.get("max_iter", 500))
    tol_pg = float(pg_cfg.get("tol", 1e-4))
    use_fista_pg = _coerce_bool(pg_cfg.get("use_fista", True), default=True)
    use_backtracking_pg = _coerce_bool(pg_cfg.get("use_backtracking", False), default=False)

    print_piecewise_summary(
        config=config,
        run_flags={
            "pp": run_pp_flag,
            "pc": run_pc_flag,
            "co": run_co_flag,
            "sgd": run_sgd_flag,
            "pg": run_pg_flag,
        },
        hyperparams={
            "pp": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda},
            "pc": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C},
            "co": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C},
            "sgd": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_sgd": beta_sgd, "C": C},
            "pg": {
                "lambda_reg": lambda_pg,
                "step_scale": step_scale_pg,
                "step_size": step_size_pg,
                "use_fista": use_fista_pg,
                "use_backtracking": use_backtracking_pg,
                "max_iter": max_iter_pg,
                "tol": tol_pg,
            },
        },
    )

    S0_pc = np.zeros((N, N))

    def run_trial(trial_seed: int):
        rng = np.random.default_rng(trial_seed)
        S_series, B_true, U, Y = generate_piecewise_X_with_exog(
            N=N,
            T=T,
            sparsity=sparsity,
            max_weight=max_weight,
            std_e=std_e,
            K=K,
            s_type="random",
            t_min=0.5,
            t_max=1.0,
            z_dist="uniform01",
            rng=rng,
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
                T_init=B_true,
            )
            estimates_pc, _ = pc.run(X, U)
            error_pc = [
                (np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['pc'] = error_pc
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
                T_init=B_true,
            )
            estimates_co, _ = co.run(X, U)
            error_co = [
                (np.linalg.norm(estimates_co[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['co'] = error_co
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
                T_init=B_true,
            )
            estimates_sgd, _ = sgd.run(X, U)
            error_sgd = [
                (np.linalg.norm(estimates_sgd[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['sgd'] = error_sgd
        if run_pg_flag:
            X = Y
            pg_config = ProximalGradientConfig(
                lambda_reg=lambda_pg,
                step_size=step_size_pg,
                step_scale=step_scale_pg,
                max_iter=max_iter_pg,
                tol=tol_pg,
                use_fista=use_fista_pg,
                use_backtracking=use_backtracking_pg,
                show_progress=False,
                name="pg_baseline",
            )
            pg_model = ProximalGradientBatchSEM(N, pg_config)
            estimates_pg, _ = pg_model.run(X, U)
            error_pg = [
                (np.linalg.norm(estimates_pg[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
                for t in range(T)
            ]
            errors['pg'] = error_pg
        return errors

    trial_seeds = [seed + i for i in range(num_trials)]
    error_pp_total = np.zeros(T) if run_pp_flag else None
    error_pc_total = np.zeros(T) if run_pc_flag else None
    error_co_total = np.zeros(T) if run_co_flag else None
    error_sgd_total = np.zeros(T) if run_sgd_flag else None
    error_pg_total = np.zeros(T) if run_pg_flag else None

    with tqdm_joblib(tqdm(desc="Progress", total=num_trials)):
        results = Parallel(n_jobs=-1, batch_size=1, prefer="threads")(delayed(run_trial)(ts) for ts in trial_seeds)

    for errs in results:
        if run_pp_flag:
            error_pp_total += np.array(errs['pp'])
        if run_pc_flag:
            error_pc_total += np.array(errs['pc'])
        if run_co_flag:
            error_co_total += np.array(errs['co'])
        if run_sgd_flag:
            error_sgd_total += np.array(errs['sgd'])
        if run_pg_flag:
            error_pg_total += np.array(errs['pg'])

    if run_pp_flag:
        error_pp_mean = error_pp_total / num_trials
    if run_pc_flag:
        error_pc_mean = error_pc_total / num_trials
    if run_co_flag:
        error_co_mean = error_co_total / num_trials
    if run_sgd_flag:
        error_sgd_mean = error_sgd_total / num_trials
    if run_pg_flag:
        error_pg_mean = error_pg_total / num_trials

    plt.figure(figsize=(10, 6))
    if run_co_flag:
        plt.plot(error_co_mean, color='blue', label='Correction Only')
    if run_pc_flag:
        plt.plot(error_pc_mean, color='limegreen', label='Prediction Correction')
    if run_sgd_flag:
        plt.plot(error_sgd_mean, color='cyan', label='SGD')
    if run_pg_flag:
        plt.plot(error_pg_mean, color='magenta', label='ProxGrad')
    if run_pp_flag:
        plt.plot(error_pp_mean, color='red', label='Proposed (PP)')
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('Average NSE')
    plt.grid(True, which='both')
    plt.legend()

    run_started_at = datetime.now()
    timestamp = run_started_at.strftime('%Y%m%d_%H%M%S')
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

    scripts_dir = Path(result_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copies: Dict[str, str] = {}
    run_script_copy = backup_script(Path(__file__), scripts_dir)
    script_copies["run_piecewise"] = str(run_script_copy)
    data_gen_path = Path(__file__).resolve().parent / "data_gen.py"
    if data_gen_path.exists():
        data_gen_copy = backup_script(data_gen_path, scripts_dir)
        script_copies["data_gen"] = str(data_gen_copy)
    if config.hyperparam_path is not None and config.hyperparam_path.is_file():
        hyper_copy = backup_script(config.hyperparam_path, scripts_dir)
        script_copies["hyperparams_json"] = str(hyper_copy)

    metadata = {
        "created_at": run_started_at.isoformat(),
        "command": sys.argv,
        "config": {
            "num_trials": num_trials,
            "seed_base": seed,
            "trial_seeds": trial_seeds,
            "N": N,
            "T": T,
            "sparsity": sparsity,
            "max_weight": max_weight,
            "std_e": std_e,
            "K": K,
        },
        "methods": {
            "pp": {"enabled": run_pp_flag, "hyperparams": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda}},
            "pc": {"enabled": run_pc_flag, "hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C}},
            "co": {"enabled": run_co_flag, "hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C}},
            "sgd": {"enabled": run_sgd_flag, "hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_sgd": beta_sgd, "C": C}},
            "pg": {
                "enabled": run_pg_flag,
                "hyperparams": {
                    "lambda_reg": lambda_pg,
                    "step_scale": step_scale_pg,
                    "step_size": step_size_pg,
                    "use_fista": use_fista_pg,
                    "use_backtracking": use_backtracking_pg,
                    "max_iter": max_iter_pg,
                    "tol": tol_pg,
                },
            },
        },
        "generator": {
            "function": "code.data_gen.generate_piecewise_X_with_exog",
            "kwargs": {
                "s_type": "random",
                "t_min": 0.5,
                "t_max": 1.0,
                "z_dist": "uniform01",
            },
        },
        "results": {
            "figure": filename,
            "figure_path": str(Path(result_dir) / filename),
            "metrics": {
                "pp": error_pp_mean.tolist() if run_pp_flag else None,
                "pc": error_pc_mean.tolist() if run_pc_flag else None,
                "co": error_co_mean.tolist() if run_co_flag else None,
                "sgd": error_sgd_mean.tolist() if run_sgd_flag else None,
                "pg": error_pg_mean.tolist() if run_pg_flag else None,
            },
        },
        "snapshots": script_copies,
        "hyperparam_json": str(config.hyperparam_path) if config.hyperparam_path is not None else None,
        "result_dir": str(result_dir),
    }
    meta_name = f"{Path(filename).stem}_meta.json"
    save_json(metadata, Path(result_dir), name=meta_name)


if __name__ == "__main__":
    main()
