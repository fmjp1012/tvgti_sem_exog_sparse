"""
実データ（real/）に対して system mismatch と再構成誤差を比較するスクリプト。

- system mismatch (ユーザ指定 1A):
    ||x_t - S_t x_t - T_t z_t||_2^2
  ※ Z=0 の場合は ||x_t - S_t x_t||_2^2

- 再構成誤差 (ユーザ指定 2B):
    x_hat_t = (I - S_t)^{-1} (T_t z_t)
    ||x_t - x_hat_t||_2^2
  ※ Z=0 の場合は x_hat_t = 0 になりやすい点に注意。

- 再構成誤差 (ユーザ指定 2C):
    欠損（マスク）を入れた上で、(I - S_t) x_t = T_t z_t を最小二乗で満たすよう
    欠損成分を復元し、欠損成分上の誤差を計算。

使用方法:
    /Users/fmjp/venv/default/bin/python -m code.run_real_mismatch_recon
    /Users/fmjp/venv/default/bin/python -m code.run_real_mismatch_recon --hyperparam_json path/to/hyperparams.json
    /Users/fmjp/venv/default/bin/python -m code.run_real_mismatch_recon --N 50 --T 2000
    /Users/fmjp/venv/default/bin/python -m code.run_real_mismatch_recon --test

注意:
- 既定の N, T, 実行手法フラグ等は code/real_config.py を参照します。
- 実データの外生入力はとりあえず Z=0 とします（ユーザ指定 4A）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from code.config import get_config
from code.hyperparam_utils import hyperparams_to_dict, load_hyperparams_json, resolve_hyperparams
from code.real_config import get_real_config
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import backup_script, create_result_dir, make_result_filename, save_json
from utils.offline_solver import solve_offline_sem_lasso_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realデータで system mismatch / reconstruction error を比較")
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="軽量テスト実行（N/Tを小さく上書き）",
    )
    parser.add_argument(
        "--hyperparam_json",
        type=Path,
        default=None,
        help="ハイパーパラメータJSONのパス（省略時はconfig.pyのデフォルト値）",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=None,
        help="入力CSV（1列目timestamp、以降が系列）",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="使用するノード数（CSVの先頭からN列分）。省略時はconfig.pyのN。",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="使用する時系列長。省略時はconfig.pyのT。",
    )
    parser.add_argument(
        "--tail",
        "--use_last_rows",
        dest="use_last_rows",
        action="store_true",
        default=None,
        help="T個を末尾から切り出す（デフォルトON）",
    )
    parser.add_argument(
        "--head",
        dest="use_last_rows",
        action="store_false",
        help="T個を先頭から切り出す",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help="各ノード系列をz-score標準化する（デフォルトON）",
    )
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        default=False,
        help="標準化を無効化（--standardize より優先）",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        default=False,
        help="log1p変換（標準化より前に適用）",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=None,
        help="再構成誤差(2C)の欠損率（0-1）",
    )
    parser.add_argument(
        "--mask_seed",
        type=int,
        default=None,
        help="マスク用seed（省略時はconfig.pyのseedを使用）",
    )
    parser.add_argument(
        "--recon_ridge",
        type=float,
        default=None,
        help="再構成(2C)のリッジ（数値安定化）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="図を表示する（保存は常に行う）",
    )
    return parser.parse_args()


def _load_real_csv(csv_path: Path, N: int) -> Tuple[np.ndarray, List[str]]:
    """CSVから (N, T_total) の行列を作る（timestamp列は無視）。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    # ヘッダ行から列名を取る（timestamp + MT_*** ...）
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    if len(header) < 2:
        raise ValueError("CSVヘッダが不正です（列が足りません）")

    available_cols = header[1:]
    if N > len(available_cols):
        N = len(available_cols)

    # timestamp列を避けて 1..N を読む
    usecols = tuple(range(1, 1 + N))
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, usecols=usecols)
    # data: (T_total, N)

    colnames = available_cols[:N]
    X = data.T.astype(float, copy=False)  # (N, T_total)

    # NaN があれば列平均で補完
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=1)
        inds = np.where(np.isnan(X))
        X[inds] = col_mean[inds[0]]

    return X, colnames


def _preprocess(X: np.ndarray, log1p: bool, standardize: bool, eps: float = 1e-12) -> np.ndarray:
    Xp = X.copy()
    if log1p:
        Xp = np.log1p(np.maximum(Xp, 0.0))
    if standardize:
        mu = Xp.mean(axis=1, keepdims=True)
        sigma = Xp.std(axis=1, keepdims=True)
        sigma = np.maximum(sigma, eps)
        Xp = (Xp - mu) / sigma
    return Xp


def _safe_solve(A: np.ndarray, b: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """(A)x=b を安全に解く。失敗時は最小二乗。"""
    if ridge and ridge > 0.0:
        # (A^T A + ridge I) x = A^T b
        AtA = A.T @ A
        AtA = AtA + ridge * np.eye(AtA.shape[0])
        return np.linalg.solve(AtA, A.T @ b)

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _system_mismatch_series(
    S_list: List[np.ndarray],
    tdiag_list: List[np.ndarray],
    X: np.ndarray,
    Z: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    T = X.shape[1]
    raw: List[float] = []
    rel: List[float] = []
    for t in range(T):
        S_t = S_list[t]
        tdiag = tdiag_list[t]
        x = X[:, t]
        z = Z[:, t]
        pred = S_t @ x + (tdiag * z)
        r = x - pred
        num = float(np.dot(r, r))
        raw.append(num / len(x))
        rel.append(num / (float(np.dot(x, x)) + eps))
    return {"raw_mse": raw, "rel": rel}


def _recon_error_B_series(
    S_list: List[np.ndarray],
    tdiag_list: List[np.ndarray],
    X: np.ndarray,
    Z: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    T = X.shape[1]
    N = X.shape[0]
    I = np.eye(N)
    raw: List[float] = []
    rel: List[float] = []

    for t in range(T):
        S_t = S_list[t]
        tdiag = tdiag_list[t]
        x = X[:, t]
        z = Z[:, t]
        rhs = tdiag * z
        x_hat = _safe_solve(I - S_t, rhs)
        r = x - x_hat
        num = float(np.dot(r, r))
        raw.append(num / N)
        rel.append(num / (float(np.dot(x, x)) + eps))

    return {"raw_mse": raw, "rel": rel}


def _recon_error_C_series(
    S_list: List[np.ndarray],
    tdiag_list: List[np.ndarray],
    X: np.ndarray,
    Z: np.ndarray,
    mask_ratio: float,
    rng: np.random.Generator,
    ridge: float = 1e-6,
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    """欠損復元（2C）。欠損成分上の誤差を時系列で返す。"""
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio は (0,1) の範囲で指定してください")

    T = X.shape[1]
    N = X.shape[0]
    k_miss = max(1, int(round(mask_ratio * N)))

    raw: List[float] = []
    rel: List[float] = []

    for t in range(T):
        S_t = S_list[t]
        tdiag = tdiag_list[t]
        x_true = X[:, t]
        z = Z[:, t]
        rhs = tdiag * z

        miss = rng.choice(N, size=k_miss, replace=False)
        obs_mask = np.ones(N, dtype=bool)
        obs_mask[miss] = False
        obs = np.where(obs_mask)[0]

        x_obs = x_true[obs]

        # (I - S) x = rhs を最小二乗で満たすように欠損成分を推定
        A = np.eye(N) - S_t
        A_u = A[:, miss]  # (N, k)
        A_o = A[:, obs]   # (N, N-k)
        y = rhs - (A_o @ x_obs)

        # argmin ||A_u x_u - y||^2 + ridge||x_u||^2
        x_u_hat = _safe_solve(A_u, y, ridge=ridge)
        x_u_true = x_true[miss]

        diff = x_u_hat - x_u_true
        num = float(np.dot(diff, diff))
        raw.append(num / len(miss))
        denom = float(np.dot(x_u_true, x_u_true)) + eps
        rel.append(num / denom)

    return {"raw_mse": raw, "rel": rel}


def _summarize(series: List[float]) -> Dict[str, float]:
    arr = np.asarray(series, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    cfg = get_config()
    real_cfg = get_real_config()

    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    hyperparam_path = args.hyperparam_json if args.hyperparam_json is not None else real_cfg.hyperparam_json
    loaded_hp = load_hyperparams_json(hyperparam_path)
    hp = resolve_hyperparams(loaded_hp, cfg)

    # N/T は config.py がデフォルト。CLI指定があれば上書き。
    # --test はさらに上書き（短時間で終わるためのプリセット）。
    N_cfg = int(real_cfg.data.N)
    T_cfg = int(real_cfg.data.T)
    if args.N is not None:
        N_cfg = int(args.N)
    if args.T is not None:
        T_cfg = int(args.T)
    if bool(args.test):
        N_cfg = int(min(N_cfg, 5))
        T_cfg = int(min(T_cfg, 200))

    csv_path = args.csv_path if args.csv_path is not None else real_cfg.data.csv_path
    X_full, colnames = _load_real_csv(csv_path, N=N_cfg)

    # 切り出し方向（CLI優先、未指定なら real_config.data.slice_mode）
    use_last_rows = args.use_last_rows
    if use_last_rows is None:
        use_last_rows = str(real_cfg.data.slice_mode).lower() != "head"

    if use_last_rows:
        T_use = min(T_cfg, X_full.shape[1])
        X_raw = X_full[:, -T_use:]
    else:
        T_use = min(T_cfg, X_full.shape[1])
        X_raw = X_full[:, :T_use]

    log1p = bool(args.log1p) if args.log1p else bool(real_cfg.data.log1p)
    standardize = (bool(args.standardize) or bool(real_cfg.data.standardize)) and (not bool(args.no_standardize))
    X = _preprocess(X_raw, log1p=log1p, standardize=standardize)

    N = X.shape[0]
    T = X.shape[1]

    # 外生入力はとりあえず Z=0
    Z = np.zeros_like(X)

    # 2C の欠損設定（実データconfigがデフォルト、CLIで上書き）
    mask_ratio = float(real_cfg.masking.mask_ratio) if args.mask_ratio is None else float(args.mask_ratio)
    recon_ridge = float(real_cfg.masking.recon_ridge) if args.recon_ridge is None else float(args.recon_ridge)

    # 実行フラグ（実データ設定から）
    run_pp = bool(real_cfg.methods.pp)
    run_pc = bool(real_cfg.methods.pc)
    run_co = bool(real_cfg.methods.co)
    run_sgd = bool(real_cfg.methods.sgd)
    run_pg = bool(real_cfg.methods.pg)

    # 乱数（欠損復元用）
    default_mask_seed = real_cfg.masking.mask_seed if real_cfg.masking.mask_seed is not None else int(cfg.common.seed)
    mask_seed = int(default_mask_seed) if args.mask_seed is None else int(args.mask_seed)
    rng = np.random.default_rng(mask_seed)

    # 出力ディレクトリ
    scenario_name = real_cfg.output.subdir_real_test if bool(args.test) else real_cfg.output.subdir_real
    result_dir = create_result_dir(real_cfg.output.result_root, scenario_name, extra_tag="images")

    # ------------------------------------------------------------
    # 推定を実行
    # ------------------------------------------------------------
    estimates_S: Dict[str, List[np.ndarray]] = {}
    estimates_Tdiag: Dict[str, List[np.ndarray]] = {}

    # Offline（定常）
    offline_lambda = hp.offline_lambda_l1
    if offline_lambda is None:
        offline_space = cfg.search_spaces.offline.offline_lambda_l1
        offline_lambda = math.sqrt(float(offline_space.low) * float(offline_space.high))

    S_offline = solve_offline_sem_lasso_batch(X, Z, offline_lambda)
    estimates_S["Offline"] = [S_offline.copy() for _ in range(T)]
    estimates_Tdiag["Offline"] = [np.zeros(N, dtype=float) for _ in range(T)]

    # PP
    if run_pp:
        S0 = np.zeros((N, N))
        b0 = np.ones(N)
        pp_model = PPExogenousSEM(
            N,
            S0,
            b0,
            r=hp.pp.r,
            q=hp.pp.q,
            rho=hp.pp.rho,
            mu_lambda=hp.pp.mu_lambda,
            lambda_S=hp.pp.lambda_S,
        )
        S_list, T_list = pp_model.run(X, Z)
        estimates_S["PP"] = S_list
        estimates_Tdiag["PP"] = [np.diag(Tm).copy() for Tm in T_list]

    # PC / CO / SGD
    def _run_pc_like(
        name: str,
        lambda_reg: float,
        alpha: float,
        beta: float,
        gamma: float,
        P: int,
        C: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        S0_pc = np.zeros((N, N))
        T_init = np.eye(N)
        model = PCSEM(
            N,
            S0_pc,
            lambda_reg,
            alpha,
            beta,
            gamma,
            P,
            C,
            show_progress=False,
            name=name,
            T_init=T_init,
        )
        S_list, _ = model.run(X, Z)
        tdiag_list = [b.copy() for b in model.b_history] if model.b_history else [np.zeros(N) for _ in range(T)]
        if len(tdiag_list) != T:
            # 念のため揃える
            tdiag_list = (tdiag_list + [tdiag_list[-1].copy()])[:T]
        return S_list, tdiag_list

    if run_pc:
        S_list, tdiag_list = _run_pc_like(
            name="pc_real",
            lambda_reg=hp.pc.lambda_reg,
            alpha=hp.pc.alpha,
            beta=hp.pc.beta,
            gamma=hp.pc.gamma,
            P=hp.pc.P,
            C=hp.pc.C,
        )
        estimates_S["PC"] = S_list
        estimates_Tdiag["PC"] = tdiag_list

    if run_co:
        S_list, tdiag_list = _run_pc_like(
            name="co_real",
            lambda_reg=hp.co.lambda_reg,
            alpha=hp.co.alpha,
            beta=hp.co.beta_co,
            gamma=hp.co.gamma,
            P=0,
            C=hp.co.C,
        )
        estimates_S["CO"] = S_list
        estimates_Tdiag["CO"] = tdiag_list

    if run_sgd:
        S_list, tdiag_list = _run_pc_like(
            name="sgd_real",
            lambda_reg=hp.sgd.lambda_reg,
            alpha=hp.sgd.alpha,
            beta=hp.sgd.beta_sgd,
            gamma=0.0,
            P=0,
            C=hp.sgd.C,
        )
        estimates_S["SGD"] = S_list
        estimates_Tdiag["SGD"] = tdiag_list

    # PG
    if run_pg:
        pg_config = ProximalGradientConfig(
            lambda_reg=hp.pg.lambda_reg,
            step_size=hp.pg.step_size,
            step_scale=hp.pg.step_scale,
            max_iter=hp.pg.max_iter,
            tol=hp.pg.tol,
            use_fista=hp.pg.use_fista,
            use_backtracking=hp.pg.use_backtracking,
            show_progress=False,
            name="pg_real",
        )
        pg_model = ProximalGradientBatchSEM(N, pg_config)
        S_list, info = pg_model.run(X, Z)
        tdiag_final = np.asarray(info.get("T_diag", np.zeros(N)), dtype=float)
        estimates_S["PG"] = S_list
        estimates_Tdiag["PG"] = [tdiag_final.copy() for _ in range(T)]

    # ------------------------------------------------------------
    # 指標計算
    # ------------------------------------------------------------
    metrics: Dict[str, Any] = {
        "system_mismatch": {},
        "recon_error_B": {},
        "recon_error_C": {},
        "summaries": {},
    }

    for method_name, S_list in estimates_S.items():
        tdiag_list = estimates_Tdiag.get(method_name)
        if tdiag_list is None:
            tdiag_list = [np.zeros(N, dtype=float) for _ in range(T)]

        sm = _system_mismatch_series(S_list, tdiag_list, X, Z)
        rb = _recon_error_B_series(S_list, tdiag_list, X, Z)
        rc = _recon_error_C_series(
            S_list,
            tdiag_list,
            X,
            Z,
            mask_ratio=float(mask_ratio),
            rng=rng,
            ridge=float(recon_ridge),
        )

        metrics["system_mismatch"][method_name] = sm
        metrics["recon_error_B"][method_name] = rb
        metrics["recon_error_C"][method_name] = rc

        metrics["summaries"][method_name] = {
            "system_mismatch_raw": _summarize(sm["raw_mse"]),
            "system_mismatch_rel": _summarize(sm["rel"]),
            "recon_B_raw": _summarize(rb["raw_mse"]),
            "recon_B_rel": _summarize(rb["rel"]),
            "recon_C_raw": _summarize(rc["raw_mse"]),
            "recon_C_rel": _summarize(rc["rel"]),
        }

    # ------------------------------------------------------------
    # プロット
    # ------------------------------------------------------------
    plot_order = [
        ("CO", "blue", "Correction Only"),
        ("PC", "limegreen", "Prediction Correction"),
        ("SGD", "cyan", "SGD"),
        ("PG", "magenta", "ProxGrad"),
        ("PP", "red", "Proposed (PP)"),
        ("Offline", "black", "Offline"),
    ]

    def _plot_metric(title: str, ylabel: str, key1: str, key2: str, out_name: str) -> Path:
        plt.figure(figsize=(10, 6))
        for method, color, label in plot_order:
            if method not in metrics[key1]:
                continue
            y = metrics[key1][method][key2]
            plt.plot(y, color=color, label=label)
        plt.yscale("log")
        plt.xlim(left=0, right=T)
        plt.xlabel("t")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        save_path = Path(result_dir) / out_name
        plt.savefig(str(save_path), bbox_inches="tight")
        if bool(args.show) or bool(real_cfg.run.show):
            plt.show()
        else:
            plt.close()
        return save_path

    filename_base = make_result_filename(
        prefix="real",
        params={
            "N": N,
            "T": T,
            "csv": csv_path.stem,
            "mask": mask_ratio,
            "seed": cfg.common.seed,
            "log1p": bool(log1p),
            "zscore": bool(standardize),
        },
        suffix=".png",
    ).replace(".png", "")

    fig_paths: Dict[str, str] = {}

    if bool(real_cfg.plot.plot_system_mismatch):
        fig_paths["system_mismatch_rel"] = str(
            _plot_metric(
                title="System mismatch (relative)",
                ylabel=r"$\\frac{\\|x_t - S_t x_t - T_t z_t\\|_2^2}{\\|x_t\\|_2^2}$",
                key1="system_mismatch",
                key2="rel",
                out_name=f"{filename_base}_system_mismatch_rel.png",
            )
        )

    if bool(real_cfg.plot.plot_recon_B):
        fig_paths["recon_B_rel"] = str(
            _plot_metric(
                title="Reconstruction error B (relative)",
                ylabel=r"$\\frac{\\|x_t - (I-S_t)^{-1}(T_t z_t)\\|_2^2}{\\|x_t\\|_2^2}$",
                key1="recon_error_B",
                key2="rel",
                out_name=f"{filename_base}_reconB_rel.png",
            )
        )

    if bool(real_cfg.plot.plot_recon_C):
        fig_paths["recon_C_rel"] = str(
            _plot_metric(
                title="Reconstruction error C (masked, relative)",
                ylabel=r"$\\frac{\\|x_{\\mathrm{miss}}-\\hat x_{\\mathrm{miss}}\\|_2^2}{\\|x_{\\mathrm{miss}}\\|_2^2}$",
                key1="recon_error_C",
                key2="rel",
                out_name=f"{filename_base}_reconC_rel.png",
            )
        )

    # ------------------------------------------------------------
    # メタデータ保存（既存run_*に倣う）
    # ------------------------------------------------------------
    run_started_at = datetime.now()

    scripts_dir = Path(result_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copies: Dict[str, str] = {}

    script_copies["run_real_mismatch_recon"] = str(backup_script(Path(__file__), scripts_dir))

    config_path = Path(__file__).resolve().parent / "config.py"
    if config_path.exists():
        script_copies["config"] = str(backup_script(config_path, scripts_dir))

    real_config_path = Path(__file__).resolve().parent / "real_config.py"
    if real_config_path.exists():
        script_copies["real_config"] = str(backup_script(real_config_path, scripts_dir))

    if hyperparam_path is not None and Path(hyperparam_path).is_file():
        script_copies["hyperparams_json"] = str(backup_script(Path(hyperparam_path), scripts_dir))

    metadata: Dict[str, Any] = {
        "created_at": run_started_at.isoformat(),
        "command": sys.argv,
        "scenario": scenario_name,
        "data": {
            "csv_path": str(csv_path),
            "columns_used": colnames,
            "N": N,
            "T": T,
            "slice": "tail" if use_last_rows else "head",
            "preprocess": {"log1p": bool(log1p), "zscore": bool(standardize)},
            "Z": "zeros",
        },
        "masking": {
            "mask_ratio": float(mask_ratio),
            "mask_seed": int(mask_seed),
            "recon_ridge": float(recon_ridge),
        },
        "methods": {
            "enabled_flags": {
                "pp": run_pp,
                "pc": run_pc,
                "co": run_co,
                "sgd": run_sgd,
                "pg": run_pg,
                "offline": True,
            },
            "hyperparams": {**hyperparams_to_dict(hp), "offline_lambda_l1": float(offline_lambda)},
        },
        "results": {
            "figures": fig_paths,
            "metrics": metrics,
            "summaries": metrics["summaries"],
        },
        "snapshots": script_copies,
        "result_dir": str(result_dir),
    }

    meta_name = f"{filename_base}_meta.json"
    save_json(metadata, Path(result_dir), name=meta_name)

    print("保存先:")
    print(f"  result_dir = {result_dir}")
    for k, v in fig_paths.items():
        print(f"  {k}: {v}")
    print(f"  meta: {Path(result_dir) / meta_name}")


if __name__ == "__main__":
    main()
