import sys
import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from code.config import get_config
from code.hyperparam_utils import load_hyperparams_json, resolve_hyperparams
from code.data_gen import generate_linear_X_with_exog
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from utils.io.plotting import apply_style, plot_heatmaps
from utils.io.results import create_result_dir, backup_script, make_result_filename, save_json


def main():
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)
    run_wall_start = time.perf_counter()

    if len(sys.argv) > 1:
        raise SystemExit("CLI 引数は使用できません。code/config.py を編集してください。")

    cfg = get_config()
    hp = resolve_hyperparams(load_hyperparams_json(cfg.hyperparam_json), cfg)

    N = int(cfg.common.N)
    T = int(cfg.common.T)
    sparsity = float(cfg.common.sparsity)
    max_weight = float(cfg.common.max_weight)
    std_e = float(cfg.common.std_e)
    seed = int(cfg.common.seed)
    rng = np.random.default_rng(seed)

    S_series, T_mat, Z, Y = generate_linear_X_with_exog(
        N=N,
        T=T,
        sparsity=sparsity,
        max_weight=max_weight,
        std_e=std_e,
        s_type=cfg.data_gen.s_type,
        t_min=cfg.data_gen.t_min,
        t_max=cfg.data_gen.t_max,
        z_dist=cfg.data_gen.z_dist,
        rng=rng,
    )

    S0 = np.zeros((N, N)); b0 = np.ones(N)
    r = int(hp.pp.r)
    q = int(hp.pp.q)
    rho = float(hp.pp.rho)
    mu_lambda = float(hp.pp.mu_lambda)
    pp = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = pp.run(Y, Z)

    # PC/CO/SGD baselines
    X = Y
    S0_pc = np.zeros((N, N))
    lambda_reg = float(hp.pc.lambda_reg)
    alpha = float(hp.pc.alpha)
    beta = float(hp.pc.beta)
    gamma = float(hp.pc.gamma)
    P = int(hp.pc.P)
    C = int(hp.pc.C)
    pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline", T_init=T_mat)
    estimates_pc, _ = pc.run(X, Z)
    beta_co = float(hp.co.beta_co)
    co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline", T_init=T_mat)
    estimates_co, _ = co.run(X, Z)
    beta_sgd = float(hp.sgd.beta_sgd)
    sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline", T_init=T_mat)
    estimates_sgd, _ = sgd.run(X, Z)

    err_pp = [
        float((np.linalg.norm(S_hat_list[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]
    err_pc = [
        float((np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]
    err_co = [
        float((np.linalg.norm(estimates_co[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
        for t in range(T)
    ]
    err_sgd = [
        float((np.linalg.norm(estimates_sgd[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12))
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
    filename = make_result_filename(
        prefix="linear_once",
        params={
            "N": N,
            "T": T,
            "sparsity": sparsity,
            "maxweight": max_weight,
            "stde": std_e,
            "seed": seed,
            "r": r,
            "q": q,
            "rho": rho,
            "mulambda": mu_lambda,
        },
        suffix=".png",
    )
    print(filename)
    result_dir = create_result_dir(cfg.output.result_root, cfg.scripts.run_linear_once_output_subdir, extra_tag='images')
    figure_path = Path(result_dir) / filename
    plt.savefig(str(figure_path))
    plt.show()
    
    # ヒートマップ表示（最終時刻）
    cfg = get_config()
    if cfg.output.save_heatmap:
        t_idx = T - 1
        heatmap_matrices = {
            'True': S_series[t_idx],
            'PP': S_hat_list[t_idx],
            'PC': estimates_pc[t_idx],
            'CO': estimates_co[t_idx],
            'SGD': estimates_sgd[t_idx],
        }
        heatmap_filename = filename.replace(".png", f"_heatmap_t{t_idx}.png")
        plot_heatmaps(
            matrices=heatmap_matrices,
            save_path=Path(result_dir) / heatmap_filename,
            title=f"Estimated vs True at t={t_idx}",
            show=True,
        )

    scripts_dir = Path(result_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copies = {
        "run_linear_once": str(backup_script(Path(__file__), scripts_dir)),
    }
    data_gen_path = Path(__file__).resolve().parent / "data_gen.py"
    if data_gen_path.exists():
        script_copies["data_gen"] = str(backup_script(data_gen_path, scripts_dir))

    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "command": sys.argv,
        "config": {
            "N": N,
            "T": T,
            "sparsity": sparsity,
            "max_weight": max_weight,
            "std_e": std_e,
            "seed": seed,
        },
        "methods": {
            "pp": {"hyperparams": {"r": r, "q": q, "rho": rho, "mu_lambda": mu_lambda}},
            "pc": {"hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta": beta, "gamma": gamma, "P": P, "C": C}},
            "co": {"hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_co": beta_co, "gamma": gamma, "C": C}},
            "sgd": {"hyperparams": {"lambda_reg": lambda_reg, "alpha": alpha, "beta_sgd": beta_sgd, "C": C}},
        },
        "generator": {
            "function": "code.data_gen.generate_linear_X_with_exog",
            "kwargs": {
                "t_min": cfg.data_gen.t_min,
                "t_max": cfg.data_gen.t_max,
                "z_dist": cfg.data_gen.z_dist,
            },
        },
        "results": {
            "figure": filename,
            "figure_path": str(figure_path),
            "timings": {
                "overall_sec": float(time.perf_counter() - run_wall_start),
            },
            "metrics": {
                "pp": err_pp,
                "pc": err_pc,
                "co": err_co,
                "sgd": err_sgd,
            },
        },
        "snapshots": script_copies,
        "exogenous_matrix_norm": float(np.linalg.norm(T_mat, ord="fro")),
    }
    save_json(metadata, Path(result_dir), name=f"{figure_path.stem}_meta.json")


if __name__ == "__main__":
    main()
