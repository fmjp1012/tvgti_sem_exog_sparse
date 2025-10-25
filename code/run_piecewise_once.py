import os
import shutil
import datetime
import argparse
import json
import cvxpy as cp

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from code.data_gen import generate_piecewise_Y_with_exog
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varying_sem import TimeVaryingSEM as PCSEM
from utils.io.plotting import apply_style
from utils.io.results import create_result_dir, backup_script
from models.tvgti_pc.time_varying_sem import TimeVaryingSEMWithL1Correction as PCSEM_L1C


def main():
    # プロット設定
    apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)

    # パラメータ
    N = 20
    T = 1000
    sparsity = 0.7
    max_weight = 0.5
    std_e = 0.05
    K = 4
    seed = 3
    np.random.seed(seed)

    # 外部設定の読み込み（JSON/CLI）
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="ハイパラ設定JSONのパス")
    parser.add_argument("--show_offline_line", type=int, default=0, help="オフラインNSEの横線を描画する(1)/しない(0)")
    parser.add_argument("--save_heatmaps", type=int, default=0, help="推定と真値のヒートマップ比較を保存する(1)/しない(0)")
    parser.add_argument("--heatmap_time", type=int, default=-1, help="ヒートマップを描画する時刻t（-1で最終時刻）")
    # 問題スケール上書き
    parser.add_argument("--N", type=int, default=None, help="ノード数Nを上書き（既定20）")
    # PPのハイパラ上書き用（任意）
    parser.add_argument("--pp_r", type=int, default=None, help="PPのrを上書き")
    parser.add_argument("--pp_q", type=int, default=None, help="PPのqを上書き")
    parser.add_argument("--pp_rho", type=float, default=None, help="PPのrhoを上書き")
    parser.add_argument("--pp_mu_lambda", type=float, default=None, help="PPのmu_lambdaを上書き")
    args, _ = parser.parse_known_args()

    # 既定ハイパラ（fallback）
    r = 50; q = 5; rho = 1e-3; mu_lambda = 0.05
    lambda_reg = 1e-3; alpha = 1e-2; beta = 1e-2; gamma = 0.9; P = 1; C = 1
    beta_co = 0.02
    beta_sgd = 0.0269

    # JSON 設定で上書き
    if args.config is not None and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        pp_cfg = cfg.get('pp', {})
        r = pp_cfg.get('r', r)
        q = pp_cfg.get('q', q)
        rho = pp_cfg.get('rho', rho)
        mu_lambda = pp_cfg.get('mu_lambda', mu_lambda)

        pc_cfg = cfg.get('pc', {})
        lambda_reg = pc_cfg.get('lambda_reg', lambda_reg)
        alpha = pc_cfg.get('alpha', alpha)
        beta = pc_cfg.get('beta', beta)
        gamma = pc_cfg.get('gamma', gamma)
        P = pc_cfg.get('P', P)
        C = pc_cfg.get('C', C)

        co_cfg = cfg.get('co', {})
        beta_co = co_cfg.get('beta_co', beta_co)

        sgd_cfg = cfg.get('sgd', {})
        beta_sgd = sgd_cfg.get('beta_sgd', beta_sgd)

    # CLIでスケール/PPを上書き
    if args.N is not None:
        N = int(args.N)
    if args.pp_r is not None:
        r = args.pp_r
    if args.pp_q is not None:
        q = args.pp_q
    if args.pp_rho is not None:
        rho = args.pp_rho
    if args.pp_mu_lambda is not None:
        mu_lambda = args.pp_mu_lambda

    # 生成
    S_series, B_true, U, Y = generate_piecewise_Y_with_exog(
        N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
        s_type="random", b_min=0.5, b_max=1.0, u_dist="uniform01"
    )

    # PP
    S0 = np.zeros((N, N))
    b0 = np.ones(N)
    pp = PPExogenousSEM(N, S0, b0, r=r, q=q, rho=rho, mu_lambda=mu_lambda)
    S_hat_list, _ = pp.run(Y, U)

    # PC/CO/SGD baseline
    X = Y
    S0_pc = np.zeros((N, N))
    pc = PCSEM(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline")
    estimates_pc, _ = pc.run(X)
    co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co, gamma, 0, C, show_progress=False, name="co_baseline")
    estimates_co, _ = co.run(X)
    sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd, 0.0, 0, C, show_progress=False, name="sgd_baseline")
    estimates_sgd, _ = sgd.run(X)

    # 新手法: PC + L1 correction
    pc_l1c = PCSEM_L1C(N, S0_pc, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_l1corr")
    estimates_pc_l1c, _ = pc_l1c.run(X)

    # 誤差プロット（Frobenius error）
    err_pp = [np.linalg.norm(S_hat_list[t] - S_series[t], ord='fro') for t in range(T)]
    err_pc = [np.linalg.norm(estimates_pc[t] - S_series[t], ord='fro') for t in range(T)]
    err_co = [np.linalg.norm(estimates_co[t] - S_series[t], ord='fro') for t in range(T)]
    err_sgd = [np.linalg.norm(estimates_sgd[t] - S_series[t], ord='fro') for t in range(T)]
    err_pc_l1c = [np.linalg.norm(estimates_pc_l1c[t] - S_series[t], ord='fro') for t in range(T)]

    # オフライン最適（SEM+L1）の平均誤差を計算（オプションの横線用）
    def compute_offline_mean_error_l1(X_mat: np.ndarray, S_series_true, lambda_l1: float) -> float:
        N_loc, T_loc = X_mat.shape
        S_var = cp.Variable((N_loc, N_loc), symmetric=True)
        data_fit = (1/(2*T_loc)) * cp.norm(X_mat - S_var @ X_mat, 'fro')
        objective = data_fit + lambda_l1 * cp.norm1(S_var)
        constraints = [cp.diag(S_var) == 0]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if S_var.value is None:
            prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
        S_opt = S_var.value
        errors = [np.linalg.norm(S_opt - S_series_true[t], ord='fro') for t in range(T_loc)]
        return float(np.mean(errors))

    plt.figure(figsize=(10, 6))
    plt.plot(err_co, color='blue', label='Correction Only')
    plt.plot(err_pc, color='limegreen', label='Prediction Correction')
    plt.plot(err_sgd, color='cyan', label='SGD')
    plt.plot(err_pp, color='red', label='Proposed (PP)')
    plt.plot(err_pc_l1c, color='magenta', label='PC + L1 correction')
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('t')
    plt.ylabel('Frobenius error')
    plt.grid(True, which='both')
    if args.show_offline_line:
        offline_err = compute_offline_mean_error_l1(X, S_series, lambda_reg)
        plt.axhline(y=offline_err, color='black', linestyle='--', alpha=0.8, label='Offline SEM+L1 (mean)')
    plt.legend()

    # 保存
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    notebook_filename = os.path.basename(__file__)
    filename = (f'timestamp{timestamp}_result_N{N}_notebook_filename{notebook_filename}_'
                f'T{T}_K{K}_seed{seed}_r{r}_q{q}_rho{rho}_mulambda{mu_lambda}.png')
    print(filename)
    result_dir = create_result_dir(Path('./result'), 'exog_sparse_piecewise_once', extra_tag='images')
    plt.savefig(str(Path(result_dir) / filename))
    plt.show()

    backup_script(Path(__file__), Path(result_dir))

    # ヒートマップ比較の保存（オプション）
    if args.save_heatmaps:
        t_idx = args.heatmap_time if args.heatmap_time >= 0 else (T - 1)
        t_idx = max(0, min(T - 1, t_idx))
        mats = {
            'True': S_series[t_idx],
            'PP': S_hat_list[t_idx],
            'PC': estimates_pc[t_idx],
            'CO': estimates_co[t_idx],
            'SGD': estimates_sgd[t_idx],
            'PC+L1C': estimates_pc_l1c[t_idx],
        }
        max_abs = max(float(np.max(np.abs(m))) for m in mats.values()) + 1e-12
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        axes = axes.ravel()
        for ax, (title, mat) in zip(axes, mats.items()):
            im = ax.imshow(mat, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='equal', interpolation='nearest')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.9, pad=0.02)
        # ラベルは不要
        cbar.set_label("")
        heatmap_filename = (f'timestamp{timestamp}_heatmaps_N{N}_notebook_filename{notebook_filename}_'
                            f'T{T}_K{K}_seed{seed}_r{r}_q{q}_rho{rho}_mulambda{mu_lambda}_t{t_idx}.png')
        fig.savefig(os.path.join(save_path, heatmap_filename))
        # 画面にも表示
        plt.show()


if __name__ == "__main__":
    main()


