import numpy as np
import matplotlib.pyplot as plt

from models.tvgti_pc.time_varying_sem import TimeVaryingSEM as PCSEM
from code.data_gen import generate_piecewise_Y_with_exog


def main():
    # 外因性を無視して既存PCをベースラインとして実行（従来法）
    N = 20
    T = 2000
    sparsity = 0.7
    max_weight = 0.5
    std_e = 0.05
    K = 3

    S_series, B_true, U, Y = generate_piecewise_Y_with_exog(
        N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
        s_type="random"
    )

    # 既存PCは X を入力とする想定のため Y をそのまま X として渡す
    X = Y

    # 適当な初期値（既存の慣例に合わせる）
    S0 = np.zeros((N, N))
    lambda_reg = 1e-3
    alpha = 1e-2
    beta = 1e-2
    gamma = 0.9
    P = 1
    C = 1

    pc = PCSEM(N, S0, lambda_reg, alpha, beta, gamma, P, C, show_progress=False, name="pc_baseline")
    estimates_pc, _ = pc.run(X)

    nse_series = [
        (np.linalg.norm(estimates_pc[t] - S_series[t]) ** 2) / (np.linalg.norm(S_series[t]) ** 2 + 1e-12)
        for t in range(len(estimates_pc))
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(nse_series, color='limegreen', label='PC baseline (NSE)')
    plt.yscale('log')
    plt.xlim(left=0, right=len(frob_series))
    plt.xlabel('t')
    plt.ylabel('NSE')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


