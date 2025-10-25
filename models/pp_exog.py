import numpy as np
from scipy.linalg import norm


class PPExogenousSEM:
    def __init__(self, N: int, S_init: np.ndarray, b_init: np.ndarray, r: int, q: int, rho: float, mu_lambda: float):
        self.N = N
        self.S = S_init.copy()  # 非対称、対角0
        np.fill_diagonal(self.S, 0.0)
        self.b = b_init.copy()  # 対角成分 b_i
        self.r = r
        self.q = q
        self.rho = rho
        self.mu_lambda = mu_lambda
        self.w = 1.0 / q

    def g_l(self, Y_block: np.ndarray, U_block: np.ndarray) -> float:
        # 残差 R = Y - S Y - B U, ここで B = diag(b)
        BU = (self.b.reshape(-1, 1) * U_block)
        R = Y_block - (self.S @ Y_block) - BU
        return 0.5 * (norm(R) ** 2) - self.rho

    def subgrad_projection(self, Y_block: np.ndarray, U_block: np.ndarray):
        # サブグラディエントを S と b に対して計算
        BU = (self.b.reshape(-1, 1) * U_block)
        R = Y_block - (self.S @ Y_block) - BU
        if 0.5 * (norm(R) ** 2) - self.rho > 0:
            # 勾配（最小二乗）: d/dS 0.5||R||^2 = -(R Y^T), d/db 0.5||R||^2 = -(R ⊙ U) で列和
            grad_S = -(R @ Y_block.T)
            grad_b = -np.sum(R * U_block, axis=1)
            subgrad_norm_sq = (norm(grad_S) ** 2) + (norm(grad_b) ** 2)
            if subgrad_norm_sq == 0:
                return self.S.copy(), self.b.copy()
            step = (0.5 * (norm(R) ** 2) - self.rho) / subgrad_norm_sq
            S_new = self.S - step * grad_S
            np.fill_diagonal(S_new, 0.0)
            b_new = self.b - step * grad_b
            return S_new, b_new
        else:
            return self.S.copy(), self.b.copy()

    def parallel_projection(self, Y_window: np.ndarray, U_window: np.ndarray):
        in_all_C_l = True
        sum_S = np.zeros_like(self.S)
        sum_b = np.zeros_like(self.b)
        numerator = 0.0

        # q 個のブロックに分割（等幅でスライス）
        T_w = Y_window.shape[1]
        block_size = max(1, T_w // self.q)
        for i in range(self.q):
            start = i * block_size
            end = T_w if i == self.q - 1 else (i + 1) * block_size
            Y_block = Y_window[:, start:end]
            U_block = U_window[:, start:end]

            S_proj, b_proj = self.subgrad_projection(Y_block, U_block)
            sum_S += self.w * S_proj
            sum_b += self.w * b_proj

            # 進捗のための numerator: 重み付き差分ノルム二乗の和
            numerator += self.w * ((norm(S_proj - self.S) ** 2) + (norm(b_proj - self.b) ** 2))

            if self.g_l(Y_block, U_block) > 0:
                in_all_C_l = False

        if not in_all_C_l:
            # 分母: まとめての差分
            denom_S = norm(sum_S - self.S) ** 2
            denom_b = norm(sum_b - self.b) ** 2
            denominator = denom_S + denom_b
            if denominator == 0:
                M_k = 0.0
            else:
                M_k = numerator / denominator
            self.S = self.S + self.mu_lambda * M_k * (sum_S - self.S)
            np.fill_diagonal(self.S, 0.0)
            self.b = self.b + self.mu_lambda * M_k * (sum_b - self.b)
        else:
            self.S = self.S + (sum_S - self.S)
            np.fill_diagonal(self.S, 0.0)
            self.b = self.b + (sum_b - self.b)

    def run(self, Y: np.ndarray, U: np.ndarray):
        S_estimates = []
        b_estimates = []
        for t in range(Y.shape[1]):
            if t + 1 >= self.r:
                Yw = Y[:, t + 1 - self.r:t + 1]
                Uw = U[:, t + 1 - self.r:t + 1]
            else:
                Yw = Y[:, :t + 1]
                Uw = U[:, :t + 1]
            self.parallel_projection(Yw, Uw)
            S_estimates.append(self.S.copy())
            b_estimates.append(self.b.copy())
        return S_estimates, b_estimates



