import numpy as np
import cvxpy as cp
from scipy.linalg import norm
from tqdm import tqdm
from utils import project_to_zero_diagonal_symmetric

class TimeVaryingSEM:
    def __init__(self, N, S_0, r, q, rho, mu_lambda, show_progress=True, name = "pp_nonsparse_nonoverlap"):
        self.N = N
        self.l = N * (N - 1) // 2
        self.S = S_0

        self.r = r # window size
        self.q = q # number of processors

        self.rho = rho
        self.w = 1 / q

        self.mu_lambda = mu_lambda
        assert self.mu_lambda > 0 and self.mu_lambda < 2

        # tqdmの表示／非表示を制御するフラグ
        self.show_progress = show_progress
        self.name = name

    def g_l(self, x): # per processor
        return norm(x - self.S @ x) ** 2 / 2 - self.rho

    def subgrad_projection(self, x):
        if self.g_l(x) > 0:
            subgrad = self.S @ x @ x.T - x @ x.T
            return self.S - (self.g_l(x) / (norm(subgrad) ** 2)) * subgrad
        else:
            return self.S

    def parallel_projection(self, X_partial):
        in_all_C_l = True
        sum_weighted_projection_sp = np.zeros((self.N, self.N))

        numerator = 0.0
        denominator = 0.0

        for i in range(self.q):
            x_per_processor = X_partial[:, i * self.q: i * self.q + self.r]

            projection_sp = self.subgrad_projection(x_per_processor)
            
            sum_weighted_projection_sp += self.w * projection_sp
            
            numerator += self.w * norm(projection_sp - self.S) ** 2

            if self.g_l(x_per_processor) > 0:
                in_all_C_l = False

        if not in_all_C_l:
            assert numerator > 0
            denominator = norm(sum_weighted_projection_sp - self.S) ** 2
            M_k = numerator / denominator
            self.S = self.S + self.mu_lambda * M_k * (sum_weighted_projection_sp - self.S)
            np.fill_diagonal(self.S, 0)  # Ensure diagonal elements are zero
        else:
            M_k = 1
            self.S = self.S + M_k * (sum_weighted_projection_sp - self.S)

    def run(self, X):
        estimates = []
        
        # tqdmを使うか通常のイテレータにするかをフラグで切り替え
        if self.show_progress:
            iterator = tqdm(X.T, desc=self.name)
        else:
            iterator = X.T

        for t, x in enumerate(iterator):
            if t - self.q * self.r >= 0:
                self.parallel_projection(X[:, t - self.q * self.r: t])
            else:
                self.parallel_projection(X[:, :t+1])

            estimates.append(self.S.copy())
        
        return estimates




