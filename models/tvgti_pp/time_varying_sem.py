import numpy as np
import cvxpy as cp
from scipy.linalg import norm
from tqdm import tqdm

class TimeVaryingSEM:
    def __init__(self, N, S_0, beta, window_size):
        self.N = N
        self.beta = beta
        self.window_size = window_size

        self.l = N * (N - 1) // 2
        self.S = S_0
        self.rho = 1e-10
        self.plus_rho = 10

    def parallel_projection(self, X_partial):
        in_all_C_l = True
        sum_subgrad = np.zeros((self.N, self.N))

        numerator = 0.0
        denominator = 0.0

        for x in X_partial.T:
            rho = self.solve_offline_min(x) + self.plus_rho
            projection_sp = self.subgrad_projection(x, rho)
            proj_diff_norm_sq = np.linalg.norm(projection_sp - self.S) ** 2

            if norm(x - self.S @ x) ** 2 > rho:
                in_all_C_l = False
                numerator += proj_diff_norm_sq
                denominator += projection_sp / X_partial.shape[1]
                subgrad = (self.S @ x @ x.T) - x @ x.T
                sum_subgrad += subgrad

        if not in_all_C_l:
            M_k = numerator / (denominator - self.S) ** 2
            self.S = self.S - M_k * (sum_subgrad / X_partial.shape[1])
            np.fill_diagonal(self.S, 0)
        else:
            M_k = 1
            self.S = self.S - M_k * (sum_subgrad / X_partial.shape[1])

    def subgrad_projection(self, x, rho):
        if norm(x - self.S @ x) ** 2 > rho:
            subgrad = self.S @ x @ x.T - x @ x.T
            return self.S - (norm(x - self.S @ x) - rho) * subgrad / (norm(subgrad) ** 2)
        else:
            return self.S

    def solve_offline_min(self, x):
        N = x.shape[0]
        S = cp.Variable((N, N), symmetric=True)
        objective = cp.Minimize(cp.norm(x - S@x, 'fro'))
        constraints = [cp.diag(S) == 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("CVXPY did not find an optimal solution.")
        np.fill_diagonal(S.value, 0)
        return norm(x - S.value @ x) ** 2 / 2
        
    def run(self, X):
        estimates = []
        for t, x in enumerate(tqdm(X.T)):
            if self.window_size > 1 and t >= self.window_size - 1:
                self.S = self.subgrad_projection(X[:, t - self.window_size + 1: t+1], self.rho)
                np.fill_diagonal(self.S, 0)
            estimates.append(self.S.copy())
        return estimates




