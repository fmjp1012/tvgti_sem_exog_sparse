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

    def subgrad_projection(self, X_partial, rho):
        # Directed variant placeholder, reuse non-directed formula
        subgrad = self.S @ X_partial @ X_partial.T - X_partial @ X_partial.T
        return self.S - (norm(X_partial - self.S @ X_partial) - rho) * subgrad / (norm(subgrad) ** 2)

    def solve_offline_min(self, x):
        N = x.shape[0]
        S = cp.Variable((N, N))
        objective = cp.Minimize(cp.norm(x - S@x, 'fro'))
        constraints = []
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("CVXPY did not find an optimal solution.")
        return norm(x - S.value @ x) ** 2 / 2
        
    def run(self, X):
        estimates = []
        for t, x in enumerate(tqdm(X.T)):
            if self.window_size > 1 and t >= self.window_size - 1:
                self.S = self.subgrad_projection(X[:, t - self.window_size + 1: t+1], self.rho)
            estimates.append(self.S.copy())
        return estimates




