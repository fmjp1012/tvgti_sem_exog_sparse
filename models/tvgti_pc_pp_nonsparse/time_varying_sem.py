import numpy as np
from scipy.linalg import norm
from tqdm import tqdm
from utils import elimination_matrix_hh, duplication_matrix_hh, project_to_zero_diagonal_symmetric

class TimeVaryingSEM:
    def __init__(self, N, S_0, lambda_reg, alpha, beta, gamma, P, C, window_size):
        self.N = N
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.P = P
        self.C = C
        self.window_size = window_size

        # Vectorization: hollow half-vectorization
        self.l = N * (N - 1) // 2
        self.D_h = duplication_matrix_hh(N).tocsc()
        self.D_h_T = self.D_h.T
        self.E_h = elimination_matrix_hh(N).tocsc()
        self.E_h_T = self.E_h.T
        
        # Initialize the graph shift operator as a hollow symmetric matrix
        self.S = S_0
        # Initialize the vectorized S
        self.s = self.E_h @ self.S.flatten()

        # Initialize empirical covariance matrix
        self.Sigma_t = np.eye(N)
        self.Sigma_prev = np.eye(N)
        self.sigma_t = self.E_h @ self.Sigma_t.flatten()
        self.tr_sigma_t = np.trace(self.Sigma_t)
        self.tr_sigma_prev = self.tr_sigma_t
        self.Q_t = np.zeros((self.l, self.l))
        self.Q_prev = np.zeros((self.l, self.l))

        self.grad = np.zeros(self.l)
        self.hessian = np.zeros((self.l, self.l))
        self.td_grad = np.zeros(self.l)
        
    def update_covariance(self, x):
        x = x.reshape(-1, 1)
        self.Sigma_t = self.gamma * self.Sigma_prev + (1 - self.gamma) * (x @ x.T)
        self.sigma_t = self.E_h @ self.Sigma_t.flatten()
        self.tr_sigma_t = np.trace(self.Sigma_t)
    
    def compute_gradient(self, s):
        self.grad = self.Q_t @ s - 2 * (self.sigma_t)
    
    def compute_hessian(self):
        Sigma_kron_I = np.kron(self.Sigma_t, np.eye(self.N))
        self.Q_t = self.D_h_T @ Sigma_kron_I @ self.D_h
        self.hessian = self.Q_t.copy()

    def compute_time_derivative_gradient(self):
        self.td_grad = (self.Q_t - self.Q_prev) @ self.s - 2 * (self.tr_sigma_t - self.tr_sigma_prev)

    def prediction_step(self):
        s_pred = self.s.copy()
        self.compute_hessian()
        self.compute_gradient(self.s)
        self.compute_time_derivative_gradient()
        
        for p in range(self.P):
            grad_approx = self.grad + self.hessian @ (s_pred - self.s) + self.td_grad
            s_pred = s_pred - self.alpha * grad_approx
        
        self.s = s_pred
    
    def correction_step(self):
        s_corr = self.s.copy()
        
        for c in range(self.C):
            self.compute_gradient(s_corr)
            s_corr = s_corr - self.beta * self.grad
        
        self.s = s_corr

    def pp_step(self, x):
        subgrad = self.S @ x @ x.T - x @ x.T
        self.S = self.S - self.beta * (subgrad)
        self.S = project_to_zero_diagonal_symmetric(self.S)
        self.s = self.E_h @ self.S.flatten()

    def pp_step_window(self, X_partial):
        subgrad = self.S @ X_partial @ X_partial.T - X_partial @ X_partial.T
        self.S = self.S - self.beta * (subgrad)
        self.S = project_to_zero_diagonal_symmetric(self.S)
        self.s = self.E_h @ self.S.flatten()
    
    def run(self, X):
        estimates = []
        for t, x in enumerate(tqdm(X.T)):
            self.Sigma_prev = self.Sigma_t.copy()
            self.tr_sigma_prev = self.tr_sigma_t
            self.Q_prev = self.Q_t.copy()
            
            self.prediction_step()
            self.update_covariance(x)

            if self.window_size > 1 and t >= self.window_size - 1:
                self.pp_step_window(X[:, t - self.window_size: t+1])
            else:
                self.pp_step(x)
            
            S_flat = self.D_h @ self.s
            S_matrix = np.zeros((self.N, self.N))
            idx = 0
            for i in range(self.N):
                for j in range(i+1, self.N):
                    S_matrix[i, j] = self.s[idx]
                    S_matrix[j, i] = self.s[idx]
                    idx += 1
            self.S = S_matrix
            
            estimates.append(self.S.copy())
        
        return estimates




