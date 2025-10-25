import numpy as np
from tqdm import tqdm
from utils import elimination_matrix_hh, duplication_matrix_hh

class TimeVaryingSEM:
    def __init__(self, N, S_0, alpha, beta, gamma, P, C, show_progress=True, name="pc_nonsparse"):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.P = P
        self.C = C
        
        # tqdmの表示を制御するフラグ
        self.show_progress = show_progress
        self.name = name

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
        self.Sigma_t = np.zeros((N, N))
        self.Sigma_prev = np.zeros((N, N))
        
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
            # Gradient of the approximate function
            grad_approx = self.grad + self.hessian @ (s_pred - self.s) + self.td_grad
            # Update step
            s_pred = s_pred - self.alpha * grad_approx
        
        self.s = s_pred
    
    def correction_step(self):
        s_corr = self.s.copy()
        
        for c in range(self.C):
            self.compute_gradient(s_corr)
            s_corr = s_corr - self.beta * self.grad
        
        self.s = s_corr

    def f(self):
        term1 = 0.5 * self.s.T @ self.Q_t @ self.s
        term2 = -2 * self.s.T @ self.sigma_t
        term3 = 0.5 * self.tr_sigma_t
        return term1 + term2 + term3
    
    def run(self, X):
        estimates = []
        cost_values = []
        
        # tqdm を使うかどうかを self.show_progress で制御
        if self.show_progress:
            iterator = tqdm(X.T, desc=self.name)
        else:
            # tqdmを使わずに通常のforループ
            iterator = X.T
        
        for t, x in enumerate(iterator):
            self.Sigma_prev = self.Sigma_t.copy()
            self.tr_sigma_prev = self.tr_sigma_t
            self.Q_prev = self.Q_t.copy()
            
            # Prediction step
            self.prediction_step()

            # Update empirical covariance
            self.update_covariance(x)
            
            # Correction step
            self.correction_step()
            
            # Reconstruct the symmetric hollow S matrix
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
            cost_values.append(self.f())
        
        return estimates, cost_values




