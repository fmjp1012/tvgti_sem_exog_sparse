import numpy as np
from tqdm import tqdm
from typing import Optional
from utils import elimination_matrix_hh, duplication_matrix_hh, soft_thresholding

class TimeVaryingSEM:
    def __init__(
        self,
        N,
        S_0,
        lambda_reg,
        alpha,
        beta,
        gamma,
        P,
        C,
        show_progress=True,
        name="pc_sparse",
        T_init: Optional[np.ndarray] = None,
        exog_forgetting: Optional[float] = None,
        exog_eps: float = 1e-6,
    ):
        self.N = N
        self.lambda_reg = lambda_reg
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

        # Exogenous handling (optional, enabled when run(X, Z) is used)
        self.use_exog = False
        self.exog_eps = exog_eps
        self.exog_gamma = exog_forgetting if exog_forgetting is not None else gamma
        if T_init is not None:
            self._b_init = np.diag(T_init).copy()
        else:
            self._b_init = None
        self.b = np.zeros(self.N) if self._b_init is None else self._b_init.copy()
        self.z_var = np.full(self.N, self.exog_eps)
        self.xz_cov = np.zeros(self.N)
        self.b_history = []

    def _update_exogenous_effect(self, x_vec: np.ndarray, z_vec: np.ndarray) -> np.ndarray:
        forgetting = self.exog_gamma
        self.z_var = forgetting * self.z_var + (1 - forgetting) * (z_vec ** 2)
        self.xz_cov = forgetting * self.xz_cov + (1 - forgetting) * (x_vec * z_vec)
        denom = np.maximum(self.z_var, self.exog_eps)
        self.b = self.xz_cov / denom
        return x_vec - self.b * z_vec

    def update_covariance(self, x, z=None):
        x_vec = x.reshape(-1)
        if self.use_exog and z is not None:
            z_vec = z.reshape(-1)
            x_vec = self._update_exogenous_effect(x_vec, z_vec)
        x_mat = x_vec.reshape(-1, 1)
        self.Sigma_t = self.gamma * self.Sigma_prev + (1 - self.gamma) * (x_mat @ x_mat.T)
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
            # Apply proximal operator (soft-thresholding)
            s_pred = soft_thresholding(s_pred, 2 * self.beta * self.lambda_reg)
        
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
    
    def run(self, X, Z=None):
        estimates = []
        cost_values = []
        T_total = X.shape[1]

        use_exog = Z is not None
        self.use_exog = use_exog
        if use_exog:
            if Z.shape != X.shape:
                raise ValueError("X と Z の形状が一致しません。")
            if self._b_init is not None:
                self.b = self._b_init.copy()
            else:
                self.b = np.zeros(self.N)
            self.z_var = np.full(self.N, self.exog_eps)
            self.xz_cov = np.zeros(self.N)
            self.b_history = []
        else:
            self.b_history = []
        
        # tqdm を使うかどうかを self.show_progress で制御
        if self.show_progress:
            iterator = tqdm(range(T_total), desc=self.name)
        else:
            iterator = range(T_total)
        
        for t in iterator:
            x = X[:, t]
            z = Z[:, t] if use_exog else None
            self.Sigma_prev = self.Sigma_t.copy()
            self.tr_sigma_prev = self.tr_sigma_t
            self.Q_prev = self.Q_t.copy()
            
            # Prediction step
            self.prediction_step()

            # Update empirical covariance
            self.update_covariance(x, z)
            
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
            if use_exog:
                self.b_history.append(self.b.copy())
        
        return estimates, cost_values


class TimeVaryingSEMWithL1Correction(TimeVaryingSEM):
    def __init__(
        self,
        N,
        S_0,
        lambda_reg,
        alpha,
        beta,
        gamma,
        P,
        C,
        show_progress=True,
        name="pc_sparse_l1corr",
        T_init: Optional[np.ndarray] = None,
        exog_forgetting: Optional[float] = None,
        exog_eps: float = 1e-6,
    ):
        super().__init__(
            N,
            S_0,
            lambda_reg,
            alpha,
            beta,
            gamma,
            P,
            C,
            show_progress,
            name,
            T_init=T_init,
            exog_forgetting=exog_forgetting,
            exog_eps=exog_eps,
        )

    def correction_step(self):
        s_corr = self.s.copy()

        for c in range(self.C):
            self.compute_gradient(s_corr)
            s_corr = s_corr - self.beta * self.grad
            # Apply L1 proximal operator in correction step as well
            s_corr = soft_thresholding(s_corr, 2 * self.beta * self.lambda_reg)

        self.s = s_corr
