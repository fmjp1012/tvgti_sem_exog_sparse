import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm

from utils import soft_thresholding


class PredictionCorrectionSEM:
    """
    Implements the asymmetric, matrix-form prediction–correction algorithm for the
    SEM-Exogenous model described in Section 4.4 of the accompanying manuscript.
    """

    def __init__(
        self,
        N: int,
        S_0: np.ndarray,
        lambda_reg: float,
        alpha: float,
        beta: Optional[float],
        gamma: float,
        P: int,
        C: int,
        show_progress: bool = True,
        name: str = "pc_sem_exog",
        T_init: Optional[np.ndarray] = None,
        exog_forgetting: Optional[float] = None,
        exog_eps: float = 1e-6,
    ):
        self.N = int(N)
        self.lambda_reg = float(lambda_reg)
        self.eta_S = float(alpha)
        self.eta_T = None if beta is None else float(beta)
        self.gamma = float(gamma)
        self.P = int(P)
        self.C = int(C)
        self.show_progress = show_progress
        self.name = name

        # Model parameters
        self.S = self._project_diag_zero(np.array(S_0, dtype=float))
        self.T_diag = (
            np.diag(T_init).astype(float)
            if T_init is not None
            else np.zeros(self.N, dtype=float)
        )

        # EW covariance trackers
        self.Sigma_xx = np.zeros((self.N, self.N), dtype=float)
        self.Sigma_xx_prev = self.Sigma_xx.copy()
        self.Sigma_xz = np.zeros((self.N, self.N), dtype=float)
        self.Sigma_xz_prev = self.Sigma_xz.copy()
        self.Sigma_zz = np.zeros((self.N, self.N), dtype=float)
        self.Sigma_zz_prev = self.Sigma_zz.copy()

        # Covariance increments for the prediction step
        self.delta_xx = np.zeros((self.N, self.N), dtype=float)
        self.delta_xz = np.zeros((self.N, self.N), dtype=float)
        self.delta_zz = np.zeros((self.N, self.N), dtype=float)

        # Exogenous control
        self.use_exog = False
        self.exog_eps = float(exog_eps)
        self.exog_gamma = float(exog_forgetting) if exog_forgetting is not None else self.gamma
        self.b_history: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _project_diag_zero(matrix: np.ndarray) -> np.ndarray:
        mat = matrix.copy()
        np.fill_diagonal(mat, 0.0)
        return mat

    def _prox_l1_offdiag(self, matrix: np.ndarray, threshold: float) -> np.ndarray:
        mat = matrix.copy()
        off_diag_mask = ~np.eye(self.N, dtype=bool)
        mat[off_diag_mask] = soft_thresholding(mat[off_diag_mask], threshold)
        np.fill_diagonal(mat, 0.0)
        return mat

    @staticmethod
    def _left_diag_mul(diagonal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return diagonal[:, None] * matrix

    # ------------------------------------------------------------------ #
    # Gradient related computations
    # ------------------------------------------------------------------ #
    def _gradient_S(
        self,
        S: np.ndarray,
        T_diag: np.ndarray,
        Sigma_xx: np.ndarray,
        Sigma_xz: np.ndarray,
    ) -> np.ndarray:
        grad = S @ Sigma_xx - Sigma_xx
        if self.use_exog:
            Sigma_zx = Sigma_xz.T
            grad += self._left_diag_mul(T_diag, Sigma_zx)
        return grad

    def _gradient_T_diag(
        self,
        S: np.ndarray,
        T_diag: np.ndarray,
        Sigma_xz: np.ndarray,
        Sigma_zz: np.ndarray,
    ) -> np.ndarray:
        if not self.use_exog or self.eta_T is None:
            return np.zeros(self.N, dtype=float)
        term = S @ Sigma_xz + self._left_diag_mul(T_diag, Sigma_zz) - Sigma_xz
        return np.diag(term)

    # ------------------------------------------------------------------ #
    # Algorithmic steps
    # ------------------------------------------------------------------ #
    def _prediction_step(self) -> Tuple[np.ndarray, np.ndarray]:
        S_pred = self.S.copy()
        T_pred = self.T_diag.copy()
        if self.P <= 0:
            return S_pred, T_pred

        grad_td_scale = self.lambda_reg * self.eta_S
        for _ in range(self.P):
            grad_S = self._gradient_S(S_pred, T_pred, self.Sigma_xx, self.Sigma_xz)
            td_grad_S = self._gradient_S(S_pred, T_pred, self.delta_xx, self.delta_xz)
            S_tmp = S_pred - self.eta_S * (grad_S + td_grad_S)
            S_tmp = self._prox_l1_offdiag(S_tmp, grad_td_scale)
            S_pred = self._project_diag_zero(S_tmp)

            if self.use_exog and self.eta_T not in (None, 0.0):
                grad_T = self._gradient_T_diag(S_pred, T_pred, self.Sigma_xz, self.Sigma_zz)
                td_grad_T = self._gradient_T_diag(S_pred, T_pred, self.delta_xz, self.delta_zz)
                T_pred = T_pred - self.eta_T * (grad_T + td_grad_T)
        return S_pred, T_pred

    def _correction_step(self) -> Tuple[np.ndarray, np.ndarray]:
        S_corr = self.S.copy()
        T_corr = self.T_diag.copy()
        if self.C <= 0:
            return S_corr, T_corr

        prox_threshold = self.lambda_reg * self.eta_S
        for _ in range(self.C):
            grad_S = self._gradient_S(S_corr, T_corr, self.Sigma_xx, self.Sigma_xz)
            S_corr = self._prox_l1_offdiag(S_corr - self.eta_S * grad_S, prox_threshold)

            if self.use_exog and self.eta_T not in (None, 0.0):
                grad_T = self._gradient_T_diag(S_corr, T_corr, self.Sigma_xz, self.Sigma_zz)
                T_corr = T_corr - self.eta_T * grad_T

        return S_corr, T_corr

    # ------------------------------------------------------------------ #
    # Covariance update
    # ------------------------------------------------------------------ #
    def _update_covariances(self, x_vec: np.ndarray, z_vec: Optional[np.ndarray]) -> None:
        x_vec = x_vec.reshape(-1)

        old_xx = self.Sigma_xx.copy()
        self.Sigma_xx = self.gamma * self.Sigma_xx + (1.0 - self.gamma) * np.outer(x_vec, x_vec)
        self.Sigma_xx_prev = old_xx
        self.delta_xx = self.Sigma_xx - self.Sigma_xx_prev

        if self.use_exog and z_vec is not None:
            z_vec = z_vec.reshape(-1)

            old_xz = self.Sigma_xz.copy()
            forgetting_exog = self.exog_gamma
            self.Sigma_xz = forgetting_exog * self.Sigma_xz + (1.0 - forgetting_exog) * np.outer(x_vec, z_vec)
            self.Sigma_xz_prev = old_xz
            self.delta_xz = self.Sigma_xz - self.Sigma_xz_prev

            old_zz = self.Sigma_zz.copy()
            self.Sigma_zz = forgetting_exog * self.Sigma_zz + (1.0 - forgetting_exog) * np.outer(z_vec, z_vec)
            self.Sigma_zz_prev = old_zz
            self.delta_zz = self.Sigma_zz - self.Sigma_zz_prev
        else:
            self.Sigma_xz.fill(0.0)
            self.Sigma_xz_prev.fill(0.0)
            self.Sigma_zz.fill(0.0)
            self.Sigma_zz_prev.fill(0.0)
            self.delta_xz.fill(0.0)
            self.delta_zz.fill(0.0)

    # ------------------------------------------------------------------ #
    # Loss computation
    # ------------------------------------------------------------------ #
    def _compute_loss(self, S: np.ndarray, T_diag: np.ndarray) -> float:
        loss = 0.5 * np.trace(S @ self.Sigma_xx @ S.T) - np.trace(self.Sigma_xx @ S.T)

        if self.use_exog:
            T_mat = np.diag(T_diag)
            loss += 0.5 * np.trace(T_mat @ self.Sigma_zz @ T_mat.T)
            loss += np.trace(S @ self.Sigma_xz @ T_mat.T)
            loss -= np.trace(self.Sigma_xz @ T_mat.T)

        return float(loss)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, X: np.ndarray, Z: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[float]]:
        T_total = X.shape[1]
        self.use_exog = Z is not None
        if self.use_exog and Z.shape != X.shape:
            raise ValueError("X と Z の形状が一致しません。")

        estimates: List[np.ndarray] = []
        cost_values: List[float] = []
        self.b_history = []

        iterator = tqdm(range(T_total), desc=self.name) if self.show_progress else range(T_total)

        for t in iterator:
            S_pred, T_pred = self._prediction_step()
            self.S = S_pred
            self.T_diag = T_pred

            x_vec = X[:, t]
            z_vec = Z[:, t] if self.use_exog else None
            self._update_covariances(x_vec, z_vec)

            S_corr, T_corr = self._correction_step()
            self.S = S_corr
            self.T_diag = T_corr

            estimates.append(self.S.copy())
            cost_values.append(self._compute_loss(self.S, self.T_diag))
            if self.use_exog:
                self.b_history.append(self.T_diag.copy())

        return estimates, cost_values
