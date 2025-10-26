from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise soft-thresholding."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _project_hollow(matrix: np.ndarray) -> np.ndarray:
    """Project a square matrix onto the set of hollow matrices (zero diagonal)."""
    mat = matrix.copy()
    np.fill_diagonal(mat, 0.0)
    return mat


@dataclass
class ProximalGradientConfig:
    lambda_reg: float
    step_size: Optional[float] = None
    step_scale: Optional[float] = None
    max_iter: int = 500
    tol: float = 1e-5
    use_fista: bool = True
    use_backtracking: bool = False
    backtracking_beta: float = 0.5
    backtracking_max_iter: int = 20
    show_progress: bool = False
    name: str = "pg_sem_batch"
    record_history: bool = False


class ProximalGradientBatchSEM:
    """Batch proximal-gradient estimator for SEM with exogenous inputs."""

    def __init__(
        self,
        N: int,
        config: ProximalGradientConfig,
        S_init: Optional[np.ndarray] = None,
        T_init: Optional[np.ndarray] = None,
    ):
        self.N = int(N)
        self.config = config
        if config.lambda_reg < 0.0:
            raise ValueError("lambda_reg must be non-negative.")

        self.S = _project_hollow(S_init.astype(float)) if S_init is not None else np.zeros((self.N, self.N), dtype=float)
        if T_init is None:
            self.t_diag = np.zeros(self.N, dtype=float)
        else:
            if T_init.shape != (self.N, self.N):
                raise ValueError("T_init must be an N x N diagonal matrix.")
            self.t_diag = np.diag(T_init).astype(float)

        self.history: Dict[str, List[float]] = {"objective": [], "residual_norm": []}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_residual(self, S: np.ndarray, t_diag: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return S @ X + (t_diag[:, None] * Z) - X

    def _objective(self, S: np.ndarray, t_diag: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
        residual = self._compute_residual(S, t_diag, X, Z)
        loss = 0.5 * np.linalg.norm(residual, ord="fro") ** 2
        off_diag_mask = ~np.eye(self.N, dtype=bool)
        penalty = self.config.lambda_reg * np.sum(np.abs(S[off_diag_mask]))
        return float(loss + penalty)

    def _initialize_step_size(self, X: np.ndarray, Z: np.ndarray) -> float:
        eta = self.config.step_size
        if eta is not None and eta > 0.0:
            return float(eta)

        U = np.vstack((X, Z))
        gram = U @ U.T
        lipschitz = np.linalg.norm(gram, ord=2)
        if not np.isfinite(lipschitz) or lipschitz <= 0.0:
            lipschitz = 1.0
        eta_base = 1.0 / lipschitz
        if self.config.step_scale is not None and self.config.step_scale > 0.0:
            eta_base *= float(self.config.step_scale)
        return eta_base

    def _backtracking_step(
        self,
        S_base: np.ndarray,
        t_base: np.ndarray,
        grad_S: np.ndarray,
        grad_t: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        initial_eta: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        eta = initial_eta
        beta = self.config.backtracking_beta
        max_iter = self.config.backtracking_max_iter
        lambda_reg = self.config.lambda_reg
        f_base = 0.5 * np.linalg.norm(self._compute_residual(S_base, t_base, X, Z), ord="fro") ** 2

        for _ in range(max_iter):
            S_trial = _project_hollow(_soft_threshold(S_base - eta * grad_S, eta * lambda_reg))
            t_trial = t_base - eta * grad_t
            residual_trial = self._compute_residual(S_trial, t_trial, X, Z)

            diff_S = S_trial - S_base
            diff_t = t_trial - t_base
            diff_norm_sq = np.linalg.norm(diff_S, ord="fro") ** 2 + np.linalg.norm(diff_t) ** 2
            quad = f_base + np.sum(grad_S * diff_S) + np.dot(grad_t, diff_t) + 0.5 / eta * diff_norm_sq
            smooth_trial = 0.5 * np.linalg.norm(residual_trial, ord="fro") ** 2

            if smooth_trial <= quad:
                return S_trial, t_trial, eta
            eta *= beta

        return _project_hollow(_soft_threshold(S_base - eta * grad_S, eta * lambda_reg)), t_base - eta * grad_t, eta

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, X: np.ndarray, Z: Optional[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, object]]:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        N, T = X.shape
        if N != self.N:
            raise ValueError(f"X shape {X.shape} incompatible with N={self.N}.")

        if Z is None:
            raise ValueError("Z must be provided for exogenous SEM estimation.")
        if Z.shape != X.shape:
            raise ValueError("X と Z の形状が一致しません。")

        if self.config.record_history:
            self.history["objective"] = []
            self.history["residual_norm"] = []
        eta = self._initialize_step_size(X, Z)
        lambda_reg = self.config.lambda_reg

        S_current = self.S.copy()
        t_current = self.t_diag.copy()

        Y_S = S_current.copy()
        Y_t = t_current.copy()
        theta = 1.0

        for iteration in range(self.config.max_iter):
            base_S = Y_S if self.config.use_fista else S_current
            base_t = Y_t if self.config.use_fista else t_current

            residual = self._compute_residual(base_S, base_t, X, Z)
            grad_S = residual @ X.T
            grad_t = np.sum(residual * Z, axis=1)

            if self.config.use_backtracking:
                S_next, t_next, eta = self._backtracking_step(base_S, base_t, grad_S, grad_t, X, Z, eta)
            else:
                S_tilde = base_S - eta * grad_S
                S_next = _project_hollow(_soft_threshold(S_tilde, eta * lambda_reg))
                t_next = base_t - eta * grad_t

            diff_S = S_next - S_current
            diff_t = t_next - t_current
            rel_norm = np.linalg.norm(diff_S, ord="fro") + np.linalg.norm(diff_t)
            denom = max(1.0, np.linalg.norm(S_current, ord="fro") + np.linalg.norm(t_current))
            rel_change = rel_norm / denom

            if self.config.record_history or self.config.show_progress:
                objective = self._objective(S_next, t_next, X, Z)
                self.history["objective"].append(objective)
                self.history["residual_norm"].append(float(np.linalg.norm(residual, ord="fro")))
                if self.config.show_progress:
                    print(f"[{self.config.name}] iter={iteration} obj={objective:.6e} rel_change={rel_change:.3e}")

            S_prev = S_current
            t_prev = t_current
            S_current = S_next
            t_current = t_next
            if self.config.use_fista:
                theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta**2))
                beta = (theta - 1.0) / theta_next
                Y_S = S_current + beta * (S_current - S_prev)
                Y_t = t_current + beta * (t_current - t_prev)
                theta = theta_next
            else:
                Y_S = S_current
                Y_t = t_current

            if rel_change < self.config.tol:
                break

        S_estimates = [S_current.copy() for _ in range(T)]
        iterations_used = iteration + 1 if "iteration" in locals() else 0
        info: Dict[str, object] = {
            "T_diag": t_current.copy(),
            "objective_history": list(self.history["objective"]) if self.config.record_history else [],
            "residual_history": list(self.history["residual_norm"]) if self.config.record_history else [],
            "step_size": eta,
            "iterations": iterations_used,
        }

        self.S = S_current
        self.t_diag = t_current
        return S_estimates, info
