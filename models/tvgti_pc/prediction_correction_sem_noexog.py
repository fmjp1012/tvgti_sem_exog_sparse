import numpy as np
from typing import List, Optional, Tuple

from .prediction_correction_sem import PredictionCorrectionSEM


class PredictionCorrectionSEMNoExog:
    """
    Prediction–Correction for the *no-exogenous* SEM:

        x_t = S(t) x_t + ε_t

    This is a compatibility wrapper that reuses the existing implementation but
    forces exogenous inputs to be ignored (Z is never used, T is not part of the model).

    Notes
    -----
    - The underlying implementation (`PredictionCorrectionSEM`) supports optional exog.
      Here we always run with Z=None to realize the model above.
    - We intentionally do NOT accept T_init / exog_* parameters to avoid accidental use.
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
        name: str = "pc_noexog",
    ):
        self._pc = PredictionCorrectionSEM(
            N=N,
            S_0=S_0,
            lambda_reg=lambda_reg,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            P=P,
            C=C,
            show_progress=show_progress,
            name=name,
            T_init=None,
        )

    def run(self, X: np.ndarray, Z: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[float]]:
        # Ignore Z by design (no-exog model).
        return self._pc.run(X, Z=None)

    @property
    def b_history(self) -> List[np.ndarray]:
        # Always empty in no-exog mode.
        return []

