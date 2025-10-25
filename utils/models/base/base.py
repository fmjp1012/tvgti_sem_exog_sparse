from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class TimeVaryingSEMBase(ABC):
    """Abstract base class for time-varying SEM models.

    Implementations must provide a unified run interface and basic
    parameter access. This makes orchestration and sweeps consistent.
    """

    @abstractmethod
    def run(self, X: np.ndarray) -> Dict[str, Any]:
        """Execute the algorithm on observations X and return results.

        Returns should include keys like 'metrics' and 'artifacts' where useful.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
