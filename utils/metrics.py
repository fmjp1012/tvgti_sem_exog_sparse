"""
評価指標計算モジュール

誤差計算関数を一元管理します。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_normalized_error(
    S_hat: np.ndarray,
    S_true: np.ndarray,
    S_offline: Optional[np.ndarray] = None,
    normalization: str = "true_value",
    eps: float = 1e-12,
) -> float:
    """
    正規化二乗誤差を計算する。

    Parameters
    ----------
    S_hat : np.ndarray
        推定行列
    S_true : np.ndarray
        真の行列
    S_offline : np.ndarray, optional
        オフライン解（normalization="offline_solution" の場合に使用）
    normalization : str
        正規化方法
        - "true_value": 真の値のノルムで正規化（従来方法）
        - "offline_solution": オフライン解との差のノルムで正規化
    eps : float
        ゼロ除算防止用の小さな値

    Returns
    -------
    float
        正規化二乗誤差
    """
    numerator = np.linalg.norm(S_hat - S_true) ** 2
    
    if normalization == "offline_solution" and S_offline is not None:
        denominator = np.linalg.norm(S_true - S_offline) ** 2 + eps
    else:
        denominator = np.linalg.norm(S_true) ** 2 + eps
    
    return float(numerator / denominator)


def compute_error_series(
    S_hat_list: list[np.ndarray],
    S_true_list: list[np.ndarray],
    S_offline: Optional[np.ndarray] = None,
    normalization: str = "true_value",
    eps: float = 1e-12,
) -> list[float]:
    """
    時系列の正規化二乗誤差を計算する。

    Parameters
    ----------
    S_hat_list : list[np.ndarray]
        推定行列の時系列
    S_true_list : list[np.ndarray]
        真の行列の時系列
    S_offline : np.ndarray, optional
        オフライン解
    normalization : str
        正規化方法
    eps : float
        ゼロ除算防止用の小さな値

    Returns
    -------
    list[float]
        各時刻の正規化二乗誤差
    """
    return [
        compute_normalized_error(S_hat_list[t], S_true_list[t], S_offline, normalization, eps)
        for t in range(len(S_hat_list))
    ]


def compute_frobenius_error(
    S_hat: np.ndarray,
    S_true: np.ndarray,
    S_offline: Optional[np.ndarray] = None,
    normalization: str = "true_value",
    eps: float = 1e-12,
) -> float:
    """
    フロベニウスノルム誤差を計算する（チューニング用）。

    Parameters
    ----------
    S_hat : np.ndarray
        推定行列
    S_true : np.ndarray
        真の行列
    S_offline : np.ndarray, optional
        オフライン解
    normalization : str
        正規化方法
    eps : float
        ゼロ除算防止用の小さな値

    Returns
    -------
    float
        正規化フロベニウスノルム誤差
    """
    err = np.linalg.norm(S_hat - S_true, ord="fro")
    
    if normalization == "offline_solution" and S_offline is not None:
        normalizer = np.linalg.norm(S_true - S_offline, ord="fro") + eps
        return float(err / normalizer)
    
    return float(err)

