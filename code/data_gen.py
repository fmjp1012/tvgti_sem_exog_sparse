"""
データ生成モジュール

時変 SEM (Structural Equation Model) のシミュレーションデータを生成します。
"""

from typing import List, Tuple

import numpy as np
from numpy.random import Generator


def _resolve_rng(rng: Generator | None) -> Generator:
    """乱数生成器を解決する。"""
    return rng if rng is not None else np.random.default_rng()


# =============================================================================
# 隣接行列生成関数
# =============================================================================

def generate_random_S_asymmetric_hollow(
    N: int,
    sparsity: float,
    max_weight: float,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    ランダムな非対称中空隣接行列を生成する。

    Parameters
    ----------
    N : int
        ノード数
    sparsity : float
        ゼロ要素の割合 (0.0-1.0)
    max_weight : float
        非ゼロ要素の最大絶対値
    rng : Generator, optional
        乱数生成器

    Returns
    -------
    np.ndarray
        形状 (N, N) の隣接行列（対角成分は 0）
    """
    rng = _resolve_rng(rng)
    S = rng.uniform(-max_weight, max_weight, size=(N, N))
    mask = rng.random((N, N)) < (1.0 - sparsity)
    S = S * mask
    np.fill_diagonal(S, 0.0)
    return S


def generate_regular_S_asymmetric_hollow(
    N: int,
    sparsity: float,
    max_weight: float,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    各行の非ゼロ要素数が均一な非対称中空隣接行列を生成する。

    Parameters
    ----------
    N : int
        ノード数
    sparsity : float
        ゼロ要素の割合 (0.0-1.0)
    max_weight : float
        非ゼロ要素の最大絶対値
    rng : Generator, optional
        乱数生成器

    Returns
    -------
    np.ndarray
        形状 (N, N) の隣接行列（対角成分は 0）
    """
    rng = _resolve_rng(rng)
    S = np.zeros((N, N))
    k_zero = int(round((N - 1) * sparsity))
    d_nonzero = (N - 1) - k_zero
    for i in range(N):
        cols = [j for j in range(N) if j != i]
        if d_nonzero <= len(cols):
            sel = rng.choice(cols, size=d_nonzero, replace=False)
        else:
            sel = cols
        S[i, sel] = rng.uniform(-max_weight, max_weight, size=len(sel))
    np.fill_diagonal(S, 0.0)
    return S


def _generate_S_matrix(
    N: int,
    sparsity: float,
    max_weight: float,
    s_type: str,
    rng: Generator,
) -> np.ndarray:
    """
    指定されたタイプの隣接行列を生成する（内部ヘルパー）。

    Parameters
    ----------
    N : int
        ノード数
    sparsity : float
        スパース性
    max_weight : float
        最大重み
    s_type : str
        "regular" または "random"
    rng : Generator
        乱数生成器

    Returns
    -------
    np.ndarray
        隣接行列
    """
    if s_type == "regular":
        return generate_regular_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
    else:
        return generate_random_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)


# =============================================================================
# 共通ヘルパー関数
# =============================================================================

def _generate_exog_and_noise(
    N: int,
    T: int,
    t_min: float,
    t_max: float,
    std_e: float,
    z_dist: str,
    rng: Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    外生変数係数行列 T、外生変数 Z、ノイズ E を生成する。

    Parameters
    ----------
    N : int
        ノード数
    T : int
        時系列長
    t_min : float
        外生変数係数の最小値
    t_max : float
        外生変数係数の最大値
    std_e : float
        ノイズの標準偏差
    z_dist : str
        外生変数の分布 ("uniform01" or "normal")
    rng : Generator
        乱数生成器

    Returns
    -------
    Tuple[T_mat, Z, E]
        - T_mat: 対角行列 (N x N)
        - Z: 外生変数 (N x T)
        - E: ノイズ (N x T)
    """
    # 対角 T 行列
    t_diag = rng.uniform(t_min, t_max, size=N)
    T_mat = np.diag(t_diag)
    
    # 外生変数 Z
    if z_dist == "uniform01":
        Z = rng.uniform(0.0, 1.0, size=(N, T))
    else:
        Z = rng.normal(0.0, 1.0, size=(N, T))
    
    # ノイズ E
    E = rng.normal(0.0, std_e, size=(N, T))
    
    return T_mat, Z, E


def _solve_sem_sequence(
    S_series: List[np.ndarray],
    T_mat: np.ndarray,
    Z: np.ndarray,
    E: np.ndarray,
) -> np.ndarray:
    """
    SEM 方程式の時系列を解いて観測データ X を生成する。

    x_t = S(t) x_t + T z_t + ε_t
    => (I - S(t)) x_t = T z_t + ε_t
    => x_t = (I - S(t))^{-1} (T z_t + ε_t)

    Parameters
    ----------
    S_series : List[np.ndarray]
        隣接行列の時系列
    T_mat : np.ndarray
        外生変数係数行列
    Z : np.ndarray
        外生変数 (N x T)
    E : np.ndarray
        ノイズ (N x T)

    Returns
    -------
    np.ndarray
        観測データ X (N x T)
    """
    N = T_mat.shape[0]
    T_len = Z.shape[1]
    I = np.eye(N)
    
    X_list = []
    for t in range(T_len):
        S_t = S_series[t]
        rhs = T_mat @ Z[:, t] + E[:, t]
        x_t = np.linalg.solve(I - S_t, rhs)
        X_list.append(x_t.reshape(N, 1))
    
    return np.concatenate(X_list, axis=1)


# =============================================================================
# Piecewise シナリオ
# =============================================================================

def generate_piecewise_X_with_exog(
    N: int,
    T: int,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    s_type: str = "regular",
    t_min: float = -1.0,
    t_max: float = 1.0,
    z_dist: str = "uniform01",
    rng: Generator | None = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Piecewise-constant Structural Equation Model (SEM) with exogenous variables:
        x_t = S(t) x_t + T z_t + ε_t
        ε_t ~ N(0, σ² I)
    
    Parameters
    ----------
    N : int
        Number of nodes (variables).
    T : int
        Number of time steps.
    sparsity : float
        Fraction of zero elements in S(t).
    max_weight : float
        Maximum absolute value of S(t) nonzero entries.
    std_e : float
        Standard deviation of Gaussian noise ε_t.
    K : int
        Number of segments (piecewise constant intervals for S).
    s_type : {"regular", "random"}
        Type of structural matrix generation.
    t_min, t_max : float
        Range for diagonal elements of T (influence coefficients).
    z_dist : {"uniform01", "normal"}
        Distribution of exogenous inputs z_t.
    
    Returns
    -------
    S_series : list of np.ndarray
        List of structural matrices S(t) for each time step.
    T_mat : np.ndarray
        Diagonal matrix for exogenous influence.
    Z : np.ndarray
        Exogenous variable sequence (N × T).
    X : np.ndarray
        Generated observed signals (N × T).
    """
    rng = _resolve_rng(rng)

    # (1) Piecewise-constant S(t) generation
    S_list: List[np.ndarray] = []
    for _ in range(K):
        S = _generate_S_matrix(N, sparsity, max_weight, s_type, rng)
        S_list.append(S)

    seg = [T // K] * K
    seg[-1] += T % K
    S_series: List[np.ndarray] = []
    for i, length in enumerate(seg):
        S_series.extend([S_list[i]] * length)

    # (2) Generate T, Z, E
    T_mat, Z, E = _generate_exog_and_noise(N, T, t_min, t_max, std_e, z_dist, rng)

    # (3) Generate X
    X = _solve_sem_sequence(S_series, T_mat, Z, E)

    return S_series, T_mat, Z, X


# =============================================================================
# Linear シナリオ
# =============================================================================

def generate_linear_X_with_exog(
    N: int,
    T: int,
    sparsity: float,
    max_weight: float,
    std_e: float,
    s_type: str = "random",
    t_min: float = -1.0,
    t_max: float = 1.0,
    z_dist: str = "uniform01",
    rng: Generator | None = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear-interpolated time-varying SEM with exogenous variables:
        x_t = S(t) x_t + T z_t + ε_t
        ε_t ~ N(0, σ² I)
    
    Parameters
    ----------
    N : int
        Number of nodes (variables).
    T : int
        Number of time steps.
    sparsity : float
        Fraction of zero elements in S(t).
    max_weight : float
        Maximum absolute value of S(t) nonzero entries.
    std_e : float
        Standard deviation of Gaussian noise ε_t.
    s_type : {"regular", "random"}
        Type of structural matrix generation.
    t_min, t_max : float
        Range for diagonal elements of T (influence coefficients).
    z_dist : {"uniform01", "normal"}
        Distribution of exogenous inputs z_t.
    
    Returns
    -------
    S_series : list of np.ndarray
        List of time-varying structural matrices S(t).
    T_mat : np.ndarray
        Diagonal matrix for exogenous influence.
    Z : np.ndarray
        Exogenous variable sequence (N × T).
    X : np.ndarray
        Generated observed signals (N × T).
    """
    rng = _resolve_rng(rng)

    # (1) Generate start and end S matrices and linearly interpolate
    S_start = _generate_S_matrix(N, sparsity, max_weight, s_type, rng)
    S_end = _generate_S_matrix(N, sparsity, max_weight, s_type, rng)

    S_series: List[np.ndarray] = []
    for t in range(T):
        lam = 0.0 if T == 1 else t / (T - 1)
        S_t = (1.0 - lam) * S_start + lam * S_end
        np.fill_diagonal(S_t, 0.0)
        S_series.append(S_t)

    # (2) Generate T, Z, E
    T_mat, Z, E = _generate_exog_and_noise(N, T, t_min, t_max, std_e, z_dist, rng)

    # (3) Generate X
    X = _solve_sem_sequence(S_series, T_mat, Z, E)

    return S_series, T_mat, Z, X


# =============================================================================
# Brownian シナリオ
# =============================================================================

def generate_brownian_piecewise_X_with_exog(
    N: int,
    T: int,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    std_S: float,
    s_type: str = "random",
    t_min: float = -1.0,
    t_max: float = 1.0,
    z_dist: str = "uniform01",
    rng: Generator | None = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Piecewise Brownian-evolving SEM with exogenous variables:
        x_t = S(t) x_t + T z_t + ε_t
        ε_t ~ N(0, σ² I)
    
    Parameters
    ----------
    N : int
        Number of nodes.
    T : int
        Number of time steps.
    sparsity : float
        Fraction of zero elements in S.
    max_weight : float
        Max absolute value of S entries.
    std_e : float
        Std deviation of Gaussian noise ε_t.
    K : int
        Number of piecewise segments.
    std_S : float
        Std deviation for Brownian noise added to S.
    s_type : {"regular", "random"}
        Base type for initial S.
    t_min, t_max : float
        Range for diagonal elements of T.
    z_dist : {"uniform01", "normal"}
        Distribution of exogenous inputs z_t.

    Returns
    -------
    S_series : list of np.ndarray
        List of piecewise-constant structural matrices S(t).
    T_mat : np.ndarray
        Diagonal matrix for exogenous influence.
    Z : np.ndarray
        Exogenous variable sequence (N × T).
    X : np.ndarray
        Generated observed signals (N × T).
    """
    rng = _resolve_rng(rng)

    # (1) Initialize base S and update by Brownian noise
    S0 = _generate_S_matrix(N, sparsity, max_weight, s_type, rng)

    S_list: List[np.ndarray] = [S0]
    for _ in range(1, K):
        noise = rng.normal(0.0, std_S, size=(N, N))
        S_new = S_list[-1] + noise
        np.fill_diagonal(S_new, 0.0)
        S_list.append(S_new)

    # (2) Assign each S_k to time segments
    seg = [T // K] * K
    seg[-1] += T % K
    S_series: List[np.ndarray] = []
    for i, length in enumerate(seg):
        S_series.extend([S_list[i]] * length)

    # (3) Generate T, Z, E
    T_mat, Z, E = _generate_exog_and_noise(N, T, t_min, t_max, std_e, z_dist, rng)

    # (4) Generate X
    X = _solve_sem_sequence(S_series, T_mat, Z, E)

    return S_series, T_mat, Z, X
