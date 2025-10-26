from typing import List, Tuple

import numpy as np
from numpy.random import Generator


def _resolve_rng(rng: Generator | None) -> Generator:
    return rng if rng is not None else np.random.default_rng()


def generate_random_S_asymmetric_hollow(
    N: int,
    sparsity: float,
    max_weight: float,
    rng: Generator | None = None,
) -> np.ndarray:
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

    I = np.eye(N)

    # --- (1) Piecewise-constant S(t) generation ---
    S_list: List[np.ndarray] = []
    for _ in range(K):
        if s_type == "regular":
            S = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
        else:
            S = generate_random_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
        S_list.append(S)

    seg = [T // K] * K
    seg[-1] += T % K
    S_series: List[np.ndarray] = []
    for i, length in enumerate(seg):
        S_series.extend([S_list[i]] * length)

    # --- (2) Diagonal T matrix ---
    t_diag = rng.uniform(t_min, t_max, size=N)
    T_mat = np.diag(t_diag)

    # --- (3) Generate exogenous inputs Z and Gaussian noise E ---
    if z_dist == "uniform01":
        Z = rng.uniform(0.0, 1.0, size=(N, T))
    else:
        Z = rng.normal(0.0, 1.0, size=(N, T))
    E = rng.normal(0.0, std_e, size=(N, T))

    # --- (4) Generate observed signals X(t) ---
    X_list = []
    for t in range(T):
        S_t = S_series[t]
        rhs = T_mat @ Z[:, t] + E[:, t]
        # (I - S_t) x_t = rhs  →  x_t = (I - S_t)^(-1) rhs
        x_t = np.linalg.solve(I - S_t, rhs)
        X_list.append(x_t.reshape(N, 1))
    X = np.concatenate(X_list, axis=1)

    return S_series, T_mat, Z, X


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

    # --- (1) Generate start and end S matrices and linearly interpolate ---
    if s_type == "regular":
        S_start = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
        S_end = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
    else:
        S_start = generate_random_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
        S_end = generate_random_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)

    S_series: List[np.ndarray] = []
    for t in range(T):
        lam = 0.0 if T == 1 else t / (T - 1)
        S_t = (1.0 - lam) * S_start + lam * S_end
        np.fill_diagonal(S_t, 0.0)
        S_series.append(S_t)

    # --- (2) Diagonal T matrix ---
    t_diag = rng.uniform(t_min, t_max, size=N)
    T_mat = np.diag(t_diag)

    # --- (3) Generate exogenous variables Z and noise E ---
    if z_dist == "uniform01":
        Z = rng.uniform(0.0, 1.0, size=(N, T))
    else:
        Z = rng.normal(0.0, 1.0, size=(N, T))
    E = rng.normal(0.0, std_e, size=(N, T))

    # --- (4) Generate X(t) ---
    I = np.eye(N)
    X_list = []
    for t in range(T):
        rhs = T_mat @ Z[:, t] + E[:, t]
        x_t = np.linalg.solve(I - S_series[t], rhs)
        X_list.append(x_t.reshape(N, 1))
    X = np.concatenate(X_list, axis=1)

    return S_series, T_mat, Z, X



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

    # --- (1) Initialize base S and update by Brownian noise ---
    if s_type == "regular":
        S0 = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)
    else:
        S0 = generate_random_S_asymmetric_hollow(N, sparsity, max_weight, rng=rng)

    S_list: List[np.ndarray] = [S0]
    for _ in range(1, K):
        noise = rng.normal(0.0, std_S, size=(N, N))
        S_new = S_list[-1] + noise
        np.fill_diagonal(S_new, 0.0)
        S_list.append(S_new)

    # --- (2) Assign each S_k to time segments ---
    seg = [T // K] * K
    seg[-1] += T % K
    S_series: List[np.ndarray] = []
    for i, length in enumerate(seg):
        S_series.extend([S_list[i]] * length)

    # --- (3) Generate T, Z, E ---
    t_diag = rng.uniform(t_min, t_max, size=N)
    T_mat = np.diag(t_diag)

    if z_dist == "uniform01":
        Z = rng.uniform(0.0, 1.0, size=(N, T))
    else:
        Z = rng.normal(0.0, 1.0, size=(N, T))
    E = rng.normal(0.0, std_e, size=(N, T))

    # --- (4) Generate X(t) ---
    I = np.eye(N)
    X_list = []
    for t in range(T):
        rhs = T_mat @ Z[:, t] + E[:, t]
        x_t = np.linalg.solve(I - S_series[t], rhs)
        X_list.append(x_t.reshape(N, 1))
    X = np.concatenate(X_list, axis=1)

    return S_series, T_mat, Z, X
