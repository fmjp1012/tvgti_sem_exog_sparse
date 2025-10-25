from typing import List, Tuple

import numpy as np


def generate_random_S_asymmetric_hollow(N: int, sparsity: float, max_weight: float) -> np.ndarray:
    S = np.random.uniform(-max_weight, max_weight, size=(N, N))
    mask = (np.random.rand(N, N) < (1.0 - sparsity))
    S = S * mask
    np.fill_diagonal(S, 0.0)
    return S


def generate_regular_S_asymmetric_hollow(N: int, sparsity: float, max_weight: float) -> np.ndarray:
    S = np.zeros((N, N))
    k_zero = int(round((N - 1) * sparsity))
    d_nonzero = (N - 1) - k_zero
    for i in range(N):
        cols = [j for j in range(N) if j != i]
        if d_nonzero <= len(cols):
            sel = np.random.choice(cols, size=d_nonzero, replace=False)
        else:
            sel = cols
        S[i, sel] = np.random.uniform(-max_weight, max_weight, size=len(sel))
    np.fill_diagonal(S, 0.0)
    return S


def generate_piecewise_Y_with_exog(
    N: int,
    T: int,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    s_type: str = "random",
    b_min: float = -1.0,
    b_max: float = 1.0,
    u_dist: str = "uniform01",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    # S(t) を区分一定で生成
    S_list: List[np.ndarray] = []
    I = np.eye(N)
    for _ in range(K):
        if s_type == "regular":
            S = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight)
        else:
            S = generate_random_S_asymmetric_hollow(N, sparsity, max_weight)
        S_list.append(S)
    seg = [T // K] * K
    seg[-1] += T % K
    S_series: List[np.ndarray] = []
    for i, length in enumerate(seg):
        S_series.extend([S_list[i]] * length)

    # B は時不変・対角
    b_diag = np.random.uniform(b_min, b_max, size=N)
    B = np.diag(b_diag)

    # 入力 U とノイズ E
    if u_dist == "uniform01":
        U = np.random.uniform(0.0, 1.0, size=(N, T))
    else:
        U = np.random.normal(0.0, 1.0, size=(N, T))
    E = np.random.normal(0.0, std_e, size=(N, T))

    # Y(t) = (I - S(t))^{-1} (B U(t) + E(t))
    Y_list = []
    for t in range(T):
        S_t = S_series[t]
        rhs = B @ U[:, t] + E[:, t]
        y_t = np.linalg.solve(I - S_t, rhs)
        Y_list.append(y_t.reshape(N, 1))
    Y = np.concatenate(Y_list, axis=1)
    return S_series, B, U, Y


def generate_linear_Y_with_exog(
    N: int,
    T: int,
    sparsity: float,
    max_weight: float,
    std_e: float,
    s_type: str = "random",
    b_min: float = -1.0,
    b_max: float = 1.0,
    u_dist: str = "uniform01",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    # S_start, S_end を生成し線形補間
    if s_type == "regular":
        S_start = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight)
        S_end = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight)
    else:
        S_start = generate_random_S_asymmetric_hollow(N, sparsity, max_weight)
        S_end = generate_random_S_asymmetric_hollow(N, sparsity, max_weight)
    S_series: List[np.ndarray] = []
    I = np.eye(N)
    for t in range(T):
        lam = 0.0 if T == 1 else t / (T - 1)
        S_t = (1.0 - lam) * S_start + lam * S_end
        np.fill_diagonal(S_t, 0.0)
        S_series.append(S_t)

    # B, U, E
    b_diag = np.random.uniform(b_min, b_max, size=N)
    B = np.diag(b_diag)
    if u_dist == "uniform01":
        U = np.random.uniform(0.0, 1.0, size=(N, T))
    else:
        U = np.random.normal(0.0, 1.0, size=(N, T))
    E = np.random.normal(0.0, std_e, size=(N, T))

    # Y を生成
    Y_list = []
    for t in range(T):
        rhs = B @ U[:, t] + E[:, t]
        y_t = np.linalg.solve(I - S_series[t], rhs)
        Y_list.append(y_t.reshape(N, 1))
    Y = np.concatenate(Y_list, axis=1)
    return S_series, B, U, Y


def generate_brownian_piecewise_Y_with_exog(
    N: int,
    T: int,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    std_S: float,
    s_type: str = "random",
    b_min: float = -1.0,
    b_max: float = 1.0,
    u_dist: str = "uniform01",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    # 初期Sを生成し、以降は加法ノイズで区分一定のSを更新
    if s_type == "regular":
        S0 = generate_regular_S_asymmetric_hollow(N, sparsity, max_weight)
    else:
        S0 = generate_random_S_asymmetric_hollow(N, sparsity, max_weight)
    S_list: List[np.ndarray] = [S0]
    for _ in range(1, K):
        noise = np.random.normal(0.0, std_S, size=(N, N))
        S_new = S_list[-1] + noise
        np.fill_diagonal(S_new, 0.0)
        S_list.append(S_new)

    seg = [T // K] * K
    seg[-1] += T % K
    S_series: List[np.ndarray] = []
    for i, length in enumerate(seg):
        S_series.extend([S_list[i]] * length)

    # B, U, E
    b_diag = np.random.uniform(b_min, b_max, size=N)
    B = np.diag(b_diag)
    if u_dist == "uniform01":
        U = np.random.uniform(0.0, 1.0, size=(N, T))
    else:
        U = np.random.normal(0.0, 1.0, size=(N, T))
    E = np.random.normal(0.0, std_e, size=(N, T))

    I = np.eye(N)
    Y_list = []
    for t in range(T):
        rhs = B @ U[:, t] + E[:, t]
        y_t = np.linalg.solve(I - S_series[t], rhs)
        Y_list.append(y_t.reshape(N, 1))
    Y = np.concatenate(Y_list, axis=1)
    return S_series, B, U, Y


