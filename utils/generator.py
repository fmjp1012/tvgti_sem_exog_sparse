from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from scipy.linalg import inv, eigvals, norm

from utils import *

def generate_random_S(
    N: int,
    sparsity: float,
    max_weight: float,
    S_is_symmetric: bool
) -> np.ndarray:
    """
    対称行列バージョンのサンプル実装.
    sparsity (0~1) に従って要素を0にするかどうかをランダムに決定し，
    [-max_weight, max_weight] の範囲で乱数を生成し対称化したものを返す.
    """
    S = np.random.uniform(-max_weight, max_weight, size=(N, N))
    mask = (np.random.rand(N, N) < (1.0 - sparsity))
    S = S * mask  # スパース化

    # 対称化
    if S_is_symmetric:
        S = (S + S.T) / 2

    spectral_radius = max(abs(eigvals(S)))
    if spectral_radius >= 1:
        S = S / (spectral_radius + 0.1)
    S = S / norm(S)
    return S

def generate_regular_S(
    N: int,
    sparsity: float,
    max_weight: float,
    S_is_symmetric: bool = True
) -> np.ndarray:
    """
    正則グラフに基づいて S を生成するサンプル実装.
    S_is_symmetric=True の場合：無向 d-正則グラフを生成し、対称行列を返す
    S_is_symmetric=False の場合：有向正則グラフ風の行列を生成し、非対称行列を返す
    """
    # 1行のオフダイアゴナル成分のうち，ゼロとする個数
    k = int(round((N - 1) * sparsity))
    # 非ゼロにする個数（すなわち次数）
    d = (N - 1) - k

    if S_is_symmetric:
        # d正則グラフの場合，全ノードの次数の合計は偶数である必要があるためチェック
        if (N * d) % 2 != 0:
            # 和が奇数になる場合は d を 1 減らし調整（それに伴い k も変わる）
            d -= 1
            k = (N - 1) - d

        # ランダムな d-正則グラフを生成 (無向グラフなので対称性が保証される)
        G = nx.random_regular_graph(d, N)
        A = nx.to_numpy_array(G)

        # 対称性を保証するために、上三角部分のみに重みを割り当て、下三角にコピー
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] == 1:  # エッジが存在する場合
                    weight = np.random.uniform(-max_weight, max_weight)
                    S[i, j] = weight
                    S[j, i] = weight  # 対称性を保証
    else:
        # 非対称版：各行で d 個の非ゼロ要素を持つように
        S = np.zeros((N, N))
        for i in range(N):
            # 行 i において、非ゼロにする列インデックスをランダムに選択
            # （対角成分は除外）
            available_cols = [j for j in range(N) if j != i]
            if d <= len(available_cols):
                selected_cols = np.random.choice(available_cols, size=d, replace=False)
            else:
                # d が N-1 を超える場合は全ての非対角要素を選択
                selected_cols = available_cols
            
            # 選択された列に重みを割り当て
            for j in selected_cols:
                S[i, j] = np.random.uniform(-max_weight, max_weight)

    # 対角成分は必ず0に
    np.fill_diagonal(S, 0)

    # スペクトル半径（最大固有値の絶対値）が1以上なら縮小
    spectral_radius = max(abs(eigvals(S)))
    if spectral_radius >= 1:
        S = S / (spectral_radius + 0.1)
    # ノルム正規化
    S = S / norm(S)
    return S

def generate_piecewise_X_K(
        N: int,
        T: int,
        S_is_symmetric: bool,
        sparsity: float,
        max_weight: float,
        std_e: float,
        K: int,
        s_type: str = "random"  # "random" または "regular"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    K 個のセグメントに分割して X を生成する.
    各セグメントで用いる S は s_type に応じて生成される．
    """
    S_list = []
    inv_I_S_list = []
    I = np.eye(N)
    for i in range(K):
        if s_type == "regular":
            S = generate_regular_S(N, sparsity, max_weight, S_is_symmetric)
        else:
            S = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
        S_list.append(S)
        inv_I_S_list.append(inv(I - S))
    # Divide T into K segments
    segment_lengths = [T // K] * K
    segment_lengths[-1] += T % K
    # Create S_series
    S_series = []
    for i, length in enumerate(segment_lengths):
        S_series.extend([S_list[i]] * length)
    # Generate error terms
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    # Compute X
    X_list = []
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        X_i = inv_I_S_list[i] @ e_t_series[:, start:end]
        X_list.append(X_i)
        start = end
    X = np.concatenate(X_list, axis=1)
    return S_series, X

def update_S(
    S_prev: np.ndarray,
    noise_std: float,
    S_is_symmetric: bool,
    max_weight: float
) -> np.ndarray:
    """
    前の S にノイズを加えて新しい S を生成するサンプル関数．
    - noise_std: ノイズの標準偏差
    - S_is_symmetric: True の場合は最終的に対称行列にする
    - max_weight: 行列要素の絶対値を制限したい場合に利用
    """
    N = S_prev.shape[0]
    
    # ガウスノイズを加える
    noise = np.random.normal(0, noise_std, size=(N, N))
    S_new = S_prev + noise
    
    # 対称行列にしたい場合は対称化
    if S_is_symmetric:
        S_new = (S_new + S_new.T) / 2
    
    # 要素を [-max_weight, max_weight] 以内にクリップしておく (任意)
    # （S の安定性などを考慮するなら、ここでスペクトル半径を抑える処理も可）
    S_new = np.clip(S_new, -max_weight, max_weight)
    
    return S_new

def generate_piecewise_X_K_with_snr(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    snr_target: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    s_type: str = "random"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    SNR をターゲットにしてスケーリングした S を用いて，
    K 個のセグメントで X を生成する.
    S の生成は s_type に応じて行われる．
    """
    S_list = []
    inv_I_S_list = []
    I = np.eye(N)
    
    # 1) Generate K random S and scale each to snr_target
    for i in range(K):
        # Generate a base random S
        if s_type == "regular":
            S_raw = generate_regular_S(N, sparsity, max_weight, S_is_symmetric)
        else:
            S_raw = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
        # scale_S_for_target_snr は utils に定義されている前提
        S_scaled = scale_S_for_target_snr(S_raw, snr_target)
        S_list.append(S_scaled)
        inv_I_S_list.append(np.linalg.inv(I - S_scaled))

    # 2) Divide T into K segments
    segment_lengths = [T // K] * K
    segment_lengths[-1] += T % K  # handle any remainder in the last segment

    # 3) Create S_series (length T)
    S_series = []
    for i, length in enumerate(segment_lengths):
        S_series.extend([S_list[i]] * length)

    # 4) Generate exogenous shock e(t)
    e_t_series = np.random.normal(0, std_e, size=(N, T))

    # 5) Compute X piecewise
    X_list = []
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        X_i = inv_I_S_list[i] @ e_t_series[:, start:end]
        X_list.append(X_i)
        start = end
    
    X = np.concatenate(X_list, axis=1)
    
    return S_series, X

def generate_brownian_piecewise_X_K(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    K: int,
    std_S: float,  # S の「揺らす」強度
    s_type: str = "random"
):
    """
    S を前の値にノイズを加える形で Brownian motion のように更新し，
    K 回に分けた区間ごとに X を生成する関数．
    初回の S は s_type に応じた方法で生成される．
    """
    S_list = []
    inv_I_S_list = []
    I = np.eye(N)
    
    # 初回 S の生成
    if s_type == "regular":
        S = generate_regular_S(N, sparsity, max_weight, S_is_symmetric)
    else:
        S = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    S_list.append(S)
    inv_I_S_list.append(inv(I - S))
    
    # 2回目以降は前の S にノイズを加える
    for i in range(1, K):
        S_prev = S_list[-1]
        S_new = update_S(
            S_prev, 
            noise_std=std_S,
            S_is_symmetric=S_is_symmetric,
            max_weight=max_weight
        )
        S_list.append(S_new)
        inv_I_S_list.append(inv(I - S_new))
    
    # T を K 区間に分割
    segment_lengths = [T // K] * K
    segment_lengths[-1] += T % K
    
    # 各時刻 t ごとの S を用意
    S_series = []
    for i, length in enumerate(segment_lengths):
        S_series.extend([S_list[i]] * length)
    
    # 外生ショックの生成
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    
    # X の計算
    X_list = []
    start = 0
    for i, length in enumerate(segment_lengths):
        end = start + length
        X_i = inv_I_S_list[i] @ e_t_series[:, start:end]
        X_list.append(X_i)
        start = end
    
    X = np.concatenate(X_list, axis=1)
    
    return S_series, X

def generate_linear_X(
    N: int,
    T: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    s_type: str = "random"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    t=0 から t=T-1 まで，線形補間した S(t) を用いて
    X(t) = (I - S(t))^-1 * e(t) を生成する.
    S_start, S_end の生成に s_type を利用する．
    """
    I = np.eye(N)

    # 2つの行列を生成
    if s_type == "regular":
        S_start = generate_regular_S(N, sparsity, max_weight, S_is_symmetric)
        S_end   = generate_regular_S(N, sparsity, max_weight, S_is_symmetric)
    else:
        S_start = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
        S_end   = generate_random_S(N, sparsity, max_weight, S_is_symmetric)
    
    # スペクトル半径のチェックと調整
    def spectral_radius(A: np.ndarray) -> float:
        return max(abs(eigvals(A)))
    
    rho_start = spectral_radius(S_start)
    rho_end   = spectral_radius(S_end)
    rho_max   = max(rho_start, rho_end)
    
    if rho_max >= 1.0:
        alpha = 0.99 / rho_max
        S_start *= alpha
        S_end   *= alpha
    
    S_series = []
    inv_I_S_list = []
    for t in range(T):
        lam = 0.0 if T == 1 else t / (T - 1)
        S_t = (1.0 - lam) * S_start + lam * S_end
        S_series.append(S_t)
        inv_I_S_list.append(np.linalg.inv(I - S_t))
    
    # 外生ショックの生成
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    
    # X(t) の計算
    X_list = []
    for t in range(T):
        x_t = inv_I_S_list[t] @ e_t_series[:, t]
        X_list.append(x_t.reshape(N, 1))
    
    X = np.concatenate(X_list, axis=1)
    return S_series, X

def generate_linear_X_L(
    N: int,
    T: int,
    L: int,
    S_is_symmetric: bool,
    sparsity: float,
    max_weight: float,
    std_e: float,
    s_type: str = "random"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    L 個の S 点を生成し，その間を線形補間して
    S(t) = (I - S(t))^{-1} e(t) により X を生成する.
    S の生成は s_type に従って行われる．
    """
    # L 個の S を生成
    if s_type == "regular":
        S_points = [generate_regular_S(N, sparsity, max_weight, S_is_symmetric) for _ in range(L)]
    else:
        S_points = [generate_random_S(N, sparsity, max_weight, S_is_symmetric) for _ in range(L)]
    
    S_series = []
    I = np.eye(N)
    
    for t in range(T):
        if T == 1:
            S_t = S_points[0]
        else:
            global_lambda = t / (T - 1)
            if global_lambda >= 1.0:
                S_t = S_points[-1]
            else:
                segment = int(global_lambda * (L - 1))
                local_lambda = (global_lambda * (L - 1)) - segment
                S_t = (1.0 - local_lambda) * S_points[segment] + local_lambda * S_points[segment + 1]
        S_series.append(S_t)
    
    # 各時刻 t における X(t) の生成
    e_t_series = np.random.normal(0, std_e, size=(N, T))
    X_list = []
    for t in range(T):
        inv_I_S = np.linalg.inv(I - S_series[t])
        x_t = inv_I_S @ e_t_series[:, t]
        X_list.append(x_t.reshape(N, 1))
    
    X = np.concatenate(X_list, axis=1)
    return S_series, X
