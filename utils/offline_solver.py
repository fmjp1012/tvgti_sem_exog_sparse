"""
オフライン解を計算するユーティリティ

cvxpyを使用してLasso問題（二乗誤差 + L1正則化）を解き、
各時刻tでの最適な隣接行列を推定する。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np


def solve_offline_sem_lasso(
    Y: np.ndarray,
    U: np.ndarray,
    lambda_l1: float,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    オフラインでSEMのLasso問題を解く（全時刻のデータを使用）
    
    モデル: x_t = S @ x_t + B @ u_t + e_t
    
    各ノードiについて、以下の最小化問題を解く:
    min ||x_i - S_i @ x - B_i @ u||_2^2 + lambda_l1 * ||S_i||_1
    
    ただし、対角成分は0に制約する（S[i,i] = 0）
    
    Parameters
    ----------
    Y : np.ndarray
        観測データ (N, T)
    U : np.ndarray
        外生変数 (M, T) - Mは外生変数の次元
    lambda_l1 : float
        L1正則化パラメータ
    solver : Optional[str]
        使用するソルバー（None の場合は自動選択）
    verbose : bool
        詳細出力を表示するか
    
    Returns
    -------
    S_offline : np.ndarray
        推定された隣接行列 (N, N)
    B_offline : np.ndarray
        推定された外生変数係数 (N, M)
    """
    N, T = Y.shape
    M = U.shape[0] if U.ndim == 2 else 1
    
    if U.ndim == 1:
        U = U.reshape(1, -1)
    
    S_offline = np.zeros((N, N))
    B_offline = np.zeros((N, M))
    
    for i in range(N):
        # ノードiの回帰問題を解く
        # x_i = S_i @ x + B_i @ u + e_i
        # ただし S[i,i] = 0 の制約付き
        
        # 目的変数: Y[i, :] (T,)
        y_i = Y[i, :]  # (T,)
        
        # 説明変数: [Y; U] から自分自身(i行目)を除いた部分
        # S_i (自分自身を除く) と B_i を推定
        
        # 決定変数
        s_i = cp.Variable(N)  # S[i, :] に対応
        b_i = cp.Variable(M)  # B[i, :] に対応
        
        # 残差: y_i - S_i @ Y - B_i @ U
        # Y は (N, T), Y.T @ s_i は (T,)
        # U は (M, T), U.T @ b_i は (T,)
        residual = y_i - Y.T @ s_i - U.T @ b_i
        
        # 目的関数: 二乗誤差 + L1正則化
        # 対角成分 s_i[i] は L1正則化に含めず、制約で0にする
        objective = cp.Minimize(
            cp.sum_squares(residual) / T + lambda_l1 * cp.norm1(s_i)
        )
        
        # 制約: 対角成分は0
        constraints = [s_i[i] == 0]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            if solver is not None:
                problem.solve(solver=solver, verbose=verbose)
            else:
                problem.solve(verbose=verbose)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                S_offline[i, :] = s_i.value
                B_offline[i, :] = b_i.value
            else:
                # 解が見つからない場合は0を返す
                if verbose:
                    print(f"Warning: Node {i} optimization failed with status {problem.status}")
                S_offline[i, :] = 0
                B_offline[i, :] = 0
        except Exception as e:
            if verbose:
                print(f"Warning: Node {i} optimization error: {e}")
            S_offline[i, :] = 0
            B_offline[i, :] = 0
    
    return S_offline, B_offline


def solve_offline_sem_lasso_series(
    Y: np.ndarray,
    U: np.ndarray,
    lambda_l1: float,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    オフラインでSEMのLasso問題を解く（各時刻で累積データを使用）
    
    時刻tにおいて、時刻0からtまでのデータを使って
    隣接行列を推定し、時系列として返す。
    
    Parameters
    ----------
    Y : np.ndarray
        観測データ (N, T)
    U : np.ndarray
        外生変数 (M, T)
    lambda_l1 : float
        L1正則化パラメータ
    solver : Optional[str]
        使用するソルバー
    verbose : bool
        詳細出力を表示するか
    
    Returns
    -------
    S_series : List[np.ndarray]
        各時刻での推定隣接行列のリスト (長さ T)
    """
    N, T = Y.shape
    S_series = []
    
    for t in range(T):
        # 時刻0からtまでのデータを使用
        Y_t = Y[:, :t+1]
        U_t = U[:, :t+1]
        
        if t == 0:
            # 時刻0ではデータが1サンプルしかないため推定不可
            S_series.append(np.zeros((N, N)))
        else:
            S_t, _ = solve_offline_sem_lasso(Y_t, U_t, lambda_l1, solver, verbose)
            S_series.append(S_t)
    
    return S_series


def solve_offline_sem_lasso_batch(
    Y: np.ndarray,
    U: np.ndarray,
    lambda_l1: float,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    オフラインでSEMのLasso問題を解く（全データを一括使用）
    
    全時刻のデータを使って単一の隣接行列を推定する。
    時変ではない定常SEMの推定に使用。
    
    Parameters
    ----------
    Y : np.ndarray
        観測データ (N, T)
    U : np.ndarray
        外生変数 (M, T)
    lambda_l1 : float
        L1正則化パラメータ
    solver : Optional[str]
        使用するソルバー
    verbose : bool
        詳細出力を表示するか
    
    Returns
    -------
    S_offline : np.ndarray
        推定された隣接行列 (N, N)
    """
    S_offline, _ = solve_offline_sem_lasso(Y, U, lambda_l1, solver, verbose)
    return S_offline


def compute_offline_error_normalizer(
    S_true: np.ndarray,
    S_offline: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    オフライン解を使った誤差正規化のための分母を計算
    
    ||S_true - S_offline||_F^2
    
    Parameters
    ----------
    S_true : np.ndarray
        真の隣接行列 (N, N)
    S_offline : np.ndarray
        オフライン推定された隣接行列 (N, N)
    eps : float
        ゼロ除算防止用の小さな値
    
    Returns
    -------
    normalizer : float
        正規化用の分母
    """
    return np.linalg.norm(S_true - S_offline, ord='fro') ** 2 + eps

