"""
手法実行モジュール

各推定手法の実行ロジックを一元管理します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from code.hyperparam_utils import ResolvedHyperparams
from models.pg_batch import ProximalGradientBatchSEM, ProximalGradientConfig
from models.pp_exog import PPExogenousSEM
from models.tvgti_pc.prediction_correction_sem import PredictionCorrectionSEM as PCSEM
from models.tvgti_pc.prediction_correction_sem_noexog import PredictionCorrectionSEMNoExog as PCSEMNoExog
from utils.metrics import compute_error_series
from utils.metrics import compute_normalized_error
from code.config import ComparisonParams


@dataclass
class MethodFlags:
    """実行する手法のフラグ"""
    pp: bool = False
    pc: bool = False
    co: bool = False
    sgd: bool = False
    pg: bool = False
    
    @classmethod
    def from_config(cls, cfg) -> "MethodFlags":
        """config.pyのMethodFlagsから作成"""
        return cls(
            pp=cfg.methods.pp,
            pc=cfg.methods.pc,
            co=cfg.methods.co,
            sgd=cfg.methods.sgd,
            pg=cfg.methods.pg,
        )
    
    def to_dict(self) -> Dict[str, bool]:
        """辞書形式に変換"""
        return {
            "pp": self.pp,
            "pc": self.pc,
            "co": self.co,
            "sgd": self.sgd,
            "pg": self.pg,
        }


@dataclass
class TrialResult:
    """試行結果"""
    errors: Dict[str, List[float]] = field(default_factory=dict)
    estimates_final: Dict[str, np.ndarray] = field(default_factory=dict)


class MethodExecutor:
    """
    各推定手法の実行を担当するクラス。

    Parameters
    ----------
    N : int
        ノード数
    flags : MethodFlags
        実行する手法のフラグ
    hyperparams : ResolvedHyperparams
        解決済みハイパーパラメータ
    error_normalization : str
        誤差の正規化方法
    """

    def __init__(
        self,
        N: int,
        flags: MethodFlags,
        hyperparams: ResolvedHyperparams,
        error_normalization: str = "true_value",
        comparison: Optional[ComparisonParams] = None,
    ):
        self.N = N
        self.flags = flags
        self.hp = hyperparams
        self.error_normalization = error_normalization
        self.comparison = comparison if comparison is not None else ComparisonParams()
        
        # PC法の初期行列
        self._S0_pc = np.zeros((N, N))

    def _resolve_pc_T_init(self, T_true: np.ndarray) -> np.ndarray:
        """PC/CO/SGD に与える T_init を比較設定に従って決める。"""
        if getattr(self.comparison, "pc_use_true_T_init", True):
            return T_true
        scale = float(getattr(self.comparison, "pc_T_init_identity_scale", 1.0))
        return np.eye(self.N) * scale

    def _resolve_pp_b0(self, T_true: np.ndarray) -> np.ndarray:
        """PP に与える b0（=diag(T_init)）を比較設定に従って決める。"""
        mode = str(getattr(self.comparison, "pp_init_b0", "ones")).strip()
        if mode == "true_T_diag":
            return np.diag(T_true)
        # default: ones
        return np.ones(self.N)
    
    def execute_pp(
        self,
        Y: np.ndarray,
        U: np.ndarray,
        S_series: List[np.ndarray],
        T_true: np.ndarray,
        S_offline: Optional[np.ndarray],
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """
        PP法を実行する。

        Returns
        -------
        Tuple[errors, final_estimate]
            誤差リストと最終推定値（無効の場合はNone）
        """
        if not self.flags.pp:
            return None, None
        
        S0 = np.zeros((self.N, self.N))
        b0 = self._resolve_pp_b0(T_true)
        lookahead_cfg = int(getattr(self.comparison, "pp_lookahead", 0))
        lookahead = (self.hp.pp.r + self.hp.pp.q - 2) if lookahead_cfg == -1 else max(0, lookahead_cfg)
        model = PPExogenousSEM(
            self.N, S0, b0,
            r=self.hp.pp.r,
            q=self.hp.pp.q,
            rho=self.hp.pp.rho,
            mu_lambda=self.hp.pp.mu_lambda,
            lambda_S=self.hp.pp.lambda_S,
            lookahead=lookahead,
        )
        S_hat_list, _ = model.run(Y, U)
        
        errors = compute_error_series(
            S_hat_list, S_series, S_offline, self.error_normalization
        )
        return errors, S_hat_list[-1]
    
    def execute_pc(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        S_series: List[np.ndarray],
        S_offline: Optional[np.ndarray],
        T_init: np.ndarray,
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """
        PC法を実行する。

        Returns
        -------
        Tuple[errors, final_estimate]
            誤差リストと最終推定値（無効の場合はNone）
        """
        if not self.flags.pc:
            return None, None
        pc_model = str(getattr(self.comparison, "pc_model", "exog")).strip()
        if pc_model == "noexog":
            pc = PCSEMNoExog(
                self.N,
                self._S0_pc,
                self.hp.pc.lambda_reg,
                self.hp.pc.alpha,
                self.hp.pc.beta,
                self.hp.pc.gamma,
                self.hp.pc.P,
                self.hp.pc.C,
                show_progress=False,
                name="pc_noexog",
            )
            estimates_pc, _ = pc.run(X, Z=None)
        else:
            T_init = self._resolve_pc_T_init(T_init)
            pc = PCSEM(
                self.N,
                self._S0_pc,
                self.hp.pc.lambda_reg,
                self.hp.pc.alpha,
                self.hp.pc.beta,
                self.hp.pc.gamma,
                self.hp.pc.P,
                self.hp.pc.C,
                show_progress=False,
                name="pc_baseline",
                T_init=T_init,
            )
            estimates_pc, _ = pc.run(X, Z)
        
        errors = compute_error_series(
            estimates_pc, S_series, S_offline, self.error_normalization
        )
        return errors, estimates_pc[-1]
    
    def execute_co(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        S_series: List[np.ndarray],
        S_offline: Optional[np.ndarray],
        T_init: np.ndarray,
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """
        CO法を実行する。

        Returns
        -------
        Tuple[errors, final_estimate]
            誤差リストと最終推定値（無効の場合はNone）
        """
        if not self.flags.co:
            return None, None
        pc_model = str(getattr(self.comparison, "pc_model", "exog")).strip()
        if pc_model == "noexog":
            co = PCSEMNoExog(
                self.N,
                self._S0_pc,
                self.hp.co.lambda_reg,
                self.hp.co.alpha,
                self.hp.co.beta_co,
                self.hp.co.gamma,
                0,  # P=0 for CO
                self.hp.co.C,
                show_progress=False,
                name="co_noexog",
            )
            estimates_co, _ = co.run(X, Z=None)
        else:
            T_init = self._resolve_pc_T_init(T_init)
            co = PCSEM(
                self.N,
                self._S0_pc,
                self.hp.co.lambda_reg,
                self.hp.co.alpha,
                self.hp.co.beta_co,
                self.hp.co.gamma,
                0,  # P=0 for CO
                self.hp.co.C,
                show_progress=False,
                name="co_baseline",
                T_init=T_init,
            )
            estimates_co, _ = co.run(X, Z)
        
        errors = compute_error_series(
            estimates_co, S_series, S_offline, self.error_normalization
        )
        return errors, estimates_co[-1]
    
    def execute_sgd(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        S_series: List[np.ndarray],
        S_offline: Optional[np.ndarray],
        T_init: np.ndarray,
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """
        SGD法を実行する。

        Returns
        -------
        Tuple[errors, final_estimate]
            誤差リストと最終推定値（無効の場合はNone）
        """
        if not self.flags.sgd:
            return None, None
        pc_model = str(getattr(self.comparison, "pc_model", "exog")).strip()
        if pc_model == "noexog":
            sgd = PCSEMNoExog(
                self.N,
                self._S0_pc,
                self.hp.sgd.lambda_reg,
                self.hp.sgd.alpha,
                self.hp.sgd.beta_sgd,
                0.0,  # gamma=0 for SGD
                0,    # P=0 for SGD
                self.hp.sgd.C,
                show_progress=False,
                name="sgd_noexog",
            )
            estimates_sgd, _ = sgd.run(X, Z=None)
        else:
            T_init = self._resolve_pc_T_init(T_init)
            sgd = PCSEM(
                self.N,
                self._S0_pc,
                self.hp.sgd.lambda_reg,
                self.hp.sgd.alpha,
                self.hp.sgd.beta_sgd,
                0.0,  # gamma=0 for SGD
                0,    # P=0 for SGD
                self.hp.sgd.C,
                show_progress=False,
                name="sgd_baseline",
                T_init=T_init,
            )
            estimates_sgd, _ = sgd.run(X, Z)
        
        errors = compute_error_series(
            estimates_sgd, S_series, S_offline, self.error_normalization
        )
        return errors, estimates_sgd[-1]
    
    def execute_pg(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        S_series: List[np.ndarray],
        S_offline: Optional[np.ndarray],
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """
        PG法を実行する。

        Returns
        -------
        Tuple[errors, final_estimate]
            誤差リストと最終推定値（無効の場合はNone）
        """
        if not self.flags.pg:
            return None, None
        
        pg_config = ProximalGradientConfig(
            lambda_reg=self.hp.pg.lambda_reg,
            step_size=self.hp.pg.step_size,
            step_scale=self.hp.pg.step_scale,
            max_iter=self.hp.pg.max_iter,
            tol=self.hp.pg.tol,
            use_fista=self.hp.pg.use_fista,
            use_backtracking=self.hp.pg.use_backtracking,
            show_progress=False,
            name="pg_baseline",
        )
        pg_model = ProximalGradientBatchSEM(self.N, pg_config)
        estimates_pg, _ = pg_model.run(X, Z)
        
        errors = compute_error_series(
            estimates_pg, S_series, S_offline, self.error_normalization
        )
        return errors, estimates_pg[-1]
    
    def execute_all(
        self,
        Y: np.ndarray,
        Z: np.ndarray,
        S_series: List[np.ndarray],
        T_init: np.ndarray,
        S_offline: Optional[np.ndarray] = None,
    ) -> TrialResult:
        """
        全手法を実行して結果をまとめる。

        Parameters
        ----------
        Y : np.ndarray
            観測データ (N x T)
        Z : np.ndarray
            外生変数 (N x T)
        S_series : List[np.ndarray]
            真の隣接行列の時系列
        T_init : np.ndarray
            外生変数係数行列
        S_offline : np.ndarray, optional
            オフライン解

        Returns
        -------
        TrialResult
            各手法の誤差と最終推定値
        """
        result = TrialResult()
        result.estimates_final["True"] = S_series[-1]
        
        if S_offline is not None:
            result.estimates_final["Offline"] = S_offline
        
        # PP法
        errors_pp, est_pp = self.execute_pp(Y, Z, S_series, T_init, S_offline)
        if errors_pp is not None:
            result.errors["pp"] = errors_pp
            result.estimates_final["PP"] = est_pp
        
        # PC法
        errors_pc, est_pc = self.execute_pc(Y, Z, S_series, S_offline, T_init)
        if errors_pc is not None:
            result.errors["pc"] = errors_pc
            result.estimates_final["PC"] = est_pc
        
        # CO法
        errors_co, est_co = self.execute_co(Y, Z, S_series, S_offline, T_init)
        if errors_co is not None:
            result.errors["co"] = errors_co
            result.estimates_final["CO"] = est_co
        
        # SGD法
        errors_sgd, est_sgd = self.execute_sgd(Y, Z, S_series, S_offline, T_init)
        if errors_sgd is not None:
            result.errors["sgd"] = errors_sgd
            result.estimates_final["SGD"] = est_sgd
        
        # PG法
        errors_pg, est_pg = self.execute_pg(Y, Z, S_series, S_offline)
        if errors_pg is not None:
            result.errors["pg"] = errors_pg
            result.estimates_final["PG"] = est_pg

        # 全手法で t=0 の誤差を同じ初期値（S0=0）に揃える
        baseline0 = compute_normalized_error(
            np.zeros((self.N, self.N)),
            S_series[0],
            S_offline,
            normalization=self.error_normalization,
        )
        for key in list(result.errors.keys()):
            if result.errors[key]:
                result.errors[key][0] = float(baseline0)
        
        return result

