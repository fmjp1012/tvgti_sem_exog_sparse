"""
ハイパーパラメータ解決ユーティリティ

JSONファイルまたはデフォルト設定からハイパーパラメータを解決するためのユーティリティを提供します。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from code.config import SimulationConfig, get_config, get_default_hyperparams_dict
from utils.formatting import coerce_bool


@dataclass
class PPParams:
    """PP法のハイパーパラメータ"""
    r: int = 50
    q: int = 5
    rho: float = 1e-3
    mu_lambda: float = 0.05


@dataclass
class PCParams:
    """PC法のハイパーパラメータ"""
    lambda_reg: float = 1e-3
    alpha: float = 1e-2
    beta: float = 1e-2
    gamma: float = 0.9
    P: int = 1
    C: int = 1


@dataclass
class COParams:
    """CO法のハイパーパラメータ（PC法のパラメータを継承して使用）"""
    lambda_reg: float = 1e-3
    alpha: float = 1e-2
    beta_co: float = 0.02
    gamma: float = 0.9
    C: int = 1


@dataclass
class SGDParams:
    """SGD法のハイパーパラメータ（PC法のパラメータを継承して使用）"""
    lambda_reg: float = 1e-3
    alpha: float = 1e-2
    beta_sgd: float = 0.0269
    C: int = 1


@dataclass
class PGParams:
    """PG法のハイパーパラメータ"""
    lambda_reg: float = 1e-3
    step_scale: float = 1.0
    step_size: Optional[float] = None
    use_fista: bool = True
    use_backtracking: bool = False
    max_iter: int = 500
    tol: float = 1e-4


@dataclass
class ResolvedHyperparams:
    """全手法の解決済みハイパーパラメータ"""
    pp: PPParams = field(default_factory=PPParams)
    pc: PCParams = field(default_factory=PCParams)
    co: COParams = field(default_factory=COParams)
    sgd: SGDParams = field(default_factory=SGDParams)
    pg: PGParams = field(default_factory=PGParams)
    offline_lambda_l1: Optional[float] = None


def load_hyperparams_json(json_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """
    ハイパーパラメータJSONを読み込む。

    Parameters
    ----------
    json_path : Path, optional
        JSONファイルのパス

    Returns
    -------
    Dict[str, Any], optional
        読み込んだハイパーパラメータ辞書、またはNone

    Raises
    ------
    FileNotFoundError
        指定されたパスにファイルが存在しない場合
    """
    if json_path is None:
        return None
    if not json_path.is_file():
        raise FileNotFoundError(f"ハイパラJSONが見つかりません: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_hyperparams(
    loaded: Optional[Dict[str, Any]] = None,
    cfg: Optional[SimulationConfig] = None,
) -> ResolvedHyperparams:
    """
    JSONまたはデフォルト設定からハイパーパラメータを解決する。

    Parameters
    ----------
    loaded : Dict[str, Any], optional
        JSONから読み込んだハイパーパラメータ
    cfg : SimulationConfig, optional
        設定オブジェクト（Noneの場合はget_config()を使用）

    Returns
    -------
    ResolvedHyperparams
        解決済みハイパーパラメータ
    """
    if cfg is None:
        cfg = get_config()
    
    defaults = get_default_hyperparams_dict()
    hyperparams = loaded if loaded else defaults
    
    pp_cfg = hyperparams.get("pp", {})
    pc_cfg = hyperparams.get("pc", {})
    co_cfg = hyperparams.get("co", {})
    sgd_cfg = hyperparams.get("sgd", {})
    pg_cfg = hyperparams.get("pg", {})
    
    # PP法
    pp = PPParams(
        r=int(pp_cfg.get("r", cfg.hyperparams.pp.r)),
        q=int(pp_cfg.get("q", cfg.hyperparams.pp.q)),
        rho=float(pp_cfg.get("rho", cfg.hyperparams.pp.rho)),
        mu_lambda=float(pp_cfg.get("mu_lambda", cfg.hyperparams.pp.mu_lambda)),
    )
    
    # PC法
    pc = PCParams(
        lambda_reg=float(pc_cfg.get("lambda_reg", cfg.hyperparams.pc.lambda_reg)),
        alpha=float(pc_cfg.get("alpha", cfg.hyperparams.pc.alpha)),
        beta=float(pc_cfg.get("beta", cfg.hyperparams.pc.beta)),
        gamma=float(pc_cfg.get("gamma", cfg.hyperparams.pc.gamma)),
        P=int(pc_cfg.get("P", cfg.hyperparams.pc.P)),
        C=int(pc_cfg.get("C", cfg.hyperparams.pc.C)),
    )
    
    # CO法
    co = COParams(
        lambda_reg=pc.lambda_reg,
        alpha=pc.alpha,
        beta_co=float(co_cfg.get("beta_co", cfg.hyperparams.co.beta_co)),
        gamma=pc.gamma,
        C=pc.C,
    )
    
    # SGD法
    sgd = SGDParams(
        lambda_reg=pc.lambda_reg,
        alpha=pc.alpha,
        beta_sgd=float(sgd_cfg.get("beta_sgd", cfg.hyperparams.sgd.beta_sgd)),
        C=pc.C,
    )
    
    # PG法
    step_size_raw = pg_cfg.get("step_size", cfg.hyperparams.pg.step_size)
    step_size = float(step_size_raw) if step_size_raw is not None else None
    
    pg = PGParams(
        lambda_reg=float(pg_cfg.get("lambda_reg", cfg.hyperparams.pg.lambda_reg)),
        step_scale=float(pg_cfg.get("step_scale", cfg.hyperparams.pg.step_scale)),
        step_size=step_size,
        use_fista=coerce_bool(pg_cfg.get("use_fista", cfg.hyperparams.pg.use_fista), default=True),
        use_backtracking=coerce_bool(pg_cfg.get("use_backtracking", cfg.hyperparams.pg.use_backtracking), default=False),
        max_iter=int(pg_cfg.get("max_iter", cfg.hyperparams.pg.max_iter)),
        tol=float(pg_cfg.get("tol", cfg.hyperparams.pg.tol)),
    )
    
    # オフラインλの解決
    offline_lambda_l1 = None
    if cfg.metric.error_normalization == "offline_solution":
        if hyperparams.get("offline_lambda_l1") is not None:
            offline_lambda_l1 = float(hyperparams["offline_lambda_l1"])
        else:
            offline_space = cfg.search_spaces.offline.offline_lambda_l1
            offline_lambda_l1 = math.sqrt(offline_space.low * offline_space.high)
    
    return ResolvedHyperparams(
        pp=pp,
        pc=pc,
        co=co,
        sgd=sgd,
        pg=pg,
        offline_lambda_l1=offline_lambda_l1,
    )


def hyperparams_to_dict(hp: ResolvedHyperparams) -> Dict[str, Dict[str, Any]]:
    """
    ResolvedHyperparamsを辞書形式に変換する（メタデータ保存用）。

    Parameters
    ----------
    hp : ResolvedHyperparams
        解決済みハイパーパラメータ

    Returns
    -------
    Dict[str, Dict[str, Any]]
        辞書形式のハイパーパラメータ
    """
    return {
        "pp": {
            "r": hp.pp.r,
            "q": hp.pp.q,
            "rho": hp.pp.rho,
            "mu_lambda": hp.pp.mu_lambda,
        },
        "pc": {
            "lambda_reg": hp.pc.lambda_reg,
            "alpha": hp.pc.alpha,
            "beta": hp.pc.beta,
            "gamma": hp.pc.gamma,
            "P": hp.pc.P,
            "C": hp.pc.C,
        },
        "co": {
            "lambda_reg": hp.co.lambda_reg,
            "alpha": hp.co.alpha,
            "beta_co": hp.co.beta_co,
            "gamma": hp.co.gamma,
            "C": hp.co.C,
        },
        "sgd": {
            "lambda_reg": hp.sgd.lambda_reg,
            "alpha": hp.sgd.alpha,
            "beta_sgd": hp.sgd.beta_sgd,
            "C": hp.sgd.C,
        },
        "pg": {
            "lambda_reg": hp.pg.lambda_reg,
            "step_scale": hp.pg.step_scale,
            "step_size": hp.pg.step_size,
            "use_fista": hp.pg.use_fista,
            "use_backtracking": hp.pg.use_backtracking,
            "max_iter": hp.pg.max_iter,
            "tol": hp.pg.tol,
        },
    }

