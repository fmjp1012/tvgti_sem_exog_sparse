"""
シミュレーション設定ファイル

すべてのシミュレーション条件はこのファイルで一元管理します。
他のスクリプト（tune_and_run.py, run_piecewise.py, run_linear.py, hyperparam_tuning.py）は
このファイルから設定を読み込みます。

※ コマンドライン引数は使用せず、このファイルを直接編集して設定を変更してください。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# 実行する手法のフラグ
# =============================================================================
@dataclass
class MethodFlags:
    """実行する手法を制御するフラグ"""
    # ※ デフォルトは全てFalse（CONFIGで明示的にTrueにした手法のみ実行）
    # ※ 設定を変更するにはファイル下部の CONFIG インスタンスを編集してください
    pp: bool = False    # Proposed (PP)
    pc: bool = False    # Prediction Correction
    co: bool = False    # Correction Only
    sgd: bool = False   # SGD
    pg: bool = False    # Proximal Gradient (バッチ法)


# =============================================================================
# シナリオ共通パラメータ
# =============================================================================
@dataclass
class CommonParams:
    """シナリオ共通のシミュレーションパラメータ"""
    # ※ ここはデフォルト値の定義です
    # ※ 設定を変更するにはファイル下部の CONFIG インスタンスを編集してください
    N: int = 20           # ノード数
    T: int = 1000         # 時系列長
    sparsity: float = 0.7 # スパース性 (エッジが存在しない割合)
    max_weight: float = 0.5  # 隣接行列の最大重み
    std_e: float = 0.05   # ノイズの標準偏差
    seed: int = 3         # 乱数シードの基点


# =============================================================================
# シナリオ固有パラメータ
# =============================================================================
@dataclass
class PiecewiseParams:
    """Piecewiseシナリオ固有のパラメータ"""
    K: int = 4  # 変化点の数


@dataclass
class LinearParams:
    """Linearシナリオ固有のパラメータ"""
    # 現状は特別なパラメータなし
    pass


# =============================================================================
# データ生成パラメータ
# =============================================================================
@dataclass
class DataGenParams:
    """データ生成の追加パラメータ"""
    s_type: str = "random"      # 隣接行列のタイプ
    t_min: float = 0.5          # 外生変数係数の最小値
    t_max: float = 1.0          # 外生変数係数の最大値
    z_dist: str = "uniform01"   # 外生変数の分布


# =============================================================================
# ハイパーパラメータチューニング設定
# =============================================================================
@dataclass
class TuningParams:
    """ハイパーパラメータチューニングの設定"""
    tuning_trials: int = 300         # Optunaの試行回数
    tuning_runs_per_trial: int = 1   # 各試行での実行回数（平均を取る）
    truncation_horizon: int = 400    # チューニング時の打ち切りステップ数
    tuning_seed: int = 4             # チューニング用シード


@dataclass
class SearchRange:
    """単一パラメータの探索範囲"""
    low: float
    high: float
    log: bool = False
    type: str = "float"
    step: Optional[int] = None
    choices: Optional[List[Any]] = None


@dataclass
class PPSearchSpace:
    """PP法の探索範囲"""
    rho: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    mu_lambda: SearchRange = field(default_factory=lambda: SearchRange(low=1e-4, high=1.0, log=True))
    lambda_S: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))


@dataclass
class PCSearchSpace:
    """PC法の探索範囲"""
    lambda_reg: SearchRange = field(default_factory=lambda: SearchRange(low=1e-5, high=1e-2, log=True))
    alpha: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    beta_pc: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    gamma: SearchRange = field(default_factory=lambda: SearchRange(low=0.85, high=0.999, log=False))
    P: SearchRange = field(default_factory=lambda: SearchRange(low=0, high=2, log=False, type="int", step=1))
    C: SearchRange = field(default_factory=lambda: SearchRange(low=0, high=0, log=False, type="categorical", choices=[1, 2, 5]))


@dataclass
class COSearchSpace:
    """CO法の探索範囲"""
    alpha: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    beta_co: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    gamma: SearchRange = field(default_factory=lambda: SearchRange(low=0.85, high=0.999, log=False))
    C: SearchRange = field(default_factory=lambda: SearchRange(low=0, high=0, log=False, type="categorical", choices=[1, 2, 5]))


@dataclass
class SGDSearchSpace:
    """SGD法の探索範囲"""
    alpha: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    beta_sgd: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))


@dataclass
class PGSearchSpace:
    """PG法の探索範囲"""
    lambda_reg: SearchRange = field(default_factory=lambda: SearchRange(low=1e-5, high=1e-2, log=True))
    step_scale: SearchRange = field(default_factory=lambda: SearchRange(low=1e-6, high=1e-1, log=True))
    use_fista: SearchRange = field(default_factory=lambda: SearchRange(low=0, high=0, log=False, type="categorical", choices=[True, False]))


@dataclass
class OfflineSearchSpace:
    """オフライン解のハイパラ探索範囲"""
    offline_lambda_l1: SearchRange = field(default_factory=lambda: SearchRange(low=1e-4, high=1.0, log=True))


@dataclass
class SearchSpaces:
    """全手法の探索範囲"""
    pp: PPSearchSpace = field(default_factory=PPSearchSpace)
    pc: PCSearchSpace = field(default_factory=PCSearchSpace)
    co: COSearchSpace = field(default_factory=COSearchSpace)
    sgd: SGDSearchSpace = field(default_factory=SGDSearchSpace)
    pg: PGSearchSpace = field(default_factory=PGSearchSpace)
    offline: OfflineSearchSpace = field(default_factory=OfflineSearchSpace)


# =============================================================================
# ハイパーパラメータのデフォルト値（フォールバック）
# =============================================================================
@dataclass
class PPHyperparams:
    """PP法のハイパーパラメータ"""
    r: int = 50
    q: int = 5
    rho: float = 1e-3
    mu_lambda: float = 0.05
    lambda_S: float = 0.01  # L1正則化係数（スパース性促進）


@dataclass
class PCHyperparams:
    """PC法のハイパーパラメータ"""
    lambda_reg: float = 1e-3
    alpha: float = 1e-2
    beta: float = 1e-2
    gamma: float = 0.9
    P: int = 1
    C: int = 1


@dataclass
class COHyperparams:
    """CO法のハイパーパラメータ"""
    beta_co: float = 0.02


@dataclass
class SGDHyperparams:
    """SGD法のハイパーパラメータ"""
    beta_sgd: float = 0.0269


@dataclass
class PGHyperparams:
    """PG法のハイパーパラメータ"""
    lambda_reg: float = 1e-3
    step_scale: float = 1.0
    step_size: Optional[float] = None
    use_fista: bool = True
    use_backtracking: bool = False
    max_iter: int = 500
    tol: float = 1e-4


@dataclass
class DefaultHyperparams:
    """全手法のデフォルトハイパーパラメータ"""
    pp: PPHyperparams = field(default_factory=PPHyperparams)
    pc: PCHyperparams = field(default_factory=PCHyperparams)
    co: COHyperparams = field(default_factory=COHyperparams)
    sgd: SGDHyperparams = field(default_factory=SGDHyperparams)
    pg: PGHyperparams = field(default_factory=PGHyperparams)


# =============================================================================
# 評価指標設定
# =============================================================================
@dataclass
class MetricParams:
    """評価指標の設定"""
    # 誤差の正規化方法を選択
    # "true_value": 真の値のノルムで割る（従来の方法）
    # "offline_solution": オフライン解のノルムで割る（offline_lambda_l1はOptunaで自動探索）
    error_normalization: str = "true_value"


# =============================================================================
# 実行設定
# =============================================================================
@dataclass
class RunParams:
    """シミュレーション実行設定"""
    num_trials: int = 100        # モンテカルロ試行回数


# =============================================================================
# 出力設定
# =============================================================================
@dataclass
class OutputParams:
    """出力設定"""
    result_root: Path = field(default_factory=lambda: Path("./result"))
    subdir_piecewise: str = "exog_sparse_piecewise"
    subdir_linear: str = "exog_sparse_linear"
    subdir_tuning: str = "exog_sparse_tuning"


# =============================================================================
# メイン設定クラス
# =============================================================================
@dataclass
class SimulationConfig:
    """
    シミュレーション設定のメインクラス
    
    すべての設定はこのクラスのインスタンスを通じてアクセスします。
    """
    # 実行する手法
    methods: MethodFlags = field(default_factory=MethodFlags)
    
    # シナリオ共通パラメータ
    common: CommonParams = field(default_factory=CommonParams)
    
    # シナリオ固有パラメータ
    piecewise: PiecewiseParams = field(default_factory=PiecewiseParams)
    linear: LinearParams = field(default_factory=LinearParams)
    
    # データ生成パラメータ
    data_gen: DataGenParams = field(default_factory=DataGenParams)
    
    # チューニング設定
    tuning: TuningParams = field(default_factory=TuningParams)
    
    # 探索範囲
    search_spaces: SearchSpaces = field(default_factory=SearchSpaces)
    
    # デフォルトハイパーパラメータ
    hyperparams: DefaultHyperparams = field(default_factory=DefaultHyperparams)
    
    # 実行設定
    run: RunParams = field(default_factory=RunParams)
    
    # 評価指標設定
    metric: MetricParams = field(default_factory=MetricParams)
    
    # 出力設定
    output: OutputParams = field(default_factory=OutputParams)
    
    # 実行モード
    skip_tuning: bool = False          # チューニングをスキップ
    skip_simulation: bool = False      # シミュレーションをスキップ
    hyperparam_json: Optional[Path] = None  # 既存のハイパラJSONを使用する場合のパス


# =============================================================================
# ★★★ 設定モード切り替え ★★★
# =============================================================================
# True: テスト用の軽量設定（プログラム動作確認用、すぐに終わる）
# False: 本番用設定（実際のシミュレーション用）
USE_TEST_CONFIG = True
# USE_TEST_CONFIG = False

# =============================================================================
# ★★★ 設定を変更するにはここを編集してください ★★★
# =============================================================================
# 上部のクラス定義ではなく、以下の CONFIG インスタンスの値を変更してください。
# クラス定義のデフォルト値を変更しても、ここで明示的に設定された値が優先されます。
#
CONFIG_MAIN = SimulationConfig(
    # 実行する手法（Trueにした手法のみ実行、コメントアウトで無効化可能）
    methods=MethodFlags(
        pp=True,
        pc=True,
        co=True,
        sgd=True,
        # pg=True,
    ),
    
    # シナリオ共通パラメータ
    common=CommonParams(
        N=20,
        T=1000,  # テスト用に短くしています -> 本番パラメータ
        sparsity=0.7,
        max_weight=0.5,
        std_e=0.05,
        seed=3,
    ),
    
    # Piecewiseシナリオのパラメータ
    piecewise=PiecewiseParams(
        K=4,
    ),
    
    # データ生成パラメータ
    data_gen=DataGenParams(
        s_type="random",
        t_min=0.5,
        t_max=1.0,
        z_dist="uniform01",
    ),
    
    # チューニング設定
    tuning=TuningParams(
        tuning_trials=300,
        tuning_runs_per_trial=1,
        truncation_horizon=400,
        tuning_seed=4,
    ),
    
    # ハイパーパラメータ探索範囲
    # 各パラメータの探索範囲をカスタマイズできます
    # SearchRange(low, high, log=False, type="float", step=None, choices=None)
    # - log=True: 対数スケールで探索
    # - type: "float", "int", "categorical"
    # - step: int型の場合のステップ
    # - choices: categorical型の場合の選択肢リスト
    search_spaces=SearchSpaces(
        pp=PPSearchSpace(
            rho=SearchRange(low=1e-6, high=1e-1, log=True),
            mu_lambda=SearchRange(low=1e-4, high=1.0, log=True),
            lambda_S=SearchRange(low=1e-6, high=1e-1, log=True),
        ),
        pc=PCSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True),
            alpha=SearchRange(low=1e-6, high=1e+1, log=True),
            beta_pc=SearchRange(low=1e-6, high=1e+1, log=True),
            gamma=SearchRange(low=0.85, high=0.999, log=False),
            P=SearchRange(low=1, high=1, log=False, type="int", step=1),
            C=SearchRange(low=1, high=1, log=False, type="int", step=1),
        ),
        co=COSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e+1, log=True),
            beta_co=SearchRange(low=1e-6, high=1e+1, log=True),
            gamma=SearchRange(low=0.85, high=0.999, log=False),
            C=SearchRange(low=1, high=1, log=False, type="int", step=1),
        ),
        sgd=SGDSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e+1, log=True),
            beta_sgd=SearchRange(low=1e-6, high=1e+1, log=True),
        ),
        pg=PGSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e+1, log=True),
            step_scale=SearchRange(low=1e-6, high=1e+1, log=True),
            use_fista=SearchRange(low=0, high=0, log=False, type="categorical", choices=[False]),
            # use_fista=SearchRange(low=0, high=0, log=False, type="categorical", choices=[True, False]),
        ),
        offline=OfflineSearchSpace(
            offline_lambda_l1=SearchRange(low=1e-4, high=1e+1, log=True),
        ),
    ),
    
    # 実行設定
    run=RunParams(
        num_trials=100,
    ),
    
    # 評価指標設定
    # "true_value": 真の値のノルムで割る（従来の方法）
    # "offline_solution": オフライン解のノルムで割る（offline_lambda_l1はOptunaで自動探索）
    metric=MetricParams(
        error_normalization="true_value",
        # error_normalization="offline_solution",
    ),
    
    # 出力設定
    output=OutputParams(
        result_root=Path("./result"),
    ),
    
    # 実行モード
    skip_tuning=False,
    skip_simulation=False,
    hyperparam_json=None,  # 既存JSONを使う場合: Path("path/to/hyperparams.json")
)

# =============================================================================
# テスト用設定（プログラム動作確認用・すぐに終わる軽量設定）
# =============================================================================
CONFIG_TEST = SimulationConfig(
    # 実行する手法（テストでは1つだけ有効に）
    methods=MethodFlags(
        pp=True,
        # pc=True,
        # co=True,
        # sgd=True,
        # pg=True,
    ),
    
    # テスト用の小さなパラメータ
    common=CommonParams(
        N=5,              # 小さなノード数
        T=50,             # 短い時系列
        sparsity=0.5,     # スパース性
        max_weight=0.5,
        std_e=0.05,
        seed=3,
    ),
    
    # Piecewiseシナリオのパラメータ
    piecewise=PiecewiseParams(
        K=2,  # 変化点の数（少なめ）
    ),
    
    # データ生成パラメータ
    data_gen=DataGenParams(
        s_type="random",
        t_min=0.5,
        t_max=1.0,
        z_dist="uniform01",
    ),
    
    # テスト用の軽量チューニング設定
    tuning=TuningParams(
        tuning_trials=3,           # 試行回数を大幅削減
        tuning_runs_per_trial=1,
        truncation_horizon=30,     # 打ち切りも短く
        tuning_seed=4,
    ),
    
    # 探索範囲（本番と同じ）
    search_spaces=SearchSpaces(
        pp=PPSearchSpace(
            rho=SearchRange(low=1e-6, high=1e-1, log=True),
            mu_lambda=SearchRange(low=1e-4, high=1.0, log=True),
            lambda_S=SearchRange(low=1e-6, high=1e-1, log=True),
        ),
        pc=PCSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True),
            alpha=SearchRange(low=1e-6, high=1e-1, log=True),
            beta_pc=SearchRange(low=1e-6, high=1e-1, log=True),
            gamma=SearchRange(low=0.85, high=0.999, log=False),
            P=SearchRange(low=0, high=2, log=False, type="int", step=1),
            C=SearchRange(low=0, high=0, log=False, type="categorical", choices=[1, 2, 5]),
        ),
        co=COSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e-1, log=True),
            beta_co=SearchRange(low=1e-6, high=1e-1, log=True),
            gamma=SearchRange(low=0.85, high=0.999, log=False),
            C=SearchRange(low=0, high=0, log=False, type="categorical", choices=[1, 2, 5]),
        ),
        sgd=SGDSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e-1, log=True),
            beta_sgd=SearchRange(low=1e-6, high=1e-1, log=True),
        ),
        pg=PGSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True),
            step_scale=SearchRange(low=1e-6, high=1e-1, log=True),
            use_fista=SearchRange(low=0, high=0, log=False, type="categorical", choices=[True, False]),
        ),
        offline=OfflineSearchSpace(
            offline_lambda_l1=SearchRange(low=1e-4, high=1.0, log=True),
        ),
    ),
    
    # テスト用の少ない試行回数
    run=RunParams(
        num_trials=2,  # モンテカルロ試行も最小限
    ),
    
    # 評価指標設定
    metric=MetricParams(
        # error_normalization="true_value",  # テストではシンプルな方法で
        error_normalization="offline_solution",
    ),
    
    # 出力設定
    output=OutputParams(
        result_root=Path("./result"),
    ),
    
    # 実行モード
    skip_tuning=False,
    skip_simulation=False,
    hyperparam_json=None,
)

# =============================================================================
# 設定の切り替え
# =============================================================================
# USE_TEST_CONFIG の値に基づいて CONFIG を選択
CONFIG = CONFIG_TEST if USE_TEST_CONFIG else CONFIG_MAIN


# =============================================================================
# ヘルパー関数
# =============================================================================
def get_config() -> SimulationConfig:
    """グローバル設定を取得"""
    return CONFIG


def get_enabled_methods() -> List[str]:
    """有効化されている手法のリストを取得"""
    cfg = get_config()
    methods = []
    if cfg.methods.pp:
        methods.append("pp")
    if cfg.methods.pc:
        methods.append("pc")
    if cfg.methods.co:
        methods.append("co")
    if cfg.methods.sgd:
        methods.append("sgd")
    if cfg.methods.pg:
        methods.append("pg")
    return methods


def search_range_to_dict(sr: SearchRange) -> Dict[str, Any]:
    """SearchRangeをdict形式に変換"""
    result: Dict[str, Any] = {"type": sr.type}
    if sr.type == "categorical":
        result["choices"] = sr.choices
    else:
        result["low"] = sr.low
        result["high"] = sr.high
        result["log"] = sr.log
        if sr.step is not None:
            result["step"] = sr.step
    return result


def get_search_spaces_dict() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """探索範囲を辞書形式で取得"""
    cfg = get_config()
    return {
        "pp": {
            "rho": search_range_to_dict(cfg.search_spaces.pp.rho),
            "mu_lambda": search_range_to_dict(cfg.search_spaces.pp.mu_lambda),
            "lambda_S": search_range_to_dict(cfg.search_spaces.pp.lambda_S),
        },
        "pc": {
            "lambda_reg": search_range_to_dict(cfg.search_spaces.pc.lambda_reg),
            "alpha": search_range_to_dict(cfg.search_spaces.pc.alpha),
            "beta_pc": search_range_to_dict(cfg.search_spaces.pc.beta_pc),
            "gamma": search_range_to_dict(cfg.search_spaces.pc.gamma),
            "P": search_range_to_dict(cfg.search_spaces.pc.P),
            "C": search_range_to_dict(cfg.search_spaces.pc.C),
        },
        "co": {
            "alpha": search_range_to_dict(cfg.search_spaces.co.alpha),
            "beta_co": search_range_to_dict(cfg.search_spaces.co.beta_co),
            "gamma": search_range_to_dict(cfg.search_spaces.co.gamma),
            "C": search_range_to_dict(cfg.search_spaces.co.C),
        },
        "sgd": {
            "alpha": search_range_to_dict(cfg.search_spaces.sgd.alpha),
            "beta_sgd": search_range_to_dict(cfg.search_spaces.sgd.beta_sgd),
        },
        "pg": {
            "lambda_reg": search_range_to_dict(cfg.search_spaces.pg.lambda_reg),
            "step_scale": search_range_to_dict(cfg.search_spaces.pg.step_scale),
            "use_fista": search_range_to_dict(cfg.search_spaces.pg.use_fista),
        },
        "offline": {
            "offline_lambda_l1": search_range_to_dict(cfg.search_spaces.offline.offline_lambda_l1),
        },
    }


def get_default_hyperparams_dict() -> Dict[str, Dict[str, Any]]:
    """デフォルトハイパーパラメータを辞書形式で取得"""
    cfg = get_config()
    return {
        "pp": {
            "r": cfg.hyperparams.pp.r,
            "q": cfg.hyperparams.pp.q,
            "rho": cfg.hyperparams.pp.rho,
            "mu_lambda": cfg.hyperparams.pp.mu_lambda,
            "lambda_S": cfg.hyperparams.pp.lambda_S,
        },
        "pc": {
            "lambda_reg": cfg.hyperparams.pc.lambda_reg,
            "alpha": cfg.hyperparams.pc.alpha,
            "beta": cfg.hyperparams.pc.beta,
            "gamma": cfg.hyperparams.pc.gamma,
            "P": cfg.hyperparams.pc.P,
            "C": cfg.hyperparams.pc.C,
        },
        "co": {
            "beta_co": cfg.hyperparams.co.beta_co,
        },
        "sgd": {
            "beta_sgd": cfg.hyperparams.sgd.beta_sgd,
        },
        "pg": {
            "lambda_reg": cfg.hyperparams.pg.lambda_reg,
            "step_scale": cfg.hyperparams.pg.step_scale,
            "step_size": cfg.hyperparams.pg.step_size,
            "use_fista": cfg.hyperparams.pg.use_fista,
            "use_backtracking": cfg.hyperparams.pg.use_backtracking,
            "max_iter": cfg.hyperparams.pg.max_iter,
            "tol": cfg.hyperparams.pg.tol,
        },
    }


def get_metric_params_dict() -> Dict[str, Any]:
    """評価指標設定を辞書形式で取得"""
    cfg = get_config()
    return {
        "error_normalization": cfg.metric.error_normalization,
    }


def print_config_summary() -> None:
    """設定のサマリーを表示"""
    cfg = get_config()
    
    print("=" * 60)
    print("シミュレーション設定サマリー")
    print("=" * 60)
    
    # テストモードかどうかを表示
    if USE_TEST_CONFIG:
        print("\n*** テストモード（軽量設定）で実行中 ***")
    else:
        print("\n*** 本番モードで実行中 ***")
    
    print("\n--- 実行する手法 ---")
    print(f"  PP:  {'ON' if cfg.methods.pp else 'OFF'}")
    print(f"  PC:  {'ON' if cfg.methods.pc else 'OFF'}")
    print(f"  CO:  {'ON' if cfg.methods.co else 'OFF'}")
    print(f"  SGD: {'ON' if cfg.methods.sgd else 'OFF'}")
    print(f"  PG:  {'ON' if cfg.methods.pg else 'OFF'}")
    
    print("\n--- シナリオ共通パラメータ ---")
    print(f"  N: {cfg.common.N}")
    print(f"  T: {cfg.common.T}")
    print(f"  sparsity: {cfg.common.sparsity}")
    print(f"  max_weight: {cfg.common.max_weight}")
    print(f"  std_e: {cfg.common.std_e}")
    print(f"  seed: {cfg.common.seed}")
    
    print("\n--- Piecewiseパラメータ ---")
    print(f"  K: {cfg.piecewise.K}")
    
    print("\n--- チューニング設定 ---")
    print(f"  trials: {cfg.tuning.tuning_trials}")
    print(f"  runs_per_trial: {cfg.tuning.tuning_runs_per_trial}")
    print(f"  truncation_horizon: {cfg.tuning.truncation_horizon}")
    
    print("\n--- 評価指標設定 ---")
    print(f"  error_normalization: {cfg.metric.error_normalization}")
    if cfg.metric.error_normalization == "offline_solution":
        print(f"  offline_lambda_l1 探索範囲: [{cfg.search_spaces.offline.offline_lambda_l1.low}, {cfg.search_spaces.offline.offline_lambda_l1.high}]")
    
    print("\n--- 実行設定 ---")
    print(f"  num_trials: {cfg.run.num_trials}")
    print(f"  skip_tuning: {cfg.skip_tuning}")
    print(f"  skip_simulation: {cfg.skip_simulation}")
    if cfg.hyperparam_json:
        print(f"  hyperparam_json: {cfg.hyperparam_json}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 設定ファイルを直接実行した場合は設定サマリーを表示
    print_config_summary()

