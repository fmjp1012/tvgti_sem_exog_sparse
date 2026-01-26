"""
シミュレーション設定ファイル

すべてのシミュレーション条件はこのファイルで一元管理します。
他のスクリプト（tune_and_run.py, run_piecewise.py, run_linear.py, hyperparam_tuning.py）は
このファイルから設定を読み込みます。

※ コマンドライン引数は使用せず、このファイルを直接編集して設定を変更してください。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# 実行する手法のフラグ
# =============================================================================
@dataclass
class MethodFlags:
    """実行する手法を制御するフラグ"""
    # ※ 設定を変更するにはファイル下部の CONFIG インスタンスを編集してください
    pp: bool    # Proposed (PP)
    pp_sgd: bool  # PP-SGD (q=1, r=1固定; 1データ更新)
    pc: bool    # Prediction Correction
    co: bool    # Correction Only
    sgd: bool   # SGD
    pg: bool    # Proximal Gradient (バッチ法)


# =============================================================================
# シナリオ共通パラメータ
# =============================================================================
@dataclass
class CommonParams:
    """シナリオ共通のシミュレーションパラメータ"""
    # ※ 設定を変更するにはファイル下部の CONFIG インスタンスを編集してください
    N: int           # ノード数
    T: int         # 時系列長
    sparsity: float # スパース性 (エッジが存在しない割合)
    max_weight: float  # 隣接行列の最大重み
    std_e: float   # ノイズの標準偏差
    seed: int         # 乱数シードの基点


# =============================================================================
# シナリオ固有パラメータ
# =============================================================================
@dataclass
class PiecewiseParams:
    """Piecewiseシナリオ固有のパラメータ"""
    K: int  # 変化点の数


@dataclass
class LinearParams:
    """Linearシナリオ固有のパラメータ"""
    # 現状は特別なパラメータなし
    pass


@dataclass
class BrownianParams:
    """Brownianシナリオ固有のパラメータ"""
    K: int          # 区間数 (K=Tなら毎ステップ更新)
    std_S: float # Sの更新ノイズ標準偏差



# =============================================================================
# データ生成パラメータ
# =============================================================================
@dataclass
class DataGenParams:
    """データ生成の追加パラメータ"""
    s_type: str      # 隣接行列のタイプ
    t_min: float          # 外生変数係数の最小値
    t_max: float          # 外生変数係数の最大値
    z_dist: str   # 外生変数の分布


# =============================================================================
# ハイパーパラメータチューニング設定
# =============================================================================
@dataclass
class TuningParams:
    """ハイパーパラメータチューニングの設定"""
    tuning_trials: int         # Optunaの試行回数
    tuning_runs_per_trial: int   # 各試行での実行回数（平均を取る）
    truncation_horizon: int    # チューニング時の打ち切りステップ数
    tuning_seed: int             # チューニング用シード


@dataclass
class SearchRange:
    """単一パラメータの探索範囲"""
    low: float
    high: float
    log: bool
    type: str
    step: Optional[int]
    choices: Optional[List[Any]]


@dataclass
class PPSearchSpace:
    """PP法の探索範囲"""
    # r,q も探索対象に含める（提案法が負けているケースを拾いやすくするため）
    # - r: ウィンドウ長（大きいほど平滑化されやすいが追従性は落ちる）
    # - q: 並列ブロック数（大きいほど情報を増やすが計算量/安定性に影響）
    r: SearchRange
    q: SearchRange
    rho: SearchRange
    mu_lambda: SearchRange
    lambda_S: SearchRange


@dataclass
class PCSearchSpace:
    """PC法の探索範囲"""
    lambda_reg: SearchRange
    alpha: SearchRange
    beta_pc: SearchRange
    gamma: SearchRange
    P: SearchRange
    C: SearchRange


@dataclass
class COSearchSpace:
    """CO法の探索範囲"""
    alpha: SearchRange
    beta_co: SearchRange
    gamma: SearchRange
    C: SearchRange


@dataclass
class SGDSearchSpace:
    """SGD法の探索範囲"""
    alpha: SearchRange
    beta_sgd: SearchRange


@dataclass
class PGSearchSpace:
    """PG法の探索範囲"""
    lambda_reg: SearchRange
    step_scale: SearchRange
    use_fista: SearchRange


@dataclass
class OfflineSearchSpace:
    """オフライン解のハイパラ探索範囲"""
    offline_lambda_l1: SearchRange


@dataclass
class SearchSpaces:
    """全手法の探索範囲"""
    pp: PPSearchSpace
    pp_sgd: PPSearchSpace
    pc: PCSearchSpace
    co: COSearchSpace
    sgd: SGDSearchSpace
    pg: PGSearchSpace
    offline: OfflineSearchSpace


# =============================================================================
# ハイパーパラメータのデフォルト値（フォールバック）
# =============================================================================
@dataclass
class PPHyperparams:
    """PP法のハイパーパラメータ"""
    r: int
    q: int
    rho: float
    mu_lambda: float
    lambda_S: float  # L1正則化係数（スパース性促進）


@dataclass
class PCHyperparams:
    """PC法のハイパーパラメータ"""
    lambda_reg: float
    alpha: float
    beta: float
    gamma: float
    P: int
    C: int


@dataclass
class COHyperparams:
    """CO法のハイパーパラメータ"""
    beta_co: float


@dataclass
class SGDHyperparams:
    """SGD法のハイパーパラメータ"""
    beta_sgd: float


@dataclass
class PGHyperparams:
    """PG法のハイパーパラメータ"""
    lambda_reg: float
    step_scale: float
    step_size: Optional[float]
    use_fista: bool
    use_backtracking: bool
    max_iter: int
    tol: float


@dataclass
class DefaultHyperparams:
    """全手法のデフォルトハイパーパラメータ"""
    pp: PPHyperparams
    # PP-SGD（q=1, r=1固定）: 1データのみで更新する比較用
    pp_sgd: PPHyperparams
    pc: PCHyperparams
    co: COHyperparams
    sgd: SGDHyperparams
    pg: PGHyperparams


# =============================================================================
# 評価指標設定
# =============================================================================
@dataclass
class MetricParams:
    """評価指標の設定"""
    # 誤差の正規化方法を選択
    # "true_value": 真の値のノルムで割る（従来の方法）
    # "offline_solution": オフライン解のノルムで割る（offline_lambda_l1はOptunaで自動探索）
    error_normalization: str
    # 時系列平均誤差を取る際に、先頭 burn_in ステップを無視する
    # - 0: 無視しない
    # - -1: 自動（PPの r,q から r+q-2 を使用）
    #
    # 目的:
    # - PPは序盤、(1)窓長rがまだ揃わない / (2)並列ブロック数qが揃わない ため更新が弱く、
    #   「序盤込みの平均」を取ると不利になりやすい。
    # - burn_inで評価から序盤を外し、「立ち上がり後」の性能比較をしやすくする。
    #
    # 注意:
    # - burn_in は “評価/可視化の集計” を変えるだけで、データ長Tやプロットの横軸範囲は変えない。
    burn_in: int

    # 誤差を N^2 で割るか（行列要素あたりの平均スケールにする）
    # - False: 既存の誤差（デフォルト）
    # - True: (既存の誤差) / (N^2)
    #
    # 目的:
    # - Nが変わったときに、誤差のスケールが N^2 に比例して見えるのを抑える。
    # 注意:
    # - 相対比較（どの手法が勝つか）は変えないが、縦軸スケールは変わる。
    divide_by_n2: bool


# =============================================================================
# 比較条件（公平性/見せ方）設定
# =============================================================================
@dataclass
class ComparisonParams:
    """手法間の比較条件を揃えるための設定"""
    # PC系（PC/CO/SGD）が使うモデル
    # - "exog": x = Sx + Tz + noise（従来）
    # - "noexog": x = Sx + noise（Z/Tを一切使わない）
    #   ※ "noexog" は提案法（PP）が外生を使う条件ではベースラインが不利になるので、
    #      “提案法を良く見せる/条件を変える” 意図がある場合に使う。
    pc_model: str

    # PC/CO/SGD に真の T を与えるか（Trueだとベースラインが有利になりがち）
    # - True: PC側は外生係数を“既知”として扱える（ベースライン有利）
    # - False: T_init を単位行列スケールで与える（Tを当てにしない/未知に近い条件）
    pc_use_true_T_init: bool
    # pc_use_true_T_init=False の場合に使う T_init（単位行列のスケール）
    pc_T_init_identity_scale: float

    # PP の T 初期値 b0 の作り方
    # - "ones": b0 = 1
    # - "true_T_diag": b0 = diag(T_true)
    #   ※ PPは [S|T] を同時に推定するので、b0 は “Tの初期値” に相当する。
    pp_init_b0: str

    # PP の「序盤のデータ不足」を緩和するために、ウィンドウの右側に先読みデータを含める
    # - 0: 先読みしない（オンライン）
    # - -1: 自動（r+q-2 だけ先読みしてフル窓/フル並列が効くようにする）
    #
    # 何が起きるか:
    # - PP内部で使うブロック窓が [t-r+1, t] から [t-r+1, t+lookahead] に広がる。
    # - これにより tが小さい序盤でも“情報量のある窓”を作れて、更新が進みやすくなる。
    #
    # 注意（重要）:
    # - lookahead>0 は未来データを使うため、厳密なオンライン条件ではない（チート/オフライン寄り）。
    # - ただしプロットの横軸 t=0..T-1（データ長T）は変えない。
    pp_lookahead: int


# =============================================================================
# 実行設定
# =============================================================================
@dataclass
class RunParams:
    """シミュレーション実行設定"""
    num_trials: int        # モンテカルロ試行回数


# =============================================================================
# スクリプト固有の設定
# =============================================================================
@dataclass
class DiagnoseRerunParams:
    """diagnose_rerun_sgd_piecewise の設定"""
    meta_json: Path
    hyperparam_json: Optional[Path]
    n_jobs: int
    topk: int
    exclude_top_by_max: int
    exclude_seeds: List[int]
    write_updated_meta: bool
    inplace: bool
    overwrite_figure: bool
    figure_out: Optional[Path]
    no_show: bool


@dataclass
class ScriptParams:
    """スクリプト固有の設定"""
    tune_and_run_scenario: str
    run_piecewise_once_show_offline_line: bool
    run_piecewise_once_heatmap_time: int
    run_piecewise_once_output_subdir: str
    run_linear_once_output_subdir: str
    diagnose_rerun_sgd_piecewise: DiagnoseRerunParams


# =============================================================================
# 出力設定
# =============================================================================
@dataclass
class OutputParams:
    """出力設定"""
    result_root: Path
    subdir_piecewise: str
    subdir_linear: str
    subdir_tuning: str
    save_heatmap: bool  # ヒートマップ画像を保存するか


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
    methods: MethodFlags
    
    # シナリオ共通パラメータ
    common: CommonParams
    
    # シナリオ固有パラメータ
    piecewise: PiecewiseParams
    linear: LinearParams
    brownian: BrownianParams
    
    # データ生成パラメータ
    data_gen: DataGenParams
    
    # チューニング設定
    tuning: TuningParams
    
    # 探索範囲
    search_spaces: SearchSpaces
    
    # デフォルトハイパーパラメータ
    hyperparams: DefaultHyperparams
    
    # 実行設定
    run: RunParams

    # スクリプト固有設定
    scripts: ScriptParams
    
    # 評価指標設定
    metric: MetricParams

    # 比較条件設定
    comparison: ComparisonParams
    
    # 出力設定
    output: OutputParams
    
    # 実行モード
    skip_tuning: bool          # チューニングをスキップ
    skip_simulation: bool      # シミュレーションをスキップ
    hyperparam_json: Optional[Path]  # 既存のハイパラJSONを使用する場合のパス


# =============================================================================
# ★★★ 設定モード切り替え ★★★
# =============================================================================
# True: テスト用の軽量設定（プログラム動作確認用、すぐに終わる）
# False: 本番用設定（実際のシミュレーション用）
# USE_TEST_CONFIG = True
USE_TEST_CONFIG = False

# =============================================================================
# ★★★ 設定を変更するにはここを編集してください ★★★
# =============================================================================
# 上部のクラス定義ではなく、以下の CONFIG インスタンスの値を変更してください。
# クラス定義にはデフォルト値を置かず、必ずここで全項目を明示的に指定してください。
#
CONFIG_MAIN = SimulationConfig(
    # 実行する手法（Trueにした手法のみ実行、コメントアウトで無効化可能）
    methods=MethodFlags(
        pp=True,
        pp_sgd=False,
        pc=True,
        co=True,
        sgd=True,
        pg=False,
    ),
    
    # シナリオ共通パラメータ
    common=CommonParams(
        N=20,              # ノード数（行列Sは N×N）
        T=1000,           # 再現対象に合わせる
        sparsity=0.7,      # スパース性（0要素の割合）
        max_weight=0.5,    # Sの非ゼロ重みの上限（生成時）
        std_e=0.05,        # 観測ノイズの標準偏差
        seed=3,            # 乱数シード基点（trialごとにseed+i）
    ),
    
    # Piecewiseシナリオのパラメータ
    piecewise=PiecewiseParams(
       K=4,               # 再現対象に合わせる
    ),

    # Linearシナリオのパラメータ（現状パラメータなしだが明示）
    linear=LinearParams(),

    # Brownianシナリオのパラメータ
    brownian=BrownianParams(
        K=2,
        std_S=0.05,
    ),
    
    # データ生成パラメータ
    data_gen=DataGenParams(
        s_type="random",   # Sの生成方法（"random"/"regular" 等）
        t_min=0.5,         # 外生係数T（対角）の最小値（生成時）
        t_max=1.0,         # 外生係数T（対角）の最大値（生成時）
        z_dist="uniform01",# 外生入力zの分布（生成時）
    ),
    
    # チューニング設定
    tuning=TuningParams(
        tuning_trials=500,          # Optuna試行回数
        tuning_runs_per_trial=1,    # 各試行の平均を取る回数
        truncation_horizon=800,     # チューニング時に使う時系列長（X[:, :T_tune]）
        tuning_seed=4,              # チューニング用seed
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
            # 提案法（PP）は負けるケースが出やすいので、探索を厚くする
            # - r,q も探索対象に含める（固定だとハマる条件が出る）
            r=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[200, 400, 600, 800, 1000],
            ),
            q=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[8, 10, 20, 40],
            ),
            # rho / lambda_S は log 探索で上下限を広げる（極端な条件も拾う）
            rho=SearchRange(low=1e-12, high=1e0, log=True, type="float", step=None, choices=None),           # レベル集合の閾値ρ（大きいほど制約が緩い）
            mu_lambda=SearchRange(low=1e-3, high=1e3, log=True, type="float", step=None, choices=None),     # 緩和係数スカラー（大きいほど更新が速い）
            lambda_S=SearchRange(low=1e-12, high=1e0, log=True, type="float", step=None, choices=None),      # SブロックへのL1強度（スパース誘導）
        ),
        pp_sgd=PPSearchSpace(
            r=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[1],
            ),
            q=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[1],
            ),
            rho=SearchRange(low=1e-12, high=1e0, log=True, type="float", step=None, choices=None),
            mu_lambda=SearchRange(low=1e-3, high=1e3, log=True, type="float", step=None, choices=None),
            lambda_S=SearchRange(low=1e-12, high=1e0, log=True, type="float", step=None, choices=None),
        ),
        pc=PCSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True, type="float", step=None, choices=None),   # L1正則化係数（PCの疎性）
            alpha=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),         # PCのステップ（S側）
            beta_pc=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),       # PCのステップ（補正側）
            gamma=SearchRange(low=0.85, high=0.999, log=False, type="float", step=None, choices=None),       # 忘却係数（共分散のEWMA）
            P=SearchRange(low=1, high=1, log=False, type="int", step=1, choices=None), # Prediction回数
            C=SearchRange(low=1, high=1, log=False, type="int", step=1, choices=None), # Correction回数
        ),
        co=COSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),
            beta_co=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),
            gamma=SearchRange(low=0.85, high=0.999, log=False, type="float", step=None, choices=None),
            C=SearchRange(low=1, high=1, log=False, type="int", step=1, choices=None),
        ),
        sgd=SGDSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),
            beta_sgd=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),
        ),
        pg=PGSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e+1, log=True, type="float", step=None, choices=None),
            step_scale=SearchRange(low=1e-6, high=1e+1, log=True, type="float", step=None, choices=None),
            use_fista=SearchRange(low=0, high=0, log=False, type="categorical", step=None, choices=[False]),
            # use_fista=SearchRange(low=0, high=0, log=False, type="categorical", choices=[True, False]),
        ),
        offline=OfflineSearchSpace(
            offline_lambda_l1=SearchRange(low=1e-4, high=1e+1, log=True, type="float", step=None, choices=None),
        ),
    ),

    # デフォルトハイパーパラメータ（ハイパラJSON未指定時のフォールバック）
    hyperparams=DefaultHyperparams(
        pp=PPHyperparams(
            r=50,            # PPの窓長（各ブロックが参照する過去長）
            q=5,             # PPの並列ブロック数（t,t-1,... の個数）
            rho=1e-3,        # PPのρ
            mu_lambda=1.0,   # PPの緩和スカラー
            lambda_S=1e-2,   # PPのL1強度（Sのみ）
        ),
        pp_sgd=PPHyperparams(
            r=1,
            q=1,
            rho=1e-3,
            mu_lambda=1.0,
            lambda_S=1e-2,
        ),
        pc=PCHyperparams(
            lambda_reg=1e-3, # PCのL1強度
            alpha=1e-2,      # PCのステップ（S側）
            beta=1e-2,       # PCのステップ（補正側）
            gamma=0.95,      # 忘却係数
            P=1,             # Prediction反復回数
            C=1,             # Correction反復回数
        ),
        co=COHyperparams(
            beta_co=1e-2,    # COのステップ（補正側のみ）
        ),
        sgd=SGDHyperparams(
            beta_sgd=1e-2,   # SGDのステップ
        ),
        pg=PGHyperparams(
            lambda_reg=1e-3,         # バッチPGのL1強度
            step_scale=1.0,          # ステップスケール
            step_size=None,          # 固定ステップ（Noneなら内部で決定）
            use_fista=True,          # FISTAを使うか
            use_backtracking=False,  # バックトラッキングを使うか
            max_iter=500,            # 最大反復
            tol=1e-4,                # 収束判定
        ),
    ),
    
    # 実行設定
    run=RunParams(
        num_trials=100,
    ),

    # スクリプト固有設定
    scripts=ScriptParams(
        tune_and_run_scenario="piecewise",
        run_piecewise_once_show_offline_line=False,
        run_piecewise_once_heatmap_time=-1,
        run_piecewise_once_output_subdir="exog_sparse_piecewise_once",
        run_linear_once_output_subdir="exog_sparse_linear_once",
        diagnose_rerun_sgd_piecewise=DiagnoseRerunParams(
            meta_json=Path("CHANGE_ME_meta.json"),
            hyperparam_json=None,
            n_jobs=-1,
            topk=10,
            exclude_top_by_max=0,
            exclude_seeds=[],
            write_updated_meta=False,
            inplace=False,
            overwrite_figure=False,
            figure_out=None,
            no_show=False,
        ),
    ),
    
    # 評価指標設定
    # "true_value": 真の値のノルムで割る（従来の方法）
    # "offline_solution": オフライン解のノルムで割る（offline_lambda_l1はOptunaで自動探索）
    metric=MetricParams(
        error_normalization="true_value",  # メタデータより
        # error_normalization="offline_solution",
        # PPは序盤の更新が弱く出やすいので、自動burn-inを推奨（r+q-2）
        burn_in=-1,                        # -1なら burn_in=r+q-2（“立ち上がり”区間を評価から除外）
        divide_by_n2=False,                # Trueなら (誤差) / N^2 をプロット・チューニングにも適用
    ),

    # 比較条件（「提案法をよく見せる」設定例）
    # - PCへ真のTを与えないことで、未知T推定という条件を揃える
    # - PPは真のT対角でウォームスタート
    comparison=ComparisonParams(
        pc_model="exog",                 # PC系のモデル（"noexog"だとZ/Tを使わない）
        pc_use_true_T_init=False,          # PC系に真のTを渡すか（noexogでは基本使われない）
        pc_T_init_identity_scale=1.0,      # pc_use_true_T_init=False のときの T_init=I*scale
        pp_init_b0="true_T_diag",          # PPのT初期値（b0）
        pp_lookahead=0,                   # PPの先読み（-1なら r+q-2。未来データを使うためオンラインではない）
    ),
    
    # 出力設定
    output=OutputParams(
        result_root=Path("./result"),
        subdir_piecewise="exog_sparse_piecewise",
        subdir_linear="exog_sparse_linear",
        subdir_tuning="exog_sparse_tuning",
        save_heatmap=False,
    ),
    
    # 実行モード
    skip_tuning=True,
    skip_simulation=False,
    # piecewise_K=4_N=20_T=1000 のハイパラ（251201）
    hyperparam_json=Path("result/251201/exog_sparse_tuning/piecewise_best_hyperparams_20251201_185134.json"),
)

# =============================================================================
# テスト用設定（プログラム動作確認用・すぐに終わる軽量設定）
# =============================================================================
CONFIG_TEST = SimulationConfig(
    # 実行する手法（テストでは1つだけ有効に）
    methods=MethodFlags(
        pp=True,
        pp_sgd=False,
        pc=False,
        co=False,
        sgd=False,
        pg=False,
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

    # Linearシナリオのパラメータ（現状パラメータなしだが明示）
    linear=LinearParams(),

    # Brownianシナリオのパラメータ
    brownian=BrownianParams(
        K=5,
        std_S=0.05,
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
            r=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[10, 20, 30, 40, 50, 75, 100, 150, 200],
            ),
            q=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[1, 2, 3, 5, 8, 10],
            ),
            rho=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            # 理論上 μ_t ∈ (0, 2 M_t) を狙うため、μ_lambda は 1 付近〜2 まで探索する
            mu_lambda=SearchRange(low=1e-2, high=2.0, log=True, type="float", step=None, choices=None),
            lambda_S=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
        ),
        pp_sgd=PPSearchSpace(
            r=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[1],
            ),
            q=SearchRange(
                low=0,
                high=0,
                log=False,
                type="categorical",
                step=None,
                choices=[1],
            ),
            rho=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            mu_lambda=SearchRange(low=1e-2, high=2.0, log=True, type="float", step=None, choices=None),
            lambda_S=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
        ),
        pc=PCSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True, type="float", step=None, choices=None),
            alpha=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            beta_pc=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            gamma=SearchRange(low=0.85, high=0.999, log=False, type="float", step=None, choices=None),
            P=SearchRange(low=0, high=2, log=False, type="int", step=1, choices=None),
            C=SearchRange(low=0, high=0, log=False, type="categorical", step=None, choices=[1, 2, 5]),
        ),
        co=COSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            beta_co=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            gamma=SearchRange(low=0.85, high=0.999, log=False, type="float", step=None, choices=None),
            C=SearchRange(low=0, high=0, log=False, type="categorical", step=None, choices=[1, 2, 5]),
        ),
        sgd=SGDSearchSpace(
            alpha=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            beta_sgd=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
        ),
        pg=PGSearchSpace(
            lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True, type="float", step=None, choices=None),
            step_scale=SearchRange(low=1e-6, high=1e-1, log=True, type="float", step=None, choices=None),
            use_fista=SearchRange(low=0, high=0, log=False, type="categorical", step=None, choices=[True, False]),
        ),
        offline=OfflineSearchSpace(
            offline_lambda_l1=SearchRange(low=1e-4, high=1.0, log=True, type="float", step=None, choices=None),
        ),
    ),

    # デフォルトハイパーパラメータ（ハイパラJSON未指定時のフォールバック）
    hyperparams=DefaultHyperparams(
        pp=PPHyperparams(
            r=50,
            q=5,
            rho=1e-3,
            mu_lambda=0.05,
            lambda_S=1e-2,
        ),
        pp_sgd=PPHyperparams(
            r=1,
            q=1,
            rho=1e-3,
            mu_lambda=0.05,
            lambda_S=1e-2,
        ),
        pc=PCHyperparams(
            lambda_reg=1e-3,
            alpha=1e-2,
            beta=1e-2,
            gamma=0.9,
            P=1,
            C=1,
        ),
        co=COHyperparams(
            beta_co=0.02,
        ),
        sgd=SGDHyperparams(
            beta_sgd=0.0269,
        ),
        pg=PGHyperparams(
            lambda_reg=1e-3,
            step_scale=1.0,
            step_size=None,
            use_fista=True,
            use_backtracking=False,
            max_iter=500,
            tol=1e-4,
        ),
    ),
    
    # テスト用の少ない試行回数
    run=RunParams(
        num_trials=2,  # モンテカルロ試行も最小限
    ),

    # スクリプト固有設定
    scripts=ScriptParams(
        tune_and_run_scenario="piecewise",
        run_piecewise_once_show_offline_line=False,
        run_piecewise_once_heatmap_time=-1,
        run_piecewise_once_output_subdir="exog_sparse_piecewise_once",
        run_linear_once_output_subdir="exog_sparse_linear_once",
        diagnose_rerun_sgd_piecewise=DiagnoseRerunParams(
            meta_json=Path("CHANGE_ME_meta.json"),
            hyperparam_json=None,
            n_jobs=-1,
            topk=10,
            exclude_top_by_max=0,
            exclude_seeds=[],
            write_updated_meta=False,
            inplace=False,
            overwrite_figure=False,
            figure_out=None,
            no_show=False,
        ),
    ),
    
    # 評価指標設定
    metric=MetricParams(
        # error_normalization="true_value",  # テストではシンプルな方法で
        error_normalization="offline_solution",
        burn_in=0,
        divide_by_n2=False,
    ),

    # 比較条件（テストではデフォルト＝exog/オンライン）
    comparison=ComparisonParams(
        pc_model="exog",
        pc_use_true_T_init=True,
        pc_T_init_identity_scale=1.0,
        pp_init_b0="ones",
        pp_lookahead=0,
    ),
    
    # 出力設定
    output=OutputParams(
        result_root=Path("./result"),
        subdir_piecewise="exog_sparse_piecewise",
        subdir_linear="exog_sparse_linear",
        subdir_tuning="exog_sparse_tuning",
        save_heatmap=True,
    ),
    
    # 実行モード
    skip_tuning=False,
    skip_simulation=False,
    hyperparam_json=None,
)

# USE_TEST_CONFIG の値に基づいて CONFIG を選択
CONFIG = CONFIG_TEST if USE_TEST_CONFIG else CONFIG_MAIN


# =============================================================================
# ヘルパー関数
# =============================================================================
def get_config() -> SimulationConfig:
    """グローバル設定を取得"""
    return CONFIG


def _to_jsonable(obj: Any) -> Any:
    """
    dataclass/Path を含む設定オブジェクトを JSON へ落とせる形に変換する。
    """
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def config_to_dict(cfg: Optional[SimulationConfig] = None) -> Dict[str, Any]:
    """
    config.py の全設定値をスナップショットとして dict 化する。
    - `*_meta.json` に埋め込む用途を想定。
    """
    cfg_obj = cfg if cfg is not None else get_config()
    return {
        "use_test_config": bool(USE_TEST_CONFIG),
        "config_name": "CONFIG_TEST" if USE_TEST_CONFIG else "CONFIG_MAIN",
        "config": _to_jsonable(cfg_obj),
    }


def get_enabled_methods() -> List[str]:
    """有効化されている手法のリストを取得"""
    cfg = get_config()
    methods = []
    if cfg.methods.pp:
        methods.append("pp")
    if getattr(cfg.methods, "pp_sgd", False):
        methods.append("pp_sgd")
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
            "r": search_range_to_dict(cfg.search_spaces.pp.r),
            "q": search_range_to_dict(cfg.search_spaces.pp.q),
            "rho": search_range_to_dict(cfg.search_spaces.pp.rho),
            "mu_lambda": search_range_to_dict(cfg.search_spaces.pp.mu_lambda),
            "lambda_S": search_range_to_dict(cfg.search_spaces.pp.lambda_S),
        },
        "pp_sgd": {
            "rho": search_range_to_dict(cfg.search_spaces.pp_sgd.rho),
            "mu_lambda": search_range_to_dict(cfg.search_spaces.pp_sgd.mu_lambda),
            "lambda_S": search_range_to_dict(cfg.search_spaces.pp_sgd.lambda_S),
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
        "pp_sgd": {
            "r": cfg.hyperparams.pp_sgd.r,
            "q": cfg.hyperparams.pp_sgd.q,
            "rho": cfg.hyperparams.pp_sgd.rho,
            "mu_lambda": cfg.hyperparams.pp_sgd.mu_lambda,
            "lambda_S": cfg.hyperparams.pp_sgd.lambda_S,
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
    print(f"  PP-SGD (q=1,r=1): {'ON' if getattr(cfg.methods, 'pp_sgd', False) else 'OFF'}")
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
