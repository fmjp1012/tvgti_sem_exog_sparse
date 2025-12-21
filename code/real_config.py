"""
実データ（real/）用の設定ファイル。

- `code/config.py` と同様に、このファイルを編集して実データ実験の設定を一元管理する。
- 実行スクリプトは `code/run_real_mismatch_recon.py`。

※ コマンドライン引数でも上書き可能だが、基本はここを編集して使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# NOTE:
# `code/config.py` と同様の `MethodFlags` をここでも定義して、
# `python code/real_config.py` で直接実行しても import エラーにならないようにする。


@dataclass
class MethodFlags:
    """実行する手法のフラグ（実データ用）"""

    pp: bool = False
    pp_sgd: bool = False
    pc: bool = False
    co: bool = False
    sgd: bool = False
    pg: bool = False


# =============================================================================
# 実データ入力設定
# =============================================================================
@dataclass
class RealDataParams:
    """実データの読み込み・前処理設定"""

    csv_path: Path = field(default_factory=lambda: Path("./real/LD2011_2014_tail10000.csv"))
    slice_mode: str = "tail"  # "tail" or "head"

    # 利用する次元
    N: int = 20
    T: int = 1000

    # 前処理
    log1p: bool = False
    standardize: bool = True


# =============================================================================
# 再構成(2C)の欠損設定
# =============================================================================
@dataclass
class MaskingParams:
    mask_ratio: float = 0.2
    mask_seed: Optional[int] = None
    recon_ridge: float = 1e-6


# =============================================================================
# 出力設定
# =============================================================================
@dataclass
class RealOutputParams:
    result_root: Path = field(default_factory=lambda: Path("./result"))
    subdir_real: str = "real_ld2011_2014"
    subdir_real_test: str = "real_ld2011_2014_test"


# =============================================================================
# 実行設定
# =============================================================================
@dataclass
class RealRunParams:
    show: bool = False


# =============================================================================
# プロット設定（実データ）
# =============================================================================
@dataclass
class RealPlotParams:
    """実データ実験でどの指標をプロットするか"""

    plot_system_mismatch: bool = True
    plot_recon_B: bool = True
    plot_recon_C: bool = True


# =============================================================================
# チューニング設定（実データ）
# =============================================================================
@dataclass
class RealTuningParams:
    """実データに対するOptunaチューニング設定（system mismatch最小化）"""

    tuning_trials: int = 50
    tuning_seed: int = 4
    # チューニング時は計算量削減のため短いTで評価（末尾or先頭は data.slice_mode に従う）
    truncation_T: int = 200
    # 評価で捨てる先頭のステップ数（オンライン法の初期過渡を無視したい場合）
    burn_in: int = 0


# =============================================================================
# メイン設定
# =============================================================================
@dataclass
class RealConfig:
    methods: MethodFlags = field(default_factory=MethodFlags)
    data: RealDataParams = field(default_factory=RealDataParams)
    masking: MaskingParams = field(default_factory=MaskingParams)
    output: RealOutputParams = field(default_factory=RealOutputParams)
    run: RealRunParams = field(default_factory=RealRunParams)
    plot: RealPlotParams = field(default_factory=RealPlotParams)
    tuning: RealTuningParams = field(default_factory=RealTuningParams)

    # 既存のハイパラJSONを使う場合（省略時は code/config.py のデフォルトハイパラを使用）
    hyperparam_json: Optional[Path] = None

    # 実行モード
    skip_tuning: bool = False
    skip_run: bool = False


# =============================================================================
# ★★★ 設定モード切り替え ★★★
# =============================================================================
# True: テスト用の軽量設定（すぐ終わる）
# False: 本番用設定
# USE_TEST_CONFIG = True
USE_TEST_CONFIG = False


# =============================================================================
# ★★★ 設定を変更するにはここを編集してください ★★★
# =============================================================================
CONFIG_MAIN = RealConfig(
    methods=MethodFlags(
        pp=True,
        pc=True,
        co=True,
        sgd=True,
        # pg=True,
    ),
    data=RealDataParams(
        csv_path=Path("./real/LD2011_2014_tail10000.csv"),
        slice_mode="tail",
        N=20,
        T=1000,
        log1p=False,
        standardize=True,
    ),
    masking=MaskingParams(
        mask_ratio=0.2,
        mask_seed=None,
        recon_ridge=1e-6,
    ),
    output=RealOutputParams(
        result_root=Path("./result"),
        subdir_real="real_ld2011_2014",
        subdir_real_test="real_ld2011_2014_test",
    ),
    run=RealRunParams(show=False),
    plot=RealPlotParams(
        plot_system_mismatch=True,
        plot_recon_B=True,
        plot_recon_C=True,
    ),
    tuning=RealTuningParams(
        tuning_trials=50,
        tuning_seed=4,
        truncation_T=200,
        burn_in=0,
    ),
    hyperparam_json=None,
    skip_tuning=False,
    skip_run=False,
)

CONFIG_TEST = RealConfig(
    methods=MethodFlags(
        pp=True,
        pc=True,
        co=True,
        sgd=True,
        # pg=False,
    ),
    data=RealDataParams(
        csv_path=Path("./real/LD2011_2014_tail10000.csv"),
        slice_mode="tail",
        N=5,
        T=200,
        log1p=False,
        standardize=True,
    ),
    masking=MaskingParams(
        mask_ratio=0.2,
        mask_seed=3,
        recon_ridge=1e-6,
    ),
    output=RealOutputParams(
        result_root=Path("./result"),
        subdir_real="real_ld2011_2014",
        subdir_real_test="real_ld2011_2014_test",
    ),
    run=RealRunParams(show=False),
    plot=RealPlotParams(
        plot_system_mismatch=True,
        plot_recon_B=True,
        plot_recon_C=True,
    ),
    tuning=RealTuningParams(
        tuning_trials=10,
        tuning_seed=4,
        truncation_T=100,
        burn_in=0,
    ),
    hyperparam_json=None,
    skip_tuning=False,
    skip_run=False,
)

CONFIG = CONFIG_TEST if USE_TEST_CONFIG else CONFIG_MAIN


def get_real_config() -> RealConfig:
    return CONFIG


def print_real_config_summary() -> None:
    cfg = get_real_config()

    print("=" * 60)
    print("実データ設定サマリー (code/real_config.py)")
    print("=" * 60)

    if USE_TEST_CONFIG:
        print("\n*** テストモード（軽量設定）で実行中 ***")
    else:
        print("\n*** 本番モードで実行中 ***")

    print("\n--- 入力 ---")
    print(f"  csv_path: {cfg.data.csv_path}")
    print(f"  slice_mode: {cfg.data.slice_mode}")

    print("\n--- 次元 ---")
    print(f"  N: {cfg.data.N}")
    print(f"  T: {cfg.data.T}")

    print("\n--- 前処理 ---")
    print(f"  log1p: {cfg.data.log1p}")
    print(f"  standardize: {cfg.data.standardize}")

    print("\n--- 欠損（recon 2C） ---")
    print(f"  mask_ratio: {cfg.masking.mask_ratio}")
    print(f"  mask_seed: {cfg.masking.mask_seed}")
    print(f"  recon_ridge: {cfg.masking.recon_ridge}")

    print("\n--- 実行する手法 ---")
    print(f"  PP:  {'ON' if cfg.methods.pp else 'OFF'}")
    print(f"  PP-SGD (q=1,r=1): {'ON' if getattr(cfg.methods, 'pp_sgd', False) else 'OFF'}")
    print(f"  PC:  {'ON' if cfg.methods.pc else 'OFF'}")
    print(f"  CO:  {'ON' if cfg.methods.co else 'OFF'}")
    print(f"  SGD: {'ON' if cfg.methods.sgd else 'OFF'}")
    print(f"  PG:  {'ON' if cfg.methods.pg else 'OFF'}")

    print("\n--- 出力 ---")
    print(f"  result_root: {cfg.output.result_root}")
    print(f"  subdir_real: {cfg.output.subdir_real}")
    print(f"  subdir_real_test: {cfg.output.subdir_real_test}")

    print("\n--- その他 ---")
    print(f"  show: {cfg.run.show}")
    print("\n--- プロット ---")
    print(f"  plot_system_mismatch: {cfg.plot.plot_system_mismatch}")
    print(f"  plot_recon_B: {cfg.plot.plot_recon_B}")
    print(f"  plot_recon_C: {cfg.plot.plot_recon_C}")
    print("\n--- チューニング ---")
    print(f"  tuning_trials: {cfg.tuning.tuning_trials}")
    print(f"  tuning_seed: {cfg.tuning.tuning_seed}")
    print(f"  truncation_T: {cfg.tuning.truncation_T}")
    print(f"  burn_in: {cfg.tuning.burn_in}")
    print(f"  skip_tuning: {cfg.skip_tuning}")
    print(f"  skip_run: {cfg.skip_run}")
    if cfg.hyperparam_json:
        print(f"  hyperparam_json: {cfg.hyperparam_json}")

    print("=" * 60)


if __name__ == "__main__":
    print_real_config_summary()
