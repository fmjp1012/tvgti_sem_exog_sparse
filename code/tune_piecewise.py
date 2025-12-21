"""
Piecewise シナリオ: ハイパーパラメータ **チューニングのみ** 実行する。

- 設定は code/config.py から読み込む（CLIフラグによる上書きはしない）
- 有効な手法は config.py の MethodFlags に従う
- 結果は config.py の output.subdir_tuning 配下に保存する

使い方:
    python -m code.tune_piecewise
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from code.config import get_config, get_enabled_methods
from code.hyperparam_tuning import save_best_hyperparams, tune_piecewise_all_methods


def main() -> None:
    cfg = get_config()
    enabled_methods = get_enabled_methods()
    if not enabled_methods:
        raise ValueError("実行する手法がありません。code/config.py の MethodFlags を確認してください。")

    print(f"[tune_piecewise] enabled_methods={enabled_methods}")

    best, tuning_summary = tune_piecewise_all_methods(methods=enabled_methods)

    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "scenario": "piecewise",
        "config": {
            "common": {
                "N": cfg.common.N,
                "T": cfg.common.T,
                "sparsity": cfg.common.sparsity,
                "max_weight": cfg.common.max_weight,
                "std_e": cfg.common.std_e,
                "seed": cfg.common.seed,
            },
            "piecewise": {"K": cfg.piecewise.K},
            "tuning": {
                "tuning_trials": cfg.tuning.tuning_trials,
                "tuning_runs_per_trial": cfg.tuning.tuning_runs_per_trial,
                "truncation_horizon": cfg.tuning.truncation_horizon,
                "tuning_seed": cfg.tuning.tuning_seed,
            },
            "enabled_methods": enabled_methods,
        },
        "tuning_summary": tuning_summary,
    }

    script_paths = {
        "tune_piecewise": Path(__file__),
        "hyperparam_tuning": Path(__file__).resolve().parent / "hyperparam_tuning.py",
        "data_gen": Path(__file__).resolve().parent / "data_gen.py",
        "config": Path(__file__).resolve().parent / "config.py",
    }

    out_path = save_best_hyperparams(
        best,
        scenario="piecewise",
        result_root=cfg.output.result_root,
        subdir=cfg.output.subdir_tuning,
        metadata=metadata,
        script_paths=script_paths,
    )
    print(f"[tune_piecewise] saved: {out_path}")


if __name__ == "__main__":
    main()

