from __future__ import annotations

import json
import time
from pathlib import Path
import sys

from code.config import get_config, get_enabled_methods
from code.hyperparam_tuning import (
    save_best_hyperparams,
    tune_piecewise_all_methods,
)


def main() -> None:
    run_wall_start = time.perf_counter()
    if len(sys.argv) > 1:
        raise SystemExit("CLI 引数は使用できません。code/config.py を編集してください。")

    cfg = get_config()
    methods = get_enabled_methods()

    best, summary = tune_piecewise_all_methods(
        N=cfg.common.N,
        T=cfg.common.T,
        sparsity=cfg.common.sparsity,
        max_weight=cfg.common.max_weight,
        std_e=cfg.common.std_e,
        K=cfg.piecewise.K,
        tuning_trials=cfg.tuning.tuning_trials,
        tuning_runs_per_trial=cfg.tuning.tuning_runs_per_trial,
        seed=cfg.common.seed,
        methods=methods,
    )

    print("推定されたハイパラ:")
    print(json.dumps(best, indent=2))

    metadata = {
        "scenario": "piecewise",
        "timings": {
            "overall_sec": float(time.perf_counter() - run_wall_start),
        },
        "arguments": {
            "N": cfg.common.N,
            "T": cfg.common.T,
            "sparsity": cfg.common.sparsity,
            "max_weight": cfg.common.max_weight,
            "std_e": cfg.common.std_e,
            "K": cfg.piecewise.K,
            "tuning_trials": cfg.tuning.tuning_trials,
            "tuning_runs_per_trial": cfg.tuning.tuning_runs_per_trial,
            "seed": cfg.common.seed,
            "methods": methods,
        },
        "tuning_summary": summary,
    }
    script_paths = {
        "hyperparam_tuning": Path(__file__).resolve().parent / "hyperparam_tuning.py",
        "hypara_tuning_piecewise": Path(__file__),
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
    print(f"ハイパラを {out_path} に保存しました")


if __name__ == '__main__':
    main()
