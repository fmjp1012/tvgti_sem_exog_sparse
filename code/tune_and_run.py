from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from code.hyperparam_tuning import (
    save_best_hyperparams,
    tune_linear_all_methods,
    tune_piecewise_all_methods,
)


SCENARIO_TO_TUNER = {
    "piecewise": tune_piecewise_all_methods,
    "linear": tune_linear_all_methods,
}

SCENARIO_TO_RUN_SCRIPT = {
    "piecewise": Path("code/run_piecewise.py"),
    "linear": Path("code/run_linear.py"),
}

SCENARIO_PARAM_SPEC = {
    "piecewise": {
        "tuning": {"N", "T", "sparsity", "max_weight", "std_e", "K", "tuning_trials", "tuning_runs_per_trial", "seed"},
        "run": {"N", "T", "sparsity", "max_weight", "std_e", "K", "seed", "run_pc", "no_pc", "run_co", "no_co", "run_sgd", "no_sgd", "run_pp", "no_pp"},
    },
    "linear": {
        "tuning": {"N", "T", "sparsity", "max_weight", "std_e", "tuning_trials", "tuning_runs_per_trial", "seed"},
        "run": {"N", "T", "sparsity", "max_weight", "std_e", "seed", "run_pc", "no_pc", "run_co", "no_co", "run_sgd", "no_sgd", "run_pp", "no_pp"},
    },
}


def tune_scenario(scenario: str, tuning_kwargs: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    if scenario not in SCENARIO_TO_TUNER:
        raise ValueError(f"未知のシナリオです: {scenario}")
    tuner = SCENARIO_TO_TUNER[scenario]
    return tuner(**tuning_kwargs)


def run_simulation(
    scenario: str,
    hyperparam_path: Path,
    num_trials: int,
    run_kwargs: Dict[str, Optional[str]],
) -> None:
    if scenario not in SCENARIO_TO_RUN_SCRIPT:
        raise ValueError(f"未知のシナリオです: {scenario}")
    script_path = SCENARIO_TO_RUN_SCRIPT[scenario]
    python_exec = Path("/User/fmjp/venv/default/bin/python")
    if not python_exec.exists():
        python_exec = Path("/Users/fmjp/venv/default/bin/python")
    if not python_exec.exists():
        python_exec = Path(sys.executable)
    cmd = [str(python_exec), str(script_path), "--hyperparam_json", str(hyperparam_path), "--num_trials", str(num_trials)]
    for key, value in run_kwargs.items():
        cmd.append(f"--{key}")
        if value is not None:
            cmd.append(str(value))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ハイパラ調整とシミュレーション実行の自動化ツール")
    parser.add_argument("scenario", choices=list(SCENARIO_TO_TUNER.keys()))
    parser.add_argument("--num_trials", type=int, default=100, help="シミュレーション試行回数")
    parser.add_argument("--no_run", action="store_true", help="チューニングのみ実施して実行しない")
    parser.add_argument("--result_root", type=Path, default=Path("./result"))
    parser.add_argument("--subdir", type=str, default="exog_sparse_tuning")
    args, unknown = parser.parse_known_args()

    tuning_kwargs: Dict[str, object] = {}
    run_kwargs: Dict[str, Optional[str]] = {}
    spec = SCENARIO_PARAM_SPEC[args.scenario]
    tuning_keys = spec["tuning"]
    run_keys = spec["run"]
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise ValueError(f"認識できない引数形式です: {token}")
        key = token[2:]
        value: Optional[str] = None
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            value = unknown[i + 1]
            i += 1
        target_in_tuning = key in tuning_keys
        target_in_run = key in run_keys
        if not (target_in_tuning or target_in_run):
            raise ValueError(f"シナリオ {args.scenario} の未知パラメータです: --{key}")

        if value is None:
            if target_in_tuning:
                tuning_kwargs[key] = True
            if target_in_run:
                run_kwargs[key] = None
        else:
            parsed_value: object = value
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value
            if target_in_tuning:
                tuning_kwargs[key] = parsed_value
            if target_in_run:
                run_kwargs[key] = value
        i += 1

    best = tune_scenario(args.scenario, tuning_kwargs)
    hyperparam_path = save_best_hyperparams(best, scenario=args.scenario, result_root=args.result_root, subdir=args.subdir)
    print(f"ハイパラを {hyperparam_path} に保存しました")

    if not args.no_run:
        run_simulation(args.scenario, hyperparam_path, args.num_trials, run_kwargs)


if __name__ == "__main__":
    main()


