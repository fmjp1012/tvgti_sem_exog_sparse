from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from turtle import Turtle
from typing import Any, Dict, Iterable, Optional, Tuple
import datetime as dt

from code.hyperparam_tuning import (
    SEARCH_OVERRIDE_SPECS,
    DEFAULT_SEARCH_SPACES,
    SUPPORTED_METHODS,
    save_best_hyperparams,
    tune_linear_all_methods,
    tune_piecewise_all_methods,
)


def _dict_to_cli_tokens(options: Dict[str, object]) -> list[str]:
    tokens: list[str] = []
    for key, value in options.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
            continue
        if isinstance(value, (list, tuple, set)):
            serialized = ",".join(str(item) for item in value)
        else:
            serialized = str(value)
        tokens.extend([flag, serialized])
    return tokens


def _gather_config_tokens(scenario: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(_dict_to_cli_tokens(USER_ARG_OVERRIDES.get("tuning", {})))
    tokens.extend(_dict_to_cli_tokens(USER_ARG_OVERRIDES.get("run", {})))
    scenario_cfg = USER_ARG_OVERRIDES.get("scenarios", {}).get(scenario, {})
    if scenario_cfg:
        tokens.extend(_dict_to_cli_tokens(scenario_cfg.get("tuning", {})))
        tokens.extend(_dict_to_cli_tokens(scenario_cfg.get("run", {})))
    return tokens


def _iter_main_override_sources(scenario: str) -> Iterable[Dict[str, object]]:
    yield USER_ARG_OVERRIDES.get("main", {})
    scenario_cfg = USER_ARG_OVERRIDES.get("scenarios", {}).get(scenario, {})
    if scenario_cfg:
        yield scenario_cfg.get("main", {})


def _apply_main_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    scenario: str,
) -> None:
    for overrides in _iter_main_override_sources(scenario):
        if not overrides:
            continue
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "no_run":
                if value:
                    args.no_run = True
                continue
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            if current_value == default_value:
                if key == "result_root":
                    setattr(args, key, Path(value))
                elif key == "num_trials":
                    setattr(args, key, int(value))
                else:
                    setattr(args, key, value)


SCENARIO_TO_TUNER = {
    "piecewise": tune_piecewise_all_methods,
    "linear": tune_linear_all_methods,
}

SCENARIO_TO_RUN_MODULE = {
    "piecewise": "code.run_piecewise",
    "linear": "code.run_linear",
}

SCENARIO_PARAM_SPEC = {
    "piecewise": {
        "tuning": set([
            "N",
            "T",
            "sparsity",
            "max_weight",
            "std_e",
            "K",
            "tuning_trials",
            "tuning_runs_per_trial",
            "seed",
            "methods",
        ]),
        "run": set([
            "N",
            "T",
            "sparsity",
            "max_weight",
            "std_e",
            "K",
            "seed",
            "run_pc",
            "no_pc",
            "run_co",
            "no_co",
            "run_sgd",
            "no_sgd",
            "run_pg",
            "no_pg",
            "run_pp",
            "no_pp",
            "hyperparam_json",
        ]),
    },
    "linear": {
        "tuning": set([
            "N",
            "T",
            "sparsity",
            "max_weight",
            "std_e",
            "tuning_trials",
            "tuning_runs_per_trial",
            "seed",
            "methods",
        ]),
        "run": set([
            "N",
            "T",
            "sparsity",
            "max_weight",
            "std_e",
            "seed",
            "run_pc",
            "no_pc",
            "run_co",
            "no_co",
            "run_sgd",
            "no_sgd",
            "run_pg",
            "no_pg",
            "run_pp",
            "no_pp",
            "hyperparam_json",
        ]),
    },
}

OVERRIDE_KEYS = set(SEARCH_OVERRIDE_SPECS.keys())
SCENARIO_PARAM_SPEC["piecewise"]["tuning"].update(OVERRIDE_KEYS)
SCENARIO_PARAM_SPEC["linear"]["tuning"].update(OVERRIDE_KEYS)

SCENARIO_TUNING_DEFAULTS: Dict[str, Dict[str, object]] = {
    "piecewise": {
        "N": 20,
        "T": 1000,
        "sparsity": 0.7,
        "max_weight": 0.5,
        "std_e": 0.05,
        "K": 4,
        "tuning_trials": 300,
        "tuning_runs_per_trial": 1,
        "seed": 4,
        "methods": ",".join(SUPPORTED_METHODS),
    },
    "linear": {
        "N": 20,
        "T": 1000,
        "sparsity": 0.6,
        "max_weight": 0.5,
        "std_e": 0.05,
        "tuning_trials": 30,
        "tuning_runs_per_trial": 5,
        "seed": 3,
        "methods": ",".join(SUPPORTED_METHODS),
    },
}

SCENARIO_RUN_DEFAULTS: Dict[str, Dict[str, object]] = {
    "piecewise": {
        "K": 4,
        "N": 20,
        "T": 1000,
        "sparsity": 0.7,
        "max_weight": 0.5,
        "std_e": 0.05,
        "seed": 3,
    },
    "linear": {
        "N": 20,
        "T": 1000,
        "sparsity": 0.6,
        "max_weight": 0.5,
        "std_e": 0.05,
        "seed": 3,
    },
}

METHOD_OBJECTIVE_SUMMARY: Dict[str, str] = {
    "pp": "Mean Frobenius error at final timestep between PP estimate and ground-truth adjacency over tuning runs.",
    "pc": "Mean Frobenius error across truncated horizon between PC estimates and ground-truth adjacency.",
    "co": "Mean Frobenius error across truncated horizon for correction-only model.",
    "sgd": "Mean Frobenius error across truncated horizon for SGD baseline.",
    "pg": "Mean Frobenius error across truncated horizon for batch proximal-gradient estimator.",
}


USER_ARG_OVERRIDES: Dict[str, Dict[str, Dict[str, object]]] = {
    # すべて None にしておくと CLI 既定値がそのまま利用されます。上書きしたい値だけ編集してください。
    "main": {
        "num_trials": None,
        "no_run": None,
        "result_root": None,
        "subdir": None,
    },
    "tuning": {
        # シナリオ共通のチューニング引数
        "N": None,
        "T": None,
        "sparsity": None,
        "max_weight": None,
        "std_e": None,
        "tuning_trials": None,
        "tuning_runs_per_trial": None,
        "seed": None,
        "methods": "pp,pc,co,sgd",
        # 探索範囲の上書き (SEARCH_OVERRIDE_SPECS)
        "pp_rho_low": None,
        "pp_rho_high": None,
        "pp_rho_log": None,
        "pp_mu_lambda_low": None,
        "pp_mu_lambda_high": None,
        "pp_mu_lambda_log": None,
        "pc_lambda_reg_low": None,
        "pc_lambda_reg_high": None,
        "pc_lambda_reg_log": None,
        "pc_alpha_low": None,
        "pc_alpha_high": None,
        "pc_alpha_log": None,
        "pc_beta_pc_low": None,
        "pc_beta_pc_high": None,
        "pc_beta_pc_log": None,
        "pc_gamma_low": None,
        "pc_gamma_high": None,
        "pc_gamma_log": None,
        "pc_P_min": None,
        "pc_P_max": None,
        "pc_P_step": None,
        "pc_C_choices": None,
        "co_alpha_low": None,
        "co_alpha_high": None,
        "co_alpha_log": None,
        "co_beta_co_low": None,
        "co_beta_co_high": None,
        "co_beta_co_log": None,
        "co_gamma_low": None,
        "co_gamma_high": None,
        "co_gamma_log": None,
        "co_C_choices": None,
        "sgd_alpha_low": None,
        "sgd_alpha_high": None,
        "sgd_alpha_log": None,
        "sgd_beta_sgd_low": None,
        "sgd_beta_sgd_high": None,
        "sgd_beta_sgd_log": None,
        "pg_lambda_reg_low": None,
        "pg_lambda_reg_high": None,
        "pg_lambda_reg_log": None,
        "pg_step_scale_low": None,
        "pg_step_scale_high": None,
        "pg_step_scale_log": None,
        "pg_use_fista_choices": None,
    },
    "run": {
        # シナリオ共通の実行フェーズ引数
        "N": None,
        "T": None,
        "sparsity": None,
        "max_weight": None,
        "std_e": None,
        "seed": None,
        "run_pc": None,
        "no_pc": None,
        "run_co": None,
        "no_co": None,
        "run_sgd": None,
        "no_sgd": None,
        "run_pg": None,
        "no_pg": True,
        "run_pp": None,
        "no_pp": None,
        "hyperparam_json": None,
    },
    "scenarios": {
        # シナリオ固有の追加項目
        "piecewise": {
            "main": {
                "num_trials": None,
                "no_run": None,
                "result_root": None,
                "subdir": None,
            },
            "tuning": {
                "K": None,
            },
            "run": {
                "K": None,
            },
        },
        "linear": {
            "main": {
                "num_trials": None,
                "no_run": None,
                "result_root": None,
                "subdir": None,
            },
            "tuning": {},
            "run": {},
        },
    },
}


def _format_value(value: object) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return str(value)


def _resolve_methods(methods_value: object) -> Tuple[str, ...]:
    if methods_value is None or methods_value == "<default>" or methods_value == "":
        return SUPPORTED_METHODS
    if isinstance(methods_value, str):
        raw = methods_value.split(",")
    elif isinstance(methods_value, (list, tuple, set)):
        raw = list(methods_value)
    else:
        raw = [methods_value]
    normalized: list[str] = []
    for item in raw:
        if item is None:
            continue
        name = str(item).strip().lower()
        if not name:
            continue
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        return SUPPORTED_METHODS
    return tuple(name for name in normalized if name in SUPPORTED_METHODS)


def _build_effective_search_spaces(
    search_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    spaces = copy.deepcopy(DEFAULT_SEARCH_SPACES)
    for method, params in (search_overrides or {}).items():
        method_space = spaces.setdefault(method, {})
        for param, fields in params.items():
            param_space = method_space.setdefault(param, {})
            param_space.update(fields)
    return spaces


def _describe_param_space(spec: Dict[str, Any]) -> str:
    if not spec:
        return "<unspecified>"
    type_name = spec.get("type", "?")
    if "choices" in spec:
        return f"{type_name} choices={spec['choices']}"
    parts = []
    low = spec.get("low")
    high = spec.get("high")
    if low is not None or high is not None:
        parts.append(f"range=[{low}, {high}]")
    if spec.get("log"):
        parts.append("log")
    if "step" in spec and spec["step"] not in (None, ""):
        parts.append(f"step={spec['step']}")
    extras = {k: v for k, v in spec.items() if k not in {"type", "low", "high", "log", "step"}}
    if extras:
        parts.append(", ".join(f"{k}={v}" for k, v in extras.items()))
    joined = " ".join(parts) if parts else ""
    return f"{type_name} {joined}".strip()


def _print_section(title: str, pairs: Tuple[Tuple[str, object], ...]) -> None:
    if not pairs:
        return
    print(f"--- {title} ---")
    for label, value in pairs:
        print(f"{label:>24}: {_format_value(value)}")


def _summarize_run_flags(run_kwargs: Dict[str, Optional[str]]) -> Dict[str, str]:
    defaults = {"pc": "ON", "co": "ON", "sgd": "ON", "pg": "ON", "pp": "ON"}
    flag_state = defaults.copy()
    for method in list(flag_state.keys()):
        if f"no_{method}" in run_kwargs:
            flag_state[method] = "OFF (--no)"
        if f"run_{method}" in run_kwargs:
            flag_state[method] = "ON (--run)"
    return flag_state


def print_experiment_plan(
    scenario: str,
    num_trials: int,
    no_run: bool,
    result_root: Path,
    subdir: str,
    tuning_kwargs: Dict[str, object],
    search_overrides: Dict[str, Dict[str, Dict[str, Any]]],
    run_kwargs: Dict[str, Optional[str]],
) -> None:
    print("=== Experiment Configuration ===")
    overview_pairs: Tuple[Tuple[str, object], ...] = (
        ("Scenario", scenario),
        ("Runs Simulation", "No (tuning only)" if no_run else "Yes"),
        ("Num Trials (run)", num_trials),
        ("Result Root", result_root),
        ("Result Subdir", subdir),
    )
    _print_section("Overview", overview_pairs)

    common_tune_keys = ["N", "T", "sparsity", "max_weight", "std_e", "seed", "tuning_trials", "tuning_runs_per_trial", "methods"]
    scenario_tuning_defaults = SCENARIO_TUNING_DEFAULTS.get(scenario, {})
    if scenario in ("piecewise", "brownian"):
        common_tune_keys.insert(5, "K")
    tuning_pairs: Tuple[Tuple[str, object], ...] = tuple(
        (
            f"--{key}",
            tuning_kwargs.get(
                key,
        scenario_tuning_defaults.get(key, "<default>"),
            ),
        )
        for key in common_tune_keys
    )
    _print_section("Tuning Parameters", tuning_pairs)

    if search_overrides:
        override_pairs: list[Tuple[str, object]] = []
        for method in sorted(search_overrides):
            for param in sorted(search_overrides[method]):
                for field in sorted(search_overrides[method][param]):
                    value = search_overrides[method][param][field]
                    override_pairs.append((f"{method}.{param}.{field}", value))
        _print_section("Search Overrides", tuple(override_pairs))

    effective_spaces = _build_effective_search_spaces(search_overrides or {})
    methods = _resolve_methods(tuning_kwargs.get("methods"))
    search_pairs: list[Tuple[str, object]] = []
    for method in methods:
        for param, spec in sorted(effective_spaces.get(method, {}).items()):
            search_pairs.append((f"{method.upper()}.{param}", _describe_param_space(spec)))
    _print_section("Tuning Search Space", tuple(search_pairs))

    objective_pairs: list[Tuple[str, object]] = []
    for method in methods:
        desc = METHOD_OBJECTIVE_SUMMARY.get(method, "See hyperparam_tuning objective function.")
        objective_pairs.append((method.upper(), desc))
    _print_section("Tuning Objective", tuple(objective_pairs))

    if not no_run:
        common_run_keys = ["N", "T", "sparsity", "max_weight", "std_e", "seed"]
        if scenario in ("piecewise", "brownian"):
            common_run_keys.insert(0, "K")
        scenario_run_defaults = SCENARIO_RUN_DEFAULTS.get(scenario, {})
        run_pairs: Tuple[Tuple[str, object], ...] = tuple(
            (
                f"--{key}",
                run_kwargs.get(
                    key,
                    scenario_run_defaults.get(key, "<default>"),
                ),
            )
            for key in common_run_keys
        )
        _print_section("Run Parameters", run_pairs)
        flag_state = _summarize_run_flags(run_kwargs)
        flag_pairs: Tuple[Tuple[str, object], ...] = tuple(
            (method.upper(), state) for method, state in flag_state.items()
        )
        _print_section("Run Flags", flag_pairs)
        hyperparam_source = run_kwargs.get("hyperparam_json")
        hyperparam_label = hyperparam_source if hyperparam_source else "auto (best from tuning)"
        _print_section("Hyperparam JSON", (("Source", hyperparam_label),))


def partition_tuning_kwargs(
    tuning_kwargs: Dict[str, object]
) -> Tuple[Dict[str, object], Dict[str, Dict[str, Dict[str, Any]]]]:
    base_kwargs: Dict[str, object] = {}
    overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in tuning_kwargs.items():
        if key in SEARCH_OVERRIDE_SPECS:
            method, param, field, converter = SEARCH_OVERRIDE_SPECS[key]
            overrides.setdefault(method, {}).setdefault(param, {})[field] = converter(value)
        else:
            base_kwargs[key] = value
    return base_kwargs, overrides


def tune_scenario(
    scenario: str,
    tuning_kwargs: Dict[str, object],
    search_overrides: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    if scenario not in SCENARIO_TO_TUNER:
        raise ValueError(f"未知のシナリオです: {scenario}")
    tuner = SCENARIO_TO_TUNER[scenario]
    overrides_arg = search_overrides or None
    return tuner(search_space_overrides=overrides_arg, **tuning_kwargs)


def run_simulation(
    scenario: str,
    hyperparam_path: Path,
    num_trials: int,
    run_kwargs: Dict[str, Optional[str]],
) -> None:
    if scenario not in SCENARIO_TO_RUN_MODULE:
        raise ValueError(f"未知のシナリオです: {scenario}")
    python_exec = Path("/User/fmjp/venv/default/bin/python")
    if not python_exec.exists():
        python_exec = Path("/Users/fmjp/venv/default/bin/python")
    if not python_exec.exists():
        python_exec = Path(sys.executable)
    module_name = SCENARIO_TO_RUN_MODULE[scenario]
    run_kwargs_for_cmd = dict(run_kwargs)
    hyperparam_override = run_kwargs_for_cmd.pop("hyperparam_json", None)
    hyperparam_source = Path(hyperparam_override).expanduser() if hyperparam_override else hyperparam_path
    cmd = [str(python_exec), "-m", module_name, "--hyperparam_json", str(hyperparam_source), "--num_trials", str(num_trials)]
    for key, value in run_kwargs_for_cmd.items():
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

    _apply_main_overrides(parser, args, args.scenario)
    preconfigured_tokens = _gather_config_tokens(args.scenario)
    if preconfigured_tokens:
        unknown = preconfigured_tokens + unknown

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
                if key == "methods":
                    raise ValueError("--methods には値が必要です (例: --methods pp,pc)")
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
                if key == "methods":
                    prev = tuning_kwargs.get(key)
                    prev_str: Optional[str]
                    if prev in (None, ""):
                        prev_str = None
                    else:
                        prev_str = str(prev)
                    if prev_str:
                        tuning_kwargs[key] = f"{prev_str},{parsed_value}"
                    else:
                        tuning_kwargs[key] = str(parsed_value)
                else:
                    tuning_kwargs[key] = parsed_value
            if target_in_run:
                run_kwargs[key] = value
        i += 1

    base_tuning_kwargs, search_overrides = partition_tuning_kwargs(tuning_kwargs)
    print_experiment_plan(
        scenario=args.scenario,
        num_trials=args.num_trials,
        no_run=args.no_run,
        result_root=args.result_root,
        subdir=args.subdir,
        tuning_kwargs=base_tuning_kwargs,
        search_overrides=search_overrides,
        run_kwargs=run_kwargs,
    )
    best, tuning_summary = tune_scenario(args.scenario, base_tuning_kwargs, search_overrides)
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "command": sys.argv,
        "scenario": args.scenario,
        "tuning_kwargs": base_tuning_kwargs,
        "search_overrides": search_overrides,
        "result_root": str(args.result_root),
        "subdir": args.subdir,
        "tuning_summary": tuning_summary,
    }
    script_paths = {
        "hyperparam_tuning": Path(__file__).resolve().parent / "hyperparam_tuning.py",
        "tune_and_run": Path(__file__),
        "data_gen": Path(__file__).resolve().parent / "data_gen.py",
    }
    hyperparam_path = save_best_hyperparams(
        best,
        scenario=args.scenario,
        result_root=args.result_root,
        subdir=args.subdir,
        metadata=metadata,
        script_paths=script_paths,
    )
    print(f"ハイパラを {hyperparam_path} に保存しました")

    if not args.no_run:
        run_simulation(args.scenario, hyperparam_path, args.num_trials, run_kwargs)


if __name__ == "__main__":
    main()
