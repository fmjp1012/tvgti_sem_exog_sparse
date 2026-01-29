"""
ハイパーパラメータチューニングとシミュレーション実行の統合スクリプト

設定は code/config.py で一元管理されています。
このスクリプトを実行する前に config.py を編集して設定を変更してください。

使用方法:
    python -m code.tune_and_run
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import datetime as dt

from code.config import get_config, get_enabled_methods, SimulationConfig
from code.hyperparam_tuning import (
    save_best_hyperparams,
    tune_brownian_all_methods,
    tune_linear_all_methods,
    tune_piecewise_all_methods,
    SUPPORTED_METHODS,
)
from code.run_linear import LinearRunner
from code.run_brownian import BrownianRunner
from code.run_piecewise import main as run_piecewise_main


SCENARIO_TO_TUNER = {
    "piecewise": tune_piecewise_all_methods,
    "linear": tune_linear_all_methods,
    "brownian": tune_brownian_all_methods,
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


def _print_section(title: str, pairs: Tuple[Tuple[str, object], ...]) -> None:
    if not pairs:
        return
    print(f"--- {title} ---")
    for label, value in pairs:
        print(f"{label:>24}: {_format_value(value)}")


def print_experiment_plan(scenario: str, cfg: SimulationConfig) -> None:
    """実験計画を表示"""
    print("=" * 60)
    print("実験設定 (config.py から読み込み)")
    print("=" * 60)
    
    overview_pairs: Tuple[Tuple[str, object], ...] = (
        ("Scenario", scenario),
        ("Skip Tuning", cfg.skip_tuning),
        ("Skip Simulation", cfg.skip_simulation),
        ("Num Trials (run)", cfg.run.num_trials),
        ("Result Root", cfg.output.result_root),
    )
    _print_section("Overview", overview_pairs)
    
    common_pairs: Tuple[Tuple[str, object], ...] = (
        ("N", cfg.common.N),
        ("T", cfg.common.T),
        ("sparsity", cfg.common.sparsity),
        ("max_weight", cfg.common.max_weight),
        ("std_e", cfg.common.std_e),
        ("seed", cfg.common.seed),
    )
    _print_section("Common Parameters", common_pairs)
    
    if scenario == "piecewise":
        scenario_pairs: Tuple[Tuple[str, object], ...] = (
            ("K", cfg.piecewise.K),
        )
        _print_section("Piecewise Parameters", scenario_pairs)
    elif scenario == "brownian":
        scenario_pairs = (
            ("K", cfg.brownian.K),
            ("std_S", cfg.brownian.std_S),
        )
        _print_section("Brownian Parameters", scenario_pairs)
    
    tuning_pairs: Tuple[Tuple[str, object], ...] = (
        ("tuning_trials", cfg.tuning.tuning_trials),
        ("tuning_runs_per_trial", cfg.tuning.tuning_runs_per_trial),
        ("truncation_horizon", cfg.tuning.truncation_horizon),
        ("tuning_seed", cfg.tuning.tuning_seed),
        ("objective", getattr(cfg.tuning, "objective", "mean")),
        ("drift_penalty_weight", getattr(cfg.tuning, "drift_penalty_weight", 0.0)),
        ("drift_window_frac", getattr(cfg.tuning, "drift_window_frac", 0.2)),
    )
    _print_section("Tuning Settings", tuning_pairs)
    
    enabled_methods = get_enabled_methods()
    method_pairs: Tuple[Tuple[str, object], ...] = tuple(
        (method.upper(), "ON" if method in enabled_methods else "OFF")
        for method in SUPPORTED_METHODS
    )
    _print_section("Enabled Methods", method_pairs)
    
    if cfg.hyperparam_json:
        _print_section("Hyperparam JSON", (("Path", str(cfg.hyperparam_json)),))
    
    print("=" * 60)


def tune_scenario(scenario: str, cfg: SimulationConfig) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """シナリオに対するハイパーパラメータチューニングを実行"""
    if scenario not in SCENARIO_TO_TUNER:
        raise ValueError(f"未知のシナリオです: {scenario}")
    
    tuner = SCENARIO_TO_TUNER[scenario]
    enabled_methods = get_enabled_methods()
    
    print(f"\n[tune_scenario] config.py で有効な手法: {enabled_methods}")
    
    if not enabled_methods:
        raise ValueError("実行する手法がありません。config.py の MethodFlags を確認してください。")
    
    return tuner(methods=enabled_methods)


def run_simulation(scenario: str, hyperparam_path: Path, cfg: SimulationConfig) -> None:
    """シミュレーションを実行（CLI引数は使用しない）"""
    if scenario == "piecewise":
        run_piecewise_main(hyperparam_path=hyperparam_path)
        return
    if scenario == "linear":
        runner = LinearRunner(hyperparam_path=hyperparam_path)
        runner.run()
        return
    if scenario == "brownian":
        runner = BrownianRunner(hyperparam_path=hyperparam_path)
        runner.run()
        return
    raise ValueError(f"未知のシナリオです: {scenario}")


def main() -> None:
    """メイン処理"""
    run_wall_start = time.perf_counter()
    if len(sys.argv) > 1:
        raise SystemExit("CLI 引数は使用できません。code/config.py を編集してください。")
    
    cfg = get_config()
    scenario = str(cfg.scripts.tune_and_run_scenario).lower()
    if scenario not in SCENARIO_TO_TUNER:
        raise ValueError(f"未知のシナリオです: {scenario}")
    
    # 設定サマリーを表示
    print_experiment_plan(scenario, cfg)
    
    # ハイパーパラメータのパスを決定（config.py が唯一の出所）
    hyperparam_path: Optional[Path] = None
    
    if cfg.hyperparam_json:
        # 既存のハイパラJSONを使用
        hyperparam_path = cfg.hyperparam_json
        print(f"\n既存のハイパラJSONを使用: {hyperparam_path}")
        if not hyperparam_path.is_file():
            print(f"エラー: ハイパラJSONが見つかりません: {hyperparam_path}")
            sys.exit(1)
    
    elif not cfg.skip_tuning:
        # チューニングを実行
        print("\nハイパーパラメータチューニングを開始...")
        best, tuning_summary = tune_scenario(scenario, cfg)
        
        # メタデータを作成
        metadata = {
            "created_at": dt.datetime.now().isoformat(),
            "command": sys.argv,
            "scenario": scenario,
            "timings": {
                "overall_sec": float(time.perf_counter() - run_wall_start),
            },
            "config": {
                "common": {
                    "N": cfg.common.N,
                    "T": cfg.common.T,
                    "sparsity": cfg.common.sparsity,
                    "max_weight": cfg.common.max_weight,
                    "std_e": cfg.common.std_e,
                    "seed": cfg.common.seed,
                },
                "tuning": {
                    "tuning_trials": cfg.tuning.tuning_trials,
                    "tuning_runs_per_trial": cfg.tuning.tuning_runs_per_trial,
                    "truncation_horizon": cfg.tuning.truncation_horizon,
                },
                "enabled_methods": get_enabled_methods(),
            },
            "tuning_summary": tuning_summary,
        }
        
        if scenario == "piecewise":
            metadata["config"]["piecewise"] = {"K": cfg.piecewise.K}
        elif scenario == "brownian":
            metadata["config"]["brownian"] = {"K": cfg.brownian.K, "std_S": cfg.brownian.std_S}
        
        script_paths = {
            "hyperparam_tuning": Path(__file__).resolve().parent / "hyperparam_tuning.py",
            "tune_and_run": Path(__file__),
            "data_gen": Path(__file__).resolve().parent / "data_gen.py",
            "config": Path(__file__).resolve().parent / "config.py",
        }
        
        hyperparam_path = save_best_hyperparams(
            best,
            scenario=scenario,
            result_root=cfg.output.result_root,
            subdir=cfg.output.subdir_tuning,
            metadata=metadata,
            script_paths=script_paths,
        )
        print(f"ハイパラを {hyperparam_path} に保存しました")
    
    # シミュレーション実行
    if not cfg.skip_simulation:
        if hyperparam_path is None:
            print("エラー: ハイパラJSONがありません。")
            print("  - config.py の hyperparam_json にパスを指定するか")
            print("  - skip_tuning を False にしてチューニングを実行してください")
            sys.exit(1)
        
        print(f"\nシミュレーションを開始 (num_trials={cfg.run.num_trials})...")
        run_simulation(scenario, hyperparam_path, cfg)
    else:
        print("\nシミュレーションはスキップされました (skip_simulation=True)")


if __name__ == "__main__":
    main()
