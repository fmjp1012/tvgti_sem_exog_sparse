"""
実験実行基盤モジュール

シナリオ共通の実験実行フレームワークを提供します。
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from code.config import SimulationConfig, get_config
from code.hyperparam_utils import (
    ResolvedHyperparams,
    hyperparams_to_dict,
    load_hyperparams_json,
    resolve_hyperparams,
)
from code.method_executor import MethodExecutor, MethodFlags, TrialResult
from utils.formatting import fmt_value, print_block
from utils.io.plotting import apply_style, plot_heatmaps
from utils.io.results import backup_script, create_result_dir, make_result_filename, save_json
from utils.offline_solver import solve_offline_sem_lasso_batch


@dataclass
class ExperimentResult:
    """実験結果"""
    error_means: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    last_estimates: Optional[Dict[str, np.ndarray]] = None
    result_dir: Optional[Path] = None
    figure_path: Optional[Path] = None


def parse_hyperparam_arg() -> Optional[Path]:
    """コマンドライン引数からハイパラJSONパスをパースする。"""
    parser = argparse.ArgumentParser(
        description="シミュレーション実行（設定は config.py で変更）"
    )
    parser.add_argument(
        "--hyperparam_json",
        type=Path,
        default=None,
        help="ハイパーパラメータJSONのパス（省略時はconfig.pyのデフォルト値を使用）",
    )
    args = parser.parse_args()
    return args.hyperparam_json


class BaseExperimentRunner(ABC):
    """
    シナリオ共通の実験実行基盤クラス。

    サブクラスで以下を実装する:
    - get_scenario_name(): シナリオ名を返す
    - generate_data(): データを生成する
    - get_scenario_params(): シナリオ固有パラメータを返す
    """

    def __init__(
        self,
        cfg: Optional[SimulationConfig] = None,
        hyperparam_path: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        cfg : SimulationConfig, optional
            設定オブジェクト（Noneの場合はget_config()を使用）
        hyperparam_path : Path, optional
            ハイパーパラメータJSONのパス
        """
        self.cfg = cfg if cfg is not None else get_config()
        self.hyperparam_path = hyperparam_path
        
        # ハイパーパラメータを解決
        loaded_hyperparams = load_hyperparams_json(hyperparam_path)
        self.hyperparams = resolve_hyperparams(loaded_hyperparams, self.cfg)
        
        # 手法フラグ
        self.flags = MethodFlags.from_config(self.cfg)
        
        # 共通パラメータ
        self.N = self.cfg.common.N
        self.T = self.cfg.common.T
        self.sparsity = self.cfg.common.sparsity
        self.max_weight = self.cfg.common.max_weight
        self.std_e = self.cfg.common.std_e
        self.seed = self.cfg.common.seed
        self.num_trials = self.cfg.run.num_trials
        
        # 評価指標設定
        self.error_normalization = self.cfg.metric.error_normalization

    @abstractmethod
    def get_scenario_name(self) -> str:
        """シナリオ名を返す。"""
        pass

    @abstractmethod
    def generate_data(
        self, rng: np.random.Generator
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        データを生成する。

        Parameters
        ----------
        rng : np.random.Generator
            乱数生成器

        Returns
        -------
        Tuple[S_series, T_mat, Z, Y]
            - S_series: 真の隣接行列の時系列
            - T_mat: 外生変数係数行列
            - Z: 外生変数 (N x T)
            - Y: 観測データ (N x T)
        """
        pass

    @abstractmethod
    def get_scenario_params(self) -> Dict[str, Any]:
        """シナリオ固有パラメータを辞書形式で返す。"""
        pass

    def get_output_subdir(self) -> str:
        """出力サブディレクトリ名を返す。"""
        return f"exog_sparse_{self.get_scenario_name()}"

    def print_summary(self) -> None:
        """実行設定のサマリーを表示する。"""
        print(f"=== Experiment Configuration ({self.get_scenario_name()}) ===")
        
        common_items: Dict[str, object] = {
            "Scenario": self.get_scenario_name(),
            "Hyperparam JSON": str(self.hyperparam_path) if self.hyperparam_path else "<default>",
            "Num Trials": self.num_trials,
            "Seed (base)": self.seed,
            "N": self.N,
            "T": self.T,
            "sparsity": self.sparsity,
            "max_weight": self.max_weight,
            "std_e": self.std_e,
        }
        print_block("Common Parameters", common_items)
        
        # シナリオ固有パラメータ
        scenario_params = self.get_scenario_params()
        if scenario_params:
            print_block("Scenario Parameters", scenario_params)
        
        # 手法フラグ
        flag_items = {name.upper(): "ON" if v else "OFF" for name, v in self.flags.to_dict().items()}
        print_block("Run Flags", flag_items)
        
        # 評価指標設定
        metric_items: Dict[str, object] = {
            "error_normalization": self.error_normalization,
        }
        if self.error_normalization == "offline_solution":
            offline_space = self.cfg.search_spaces.offline.offline_lambda_l1
            metric_items["offline_lambda_l1 (range)"] = f"[{offline_space.low}, {offline_space.high}]"
        print_block("Metric Settings", metric_items)
        
        # ハイパーパラメータ
        hp_dict = hyperparams_to_dict(self.hyperparams)
        for method_key, params in hp_dict.items():
            label = f"{method_key.upper()} Hyperparams"
            print_block(label, params)
        
        # データ生成設定
        print_block("Data Generation", {
            "s_type": self.cfg.data_gen.s_type,
            "t_min": self.cfg.data_gen.t_min,
            "t_max": self.cfg.data_gen.t_max,
            "z_dist": self.cfg.data_gen.z_dist,
        })
        print("------------------------------")

    def run_trial(self, trial_seed: int) -> TrialResult:
        """
        単一の試行を実行する。

        Parameters
        ----------
        trial_seed : int
            試行用の乱数シード

        Returns
        -------
        TrialResult
            試行結果
        """
        rng = np.random.default_rng(trial_seed)
        S_series, T_mat, Z, Y = self.generate_data(rng)
        
        # オフライン解を計算（必要な場合）
        S_offline = None
        if self.error_normalization == "offline_solution":
            S_offline = solve_offline_sem_lasso_batch(
                Y, Z, self.hyperparams.offline_lambda_l1
            )
        
        # 手法実行
        executor = MethodExecutor(
            N=self.N,
            flags=self.flags,
            hyperparams=self.hyperparams,
            error_normalization=self.error_normalization,
        )
        
        return executor.execute_all(Y, Z, S_series, T_mat, S_offline)

    def aggregate_results(
        self, results: List[TrialResult]
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        """
        複数試行の結果を集計する。

        Parameters
        ----------
        results : List[TrialResult]
            全試行の結果リスト

        Returns
        -------
        Tuple[error_means, last_estimates]
            - error_means: 各手法の平均誤差
            - last_estimates: 最後の試行の最終推定値
        """
        methods = ["pp", "pc", "co", "sgd", "pg"]
        error_totals: Dict[str, Optional[np.ndarray]] = {}
        
        # 初期化
        for method in methods:
            flag = getattr(self.flags, method)
            if flag:
                error_totals[method] = np.zeros(self.T)
            else:
                error_totals[method] = None
        
        # 集計
        last_estimates = None
        for result in results:
            for method in methods:
                if error_totals[method] is not None and method in result.errors:
                    error_totals[method] += np.array(result.errors[method])
            last_estimates = result.estimates_final
        
        # 平均を計算
        error_means: Dict[str, Optional[np.ndarray]] = {}
        for method in methods:
            if error_totals[method] is not None:
                error_means[method] = error_totals[method] / self.num_trials
            else:
                error_means[method] = None
        
        return error_means, last_estimates

    def plot_results(
        self, error_means: Dict[str, Optional[np.ndarray]], save_path: Path
    ) -> None:
        """
        結果をプロットする。

        Parameters
        ----------
        error_means : Dict[str, Optional[np.ndarray]]
            各手法の平均誤差
        save_path : Path
            保存先パス
        """
        plt.figure(figsize=(10, 6))
        
        # プロット順序と色を定義
        plot_order = [
            ("co", "blue", "Correction Only"),
            ("pc", "limegreen", "Prediction Correction"),
            ("sgd", "cyan", "SGD"),
            ("pg", "magenta", "ProxGrad"),
            ("pp", "red", "Proposed (PP)"),
        ]
        
        for method, color, label in plot_order:
            if error_means.get(method) is not None:
                plt.plot(error_means[method], color=color, label=label)
        
        plt.yscale("log")
        plt.xlim(left=0, right=self.T)
        plt.xlabel("t")
        
        if self.error_normalization == "offline_solution":
            plt.ylabel("Average Error Ratio (vs Offline)")
        else:
            plt.ylabel("Average NSE")
        
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig(str(save_path))
        plt.show()

    def save_metadata(
        self,
        result_dir: Path,
        filename: str,
        error_means: Dict[str, Optional[np.ndarray]],
        trial_seeds: List[int],
    ) -> None:
        """
        メタデータを保存する。

        Parameters
        ----------
        result_dir : Path
            結果ディレクトリ
        filename : str
            結果ファイル名
        error_means : Dict[str, Optional[np.ndarray]]
            各手法の平均誤差
        trial_seeds : List[int]
            試行シードのリスト
        """
        run_started_at = datetime.now()
        
        # スクリプトのバックアップ
        scripts_dir = result_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_copies: Dict[str, str] = {}
        
        # run_*.pyのバックアップ
        run_script_path = Path(__file__).resolve().parent / f"run_{self.get_scenario_name()}.py"
        if run_script_path.exists():
            run_script_copy = backup_script(run_script_path, scripts_dir)
            script_copies[f"run_{self.get_scenario_name()}"] = str(run_script_copy)
        
        # config.pyのバックアップ
        config_path = Path(__file__).resolve().parent / "config.py"
        if config_path.exists():
            config_copy = backup_script(config_path, scripts_dir)
            script_copies["config"] = str(config_copy)
        
        # data_gen.pyのバックアップ
        data_gen_path = Path(__file__).resolve().parent / "data_gen.py"
        if data_gen_path.exists():
            data_gen_copy = backup_script(data_gen_path, scripts_dir)
            script_copies["data_gen"] = str(data_gen_copy)
        
        # ハイパラJSONのバックアップ
        if self.hyperparam_path is not None and self.hyperparam_path.is_file():
            hyper_copy = backup_script(self.hyperparam_path, scripts_dir)
            script_copies["hyperparams_json"] = str(hyper_copy)
        
        # ハイパーパラメータを辞書形式に変換
        hp_dict = hyperparams_to_dict(self.hyperparams)
        
        # メタデータ構築
        metadata = {
            "created_at": run_started_at.isoformat(),
            "command": sys.argv,
            "scenario": self.get_scenario_name(),
            "config": {
                "num_trials": self.num_trials,
                "seed_base": self.seed,
                "trial_seeds": trial_seeds,
                "N": self.N,
                "T": self.T,
                "sparsity": self.sparsity,
                "max_weight": self.max_weight,
                "std_e": self.std_e,
                **self.get_scenario_params(),
            },
            "metric": {
                "error_normalization": self.error_normalization,
                "offline_lambda_l1": self.hyperparams.offline_lambda_l1,
            },
            "methods": {
                "pp": {
                    "enabled": self.flags.pp,
                    "hyperparams": hp_dict["pp"],
                },
                "pc": {
                    "enabled": self.flags.pc,
                    "hyperparams": hp_dict["pc"],
                },
                "co": {
                    "enabled": self.flags.co,
                    "hyperparams": hp_dict["co"],
                },
                "sgd": {
                    "enabled": self.flags.sgd,
                    "hyperparams": hp_dict["sgd"],
                },
                "pg": {
                    "enabled": self.flags.pg,
                    "hyperparams": hp_dict["pg"],
                },
            },
            "generator": {
                "function": f"code.data_gen.generate_{self.get_scenario_name()}_X_with_exog",
                "kwargs": {
                    "s_type": self.cfg.data_gen.s_type,
                    "t_min": self.cfg.data_gen.t_min,
                    "t_max": self.cfg.data_gen.t_max,
                    "z_dist": self.cfg.data_gen.z_dist,
                },
            },
            "results": {
                "figure": filename,
                "figure_path": str(result_dir / filename),
                "metrics": {
                    method: err.tolist() if err is not None else None
                    for method, err in error_means.items()
                },
            },
            "snapshots": script_copies,
            "hyperparam_json": str(self.hyperparam_path) if self.hyperparam_path else None,
            "result_dir": str(result_dir),
        }
        
        meta_name = f"{Path(filename).stem}_meta.json"
        save_json(metadata, result_dir, name=meta_name)

    def run(self) -> ExperimentResult:
        """
        実験を実行する。

        Returns
        -------
        ExperimentResult
            実験結果
        """
        # プロットスタイル設定
        apply_style(use_latex=True, font_family="Times New Roman", base_font_size=15)
        
        # サマリー表示
        self.print_summary()
        
        # 試行シード生成
        trial_seeds = [self.seed + i for i in range(self.num_trials)]
        
        # 並列実行
        with tqdm_joblib(tqdm(desc="Progress", total=self.num_trials)):
            results = Parallel(n_jobs=-1, batch_size=1, prefer="threads")(
                delayed(self.run_trial)(ts) for ts in trial_seeds
            )
        
        # 結果集計
        error_means, last_estimates = self.aggregate_results(results)
        
        # 結果ディレクトリ作成
        result_dir = create_result_dir(
            self.cfg.output.result_root,
            self.get_output_subdir(),
            extra_tag="images",
        )
        
        # ファイル名生成
        filename_params = {
            "N": self.N,
            "T": self.T,
            "num_trials": self.num_trials,
            "maxweight": self.max_weight,
            "stde": self.std_e,
            "seed": self.seed,
            "r": self.hyperparams.pp.r,
            "q": self.hyperparams.pp.q,
            "rho": self.hyperparams.pp.rho,
            "mulambda": self.hyperparams.pp.mu_lambda,
            "lambdaS": self.hyperparams.pp.lambda_S,
            **self.get_scenario_params(),
        }
        filename = make_result_filename(
            prefix=self.get_scenario_name(),
            params=filename_params,
            suffix=".png",
        )
        print(filename)
        
        # プロット保存
        figure_path = Path(result_dir) / filename
        self.plot_results(error_means, figure_path)
        
        # ヒートマップ表示
        if last_estimates is not None:
            heatmap_filename = filename.replace(".png", "_heatmap.png")
            plot_heatmaps(
                matrices=last_estimates,
                save_path=Path(result_dir) / heatmap_filename,
                title=f"Estimated vs True at t={self.T-1} (last trial)",
                show=True,
            )
        
        # メタデータ保存
        self.save_metadata(result_dir, filename, error_means, trial_seeds)
        
        return ExperimentResult(
            error_means=error_means,
            last_estimates=last_estimates,
            result_dir=Path(result_dir),
            figure_path=figure_path,
        )

