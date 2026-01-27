"""
実験実行基盤モジュール

シナリオ共通の実験実行フレームワークを提供します。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
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

from code.config import SimulationConfig, get_config, config_to_dict
from code.hyperparam_utils import (
    ResolvedHyperparams,
    hyperparams_to_dict,
    load_hyperparams_json,
    resolve_hyperparams,
)
from code.method_executor import MethodExecutor, MethodFlags, TrialResult
from utils.formatting import fmt_value, print_block
from utils.io.plotting import apply_style, plot_heatmaps
from utils.io.results import backup_script, create_result_dir, make_result_filename, save_json, get_run_id
from utils.offline_solver import solve_offline_sem_lasso_batch
from utils.repro import collect_environment_info


@dataclass
class ExperimentResult:
    """実験結果"""
    error_means: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    error_means_by_variant: Dict[str, Dict[str, Optional[np.ndarray]]] = field(default_factory=dict)
    last_estimates: Optional[Dict[str, np.ndarray]] = None
    result_dir: Optional[Path] = None
    figure_path: Optional[Path] = None
    figure_paths: List[Path] = field(default_factory=list)


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
        self.burn_in_cfg = int(getattr(self.cfg.metric, "burn_in", 0))
        self.divide_by_n2 = bool(getattr(self.cfg.metric, "divide_by_n2", False))
        self.plot_variants = self._init_plot_variants()
        self.primary_variant_key = self._variant_key(self.error_normalization, self.divide_by_n2)

    @staticmethod
    def _variant_key(normalization: str, divide_by_n2: bool) -> str:
        norm_tag = "true" if normalization == "true_value" else "offline"
        return f"{norm_tag}_n2={int(bool(divide_by_n2))}"

    def _init_plot_variants(self) -> List[Dict[str, object]]:
        variants = [
            {"normalization": "true_value", "divide_by_n2": False},
            {"normalization": "true_value", "divide_by_n2": True},
            {"normalization": "offline_solution", "divide_by_n2": False},
            {"normalization": "offline_solution", "divide_by_n2": True},
        ]
        for v in variants:
            v["key"] = self._variant_key(str(v["normalization"]), bool(v["divide_by_n2"]))
        return variants

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
            "burn_in": self.burn_in_cfg,
        }
        if self.error_normalization == "offline_solution":
            offline_space = self.cfg.search_spaces.offline.offline_lambda_l1
            metric_items["offline_lambda_l1 (range)"] = f"[{offline_space.low}, {offline_space.high}]"
        print_block("Metric Settings", metric_items)

        # 比較条件
        comp = getattr(self.cfg, "comparison", None)
        if comp is not None:
            print_block("Comparison Settings", {
                "pc_model": getattr(comp, "pc_model", "exog"),
                "pc_use_true_T_init": getattr(comp, "pc_use_true_T_init", True),
                "pc_T_init_identity_scale": getattr(comp, "pc_T_init_identity_scale", 1.0),
                "pp_init_b0": getattr(comp, "pp_init_b0", "ones"),
            })
        
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
        t0 = time.perf_counter()
        rng = np.random.default_rng(trial_seed)
        S_series, T_mat, Z, Y = self.generate_data(rng)
        t1 = time.perf_counter()
        
        # オフライン解を計算（必要な場合）
        S_offline = None
        if any(v["normalization"] == "offline_solution" for v in self.plot_variants):
            S_offline = solve_offline_sem_lasso_batch(
                Y, Z, self.hyperparams.offline_lambda_l1
            )
        t2 = time.perf_counter()
        
        # 手法実行
        executor = MethodExecutor(
            N=self.N,
            flags=self.flags,
            hyperparams=self.hyperparams,
            error_normalization=self.error_normalization,
            comparison=getattr(self.cfg, "comparison", None),
            divide_by_n2=bool(getattr(self.cfg.metric, "divide_by_n2", False)),
            error_variants=self.plot_variants,
        )
        
        result = executor.execute_all(Y, Z, S_series, T_mat, S_offline)
        t3 = time.perf_counter()
        result.trial_seed = trial_seed
        result.timing = {
            "total_sec": t3 - t0,
            "data_gen_sec": t1 - t0,
            "offline_sec": t2 - t1,
            "methods_sec": t3 - t2,
        }
        return result

    def aggregate_results(
        self, results: List[TrialResult]
    ) -> Tuple[Dict[str, Dict[str, Optional[np.ndarray]]], Dict[str, Optional[np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        """
        複数試行の結果を集計する。

        Parameters
        ----------
        results : List[TrialResult]
            全試行の結果リスト

        Returns
        -------
        Tuple[error_means_by_variant, error_means, last_estimates]
            - error_means_by_variant: 各バリアント×手法の平均誤差
            - error_means: 設定上のプライマリ指標の平均誤差
            - last_estimates: 最後の試行の最終推定値
        """
        methods = ["pp", "pp_sgd", "pc", "co", "sgd", "pg"]
        error_totals_by_variant: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for variant in self.plot_variants:
            v_key = str(variant["key"])
            error_totals_by_variant[v_key] = {}
            for method in methods:
                flag = getattr(self.flags, method)
                error_totals_by_variant[v_key][method] = np.zeros(self.T) if flag else None
        
        # 集計
        last_estimates = None
        for result in results:
            for variant in self.plot_variants:
                v_key = str(variant["key"])
                for method in methods:
                    if (
                        error_totals_by_variant[v_key][method] is not None
                        and method in result.errors_by_variant.get(v_key, {})
                    ):
                        error_totals_by_variant[v_key][method] += np.array(
                            result.errors_by_variant[v_key][method]
                        )
            last_estimates = result.estimates_final
        
        # 平均を計算
        error_means_by_variant: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for variant in self.plot_variants:
            v_key = str(variant["key"])
            error_means_by_variant[v_key] = {}
            for method in methods:
                if error_totals_by_variant[v_key][method] is not None:
                    error_means_by_variant[v_key][method] = (
                        error_totals_by_variant[v_key][method] / self.num_trials
                    )
                else:
                    error_means_by_variant[v_key][method] = None

        error_means = error_means_by_variant.get(self.primary_variant_key, {})
        
        return error_means_by_variant, error_means, last_estimates

    def plot_results(
        self,
        error_means: Dict[str, Optional[np.ndarray]],
        save_path: Path,
        normalization: str,
        divide_by_n2: bool,
    ) -> None:
        """
        結果をプロットする。

        Parameters
        ----------
        error_means : Dict[str, Optional[np.ndarray]]
            各手法の平均誤差
        save_path : Path
            保存先パス
        normalization : str
            正規化方法
        divide_by_n2 : bool
            N^2 で割るかどうか
        """
        plt.figure(figsize=(10, 6))
        
        # プロット順序と色を定義
        plot_order = [
            ("co", "blue", "Correction Only"),
            ("pc", "limegreen", "Prediction Correction"),
            ("sgd", "cyan", "SGD"),
            ("pg", "magenta", "ProxGrad"),
            ("pp_sgd", "orange", "PP-SGD (q=1,r=1)"),
            ("pp", "red", "Proposed (PP)"),
        ]
        
        for method, color, label in plot_order:
            if error_means.get(method) is not None:
                plt.plot(error_means[method], color=color, label=label)
        
        plt.yscale("log")
        plt.xlim(left=0, right=self.T)
        plt.xlabel("t")
        
        if normalization == "offline_solution":
            ylabel = r"Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^* - S_{\mathrm{offline}}\|_F^2}$"
        else:
            ylabel = r"Average $\frac{\|\hat{S} - S^*\|_F^2}{\|S^*\|_F^2}$"
        if divide_by_n2:
            ylabel = ylabel + r"\,$/\,N^2$"
        plt.ylabel(ylabel)
        
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig(str(save_path))
        plt.close() # Close to avoid memory leaks

    def save_metadata(
        self,
        result_dir: Path,
        filename: str,
        error_means: Dict[str, Optional[np.ndarray]],
        trial_seeds: List[int],
        error_means_by_variant: Optional[Dict[str, Dict[str, Optional[np.ndarray]]]] = None,
        figure_records: Optional[List[Dict[str, str]]] = None,
        trial_timings: Optional[Dict[str, Dict[str, float]]] = None,
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
        error_means_by_variant : Dict[str, Dict[str, Optional[np.ndarray]]], optional
            各バリアント×手法の平均誤差
        figure_records : List[Dict[str, str]], optional
            保存した図の情報
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
        
        # ハイパラJSONの内容も埋め込む（パスが変わっても再現できるように）
        hyperparam_json_content = None
        if self.hyperparam_path is not None and self.hyperparam_path.is_file():
            try:
                hyperparam_json_content = json.loads(self.hyperparam_path.read_text(encoding="utf-8"))
            except Exception:
                hyperparam_json_content = None

        # git情報（再現性用）
        def _git(args: list[str]) -> Optional[str]:
            try:
                out = subprocess.check_output(args, cwd=str(Path(__file__).resolve().parents[1]), stderr=subprocess.DEVNULL)
                return out.decode("utf-8", errors="replace").strip()
            except Exception:
                return None

        git_info = {
            "head": _git(["git", "rev-parse", "HEAD"]),
            "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "is_dirty": bool(_git(["git", "status", "--porcelain"])),
        }

        env_info = collect_environment_info(
            package_names=[
                "numpy",
                "scipy",
                "cvxpy",
                "optuna",
                "matplotlib",
                "joblib",
                "tqdm",
                "tqdm-joblib",
                "networkx",
            ]
        )

        # ハイパーパラメータを辞書形式に変換
        hp_dict = hyperparams_to_dict(self.hyperparams)
        
        # メタデータ構築
        if error_means_by_variant is None:
            error_means_by_variant = {self.primary_variant_key: error_means}
        if figure_records is None:
            figure_records = []

        timing_summary: Dict[str, Optional[float]] = {}
        if trial_timings:
            totals = [v.get("total_sec", 0.0) for v in trial_timings.values()]
            timing_summary = {
                "total_sec_sum": float(np.sum(totals)) if totals else None,
                "total_sec_mean": float(np.mean(totals)) if totals else None,
                "total_sec_median": float(np.median(totals)) if totals else None,
                "total_sec_min": float(np.min(totals)) if totals else None,
                "total_sec_max": float(np.max(totals)) if totals else None,
            }

        metadata = {
            "created_at": run_started_at.isoformat(),
            "command": sys.argv,
            "repro": {
                "git": git_info,
                "hyperparam_json_content": hyperparam_json_content,
                "environment": env_info,
            },
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
            # config.py の全設定スナップショット（CONFIG_MAIN/TEST の全フィールド）
            "config_full": config_to_dict(self.cfg),
            "metric": {
                "error_normalization": self.error_normalization,
                "offline_lambda_l1": self.hyperparams.offline_lambda_l1,
                "burn_in": self.burn_in_cfg,
                "divide_by_n2": bool(getattr(self.cfg.metric, "divide_by_n2", False)),
                "plot_variants": [
                    {
                        "key": v["key"],
                        "normalization": v["normalization"],
                        "divide_by_n2": v["divide_by_n2"],
                    }
                    for v in self.plot_variants
                ],
            },
            "methods": {
                "pp": {
                    "enabled": self.flags.pp,
                    "hyperparams": hp_dict["pp"],
                },
                "pp_sgd": {
                    "enabled": getattr(self.flags, "pp_sgd", False),
                    "hyperparams": hp_dict.get("pp_sgd", {}),
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
                "timings": {
                    "trial_sec": trial_timings or {},
                    "summary": timing_summary,
                },
                "figures": [
                    {
                        "key": v["key"],
                        "normalization": v["normalization"],
                        "divide_by_n2": v["divide_by_n2"],
                        "figure": record["figure"],
                        "figure_path": record["figure_path"],
                    }
                    for v, record in zip(self.plot_variants, figure_records)
                ],
                "metrics_by_variant": {
                    v["key"]: {
                        method: err.tolist() if err is not None else None
                        for method, err in error_means_by_variant[v["key"]].items()
                    }
                    for v in self.plot_variants
                },
            },
            "snapshots": script_copies,
            "hyperparam_json": str(self.hyperparam_path) if self.hyperparam_path else None,
            "result_dir": str(result_dir),
        }
        
        meta_name = f"{Path(filename).stem}_meta.json"
        save_json(metadata, result_dir, name=meta_name)

        run_id = get_run_id()
        used_hyperparams_payload = {
            "hyperparam_json": str(self.hyperparam_path) if self.hyperparam_path else None,
            "hyperparam_json_content": hyperparam_json_content,
            "resolved_hyperparams": hp_dict,
            "offline_lambda_l1": self.hyperparams.offline_lambda_l1,
            "note": "Resolved from hyperparam_json if provided; otherwise config defaults.",
        }
        save_json(used_hyperparams_payload, result_dir, name=f"used_hyperparams_ts={run_id}.json")

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

        # 結果ディレクトリ作成（trialデータ保存のため早めに作成）
        result_dir = create_result_dir(
            self.cfg.output.result_root,
            self.get_output_subdir(),
            extra_tag="images",
        )
        self.result_dir = Path(result_dir)
        
        # 試行シード生成
        trial_seeds = [self.seed + i for i in range(self.num_trials)]
        
        # 並列実行
        with tqdm_joblib(tqdm(desc="Progress", total=self.num_trials)):
            results = Parallel(n_jobs=-1, batch_size=1, prefer="threads")(
                delayed(self.run_trial)(ts) for ts in trial_seeds
            )
        
        # 結果集計
        error_means_by_variant, error_means, last_estimates = self.aggregate_results(results)

        trial_timings: Dict[str, Dict[str, float]] = {}
        for result in results:
            if result.trial_seed is not None and result.timing:
                trial_timings[str(result.trial_seed)] = result.timing
        
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
        figure_records: List[Dict[str, str]] = []
        figure_paths: List[Path] = []
        filename_primary = None
        for variant in self.plot_variants:
            variant_params = {
                **filename_params,
                "norm": "true" if variant["normalization"] == "true_value" else "offline",
                "n2": int(bool(variant["divide_by_n2"])),
            }
            filename = make_result_filename(
                prefix=self.get_scenario_name(),
                params=variant_params,
                suffix=".png",
            )
            if variant["key"] == self.primary_variant_key:
                filename_primary = filename
            print(filename)
            figure_path = Path(result_dir) / filename
            self.plot_results(
                error_means_by_variant[variant["key"]],
                figure_path,
                str(variant["normalization"]),
                bool(variant["divide_by_n2"]),
            )
            figure_paths.append(figure_path)
            figure_records.append({
                "figure": filename,
                "figure_path": str(figure_path),
            })
        filename = filename_primary or figure_records[0]["figure"]
        figure_path = Path(result_dir) / filename
        
        # ヒートマップ表示
        cfg = get_config()
        if cfg.output.save_heatmap and last_estimates is not None:
            heatmap_filename = filename.replace(".png", "_heatmap.png")
            plot_heatmaps(
                matrices=last_estimates,
                save_path=Path(result_dir) / heatmap_filename,
                title=f"Estimated vs True at t={self.T-1} (last trial)",
                show=True,
            )
        
        # メタデータ保存
        self.save_metadata(
            result_dir,
            filename,
            error_means,
            trial_seeds,
            error_means_by_variant=error_means_by_variant,
            figure_records=figure_records,
            trial_timings=trial_timings,
        )
        
        return ExperimentResult(
            error_means=error_means,
            error_means_by_variant=error_means_by_variant,
            last_estimates=last_estimates,
            result_dir=Path(result_dir),
            figure_path=figure_path,
            figure_paths=figure_paths,
        )
