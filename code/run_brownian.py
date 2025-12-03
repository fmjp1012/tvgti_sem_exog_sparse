"""
Brownian シナリオのシミュレーション実行スクリプト

設定は code/config.py で一元管理されています。
このスクリプトを実行する前に config.py を編集して設定を変更してください。

使用方法:
    python -m code.run_brownian
    python -m code.run_brownian --hyperparam_json path/to/hyperparams.json

Note:
    Brownian シナリオには追加パラメータ std_S と K が必要です。
    現在は config.py の piecewise.K と固定の std_S=0.05 を使用しています。
    必要に応じて BrownianParams を config.py に追加してください。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from code.data_gen import generate_brownian_piecewise_X_with_exog
from code.experiment_runner import BaseExperimentRunner, parse_hyperparam_arg


class BrownianRunner(BaseExperimentRunner):
    """Brownian シナリオの実験実行クラス"""

    # Brownian シナリオ固有のデフォルトパラメータ
    DEFAULT_STD_S = 0.05
    DEFAULT_K = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Brownian シナリオ固有パラメータ
        # config.py に BrownianParams が追加されたらそこから取得するように変更
        self.K = getattr(self.cfg.piecewise, "K", self.DEFAULT_K)
        self.std_S = self.DEFAULT_STD_S

    def get_scenario_name(self) -> str:
        return "brownian"

    def get_output_subdir(self) -> str:
        return "exog_sparse_brownian"

    def generate_data(
        self, rng: np.random.Generator
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        return generate_brownian_piecewise_X_with_exog(
            N=self.N,
            T=self.T,
            sparsity=self.sparsity,
            max_weight=self.max_weight,
            std_e=self.std_e,
            K=self.K,
            std_S=self.std_S,
            s_type=self.cfg.data_gen.s_type,
            t_min=self.cfg.data_gen.t_min,
            t_max=self.cfg.data_gen.t_max,
            z_dist=self.cfg.data_gen.z_dist,
            rng=rng,
        )

    def get_scenario_params(self) -> Dict[str, Any]:
        return {
            "K": self.K,
            "std_S": self.std_S,
        }


def main() -> None:
    """メイン処理"""
    hyperparam_path = parse_hyperparam_arg()
    runner = BrownianRunner(hyperparam_path=hyperparam_path)
    runner.run()


if __name__ == "__main__":
    main()
