"""
Piecewise シナリオのシミュレーション実行スクリプト

設定は code/config.py で一元管理されています。
このスクリプトを実行する前に config.py を編集して設定を変更してください。

使用方法:
    python -m code.run_piecewise
    python -m code.run_piecewise --hyperparam_json path/to/hyperparams.json
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from code.data_gen import generate_piecewise_X_with_exog
from code.experiment_runner import BaseExperimentRunner, parse_hyperparam_arg


class PiecewiseRunner(BaseExperimentRunner):
    """Piecewise シナリオの実験実行クラス"""

    def get_scenario_name(self) -> str:
        return "piecewise"

    def generate_data(
        self, rng: np.random.Generator
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        return generate_piecewise_X_with_exog(
            N=self.N,
            T=self.T,
            sparsity=self.sparsity,
            max_weight=self.max_weight,
            std_e=self.std_e,
            K=self.cfg.piecewise.K,
            s_type=self.cfg.data_gen.s_type,
            t_min=self.cfg.data_gen.t_min,
            t_max=self.cfg.data_gen.t_max,
            z_dist=self.cfg.data_gen.z_dist,
            rng=rng,
        )

    def get_scenario_params(self) -> Dict[str, Any]:
        return {"K": self.cfg.piecewise.K}


def main() -> None:
    """メイン処理"""
    hyperparam_path = parse_hyperparam_arg()
    runner = PiecewiseRunner(hyperparam_path=hyperparam_path)
    runner.run()


if __name__ == "__main__":
    main()
