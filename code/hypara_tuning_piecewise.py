from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from code.hyperparam_tuning import (
    save_best_hyperparams,
    tune_piecewise_all_methods,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ピースワイズシナリオのハイパラ調整ユーティリティ")
    parser.add_argument("--N", type=int, default=20, help="ノード数")
    parser.add_argument("--T", type=int, default=1000, help="系列長")
    parser.add_argument("--sparsity", type=float, default=0.7, help="隣接行列のスパース率")
    parser.add_argument("--max_weight", type=float, default=0.5, help="隣接行列の最大重み")
    parser.add_argument("--std_e", type=float, default=0.05, help="ノイズ分散の標準偏差")
    parser.add_argument("--K", type=int, default=4, help="区分数")
    parser.add_argument("--tuning_trials", type=int, default=30, help="Optunaの試行回数")
    parser.add_argument("--tuning_runs_per_trial", type=int, default=5, help="各Optuna試行での平均化回数")
    parser.add_argument("--seed", type=int, default=3, help="乱数シード")
    parser.add_argument("--result_root", type=Path, default=Path("./result"), help="結果保存ルート")
    parser.add_argument("--subdir", type=str, default="exog_sparse_tuning", help="結果保存のサブディレクトリ")
    parser.add_argument("--output_json", type=Path, default=None, help="保存先を直接指定（指定時はresult_root/subdirは無視）")
    parser.add_argument("--no_save", action="store_true", help="JSON保存をスキップ")

    args = parser.parse_args()

    best = tune_piecewise_all_methods(
        N=args.N,
        T=args.T,
        sparsity=args.sparsity,
        max_weight=args.max_weight,
        std_e=args.std_e,
        K=args.K,
        tuning_trials=args.tuning_trials,
        tuning_runs_per_trial=args.tuning_runs_per_trial,
        seed=args.seed,
    )

    print("推定されたハイパラ:")
    print(json.dumps(best, indent=2))

    if args.no_save:
        return

    if isinstance(args.output_json, Path):
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"ハイパラを {args.output_json} に保存しました")
    else:
        out_path = save_best_hyperparams(
            best,
            scenario="piecewise",
            result_root=args.result_root,
            subdir=args.subdir,
        )
        print(f"ハイパラを {out_path} に保存しました")


if __name__ == '__main__':
    main()


