## utils パッケージ概要

時変 SEM (Time-Varying Structural Equation Model) の実験を共通化するためのユーティリティ群です。

- **I/O**: `utils/io/` に描画スタイル・結果保存の補助関数
- **モデル基底**: `utils/models/base/` に抽象基底 `TimeVaryingSEMBase`


## ディレクトリ構成

- `utils/io/plotting.py`: `apply_style(use_latex, font_family, base_font_size)` で matplotlib の rc を一括設定
- `utils/io/results.py`: 出力ディレクトリ作成・スクリプトバックアップ・`meta.json` 保存・ファイル名生成
- `utils/models/base/base.py`: `TimeVaryingSEMBase` 抽象基底（`run(X) -> Dict` を必須実装）



## 必要要件（Python 環境）

- Python: ユーザ既存の仮想環境を利用
- 主要パッケージ: `numpy`, `matplotlib`

インストール例（ユーザ規約に合わせて既定 venv を使用）:

```bash
~/venv/default/bin/python -m pip install numpy matplotlib
```


## 実行方法

各 `code/run_*.py` を直接実行してください。パラメータや `seed` は各スクリプト内で設定します。


## 設定ファイル

- 例: `utils/configs/linear.yaml`
  - `scenario`, `seed`, `plot.{latex,font,size}`, `output.root`
  - `model.{name,params}` は既存実装の識別・パラメータをメタとして保持
  - 未知キーは `ExperimentConfig.extra` に格納され、将来的な拡張に対応

`types.py` の `ExperimentConfig.from_yaml(path)` で読み込み、CLI からの上書きが可能です。


## 出力

- ルート: `output.root`（既定: `result/`）
- 構成: `result/YYMMDD/{scenario}/` に `meta.json` と実行スクリプトのバックアップが保存されます。
- 画像保存時は `io/results.py` の `make_result_filename` で安定したファイル名を生成できます。


## 既知の注意点（インポートパス）

`utils/scripts/run.py` は内部で `utils...` を参照します。

回避策（推奨修正）:

1. 既存スクリプトで古い `refactor.*` や `exog_sparse_sim.*` が残っている場合は現行のパスに置換してください。
   - 例: `from refactor.exog_sparse_sim.data_gen import ...` → `from code.data_gen import ...`
   - 例: `from exog_sparse_sim.data_gen import ...` → `from code.data_gen import ...`
   - 例: `from refactor.models.tvgti_pc.time_varying_sem import TimeVaryingSEM` → `from models.tvgti_pc.time_varying_sem import TimeVaryingSEM`
2. 統一ランナーのディスパッチは `dispatch = {"linear": ("code.run_linear", "main"), ...}` の形式で、`mod = __import__(module_name, fromlist=[func_name])` 経由で `main()` を実行します。

上記を反映すると、前述の実行コマンドで動作します。


## ラテフ描画（オプション）

`plot.latex: true` の場合、システムに LaTeX が必要です。利用環境に応じて TeX 環境を用意してください。


## よくあるエラー

- `ModuleNotFoundError: No module named 'refactor'`
  - 上記「既知の注意点」に従って `run.py` のインポートを修正してください。
- LaTeX 関連のエラー
  - LaTeX をインストールするか、設定で `plot.latex: false` にしてください。


