## Time-Varying Graph Topology Identification (SEM + Exogenous Sparse)

本リポジトリは、時変SEM(Time-Varying Structural Equation Model)に基づくグラフトポロジー同定の実験コード一式です。外生変数を含む疎な設定を対象に、複数のアルゴリズム(PC: Prediction–Correction, PP: Parallel Projection 等)を比較・可視化できます。


### 主な特徴
- **シナリオ別の実行スクリプト**: linear / brownian / piecewise それぞれのデータ生成・評価コードを同梱
  
- **モデル群**: `models/` 配下に PC/PP の各バリアント実装を用意
- **結果出力の標準化**: `result/YYMMDD/{scenario}/` に `meta.json` や画像を体系的に保存


## ディレクトリ構成

- `code/`
  - シナリオ別の直接実行スクリプトと、データ生成・ベースライン等
  - 例: `run_linear.py`, `run_brownian.py`, `run_piecewise.py`, `data_gen.py`
- `models/`
  - アルゴリズム実装
  - `tvgti_pc/`: Prediction–Correction 系の `time_varying_sem.py`
  - `tvgti_pp/`: Parallel Projection 系の `time_varying_sem.py`
  - `tvgti_pp_nonsparse_*` 系: 非疎／有向・無向・非重なり等のバリアント
  - `pp_exog.py`: 外生変数付き PP モデル
- `utils/`
  - 実験共通のユーティリティ
  - `io/`: `plotting.py`(描画スタイル), `results.py`(出力ディレクトリ生成/メタ保存/ファイル名生成)
  - `models/base/`: `TimeVaryingSEMBase` の抽象基底
  - `test/`: 簡易テスト
- `matrix_cache/`
  - 反復で用いる行列(duplication/elimination)のキャッシュ `.npz`
- `result/`
  - 実行日の階層で成果物を整理。例: `result/251006/{scenario}/meta.json`
- `thesis/`
  - LaTeX 文書一式(ビルド済みの `out/thesis.pdf` 等と画像)

補助の詳細は `utils/README.md` も参照してください。


## セットアップ

以下はユーザ環境ルール(既定の仮想環境)に従った例です。

```bash
~/venv/default/bin/python -m pip install -U pip
~/venv/default/bin/python -m pip install numpy matplotlib scipy cvxpy joblib tqdm networkx
```

必要に応じて LaTeX(オプション) を用意してください。`plot.latex: true` の場合に必要です。


## 使い方

### 実行方法

各シナリオスクリプトを直接実行します。

```bash
~/venv/default/bin/python code/run_linear.py
~/venv/default/bin/python code/run_brownian.py
~/venv/default/bin/python code/run_piecewise.py
```

利用可能なシナリオ(内部ディスパッチ):
- `linear`, `brownian`, `piecewise`
- `linear_mean`, `brownian_mean`, `piecewise_mean`
- `linear_once`, `brownian_once`, `piecewise_once`

出力は `result/YYMMDD/{scenario}/` に保存され、`meta.json` とともに実行スクリプトのバックアップも行われます。

#### 統一ランナーと直接実行の違い

- 統一ランナー: ルートの `run.py` が YAML を読み込み、描画設定を適用し、`result/YYMMDD/{scenario}/` に `meta.json` と自身のスクリプトをバックアップします。シナリオは `--scenario` で切り替えでき、実行スクリプトは `code/run_*.py` にディスパッチされます。
- 直接実行: 各 `code/run_*.py` が独自に `apply_style` と出力ディレクトリ・バックアップを行います。画像は `result/YYMMDD/exog_sparse_* /images/` 配下に保存され、`meta.json` は原則生成されません（統一ランナーのみが生成）。

パラメータ（N, T, sparsity など）や乱数 `seed` は各 `code/run_*.py` 側で設定・管理してください。

### 2) シナリオスクリプトを直接実行

```bash
~/venv/default/bin/python code/run_piecewise.py
```

各スクリプトは内部で `utils.io.plotting.apply_style` と `utils.io.results` を利用して画像保存やバックアップを行います。

### 3) ハイパラチューニング(任意)

Optuna を用いた例が `code/hypara_tuning_*.py` にあります。必要に応じて `optuna` を追加インストールしてください。


## 設定

YAML 設定は廃止しました。描画スタイルや出力先は各スクリプトが `utils.io.plotting` と `utils.io.results` を直接呼び出して設定します。`seed` は各 `run_*` スクリプトで明示的に設定してください。


## 出力(成果物)

- ルート: `result/`
- 構成: 実行日(YYMMDD) / シナリオ名 で階層化
- 保存物:
  - 直接実行スクリプト: `result/YYMMDD/exog_sparse_{scenario}/images/*.png` とスクリプトのバックアップ
  - 画像(PNG)などの成果物。ファイル名は `utils/io/results.py` の `make_result_filename` で安定生成


## モデル実装(概要)

- PC(Prediction–Correction): `models/tvgti_pc/time_varying_sem.py`
- PP(Parallel Projection) および外生変数付き PP: `models/tvgti_pp/*`, `models/pp_exog.py`
- 非疎/有向/無向/非重なり等のバリアント: `models/tvgti_pp_nonsparse_*/*`, `models/tvgti_pc_nonsparse/*`, `models/tvgti_pc_pp_nonsparse/*`

すべて `numpy`/`scipy`/`cvxpy`/`tqdm` などを使用し、`utils/models/base/base.py` の抽象基底を土台としています。


## データ生成

- `code/data_gen.py` に linear / brownian / piecewise 用の観測 `Y` と外生 `U` の生成関数を実装
- 例: `generate_piecewise_Y_with_exog(N, T, sparsity, max_weight, std_e, K, ...)`


## 既知の注意点 / トラブルシュート

- 古いインポート残骸に注意: `refactor.*` や `exog_sparse_sim.*` が一部のスクリプト/テストに残っている場合があります。`utils.*` や `code.data_gen` など現行のパスに置換してください。
  - 例: `from refactor.exog_sparse_sim.data_gen import ...` → `from code.data_gen import ...`
  - 例: `from exog_sparse_sim.data_gen import ...` → `from code.data_gen import ...`
  - 例: `from refactor.models.tvgti_pc.time_varying_sem import TimeVaryingSEM` → `from models.tvgti_pc.time_varying_sem import TimeVaryingSEM`
- `ModuleNotFoundError: No module named 'refactor'`
  - 上記のとおり該当インポートを修正
- LaTeX 関連のエラー
  - `plot.latex: false` にするか、TeX 環境を導入


## テスト(任意)

簡易テストは `utils/test/` にあります。必要に応じて `pytest` を導入し実行してください。

```bash
~/venv/default/bin/python -m pip install pytest
~/venv/default/bin/python -m pytest -q
```


## ライセンス / 引用

研究・学習目的のコードです。学術利用での引用・参考文献への記載等は、実験内容に応じて適宜行ってください。`thesis/` には関連の LaTeX 文書例が含まれています。

