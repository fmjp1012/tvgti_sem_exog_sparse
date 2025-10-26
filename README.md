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

### Makefile ワークフロー

`make help` で主要ターゲットと共通変数を確認できます。よく使うターゲットは次の通りです。

| ターゲット | 内容 | 主な用途 |
| --- | --- | --- |
| `make help` | 利用可能ターゲットと変数を表示 | 全体の確認 |
| `make tune_piecewise` | `code.tune_and_run piecewise --no_run` を呼び出しハイパラ探索のみ実行 | ピースワイズの事前チューニング |
| `make tune_linear` | `code.tune_and_run linear --no_run` を呼び出しハイパラ探索のみ実行 | 線形成形の事前チューニング |
| `make run_piecewise` | `code/run_piecewise.py` を使い訓練・評価を実行 | ピースワイズの本番実験 |
| `make run_linear` | `code/run_linear.py` を使い訓練・評価を実行 | 線形成形の本番実験 |
| `make tune_run_piecewise` | `code.tune_and_run piecewise` を実行し、探索後に `NUM_TRIALS` 回の評価をまとめて実行 | チューニングと実験を一括処理(ピースワイズ) |
| `make tune_run_linear` | `code.tune_and_run linear` を実行し、探索後に `NUM_TRIALS` 回の評価をまとめて実行 | チューニングと実験を一括処理(線形) |

共通で上書きできる make 変数は次のとおりです。

| 変数 | 既定値 | 意味 |
| --- | --- | --- |
| `PYTHON` | `/Users/fmjp/venv/default/bin/python` | 利用する Python 実行ファイル |
| `NUM_TRIALS` | `100` | `--num_trials` に渡す試行回数 (`run_*`, `tune_and_run`) |
| `TUNE_ARGS` | 空文字 | `code.tune_and_run` へそのまま渡される追加引数 |
| `RUN_ARGS` | 空文字 | `code/run_*.py` へそのまま渡される追加引数 |
| `HYPER` | 空文字 | 既存ハイパラ JSON のパス (`--hyperparam_json`) |

### TUNE_ARGS / RUN_ARGS の使い分け

- `TUNE_ARGS` は `code.tune_and_run` のハイパラ探索パートに渡されます。`--methods` や `--tuning_trials`、探索範囲の上書きなどはこちらで指定します。
- `RUN_ARGS` は実行フェーズ向けの引数です。`make run_*` では `code/run_*.py` に、`make tune_run_*` では `code.tune_and_run` 内で生成される実行フェーズにそのまま渡されます。
- `tune_run_*` ターゲットでは `$(TUNE_ARGS) $(RUN_ARGS)` の順で同じコマンドに連結されますが、`code/tune_and_run.py` がパラメータをチューニング用/実行用に振り分けるため混在していても問題ありません。
- フラグ形式の引数（例: `--no_pp`）を複数指定する場合は `RUN_ARGS="--no_pc --no_co --run_pp"` のように引用符でまとめて渡してください。
- 既存のハイパラ JSON を使いたい場合は `HYPER=path/to.json` を指定するか、`RUN_ARGS="--hyperparam_json path/to.json"` を追加します。

#### 主なチューニング引数 (`TUNE_ARGS`)

| 引数 | 対象 | 概要 | 備考 | 例 |
| --- | --- | --- | --- | --- |
| `--N`, `--T` | 共通 | ノード数、サンプル長 | 実験スケールを変更 | `--N 40 --T 1500` |
| `--sparsity` | 共通 | グラフの疎度 | 0〜1 で指定 | `--sparsity 0.55` |
| `--max_weight` | 共通 | エッジ重みの上限 | 絶対値の最大値 | `--max_weight 0.35` |
| `--std_e` | 共通 | ノイズ標準偏差 |  | `--std_e 0.1` |
| `--seed` | 共通 | 乱数シード | チューニング・実行のベース | `--seed 123` |
| `--K` | piecewise | 区分数 | piecewise シナリオのみ | `--K 6` |
| `--methods` | 共通 | チューニング対象メソッド |  | `--methods pp,pc` |
| `--tuning_trials` | 共通 | Optuna の試行回数 | 既定: piecewise=300, linear=30 | `--tuning_trials 200` |
| `--tuning_runs_per_trial` | 共通 | 1 トライアル内でのシミュレーション回数 |  | `--tuning_runs_per_trial 3` |
| `--result_root`, `--subdir` | 共通 | 保存先のルート/サブディレクトリ |  | `--result_root ./result --subdir debug_run` |

#### 実行フェーズ向け引数 (`RUN_ARGS`)

| 引数 | 概要 | 備考 | 例 |
| --- | --- | --- | --- |
| `--num_trials` | 実行時のシミュレーション回数 | `make run_*` では `NUM_TRIALS` が優先 | `--num_trials 50` |
| `--hyperparam_json` | 既存ハイパラ JSON のパス | `HYPER` 変数でも指定可 | `--hyperparam_json result/241201/piecewise/meta/best.json` |
| `--seed` | 実行時の基本シード | チューニングと別個に渡せます | `--seed 42` |
| `--N`, `--T`, `--sparsity`, `--max_weight`, `--std_e`, `--K` | 実行パラメータの上書き | チューニング値とは独立 | `--N 30 --sparsity 0.6` |

実行メソッドの有効・無効は次のフラグでまとめています。

| フラグ | 効果 | 例 |
| --- | --- | --- |
| `--run_pc` / `--no_pc` | Prediction–Correction を強制実行 / スキップ | `--no_pc` |
| `--run_pp` / `--no_pp` | Parallel Projection を強制実行 / スキップ | `--run_pp` |
| `--run_co` / `--no_co` | Correction Only を強制実行 / スキップ | `--no_co` |
| `--run_sgd` / `--no_sgd` | SGD ベースラインを強制実行 / スキップ | `--no_sgd` |

#### 実行例

```bash
# PP 手法だけをチューニングして 100 試行実行（ピースワイズ）
make tune_run_piecewise TUNE_ARGS="--methods pp" RUN_ARGS="--no_pc --no_co --no_sgd"

# 既存ハイパラ JSON を使って 200 試行だけ実行
make run_piecewise NUM_TRIALS=200 HYPER=result/241201/piecewise/meta/best.json RUN_ARGS="--no_sgd --seed 42"

# 線形成形をチューニング後に 50 試行実行（PC/PP のみ）
make tune_run_linear NUM_TRIALS=50 TUNE_ARGS="--N 30 --T 1500 --sparsity 0.5 --tuning_trials 40" RUN_ARGS="--run_pc --run_pp --no_co --no_sgd"
```

### ハイパラ探索で使える主なオーバーライド

Optuna で探索するパラメータ範囲を `TUNE_ARGS` で上書きできます。代表的なキーをメソッド別にまとめます（カンマ区切り指定は文字列で渡してください）。

| メソッド | 引数 | 概要 | 例 |
| --- | --- | --- | --- |
| PP | `--pp_rho_low` / `high` / `log` | ρ の探索範囲と対数探索指定 | `--pp_rho_low 1e-5 --pp_rho_high 1e-2 --pp_rho_log true` |
| PP | `--pp_mu_lambda_low` / `high` / `log` | μλ の探索範囲 | `--pp_mu_lambda_low 0.1 --pp_mu_lambda_high 2.0` |
| PC | `--pc_lambda_reg_low` / `high` / `log` | λ_reg の探索範囲 | `--pc_lambda_reg_low 1e-4 --pc_lambda_reg_high 1e-1` |
| PC | `--pc_alpha_low` / `high` / `log` | α の探索範囲 | `--pc_alpha_low 0.05 --pc_alpha_high 0.5` |
| PC | `--pc_beta_pc_low` / `high` / `log` | β_pc の探索範囲 | `--pc_beta_pc_low 0.01 --pc_beta_pc_high 0.2` |
| PC | `--pc_gamma_low` / `high` / `log` | γ の探索範囲 | `--pc_gamma_low 0.5 --pc_gamma_high 5` |
| PC | `--pc_P_min` / `max` / `step` | P 値の候補集合 | `--pc_P_min 1 --pc_P_max 5 --pc_P_step 1` |
| PC | `--pc_C_choices` | C の候補（カンマ区切り） | `--pc_C_choices 1,3,5` |
| CO | `--co_beta_co_low` / `high` / `log` | β_co の探索範囲 | `--co_beta_co_low 0.01 --co_beta_co_high 0.3` |
| CO | `--co_gamma_low` / `high` | γ の探索範囲 | `--co_gamma_low 0.5 --co_gamma_high 3.0` |
| CO | `--co_C_choices` | C の候補（カンマ区切り） | `--co_C_choices 1,2,4` |
| SGD | `--sgd_beta_sgd_low` / `high` / `log` | β_sgd の探索範囲 | `--sgd_beta_sgd_low 1e-3 --sgd_beta_sgd_high 1e-1` |

値は `int`・`float`・カンマ区切り文字列で渡せば `code/tune_and_run.py` 側で適切にキャストされます。

### スクリプトを直接実行したい場合

Makefile を使わずに直接スクリプトを走らせることもできます。

```bash
~/venv/default/bin/python code/run_piecewise.py --num_trials 100 --no_co --seed 3
~/venv/default/bin/python -m code.tune_and_run piecewise --no_run --methods pp,pc
```

この場合も前述の `TUNE_ARGS` / `RUN_ARGS` 向け引数と同じ指定をそのまま CLI に渡せます。`code/run_*.py` は内部で `utils.io.plotting.apply_style` と `utils.io.results` を呼び出し、結果ディレクトリやスクリプトのバックアップを自動で行います。


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

- `code/data_gen.py` に linear / brownian / piecewise 用の構造系列 `S(t)` と観測 `X`(=Y)・外生 `Z` を生成する関数を実装
- 例: `generate_piecewise_X_with_exog(N, T, sparsity, max_weight, std_e, K, ...)`


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
