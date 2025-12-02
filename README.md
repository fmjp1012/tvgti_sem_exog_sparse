# Time-Varying Graph Topology Identification with Exogenous Inputs

外因性入力を含む時変構造方程式モデル（Time-Varying SEM）におけるグラフ構造推定アルゴリズムのシミュレーション実装プロジェクトです。
主に Prediction-Correction (PC) 法および Proximal Projection (PP) 法を用いた手法の検証・比較を行います。

## プロジェクト構造

- **`code/`**: 実験実行用スクリプト、データ生成、ハイパーパラメータチューニング
  - **`config.py`**: 🔧 **すべてのシミュレーション設定を一元管理**
  - `run_*.py`: 各種シナリオ（Piecewise, Linear, Brownian）のモンテカルロシミュレーション実行
  - `run_*_once.py`: 各種シナリオの単発実行用スクリプト（デバッグ・確認用）
  - `data_gen.py`: シミュレーションデータ生成
  - `tune_and_run.py`: チューニングと実行を一括で行うスクリプト
  - `hyperparam_tuning.py`: ハイパーパラメータチューニング
- **`models/`**: アルゴリズムの実装
  - `tvgti_pc/`: Prediction-Correction 法 (PC-SEM)
  - `pp_exog.py`: Proximal Projection 法 (PP-SEM)
- **`utils/`**: ユーティリティ（プロット、データIO、行列計算など）
- **`result/`**: 実験結果（画像、メタデータ、実行スクリプトのバックアップ）の出力先
- **`thesis/`**: 論文原稿（LaTeX）
- **`Makefile`**: 実行コマンド管理

## 環境要件

- Python 3.x
- 推奨仮想環境: `/User/fmjp/venv/default`

### 主な依存ライブラリ
- numpy
- matplotlib
- tqdm
- joblib
- scipy
- optuna
- cvxpy

## 使用方法

### ⚠️ 重要: 設定は `code/config.py` で一元管理

**すべてのシミュレーション条件は `code/config.py` で設定してください。**
コマンドライン引数による設定は非推奨です。

### 設定の変更方法

`code/config.py` を開いて、`CONFIG` インスタンスの値を編集します。
（詳細は `code/config.py` 内のコメントを参照）

### 1. 現在の設定を確認

```bash
make config
# または
/Users/fmjp/venv/default/bin/python code/config.py
```

### 2. 実験の実行（Makefile利用）

`Makefile` を使用して、チューニングからシミュレーションまでを一括実行できます。

#### フォアグラウンド実行（SSH接続中のみ）

```bash
# Piecewise シナリオ
make piecewise

# Linear シナリオ
make linear

# チューニングのみ
make tune_piecewise
make tune_linear

# シミュレーションのみ
make run_piecewise
make run_linear
```

#### バックグラウンド実行（SSH切断後も継続）

長時間かかる実験はバックグラウンド実行を推奨します。ログは `logs/` ディレクトリに保存されます。

```bash
# Piecewise シナリオ（チューニング＋実行）
make bg_piecewise

# Linear シナリオ（チューニング＋実行）
make bg_linear

# シミュレーションのみ（バックグラウンド）
make bg_run_piecewise
make bg_run_linear

# ステータス確認
make bg_status

# ログのリアルタイム確認
make bg_tail

# 実行停止
make bg_stop
```

### 3. 個別スクリプトの実行

Brownian Motion シナリオや単発実行など、Makefile にターゲットがない場合は直接 Python モジュールとして実行します。

```bash
# Brownian Motion シナリオ
/Users/fmjp/venv/default/bin/python -m code.run_brownian

# 単発実行（Piecewise）
/Users/fmjp/venv/default/bin/python -m code.run_piecewise_once
```

## 設定項目一覧

| カテゴリ | 設定項目 | 説明 |
|---------|---------|------|
| **methods** | pp, pc, co, sgd, pg | 実行する手法のON/OFF |
| **common** | N, T, sparsity, max_weight, std_e, seed | シミュレーション共通パラメータ |
| **piecewise** | K | 変化点の数（Piecewiseシナリオ用） |
| **tuning** | tuning_trials, tuning_runs_per_trial, truncation_horizon | チューニング設定 |
| **search_spaces** | pp, pc, co, sgd, pg | 各手法のハイパラ探索範囲 |
| **hyperparams** | pp, pc, co, sgd, pg | デフォルトハイパーパラメータ |
| **run** | num_trials | モンテカルロ試行回数 |
| **output** | result_root, subdir_* | 出力先ディレクトリ |

## 出力

実験結果は `result/` ディレクトリ内に日付・シナリオごとのフォルダで保存されます。
- **images/**: 結果のプロット画像
- **meta.json**: 実験設定や結果の数値データを含むメタデータ
- **scripts/**: 実行時のスクリプトのバックアップ（config.py を含む）
