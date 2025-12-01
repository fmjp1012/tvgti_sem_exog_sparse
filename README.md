# Time-Varying Graph Topology Identification with Exogenous Inputs

外因性入力を含む時変構造方程式モデル（Time-Varying SEM）におけるグラフ構造推定アルゴリズムのシミュレーション実装プロジェクトです。
主に Prediction-Correction (PC) 法および Proximal Projection (PP) 法を用いた手法の検証・比較を行います。

## プロジェクト構造

- **`code/`**: 実験実行用スクリプト、データ生成、ハイパーパラメータチューニング
  - `run_*.py`: 各種シナリオの実行スクリプト
  - `data_gen.py`: シミュレーションデータ生成
  - `tune_and_run.py`: チューニングと実行を一括で行うスクリプト
- **`models/`**: アルゴリズムの実装
  - `tvgti_pc/`: Prediction-Correction 法 (PC-SEM)
  - `pp_exog.py`: Proximal Projection 法 (PP-SEM)
- **`utils/`**: ユーティリティ（プロット、データIO、行列計算など）
- **`result/`**: 実験結果（画像、メタデータ、実行スクリプトのバックアップ）の出力先
- **`thesis/`**: 論文原稿（LaTeX）

## 環境要件

- Python 3.x
- 推奨仮想環境: `/User/fmjp/venv/default`

### 主な依存ライブラリ
- numpy
- matplotlib
- tqdm
- joblib
- scipy

## 使用方法

本プロジェクトはパッケージとして構成されているため、ルートディレクトリから `-m` オプションを使用してモジュールとして実行します。

### 1. 実験の実行

3つの主要な構造変化シナリオ（Piecewise, Linear, Brownian）に対応したスクリプトが用意されています。

#### Piecewise (区分一定の変化)
```bash
/User/fmjp/venv/default/bin/python -m code.run_piecewise
```

#### Linear (線形変化)
```bash
/User/fmjp/venv/default/bin/python -m code.run_linear
```

#### Brownian (ブラウン運動による変化)
```bash
/User/fmjp/venv/default/bin/python -m code.run_brownian
```

### 2. ハイパーパラメータチューニングと実行

`tune_and_run.py` を使用して、ハイパーパラメータの探索と、最適パラメータを用いた実験を一括で行うことができます。

**実行例 (Piecewiseシナリオ):**
```bash
/User/fmjp/venv/default/bin/python -m code.tune_and_run piecewise \
  --N 20 --T 1000 --sparsity 0.7 --max_weight 0.5 --std_e 0.05 --K 4 \
  --tuning_trials 30 --tuning_runs_per_trial 5
```

### パラメータ設定

各実行スクリプトはコマンドライン引数で実験条件（ノード数 `N`、時系列長 `T`、スパース性 `sparsity` など）を調整可能です。
詳細なパラメータ設定やシナリオごとの仕様については、[code/README.md](code/README.md) および [agent.md](agent.md) を参照してください。

## 出力

実験結果は `result/` ディレクトリ内に日付・シナリオごとのフォルダで保存されます。
- **images/**: 結果のプロット画像
- **meta.json**: 実験設定や結果の数値データを含むメタデータ
- **scripts/**: 実行時のスクリプトのバックアップ

