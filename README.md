# Time-Varying Graph Topology Identification with Exogenous Inputs

外因性入力を含む時変構造方程式モデル（Time-Varying SEM）におけるグラフ構造推定アルゴリズムのシミュレーション実装プロジェクトです。
主に Prediction-Correction (PC) 法および Proximal Projection (PP) 法を用いた手法の検証・比較を行います。

## プロジェクト構造

- **`code/`**: 実験実行用スクリプト、データ生成、ハイパーパラメータチューニング
  - **`config.py`**: 🔧 **すべてのシミュレーション設定を一元管理**
  - `run_*.py`: 各種シナリオの実行スクリプト
  - `data_gen.py`: シミュレーションデータ生成
  - `tune_and_run.py`: チューニングと実行を一括で行うスクリプト
  - `hyperparam_tuning.py`: ハイパーパラメータチューニング
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
- optuna

## 使用方法

### ⚠️ 重要: 設定は `code/config.py` で一元管理

**すべてのシミュレーション条件は `code/config.py` で設定してください。**
コマンドライン引数による設定は非推奨です。

### 設定の変更方法

`code/config.py` を開いて、`CONFIG` インスタンスの値を編集します:

```python
# code/config.py の CONFIG を編集

CONFIG = SimulationConfig(
    # 実行する手法
    methods=MethodFlags(
        pp=True,   # Proposed (PP)
        pc=True,   # Prediction Correction
        co=True,   # Correction Only
        sgd=True,  # SGD
        pg=False,  # Proximal Gradient (バッチ法)
    ),
    
    # シナリオ共通パラメータ
    common=CommonParams(
        N=20,           # ノード数
        T=1000,         # 時系列長
        sparsity=0.7,   # スパース性
        max_weight=0.5, # 最大重み
        std_e=0.05,     # ノイズ標準偏差
        seed=3,         # 乱数シード
    ),
    
    # Piecewiseシナリオのパラメータ
    piecewise=PiecewiseParams(
        K=4,  # 変化点の数
    ),
    
    # チューニング設定
    tuning=TuningParams(
        tuning_trials=300,        # Optunaの試行回数
        tuning_runs_per_trial=1,  # 各試行での実行回数
        truncation_horizon=400,   # 打ち切りステップ数
    ),
    
    # 実行設定
    run=RunParams(
        num_trials=100,  # モンテカルロ試行回数
    ),
    
    # 実行モード
    skip_tuning=False,      # True: チューニングをスキップ
    skip_simulation=False,  # True: シミュレーションをスキップ
    hyperparam_json=None,   # 既存のハイパラJSONを使用する場合のパス
)
```

### 1. 現在の設定を確認

```bash
make config
# または
/Users/fmjp/venv/default/bin/python code/config.py
```

### 2. 実験の実行

#### Piecewise シナリオ（チューニング → シミュレーション）
```bash
make piecewise
# または
/Users/fmjp/venv/default/bin/python -m code.tune_and_run piecewise
```

#### Linear シナリオ（チューニング → シミュレーション）
```bash
make linear
# または
/Users/fmjp/venv/default/bin/python -m code.tune_and_run linear
```

### 3. シミュレーションのみ実行

既存のハイパラJSONを使用する場合:

```bash
# config.py の hyperparam_json にパスを設定するか、コマンドラインで指定
/Users/fmjp/venv/default/bin/python -m code.run_piecewise --hyperparam_json path/to/hyperparams.json
```

### 設定項目一覧

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

### 探索範囲のカスタマイズ

ハイパーパラメータチューニングの探索範囲も `config.py` で設定できます:

```python
# config.py の search_spaces を編集
search_spaces=SearchSpaces(
    pp=PPSearchSpace(
        rho=SearchRange(low=1e-6, high=1e-1, log=True),
        mu_lambda=SearchRange(low=1e-4, high=1.0, log=True),
    ),
    pc=PCSearchSpace(
        lambda_reg=SearchRange(low=1e-5, high=1e-2, log=True),
        # ... 他のパラメータ
    ),
),
```

## 出力

実験結果は `result/` ディレクトリ内に日付・シナリオごとのフォルダで保存されます。
- **images/**: 結果のプロット画像
- **meta.json**: 実験設定や結果の数値データを含むメタデータ
- **scripts/**: 実行時のスクリプトのバックアップ（config.py を含む）
