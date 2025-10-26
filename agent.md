## シナリオ一覧
- **piecewise**: 区分一定 S(t)。主要条件: N, T, sparsity, max_weight, std_e, K, seed, num_trials, run_{pc,co,sgd,pp}, hyperparam_json
- **linear**: 端点間線形補間 S(t)。主要条件: N, T, sparsity, max_weight, std_e, seed, num_trials, run_{pc,co,sgd,pp}, hyperparam_json
- **brownian**: 区分ごとに S にブラウン運動ノイズ。主要条件: N, T, sparsity, max_weight, std_e, K, std_S, seed, num_trials, run_{pc,co,sgd,pp}

## 実験条件（共通）
- **N**: ノード数
- **T**: 時系列長
- **sparsity**: 構造行列のゼロ率
- **max_weight**: 構造行列の非ゼロ重み最大絶対値
- **std_e**: 観測ノイズ標準偏差
- **seed**: 乱数シード基点
- **num_trials**: 実験試行回数
- **hyperparam_json**: 手法ハイパラJSONパス
- **run_pc/no_pc, run_co/no_co, run_sgd/no_sgd, run_pp/no_pp**: 各手法の実行フラグ

## シナリオ固有の条件
- **piecewise**: K（区分数）
- **brownian**: K（区分数）, std_S（S の区分間ノイズ強度）

## 手法ハイパーパラメータ
- **PP**: r, q, rho, mu_lambda
- **PC**: lambda_reg, alpha, beta, gamma, P, C
- **CO**: beta_co
- **SGD**: beta_sgd

## チューニング関連
- **tuning_trials**: 1手法あたりの試行回数
- **tuning_runs_per_trial**: 各試行内の評価反復数
- （内部）**truncation_horizon**: 評価で用いる時系列の最大長
- **seed**: チューニング用シード

## データ生成オプション（code/data_gen.py）
- **s_type**: 構造行列生成タイプ ("regular"|"random")
- **t_min, t_max**: 外因性影響行列 T の対角成分一様分布範囲
- **z_dist**: 外因性入力 Z の分布 ("uniform01"|"normal")
- （brownian 追加）**std_S**: S の区分間ノイズ強度

## 出力メタ（保存名・記録のキー）
- **maxweight** = max_weight, **stde** = std_e, **mulambda** = mu_lambda

## 実行例（Python 実行パスは `/User/fmjp/venv/default/bin/python`）
- **チューニング+実行（piecewise）**
```bash
/User/fmjp/venv/default/bin/python -m code.tune_and_run piecewise \
  --N 20 --T 1000 --sparsity 0.7 --max_weight 0.5 --std_e 0.05 --K 4 \
  --tuning_trials 30 --tuning_runs_per_trial 5
```
- **実行のみ（linear）**
```bash
/User/fmjp/venv/default/bin/python -m code.run_linear \
  --hyperparam_json <保存したJSONパス> \
  --num_trials 100 --N 20 --T 1000 --sparsity 0.6 --max_weight 0.5 --std_e 0.05
```
- **brownian 実行（手動）**
```bash
/User/fmjp/venv/default/bin/python -m code.run_brownian
```
