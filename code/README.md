exog_sparse_sim: 外因性あり・スパース性（Sのみ）を想定したシミュレーション一式。

前提
- モデル: Y(t) = S(t) Y(t) + B U(t) + noise,  with diag(S)=0, Sは非対称, Bは時不変の対角行列（符号制約なし）
- スパース性: Sのみ（スパース率は可変）
- 指標: Frobenius誤差（S推定）

構成
- data_gen.py: Sの時変系列とU(外因性)を生成し、Yを作る
- models/pp_exog.py: PPによる S, B 同時推定
- run_piecewise.py / run_linear.py / run_brownian.py: 生成と推定の実行
- hypara_tuning_pp.py: PPのハイパラ探索（r,q,mu_lambda,rho, sparsity）

実行例
```bash
~/venv/default/bin/python exog_sparse_sim/run_piecewise.py
~/venv/default/bin/python exog_sparse_sim/run_linear.py
~/venv/default/bin/python exog_sparse_sim/run_brownian.py
~/venv/default/bin/python exog_sparse_sim/hypara_tuning_pp.py
```

注意
- 既存コードは変更せず、本ディレクトリ以下に新規実装。
- 既存 utils の関数は必要に応じてローカルに同等処理を実装し、今回仕様（スペクトル半径の強制なし）に合わせています。


