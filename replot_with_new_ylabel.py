"""
縦軸を 'Average Normalized Squared Errors' に変更してプロットを再生成
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 15

# プロット対象のメタデータファイル
meta_files = [
    "/Users/fmjp/Desktop/lab/simu/tvgti_sem_exog_sparse/result/251203/exog_sparse_piecewise/images/piecewise_K=4_N=50_T=1000_maxweight=0.5_mulambda=0.9872907565269483_num_trials=100_q=5_r=50_rho=0.03956111183628299_seed=3_stde=0.05_meta.json",
    "/Users/fmjp/Desktop/lab/simu/tvgti_sem_exog_sparse/result/251201/exog_sparse_piecewise/images/piecewise_K=4_N=20_T=1000_maxweight=0.5_mulambda=0.8288774126612185_num_trials=100_q=5_r=50_rho=6.532215770893988e-06_seed=3_stde=0.05_meta.json",
]

output_dir = Path("/Users/fmjp/Desktop/lab/simu/tvgti_sem_exog_sparse/result/replot")
output_dir.mkdir(parents=True, exist_ok=True)

for meta_path in meta_files:
    with open(meta_path, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    N = config['N']
    T = config['T']
    K = config['K']
    
    metrics = data['results']['metrics']
    
    plt.figure(figsize=(10, 6))
    
    # 各手法のプロット
    if metrics.get('co') is not None:
        plt.plot(metrics['co'], color='blue', label='Correction Only')
    if metrics.get('pc') is not None:
        plt.plot(metrics['pc'], color='limegreen', label='Prediction Correction')
    if metrics.get('sgd') is not None:
        plt.plot(metrics['sgd'], color='cyan', label='SGD')
    if metrics.get('pg') is not None:
        plt.plot(metrics['pg'], color='magenta', label='ProxGrad')
    if metrics.get('pp') is not None:
        plt.plot(metrics['pp'], color='red', label='Proposed (PP)')
    
    plt.yscale('log')
    plt.xlim(left=0, right=T)
    plt.xlabel('$t$')
    plt.ylabel('Average Normalized Squared Errors')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    
    # 保存
    output_name = f"piecewise_K={K}_N={N}_T={T}_avg_nse.png"
    output_path = output_dir / output_name
    plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

print("\nDone!")






