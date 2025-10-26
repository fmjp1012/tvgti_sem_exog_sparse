import numpy as np
from numpy.linalg import norm


def soft_threshold_masked(W, lam, eta):
    """
    マスク付き ℓ1 近接写像（式 (masked-soft)）
    左ブロック S (W[:, :N]) のみソフトしきい値、右ブロック T (W[:, N:]) はそのまま通す。

    prox_{eta * lam * ||·||_1 on S}(W)
    """
    N = W.shape[0]
    S_block = W[:, :N]
    T_block = W[:, N:]

    # ソフトしきい値 shrinkage: sign(x)*max(|x|-tau,0)
    tau = eta * lam
    S_shrunk = np.sign(S_block) * np.maximum(np.abs(S_block) - tau, 0.0)

    # T 側は正則化しないのでそのまま
    T_pass = T_block.copy()

    # 再結合
    W_out = np.concatenate([S_shrunk, T_pass], axis=1)
    return W_out


def project_structure(W):
    """
    構造射影 𝒫_𝒟(W)
    - S の対角成分は 0 にする
    - T は対角成分のみ残し、それ以外は 0 にする
      （つまり各ノードごとにそのノード自身の外因性 z_i,t だけ許す）

    W = [S | T], 形状 (N, 2N)
    """
    N = W.shape[0]
    S_block = W[:, :N].copy()
    T_block = W[:, N:].copy()

    # S の自己ループ禁止: diag(S)=0
    np.fill_diagonal(S_block, 0.0)

    # T は対角行列のみ残す
    T_diag_only = np.zeros_like(T_block)
    np.fill_diagonal(T_diag_only, np.diag(T_block))

    W_proj = np.concatenate([S_block, T_diag_only], axis=1)
    return W_proj


def build_window_blocks(X, Z, center_idx, r):
    """
    ウィンドウ行列 X_iota, Z_iota を作る（式 (X) に対応）
    - center_idx = ι
    - 取り出す区間は [ι-r+1, ι] の時系列スライス（時間が足りなければ先頭まで）
    - 列方向が時間方向。並び順は古い→新しいでも新しい→古いでも
      ||·||_F^2 や勾配計算には影響しないので、ここでは古い→新しいで統一

    戻り値:
        X_win: (N, L)
        Z_win: (N, L)
    """
    start = max(0, center_idx - r + 1)
    end = center_idx + 1  # python sliceはend非含まないので+1
    X_win = X[:, start:end]
    Z_win = Z[:, start:end]
    return X_win, Z_win


class APSPExogenousSEM:
    """
    並列サブグラディエント射影 (APSP) + ℓ1近接 + 構造射影 による SEM 推定器
    ------------------------------------------------------------------
    本クラスは LaTeX の以下の流れを忠実に実装するもの:
      - 問題設定の SEM: x_t = S x_t + T z_t + ε_t           (式 (SEM_matrix))
      - レベル集合 C_ι(ρ) と g_ι(W)                        (式 (C), (g), (C_level))
      - 劣勾配射影 T_sp(g_ι)(W)                            (式 (Tsp))
      - 並列合成と緩和 (widehat{W}_t, μ_t, ℳ_t)             (式 (parallel), (eq:apsp_M))
      - ℓ1 近接写像 (masked soft-threshold)                (式 (eq:masked-soft))
      - 構造射影 𝒫_𝒟(W) で diag(S)=0, Tは対角のみ

    オンライン運用イメージ：
      各時刻 t で最新データ周りの q 個のブロック {ι = t, t-1, ..., t-q+1}
      それぞれに対し制約 C_ι(ρ) を作って並列射影 → 平均 → 緩和
      → ℓ1近接 → 構造射影 → 次の推定 W_{t+1}

    パラメータ
    ----------
    N : int
        ノード数
    r : int
        各ブロックのウィンドウ長 (式 (X) の r)
    q : int
        並列に使うブロック数 (#processors)
    rho : float
        レベル集合のしきい値 ρ
    mu_lambda : float
        緩和ステップ係数用のスカラー μ_λ.
        実装では μ_t = μ_λ * ℳ_t として使う。
        （理論上 μ_t ∈ (0, 2 ℳ_t) なので、μ_λ はだいたい (0,2) にチューニングするイメージ）
    lambda_S : float
        ℓ1 正則化係数（S ブロックのみ）
    eta : float
        近接写像のステップ幅 η (>0)
    S_init : np.ndarray shape (N,N)
        初期隣接行列推定
    T_init : np.ndarray shape (N,N), 対角行列想定
        初期外因性ゲイン行列推定（対角のみ想定だが行列で渡す）

    主要メソッド
    ------------
    step_update(t, X, Z):
        時刻 t までの観測を使って W を 1 ステップ更新
    run(X, Z):
        t=0..T-1 全時刻に対して逐次 step_update し，推定履歴を返す
    """

    def __init__(
        self,
        N: int,
        r: int,
        q: int,
        rho: float,
        mu_lambda: float,
        lambda_S: float,
        eta: float,
        S_init: np.ndarray,
        T_init: np.ndarray,
    ):
        self.N = N
        self.r = r
        self.q = q
        self.rho = rho
        self.mu_lambda = mu_lambda
        self.lambda_S = lambda_S
        self.eta = eta

        # W = [S | T] in R^{N x (2N)}
        S0 = S_init.copy()
        np.fill_diagonal(S0, 0.0)  # diag(S)=0
        T0 = T_init.copy()
        # T は対角のみ残す
        T0_diag = np.zeros_like(T0)
        np.fill_diagonal(T0_diag, np.diag(T0))

        self.W = np.concatenate([S0, T0_diag], axis=1)  # shape (N, 2N)

    def _block_projection(self, X_block, Z_block):
        """
        1つのブロック (ι) に対して、劣勾配射影 T_sp(g_ι)(W) を計算する部分
        式:
          g_ι(W) = 0.5 || X_ι - W U_ι ||_F^2 - ρ
          ∇g_ι(W) = (W U_ι - X_ι) U_ι^T
          T_sp(g_ι)(W) = W - (g_ι / ||∇g_ι||^2) ∇g_ι   if g_ι>0
                         W                              otherwise
        """
        # U_ι = [X_ι; Z_ι] ∈ R^{2N x L}
        U_block = np.concatenate([X_block, Z_block], axis=0)  # (2N, L)
        # 予測 WX
        pred = self.W @ U_block  # (N, L)
        residual = X_block - pred  # (N, L)

        g_val = 0.5 * (norm(residual) ** 2) - self.rho

        if g_val > 0:
            grad = (pred - X_block) @ U_block.T  # (N,2N)
            grad_norm_sq = norm(grad) ** 2
            if grad_norm_sq == 0.0:
                # 変化なし（数値的な特異ケース）
                return self.W.copy(), g_val
            step = g_val / grad_norm_sq
            W_proj = self.W - step * grad
            return W_proj, g_val
        else:
            # すでにブロック制約 C_ι(ρ) 内
            return self.W.copy(), g_val

    def step_update(self, t: int, X: np.ndarray, Z: np.ndarray):
        """
        時刻 t の観測までを使って 1 ステップ更新し、self.W を上書きする。
        LaTeX 式 (parallel), (eq:apsp_M), (eq:masked-soft) の流れに相当。

        引数
        ----
        t : int
            現在の時刻インデックス (0-based を想定)
        X : np.ndarray, shape (N, T_total)
            観測 x_t の時系列 (列が時刻)
        Z : np.ndarray, shape (N, T_total)
            外因性 z_t の時系列
        """
        N = self.N

        # --- 1. 並列インデックス集合 I_t = {t, t-1, ..., t-q+1} （境界は >=0）
        idx_list = list(range(t, max(-1, t - self.q), -1))
        # t, t-1, ..., t-q+1 but stop at -1
        # 実際に存在するインデックスのみ
        idx_list = [idx for idx in idx_list if idx >= 0]
        Lq = len(idx_list)
        w = 1.0 / Lq  # 均等重み w_ι^{(t)}

        # --- 2. 各ブロックのサブグラ射影 T_sp(g_ι)(W) を並列的に計算
        W_proj_list = []
        g_list = []

        for iota in idx_list:
            X_blk, Z_blk = build_window_blocks(X, Z, iota, self.r)
            W_iota_proj, g_iota_val = self._block_projection(X_blk, Z_blk)
            W_proj_list.append(W_iota_proj)
            g_list.append(g_iota_val)

        # --- 3. 加重平均 \hat{W}_t = sum w_ι T_sp(g_ι)(W)
        W_hat = np.zeros_like(self.W)
        for Wp in W_proj_list:
            W_hat += w * Wp

        # --- 4. 交差判定: すべてのブロックで g_ι <= 0 か？
        all_satisfied = all(g_val <= 0 for g_val in g_list)

        # --- 5. M_t の計算 (式 (eq:apsp_M) の分数部分)
        # num = Σ w_ι ||Wp - W||^2
        # den = ||(Σ w_ι Wp) - W||^2 = ||W_hat - W||^2
        num = 0.0
        for Wp in W_proj_list:
            num += w * (norm(Wp - self.W) ** 2)

        den = norm(W_hat - self.W) ** 2

        if all_satisfied:
            # ℳ_t = 1 とする扱い（理論式では all satisfied のとき ℳ_t=1）
            M_t = 1.0
        else:
            if den == 0.0:
                M_t = 0.0
            else:
                M_t = num / den

        # --- 6. 緩和ステップ
        # W_tilde = W + μ_t (W_hat - W)
        # μ_t ∈ (0, 2ℳ_t). ここでは μ_t = mu_lambda * M_t.
        mu_t = self.mu_lambda * M_t
        W_tilde = self.W + mu_t * (W_hat - self.W)

        # --- 7. ℓ1 近接（S ブロックのみソフトしきい値; 式 (eq:masked-soft)）
        W_after_l1 = soft_threshold_masked(W_tilde, lam=self.lambda_S, eta=self.eta)

        # --- 8. 構造射影（diag(S)=0, T=diag(T); 「構造制約と ℓ1 近接の組み込み」）
        W_new = project_structure(W_after_l1)

        # --- 9. 更新
        self.W = W_new

        # 返り値として現在の S, T も返しておくと便利
        S_est = self.W[:, :N].copy()
        T_est = self.W[:, N:].copy()  # 対角行列
        return S_est, T_est

    def run(self, X: np.ndarray, Z: np.ndarray):
        """
        全時系列 (t = 0,...,T-1) で逐次 step_update を回す
        戻り値: 推定履歴 (S_list, T_list)
          S_list[t] = 推定 S_t (N,N)
          T_list[t] = 推定 T_t (N,N, 対角)
        """
        T_total = X.shape[1]
        S_list = []
        T_list = []

        for t in range(T_total):
            S_est, T_est = self.step_update(t, X, Z)
            S_list.append(S_est)
            T_list.append(T_est)

        return S_list, T_list

    def get_current_ST(self):
        """
        現在の推定値 (S, T) を返す
        S: (N,N), T: (N,N)
        """
        N = self.N
        S_est = self.W[:, :N].copy()
        T_est = self.W[:, N:].copy()
        return S_est, T_est


# ============================================================
# 参考: SNR制御つきデータ生成（アルゴリズム alg:snr-sim）
# ============================================================

def generate_sem_data(S_true, T_true, R_z, snr_target, T_len, rng=None):
    """
    LaTeX中の SNR 制御付きデータ生成 (Algorithm 'snr-sim') を実装する。
    モデル:
        x_t = (I - S_true)^{-1} ( T_true z_t + ε_t )

    入力
    -----
    S_true : (N,N)
        真の隣接行列 S（スペクトル半径 < 1 を仮定）
    T_true : (N,N)
        真の外因性ゲイン行列（対角を想定）
    R_z : (N,N)
        外因性入力 z_t の共分散行列
    snr_target : float
        目標 SNR_* (式 (eq:sigma2-for-target-snr) の SNR_★)
    T_len : int
        サンプル長 T
    rng : np.random.Generator or None
        乱数生成器（指定がなければ np.random.default_rng()）

    戻り値
    -------
    X : (N,T_len)
        観測系列 x_t の行列
    Z : (N,T_len)
        外因性系列 z_t の行列
    sigma2 : float
        使用したノイズ分散 σ^2
    """
    if rng is None:
        rng = np.random.default_rng()

    N = S_true.shape[0]
    I = np.eye(N)
    A = np.linalg.inv(I - S_true)  # A = (I - S)^{-1}

    # 式 (eq:sigma2-for-target-snr)
    # sigma^2 = tr(A T R_z T A^T) / ( SNR_* * tr(A A^T) )
    ATRzTA_T = A @ T_true @ R_z @ T_true @ A.T
    numerator = np.trace(ATRzTA_T)

    AAT = A @ A.T
    denom = snr_target * np.trace(AAT)

    sigma2 = float(numerator / denom)

    # 生成ループ (Algorithm alg:snr-sim)
    X = np.zeros((N, T_len))
    Z = np.zeros((N, T_len))

    for t in range(T_len):
        # z_t ~ N(0, R_z)
        z_t = rng.multivariate_normal(mean=np.zeros(N), cov=R_z)
        # ε_t ~ N(0, σ^2 I)
        eps_t = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=N)

        x_t = A @ (T_true @ z_t + eps_t)

        X[:, t] = x_t
        Z[:, t] = z_t

    return X, Z, sigma2


# 互換ラッパー: 既存コード（hyperparam_tuning.py）が期待する API
class PPExogenousSEM:
    """
    互換レイヤー。古い PPExogenousSEM API を APSPExogenousSEM に委譲する。

    期待される使用方法（hyperparam_tuning.py より）:
        model = PPExogenousSEM(N, S0, b0, r, q, rho, mu_lambda)
        S_list, _ = model.run(X, Z)

    注意: APSP 実装では ℓ1 近接に lambda_S, eta が必要だが、
    チューニング対象外のためここではデフォルト値を用いる。
      - lambda_S = 0.0 (ソフトしきい値で縮まない)
      - eta = 1.0
    """

    def __init__(
        self,
        N: int,
        S_init: np.ndarray,
        b_init: np.ndarray,
        *,
        r: int,
        q: int,
        rho: float,
        mu_lambda: float,
    ) -> None:
        if b_init.ndim == 1:
            T_init = np.diag(b_init)
        else:
            T_init = b_init

        # 互換目的のデフォルト
        lambda_S = 0.0
        eta = 1.0

        self._apsp = APSPExogenousSEM(
            N=N,
            r=r,
            q=q,
            rho=rho,
            mu_lambda=mu_lambda,
            lambda_S=lambda_S,
            eta=eta,
            S_init=S_init,
            T_init=T_init,
        )

    def run(self, X: np.ndarray, Z: np.ndarray):
        return self._apsp.run(X, Z)
