"""KI（knowledge insulation）的可复现内核：一个 2x2 块注意力 + 手写反向。

对应文章：π 系列具身模型（03）KI —— 断在哪两个矩阵上。

为什么要手写反向：本实验要验证的核心事实是 **stop-gradient 之后某个梯度恒等于 0**。
有限差分只测「前向函数对参数的敏感度」，而 stop-gradient 在前向里是恒等映射——
用有限差分去测它，测出来的是「如果不断梯度会是多少」，永远不为 0。
所以只能手写反向，再用有限差分（在 sg 关闭时）去校准这份手写反向。

结构（对齐 π0 / KI 论文的 2x2 块分解）：

    P = ( P_bb    0  )      E = ( P_bb · V_b                        )
        ( P_ab  P_aa )          ( P_ab · sg(V_b) + P_aa · V_a       )

token 分三块：
    O —— 观测与指令（4 个物体 token + 1 个指令 token + 1 个回答 token），backbone
    R —— AR 动作 token 的槽位（5 个），backbone，KI 论文里 FAST token 就落在这儿
    A —— action expert 的连续动作 token（5 个）

掩码：O 双向；R 看 O + R；A 看 O + A，**A 看不见 R**（论文原文：两类动作表示互不 attend）；
任何 backbone token 都看不见 A（π0 的掩码已经如此）。
"""

import numpy as np

# ------------------------------------------------------------------ 尺寸
D_B, D_A, H = 32, 16, 8          # backbone 宽 / expert 宽 / 单头 head_dim（可不同宽、同 head_dim）
N_OBJ, N_R, N_A, D_ACT = 4, 5, 5, 7
N_O = N_OBJ + 2                  # 4 图像 + 指令 + 答案
N_B = N_O + N_R                  # backbone token 数（含 AR 槽位）
N_ALL = N_B + N_A
POS_LANG, POS_ANS = 4, 5         # O 内部的指令位与回答位
V, C_LANG = 16, 4                # 动作 token 词表 / 语言任务的类别数
R_A = 3                          # 动作真正依赖的读数个数（A 的秩）

R_SLICE = slice(N_O, N_B)
A_SLICE = slice(N_B, N_ALL)
A_ROWS = np.zeros(N_ALL, dtype=bool)
A_ROWS[N_B:] = True
NON_A_ROWS = ~A_ROWS


def build_mask():
    """注意力掩码。行 = 查询位置，列 = 被 attend 的位置。"""
    m = np.zeros((N_ALL, N_ALL), dtype=bool)
    O = slice(0, N_O)
    m[O, O] = True                       # 观测块内部双向
    m[R_SLICE, O] = True                 # AR 槽位看得见观测
    m[R_SLICE, R_SLICE] = True           # AR 槽位之间双向
    m[A_SLICE, O] = True                 # expert 看得见观测（读 K_b / V_b 的地方）
    m[A_SLICE, A_SLICE] = True           # expert 内部双向
    # 刻意留空：A 看不见 R（两类动作表示互不 attend）；backbone 永远看不见 A
    return m


MASK = build_mask()
BIG_NEG = -1e9


# ------------------------------------------------------------------ 数据
class Scene:
    """一次前向需要的全部合成数据。物体向量 -> 真值动作块 / 语言标签 / 离散动作 token。"""

    def __init__(self, seed, n, noise=0.6):
        rng = np.random.default_rng(seed)
        # 固定的「世界」参数（真值生成器），所有 recipe 共用。
        # P_gt 与 NAME 都取**正交**行：随机高斯行会让 4 个类别的 argmax 区域互相纠缠，
        # 连「在精确特征上跑多项 logistic」都只能到 0.85（实测），语言任务的天花板就被
        # 标签本身的几何卡住了，看不出配方差异。正交行把天花板抬到 0.97（实测）。
        Qg, _ = np.linalg.qr(rng.normal(size=(D_B, D_B)))
        self.P_gt = Qg[:, :C_LANG].T
        # 动作只依赖「语言任务用不到的那部分特征」：把 W_gt 的行空间投影到 P_gt 行空间的正交补上。
        # 这是「VLM 预训练里没有机器人数据、冻住的表示不够用」的合成对应物。
        Qp, _ = np.linalg.qr(self.P_gt.T)                 # (D_B, C_LANG)
        # 动作只依赖 R_A 个标量读数（U 的行），且 U ⊥ P_gt 的行空间：
        # 语言任务与动作任务需要的特征正交，这正是 rank-H 的 K/V 瓶颈能卡住的地方
        U = rng.normal(size=(R_A, D_B))
        U = U - (U @ Qp) @ Qp.T          # 投到 P_gt 行空间的正交补上
        Uq, _ = np.linalg.qr(U.T)        # 再正交化（投影后各行已不互相正交，
        self.U = Uq.T                    # 不补这一步 Cov(s) ≠ σ_s²·I，Σ_A 就配不准）
        # Vg 取正交阵的前 R_A 列：这样 s @ Vg 在 35 个坐标上的方差**处处相等**
        # （正交阵的行与列都是单位范数）。用 iid 高斯的话各维方差会按 chi^2_3 波动，
        # 低方差维上 A 变弱，u 就能从 x_t 里直接反解出来（见下面 UNPRED 那段）。
        Qv, _ = np.linalg.qr(rng.normal(size=(N_A * D_ACT, N_A * D_ACT)))
        self.Vg = Qv[:, :R_A].T
        # 物体的「名字」嵌入：跨样本一致，视觉编码器把图像映射到与它同一个语义空间，
        # 这是 VLM 预训练的基本假设，也是「按指令挑物体」能成立的前提
        Qn, _ = np.linalg.qr(rng.normal(size=(D_B, D_B)))
        self.NAME = Qn[:, :N_OBJ].T
        self.ans_tok = rng.normal(size=(D_B,)) / np.sqrt(D_B)
        self.Slot_emb = rng.normal(size=(N_R, D_B)) / np.sqrt(D_B)
        self.Wa_in = rng.normal(size=(D_ACT + 1, D_A)) / np.sqrt(D_ACT + 1)
        self.w_bin = rng.normal(size=(D_ACT,))
        self.w_bin /= np.linalg.norm(self.w_bin)

        self.n = n
        # 每个物体 = 名字嵌入 + 当次的外观扰动
        self.eps = noise * rng.normal(size=(n, N_OBJ, D_B)) / np.sqrt(D_B)
        self.obj = self.NAME[None, :, :] + self.eps
        self.c = rng.integers(0, N_OBJ, size=n)                      # 指令挑中的那个物体

        # 动作块只由「当次实例的外观扰动」决定：这部分信息语言任务用不到（U ⊥ P_gt），
        # 所以预训练过的 backbone 在冻住时把它压掉了 —— 这正是论文里「冻住不行」的合成对应物
        picked_eps = self.eps[np.arange(n), self.c]
        # 动作依赖**整幅场景**（4 个物体扰动的均值），语言依赖**指令挑中的那一个**。
        # 这不是随便选的：expert 的 query 只来自带噪动作与时间，不含指令，
        # 注意力只能摊在 4 个物体上；若动作只依赖被选中那个，信号会被另外 3 个的
        # 独立噪声稀释 4 倍（实测 act 全卡在 0.52，与「不看 backbone」的 0.523 一致）。
        self.s = self.eps.mean(axis=1) @ self.U.T                     # (n, R_A) 动作真正依赖的读数
        A_sig = (self.s @ self.Vg).reshape(n, N_A, D_ACT)             # 可预测部分（秩 R_A）
        # 不可预测部分：观测里根本没有的信息，谁也学不会（对应真实机器人里
        # 「同一场景下两次示范也不一样」）。它必须存在，否则 A 严格低秩，
        # 秩之外的方向上 A≡0，那些方向的 u 可以直接从 x_t 反解出来。
        self.unpred = 0.5                                            # 噪声标准差 / 信号标准差
        A_raw = A_sig + self.unpred * A_sig.std() * rng.normal(size=A_sig.shape)
        self.a_scale = float(1.0 / A_raw.std())                      # 归一化到 std≈1
        self.A = A_raw * self.a_scale
        # 流匹配的噪声 ω 取「与 A 同协方差」的高斯，而不是标准高斯。
        # 原因：x_t = τA + (1−τ)ω 里本来就带着 A 的信息，expert 哪怕完全不读 backbone
        # 也能反解出一部分 u —— 实测这会让 freeze 与 ki 的动作误差一样（0.428 / 0.429），
        # 指标彻底失去分辨力。cov(u, x_t) = (1−τ)Σ_ω − τΣ_A，令 Σ_ω = Σ_A 则在 τ=0.5 处
        # 严格为 0；A 与 ω 都是高斯，不相关即独立，捷径被完全堵死。
        # 代价：τ≠0.5 时仍有 (1−2τ) 量级的残余相关（τ→0 或 1 时最多白送 50%），
        # 所以评估一律钉在 τ=0.5（`sample_batch(fixed_tau=0.5)`）。
        # Σ_A = a_scale²(σ_s²·Π + (u·σ_A)²·I)，Π = Vg^T Vg 是秩 R_A 的投影。
        # 用闭式而不是经验协方差：A 是高斯的，协方差对得上就是严格独立。
        Cs = np.cov(self.s.T)                                        # 用样本协方差，避开抽样涨落
        Sig = self.a_scale ** 2 * (self.Vg.T @ Cs @ self.Vg
                                   + (self.unpred * A_sig.std()) ** 2 * np.eye(N_A * D_ACT))
        self.omega_L = np.linalg.cholesky(Sig)                        # Σ_A^{1/2}（35×35）
        # 语言标签 = 被选中物体「当次外观」在 P_gt 上的读数取 argmax。
        # 关键：标签只依赖 eps，不依赖 NAME —— 否则回答位自带的 NAME[c] 就是泄漏，
        # 模型不用看图也能猜对（负对照会打不下去）。同时在 x_b 里 NAME[c] 只作为
        # 「指令」出现在查询侧，模型必须靠它去 4 个物体里挑出对应的那个，
        # 这才是「语言跟随」在合成任务里的最小形式。
        self.y_lang = np.argmax(picked_eps @ self.P_gt.T, axis=-1)

        # 离散动作 token：把每个动作步沿固定方向投影后分箱（就是 02 篇讲的 naive binning）
        z = self.A @ self.w_bin                                      # (n, N_A)
        lo, hi = z.min(), z.max()
        edges = lo + (hi - lo) * np.arange(1, V) / V
        self.y_ar = np.clip(np.searchsorted(edges, z), 0, V - 1).astype(np.int64)
        self.edges = edges

    def x_b(self, idx):
        """backbone 的输入 token（嵌入层冻结，等价于预训练好的 SigLIP + 词嵌入）。"""
        n = len(idx)
        X = np.zeros((n, N_B, D_B))
        X[:, :N_OBJ, :] = self.obj[idx]                              # 4 个图像 token
        X[:, POS_LANG, :] = self.NAME[self.c[idx]]                   # 指令 token（= 名字嵌入）
        X[:, POS_ANS, :] = self.NAME[self.c[idx]] + 0.5 * self.ans_tok  # 回答位：带着指令
        X[:, R_SLICE, :] = self.Slot_emb[None, :, :]                 # AR 槽位
        return X

    def x_a(self, idx, tau, omega):
        """expert 的输入：[加噪动作, 流时间] 过一层固定投影。"""
        A = self.A[idx]
        x_t = tau[:, None, None] * A + (1.0 - tau[:, None, None]) * omega
        t_col = np.broadcast_to(tau[:, None, None], (x_t.shape[0], N_A, 1))
        raw = np.concatenate([x_t, t_col], axis=-1)          # (n, N_A, D_ACT+1)
        return np.tanh(raw @ self.Wa_in), x_t, A


def sample_batch(sc, idx, rng, fixed_tau=None):
    """抽一批，返回 forward 需要的全部量。fixed_tau 用于评估时钉住噪声水平。"""
    n = len(idx)
    if fixed_tau is None:
        tau = rng.beta(1.5, 1.0, size=n) * 0.999 + 0.001
    else:
        tau = np.full(n, fixed_tau)
    z = rng.normal(size=(n, N_A * D_ACT))
    omega = (z @ sc.omega_L.T).reshape(n, N_A, D_ACT)          # ω ~ N(0, Σ_A)
    X_b = sc.x_b(idx)
    X_a, x_t, A = sc.x_a(idx, tau, omega)
    return dict(
        X_b=X_b, X_a=X_a, tau=tau, u=omega - A, x_t=x_t,
        y_lang=sc.y_lang[idx], y_ar=sc.y_ar[idx], A=A,
        # 预训练阶段给 AR 头用的「语言类」目标：让离散头像真实系统那样从预训练的输出层起步
        y_ar_lang=np.broadcast_to(sc.y_lang[idx][:, None], (len(idx), N_R)),
    )


# ------------------------------------------------------------------ 参数
def init_params(rng):
    def w(shape):
        return rng.normal(size=shape) / np.sqrt(shape[0])

    return dict(
        Wq_b=w((D_B, H)), Wk_b=w((D_B, H)), Wv_b=w((D_B, H)), Wo_b=w((H, D_B)),
        Wq_a=w((D_A, H)), Wk_a=w((D_A, H)), Wv_a=w((D_A, H)), Wo_a=w((H, D_A)),
        W_lang=w((D_B, C_LANG)), W_ar=w((D_B, V)), W_flow=w((D_A, D_ACT)),
    )


BONE = ["Wq_b", "Wk_b", "Wv_b", "Wo_b", "W_lang", "W_ar"]      # backbone 侧
EXPERT = ["Wq_a", "Wk_a", "Wv_a", "Wo_a", "W_flow"]            # expert 侧
ALL_P = BONE + EXPERT


def zeros_grad(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


# ------------------------------------------------------------------ 前向
def forward(P, b, sg):
    """sg=True 时对 P_ab / V_b 的跨模块通路做 stop-gradient。返回 (输出, cache)。"""
    X_b, X_a = b["X_b"], b["X_a"]
    q_b = X_b @ P["Wq_b"]
    k_b = X_b @ P["Wk_b"]
    v_b = X_b @ P["Wv_b"]
    q_a = X_a @ P["Wq_a"]
    k_a = X_a @ P["Wk_a"]
    v_a = X_a @ P["Wv_a"]

    q = np.concatenate([q_b, q_a], axis=1)
    k = np.concatenate([k_b, k_a], axis=1)
    v = np.concatenate([v_b, v_a], axis=1)
    scale = H ** -0.5

    logits = np.einsum("bth,bsh->bts", q, k) * scale
    if sg:
        # 只有 A 行会用到「断掉 backbone 梯度」的那一版 logits
        k_sg = np.concatenate([k_b, k_a], axis=1)          # 前向值与 k 相同
        logits_sg = np.einsum("bth,bsh->bts", q, k_sg) * scale
        logits = np.where(A_ROWS[None, :, None], logits_sg, logits)
    else:
        k_sg = None
        logits_sg = None

    masked = np.where(MASK[None, :, :], logits, BIG_NEG)
    probs = _softmax(masked)

    enc_full = np.einsum("bts,bsh->bth", probs, v)
    if sg:
        v_sg = np.concatenate([v_b, v_a], axis=1)          # 前向值与 v 相同
        enc_sg = np.einsum("bts,bsh->bth", probs, v_sg)
        enc = np.where(A_ROWS[None, :, None], enc_sg, enc_full)
    else:
        enc_sg, v_sg = None, None
        enc = enc_full

    # 两套输出投影（宽度不同，所以各回各的宽）
    out_b = enc[:, :N_B, :] @ P["Wo_b"]
    out_a = enc[:, N_B:, :] @ P["Wo_a"]
    H_b = X_b + out_b                       # 残差
    H_a = X_a + out_a

    out = dict(
        logits_lang=H_b[:, POS_ANS, :] @ P["W_lang"],
        logits_ar=H_b[:, R_SLICE, :] @ P["W_ar"],
        pred_flow=H_a @ P["W_flow"],
    )
    cache = dict(q=q, k=k, v=v, k_sg=k_sg, v_sg=v_sg, probs=probs, enc=enc,
                 enc_full=enc_full, enc_sg=enc_sg, H_b=H_b, H_a=H_a,
                 X_b=X_b, X_a=X_a, sg=sg, out=out)
    return out, cache


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def ce_loss(logits, y):
    """返回 (均值损失, dlogits)。"""
    n = logits.shape[0]
    z = logits - logits.max(axis=-1, keepdims=True)
    p = _softmax(z)
    m = logits.shape[1] if logits.ndim == 3 else 1
    loss = -np.mean(np.log(np.take_along_axis(p, y[..., None], axis=-1) + 1e-12))
    d = p.copy()
    np.put_along_axis(d, y[..., None], np.take_along_axis(d, y[..., None], axis=-1) - 1.0, axis=-1)
    d /= (n * (m if logits.ndim == 3 else 1))
    return loss, d


def mse_loss(pred, target):
    numel = pred.size
    loss = np.mean(np.square(pred - target))
    return loss, 2.0 * (pred - target) / numel


# ------------------------------------------------------------------ 反向
def backward(P, b, cache, g):
    """g: dict，各输出的上游梯度。返回 grads。"""
    G = zeros_grad(P)
    H_b, H_a = cache["H_b"], cache["H_a"]
    X_b, X_a = cache["X_b"], cache["X_a"]

    dH_b = np.zeros_like(H_b)
    dH_a = np.zeros_like(H_a)

    # ---- 三个头
    dH_b[:, POS_ANS, :] += g["logits_lang"] @ P["W_lang"].T
    G["W_lang"] += H_b[:, POS_ANS, :].T @ g["logits_lang"]

    d_ar = g["logits_ar"]                                    # (n, N_R, V)
    dH_b[:, R_SLICE, :] += d_ar @ P["W_ar"].T
    G["W_ar"] += H_b[:, R_SLICE, :].reshape(-1, D_B).T @ d_ar.reshape(-1, V)

    d_flow = g["pred_flow"]                                  # (n, N_A, D_ACT)
    dH_a += d_flow @ P["W_flow"].T
    G["W_flow"] += H_a.reshape(-1, D_A).T @ d_flow.reshape(-1, D_ACT)

    # ---- 输出投影（残差：dout = dH）
    enc_b, enc_a = cache["enc"][:, :N_B, :], cache["enc"][:, N_B:, :]
    G["Wo_b"] += enc_b.reshape(-1, H).T @ dH_b.reshape(-1, D_B)
    G["Wo_a"] += enc_a.reshape(-1, H).T @ dH_a.reshape(-1, D_A)
    denc = np.zeros_like(cache["enc"])
    denc[:, :N_B, :] += dH_b @ P["Wo_b"].T
    denc[:, N_B:, :] += dH_a @ P["Wo_a"].T

    # ---- enc 的两条支路：sg 时 A 行走 sg 版，其余走普通版；非 sg 时全走普通版
    probs, v = cache["probs"], cache["v"]
    sg = cache["sg"]
    denc_full = np.where(A_ROWS[None, :, None], 0.0, denc) if sg else denc
    dprobs = np.einsum("bth,bsh->bts", denc_full, v)
    dv = np.einsum("bts,bth->bsh", probs, denc_full)

    if sg:
        denc_sg = np.where(A_ROWS[None, :, None], denc, 0.0)
        dprobs += np.einsum("bth,bsh->bts", denc_sg, cache["v_sg"])
        # sg(V_b) 这一支不给 V_b 任何梯度；只把 expert 自己的 V_a 接上
        dv[:, N_B:, :] += np.einsum("bts,bth->bsh", probs, denc_sg)[:, N_B:, :]

    # ---- softmax（probs 本身是用混合后的 logits 算的，所以这里只有一份）
    dlogits = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
    dlogits = np.where(MASK[None, :, :], dlogits, 0.0)

    scale = H ** -0.5
    dlogits_full = np.where(A_ROWS[None, :, None], 0.0, dlogits) if sg else dlogits
    dq = np.einsum("bts,bsh->bth", dlogits_full, cache["k"]) * scale
    dk = np.einsum("bts,bth->bsh", dlogits_full, cache["q"]) * scale

    if sg:
        dlogits_sg = np.where(A_ROWS[None, :, None], dlogits, 0.0)
        dq += np.einsum("bts,bsh->bth", dlogits_sg, cache["k_sg"]) * scale
        # sg(K_b) 这一支不给 K_b 任何梯度；只把 expert 自己的 K_a 接上
        dk[:, N_B:, :] += np.einsum("bts,bth->bsh", dlogits_sg, cache["q"])[:, N_B:, :] * scale

    # ---- Q/K/V 投影
    G["Wq_b"] += X_b.reshape(-1, D_B).T @ dq[:, :N_B, :].reshape(-1, H)
    G["Wq_a"] += X_a.reshape(-1, D_A).T @ dq[:, N_B:, :].reshape(-1, H)
    G["Wk_b"] += X_b.reshape(-1, D_B).T @ dk[:, :N_B, :].reshape(-1, H)
    G["Wk_a"] += X_a.reshape(-1, D_A).T @ dk[:, N_B:, :].reshape(-1, H)
    G["Wv_b"] += X_b.reshape(-1, D_B).T @ dv[:, :N_B, :].reshape(-1, H)
    G["Wv_a"] += X_a.reshape(-1, D_A).T @ dv[:, N_B:, :].reshape(-1, H)

    return G


def grads(P, b, sg, w_flow=1.0, w_ar=1.0, w_lang=0.0, ar_tgt=None):
    """一步前向 + 反向，返回 (损失 dict, grads)。ar_tkg 可把 AR 目标换成预训练阶段的替代目标。"""
    tgt = b["y_ar"] if ar_tgt is None else ar_tgt
    out, cache = forward(P, b, sg)
    losses = {}
    g = {}
    if w_lang > 0:
        losses["lang"], g["logits_lang"] = ce_loss(out["logits_lang"], b["y_lang"])
        g["logits_lang"] *= w_lang
    else:
        g["logits_lang"] = np.zeros_like(out["logits_lang"])
    if w_ar > 0:
        losses["ar"], g["logits_ar"] = ce_loss(out["logits_ar"], tgt)
        g["logits_ar"] *= w_ar
    else:
        g["logits_ar"] = np.zeros_like(out["logits_ar"])
    if w_flow > 0:
        losses["flow"], g["pred_flow"] = mse_loss(out["pred_flow"], b["u"])
        g["pred_flow"] *= w_flow
    else:
        g["pred_flow"] = np.zeros_like(out["pred_flow"])
    return losses, backward(P, b, cache, g), out, cache


# ------------------------------------------------------------------ 优化器
class Adam:
    def __init__(self, P, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = zeros_grad(P)
        self.v = zeros_grad(P)
        self.t = 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, P, G, keys):
        self.t += 1
        for k in keys:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * G[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * np.square(G[k])
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class SGD:
    """纯 SGD（无动量）。只用来做 B 组的负对照：它没有 Adam 的逐参数归一化，
    α 会原封不动地放大步长，正好把「α 不变性到底是谁给的」切开。"""

    def __init__(self, P, lr=1e-2):
        self.lr = lr
        self.acc = zeros_grad(P)          # 累计位移，用来核对「位移比 = α」

    def step(self, P, G, keys):
        for k in keys:
            d = self.lr * G[k]
            P[k] -= d
            self.acc[k] = self.acc[k] + d


def clone(P):
    return {k: v.copy() for k, v in P.items()}


def acc(logits, y):
    if logits.ndim == 3:
        return float(np.mean(np.argmax(logits, axis=-1) == y))
    return float(np.mean(np.argmax(logits, axis=-1) == y))
