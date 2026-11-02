"""KI（knowledge insulation）companion lab —— 断在哪两个矩阵上，断了之后 α 还是不是旋钮。

对应文章：π 系列具身模型（03）KI。

设计原则：**只主张能被核到的事实**。论文层面的数字（7.5 倍步数、LIBERO 表、DROID 分数）
是引用，本实验不重算；这里只负责四件能在 CPU 上钉死的事：

  A 组  梯度通路的结构：为什么断 K_b / V_b 就够了，Q_b 和输出投影为什么不用管
  B 组  α 的角色：断梯度之后 backbone 的训练轨迹与 α 完全无关（逐位）
  C 组  语言跟随：几种配方各自拿什么换了什么
  D 组  信息瓶颈：断梯度成立的前提是 backbone 还得被离散动作训练着

纯 numpy / CPU / 固定 seed / 每组末尾断言 / 最后一个 cell 打出断言总数。
反向是手写的（stop-gradient 在前向里是恒等映射，有限差分测不出它），
再用有限差分在 sg 关闭时校准这份手写反向。

一个必须说清的前提：本实验验的是**机制**（哪条梯度到得了哪个权重、代价是什么），
不是论文的最终成绩。玩具 backbone 只有 32 维、K/V 瓶颈秩 8，三千步就能从零重新
学会动作表示，所以「KI 收敛更快」这类规模相关的结论在这里体现不出来（§D 会给出
实测的反向结果并说明原因）。
"""

import numpy as np

import ki_lib as K

SEED = 20261103
N_TRAIN, N_EVAL = 2048, 512
N_ASSERT = 0


def ok(cond, msg):
    global N_ASSERT
    assert cond, f"FAILED: {msg}"
    N_ASSERT += 1
    print(f"  [ok] {msg}")


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ================================================================== A 组
section("A 组：梯度通路的结构（单批，无需训练）")

rng = np.random.default_rng(SEED)
sc = K.Scene(seed=4242, n=N_TRAIN + N_EVAL)
PA = K.init_params(np.random.default_rng(1))
bA = K.sample_batch(sc, np.arange(64), rng)

out_n, cache_n = K.forward(PA, bA, sg=False)
out_s, cache_s = K.forward(PA, bA, sg=True)

M = K.MASK
ok(not M[:K.N_B, K.N_B:].any(), "掩码：backbone 的任何位置都看不见 expert token（P_ba = 0）")
ok(not M[K.N_B:, K.N_O:K.N_B].any(), "掩码：expert 看不见 AR 动作 token 槽位（两类动作表示互不 attend）")
ok(M[K.N_B:, :K.N_O].any() and M[:K.N_O, :K.N_O].all(),
   "掩码：expert 看得见观测块，观测块内部双向")

_, Gn, _, _ = K.grads(PA, bA, sg=False, w_flow=1.0, w_ar=0.0, w_lang=0.0)
_, Gs, _, _ = K.grads(PA, bA, sg=True, w_flow=1.0, w_ar=0.0, w_lang=0.0)

gb_k, gb_v = np.abs(Gn["Wk_b"]).max(), np.abs(Gn["Wv_b"]).max()
ok(gb_k > 1e-4 and gb_v > 1e-4,
   f"不断梯度时，流匹配损失确实会流进 backbone：|dWk_b|={gb_k:.3e}  |dWv_b|={gb_v:.3e}")
ok(np.abs(Gn["Wq_b"]).max() == 0.0 and np.abs(Gn["Wo_b"]).max() == 0.0,
   "不断梯度时 ∂L_flow/∂Wq_b 与 ∂L_flow/∂Wo_b 本来就是 0（backbone 行读不到 action token）")
ok(np.abs(Gs["Wk_b"]).max() == 0.0 and np.abs(Gs["Wv_b"]).max() == 0.0,
   "断梯度后 ∂L_flow/∂Wk_b 与 ∂L_flow/∂Wv_b 严格为 0（逐位）")
ok(np.abs(Gs["Wq_a"]).max() > 1e-6,
   f"断梯度后 expert 自己的 query 照常学习：|dWq_a|={np.abs(Gs['Wq_a']).max():.3e}")

dmax = max(np.abs(out_n[k] - out_s[k]).max() for k in out_n)
ok(dmax == 0.0, f"断梯度不改变前向：三个头输出的逐位最大差 = {dmax}")
dmax_e = max(np.abs(Gs[k] - Gn[k]).max() for k in K.EXPERT)
ok(dmax_e == 0.0, f"断梯度不改变 expert 侧梯度：逐位最大差 = {dmax_e}")

_, Gar, _, _ = K.grads(PA, bA, sg=True, w_flow=0.0, w_ar=1.0, w_lang=0.0)
ok(np.abs(Gar["Wk_b"]).max() > 1e-6 and np.abs(Gar["Wv_b"]).max() > 1e-6,
   "断梯度后离散动作那条学习通路完好：∂L_AR/∂Wk_b、∂L_AR/∂Wv_b 非零")


def loss_sum(P_, sg):
    o, _ = K.forward(P_, bA, sg)
    return (K.ce_loss(o["logits_lang"], bA["y_lang"])[0]
            + K.ce_loss(o["logits_ar"], bA["y_ar"])[0]
            + K.mse_loss(o["pred_flow"], bA["u"])[0])


_, Gfd, _, _ = K.grads(PA, bA, sg=False, w_flow=1.0, w_ar=1.0, w_lang=1.0)
eps, worst = 1e-6, 0.0
for name in K.ALL_P:
    W = PA[name]
    for t in np.argsort(np.abs(Gfd[name]).ravel())[-4:]:
        i = np.unravel_index(t, W.shape)
        old = W[i]
        W[i] = old + eps
        lp = loss_sum(PA, False)
        W[i] = old - eps
        lm = loss_sum(PA, False)
        W[i] = old
        fd, an = (lp - lm) / (2 * eps), Gfd[name][i]
        worst = max(worst, abs(fd - an) / max(abs(fd), abs(an), 1e-9))
ok(worst < 1e-6, f"手写反向用有限差分校准（sg 关闭）：最大相对误差 {worst:.2e}")

# ================================================================== 训练
section("训练：先预训练 backbone，再把随机初始化的 expert 嫁接上去")

EVal = K.sample_batch(sc, np.arange(N_TRAIN, N_TRAIN + N_EVAL),
                      np.random.default_rng(999), fixed_tau=0.5)
# 「什么都不看」的下界：预测恒为 0（u 的均值就是 0）。评估一律钉在 τ=0.5，
# 那里 u 与 x_t 严格独立（见 ki_lib 里 ω 的协方差构造），这条下界才是干净的。
ACT_FLOOR = 0.25 * float(np.mean(np.square(EVal["u"])))
print(f"  动作误差下界（预测恒 0）= {ACT_FLOOR:.4f}")

# 负对照：把 4 个图像 token 换成别的样本的，语言任务应当掉到随机（1/4）
EVal_shuf = dict(EVal)
EVal_shuf["X_b"] = EVal["X_b"].copy()
EVal_shuf["X_b"][:, :K.N_OBJ, :] = EVal["X_b"][
    np.random.default_rng(4242).permutation(N_EVAL)][:, :K.N_OBJ, :]


def metrics(P, batch=None):
    b = EVal if batch is None else batch
    o, _ = K.forward(P, b, sg=True)
    return dict(act=float(np.mean(np.square(b["x_t"] - 0.5 * o["pred_flow"] - b["A"]))),
                lang=K.acc(o["logits_lang"], b["y_lang"]),
                ar=K.acc(o["logits_ar"], b["y_ar"]))


def feats(P, idx):
    """expert 真正能读到的东西：V_b = X_b @ Wv_b，注意力之前。"""
    return (sc.x_b(idx) @ P["Wv_b"])[:, :K.N_O, :].mean(axis=1)


def probe_r2(P, n_fit=1024, lam=1e-3):
    Xf = np.concatenate([feats(P, np.arange(n_fit)), np.ones((n_fit, 1))], 1)
    Xe = np.concatenate([feats(P, np.arange(N_TRAIN, N_TRAIN + N_EVAL)), np.ones((N_EVAL, 1))], 1)
    Yf, Ye = sc.A[:n_fit].reshape(n_fit, -1), sc.A[N_TRAIN:].reshape(N_EVAL, -1)
    W = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ Yf)
    return float(1 - np.sum((Xe @ W - Ye) ** 2) / np.sum((Ye - Ye.mean(0)) ** 2))


def train(P, keys, sg, alpha, wa, wl, steps, lr, seed, pre=False):
    r = np.random.default_rng(seed)
    P = K.clone(P)
    opt = K.Adam(P, lr=lr)
    for _ in range(steps):
        idx = r.integers(0, N_TRAIN, 64)
        b = K.sample_batch(sc, idx, r)
        _, G, _, _ = K.grads(P, b, sg, w_flow=alpha, w_ar=wa, w_lang=wl,
                             ar_tgt=(b["y_ar_lang"] if pre else None))
        opt.step(P, G, keys)
    return P


PRE_LR, PRE_STEPS = 3e-3, 1200
LR, STEPS = 5e-3, 3000
P0 = K.init_params(np.random.default_rng(1))
P0 = train(P0, K.BONE, False, 0.0, 1.0, 1.0, PRE_STEPS, PRE_LR, 7, pre=True)
base = metrics(P0)
R2_0 = probe_r2(P0)
print(f"  预训练后：lang={base['lang']:.3f}  act={base['act']:.4f}  探针 R2={R2_0:.4f}")
ok(base["lang"] > 0.70, f"预训练把语言任务做到了 {base['lang']:.3f}（后面所有配方都从这份权重出发）")
ok(metrics(P0, EVal_shuf)["lang"] < 0.40,
   f"负对照：打乱图像 token 后语言准确率掉到 {metrics(P0, EVal_shuf)['lang']:.3f}"
   f"（随机水平 0.25）—— 说明这个任务真的在看图，不是背指令")

RECIPES = dict(
    naive=dict(sg=False, wa=0.0, wl=0.0, keys=K.ALL_P),       # π0：只有流匹配，不断梯度
    joint=dict(sg=False, wa=1.0, wl=0.0, keys=K.ALL_P),       # 联合训练（离散 + 连续），不断梯度
    joint_vlm=dict(sg=False, wa=1.0, wl=1.0, keys=K.ALL_P),
    ki=dict(sg=True, wa=1.0, wl=0.0, keys=K.ALL_P),           # 断梯度 + 离散动作
    ki_vlm=dict(sg=True, wa=1.0, wl=1.0, keys=K.ALL_P),       # 论文完整配方
    freeze=dict(sg=True, wa=0.0, wl=0.0, keys=K.EXPERT),      # 冻住 backbone 只训 expert
)
RUN = {}
for name, cfg in RECIPES.items():
    for a in (1.0, 8.0, 32.0):
        P = K.clone(K.init_params(np.random.default_rng(555)))
        for k in K.BONE:
            P[k] = P0[k].copy()                    # backbone 与离散头沿用预训练权重
        P = train(P, cfg["keys"], cfg["sg"], a, cfg["wa"], cfg["wl"], STEPS, LR, 31)
        RUN[(name, a)] = dict(P=P, m=metrics(P), r2=probe_r2(P))
    print(f"  {name:10s} " + " | ".join(
        f"a={a:<4g} lang={RUN[(name, a)]['m']['lang']:.3f} ar={RUN[(name, a)]['m']['ar']:.3f} "
        f"act={RUN[(name, a)]['m']['act']:.4f} R2={RUN[(name, a)]['r2']:.3f}"
        for a in (1.0, 8.0, 32.0)))


def rel_diff(Pa, Pb, keys):
    """平均逐元素差 / 平均逐元素幅值。"""
    return max(float(np.abs(Pa[k] - Pb[k]).mean() / max(np.abs(Pb[k]).mean(), 1e-12))
               for k in keys)


# ================================================================== B 组
section("B 组：α 的角色 —— 断梯度之后它就不是旋钮了")

for name in ("ki", "ki_vlm"):
    dB = max(np.abs(RUN[(name, a)]["P"][k] - RUN[(name, 1.0)]["P"][k]).max()
             for a in (8.0, 32.0) for k in K.BONE)
    ok(dB == 0.0,
       f"{name}：α = 1 / 8 / 32 三跑的 backbone 参数**逐位相同**（max 差 {dB:.1e}）")
    dE = max(rel_diff(RUN[(name, a)]["P"], RUN[(name, 1.0)]["P"], K.EXPERT)
             for a in (8.0, 32.0))
    ok(dE < 1e-3, f"{name}：expert 参数也几乎不动（相对差 {dE:.1e}，残差只来自 Adam 的 ε）")

for name in ("joint", "joint_vlm"):
    d = max(rel_diff(RUN[(name, a)]["P"], RUN[(name, 1.0)]["P"], K.BONE)
            for a in (8.0, 32.0))
    ok(d > 0.05, f"{name}：不断梯度时 α 一变，backbone 轨迹就变（相对差 {d * 100:.0f}%）")

j1, j8 = RUN[("joint_vlm", 1.0)], RUN[("joint_vlm", 8.0)]
ok(abs(j8["r2"] - j1["r2"]) > 0.1,
   f"joint_vlm：α 从 1 抬到 8，backbone 探针 R2 {j1['r2']:.3f} → {j8['r2']:.3f}"
   f"（α 是真的旋钮，调大就是让流匹配梯度压过离散动作梯度）")
ok(abs(RUN[("ki_vlm", 32.0)]["m"]["lang"] - RUN[("ki_vlm", 1.0)]["m"]["lang"]) < 1e-12,
   "ki_vlm：语言跟随对 α 完全不敏感（逐位相同）")

# 负对照：naive 只有一个损失，α 也不影响结果 —— 但机制不同：
# 它的 backbone 梯度**带** α 这个因子，只是被 Adam 的逐参数归一化消掉了，
# 所以只能做到「近似相同」，做不到 ki 那种「逐位相同」。
dN = rel_diff(RUN[("naive", 32.0)]["P"], RUN[("naive", 1.0)]["P"], K.BONE)
ok(0.0 < dN < 1e-2,
   f"负对照 naive：单一损失下 α 也不改变结果（相对差 {dN:.1e}），"
   f"但只是「近似」相同 —— 与 ki 的逐位为 0 不是一回事")

# 把「梯度本身」和「优化器」切开：梯度仍然严格按 α 线性放大，
# 是 Adam 的逐参数归一化把 α 消掉的（换成 SGD，步长就会原样放大 α 倍）。
P_start = K.clone(K.init_params(np.random.default_rng(555)))
for k in K.BONE:
    P_start[k] = P0[k].copy()
GB = {}
for a in (1.0, 8.0, 32.0):
    _, GB[a], _, _ = K.grads(P_start, bA, sg=True, w_flow=a, w_ar=0.0, w_lang=0.0)
ok(all(np.abs(GB[a][k]).max() == 0.0 for a in GB for k in K.BONE),
   "断梯度后，流匹配损失对 backbone 的梯度恒为 0 —— 与 α 无关，连「放大」都谈不上")
worst = max(float(np.abs(GB[a][k] - a * GB[1.0][k]).max()
                  / max(np.abs(a * GB[1.0][k]).max(), 1e-12))
            for a in (8.0, 32.0) for k in K.EXPERT)
ok(worst < 1e-12,
   f"同一批数据上 expert 的梯度严格按 α 线性放大（相对误差 {worst:.1e}）——"
   f"α 之所以不影响结果，是 Adam 逐参数归一化的功劳，不是梯度变小了")

# ================================================================== C 组
section("C 组：语言跟随 —— 各配方拿什么换了什么")

BL = base["lang"]
for name in ("naive", "joint", "ki"):
    ok(RUN[(name, 1.0)]["m"]["lang"] < BL - 0.15,
       f"{name}（不带 VLM 协同训练）：语言跟随 {BL:.3f} → {RUN[(name, 1.0)]['m']['lang']:.3f}")
for name in ("ki_vlm", "joint_vlm"):
    ok(RUN[(name, 1.0)]["m"]["lang"] > BL - 0.05,
       f"{name}（带 VLM 协同训练）：语言跟随保住在 {RUN[(name, 1.0)]['m']['lang']:.3f}")
ok(abs(RUN[("freeze", 1.0)]["m"]["lang"] - BL) < 1e-12,
   f"freeze：backbone 没动，语言跟随恒等于预训练值 {BL:.3f}")

for name in ("joint", "joint_vlm", "ki", "ki_vlm"):
    ok(RUN[(name, 1.0)]["m"]["ar"] > RUN[("naive", 1.0)]["m"]["ar"] + 0.2,
       f"{name}：离散动作 token 准确率 {RUN[(name, 1.0)]['m']['ar']:.3f} "
       f"远高于没训过它的 naive {RUN[('naive', 1.0)]['m']['ar']:.3f}")

# ================================================================== D 组
section("D 组：信息瓶颈 —— 断梯度成立的前提")

fr = RUN[("freeze", 1.0)]
ok(fr["m"]["act"] > ACT_FLOOR - 0.01,
   f"冻住 backbone：动作误差 {fr['m']['act']:.4f} 就卡在「什么都不看」的下界 "
   f"{ACT_FLOOR:.4f} 上 —— 论文说的「冻住不行」在这里就是这样")
ok(fr["r2"] < 0.25,
   f"冻住的 backbone 里几乎没有动作信息：探针 R2 = {fr['r2']:.3f}"
   f"（与预训练完的 {R2_0:.3f} 同量级，等于没多学）")
ok(RUN[("ki_vlm", 1.0)]["r2"] > 1.5 * fr["r2"],
   f"断梯度 + 离散动作训练后，backbone 学会承载动作信息：R2 {fr['r2']:.3f} → "
   f"{RUN[('ki_vlm', 1.0)]['r2']:.3f}")
ok(RUN[("ki_vlm", 1.0)]["m"]["act"] < fr["m"]["act"] - 0.02,
   f"同一件事在动作误差上看：{fr['m']['act']:.4f} → {RUN[('ki_vlm', 1.0)]['m']['act']:.4f}")
ok(RUN[("naive", 1.0)]["r2"] > 1.5 * RUN[("ki_vlm", 1.0)]["r2"],
   f"代价：断掉梯度后动作信息只能从离散动作那条通路慢慢灌进 backbone，"
   f"同样步数下探针 R2 {RUN[('naive', 1.0)]['r2']:.3f} vs {RUN[('ki_vlm', 1.0)]['r2']:.3f} "
   f"（论文在真实规模上反而更快，玩具里 backbone 太小，体现不出来）")

# 步数趋势：KI 到底追不追得上（玩具里追不上，这是规模问题不是机制问题）
TREND = {}
for tname, tsg, twa, twl in (("naive", False, 0.0, 0.0), ("ki_vlm", True, 1.0, 1.0)):
    rows = []
    for st in (1000, 3000, 9000):
        P = K.clone(K.init_params(np.random.default_rng(555)))
        for k in K.BONE:
            P[k] = P0[k].copy()
        P = train(P, K.ALL_P, tsg, 1.0, twa, twl, st, LR, 31)
        rows.append((metrics(P), probe_r2(P)))
    TREND[tname] = rows
    print(f"  {tname:8s} " + " | ".join(
        f"{st:5d} 步 lang={m['lang']:.3f} act={m['act']:.4f} R2={r:.3f}"
        for st, (m, r) in zip((1000, 3000, 9000), rows)))
ok(TREND["ki_vlm"][-1][1] < TREND["naive"][0][1],
   f"步数趋势：KI 训到 9000 步（3 倍）主干探针 R2 {TREND['ki_vlm'][-1][1]:.3f} 仍低于 "
   f"naive 1000 步的 {TREND['naive'][0][1]:.3f} —— 玩具里 KI 追不上，"
   f"与论文「KI 收敛更快」方向相反（原因见文件头：玩具主干太小）")
ok(TREND["ki_vlm"][1][0]["lang"] > TREND["naive"][1][0]["lang"] + 0.3,
   f"但语言跟随的差距不随步数收窄：3000 步时 {TREND['naive'][1][0]['lang']:.3f} vs "
   f"{TREND['ki_vlm'][1][0]['lang']:.3f}")

# 负对照：动作只依赖当次实例，不依赖物体「名字」
Xf = np.concatenate([sc.NAME[sc.c[:1024]], np.ones((1024, 1))], 1)
Xe = np.concatenate([sc.NAME[sc.c[N_TRAIN:N_TRAIN + N_EVAL]], np.ones((N_EVAL, 1))], 1)
Yf, Ye = sc.A[:1024].reshape(1024, -1), sc.A[N_TRAIN:].reshape(N_EVAL, -1)
W = np.linalg.solve(Xf.T @ Xf + 1e-3 * np.eye(Xf.shape[1]), Xf.T @ Yf)
r2_name = float(1 - np.sum((Xe @ W - Ye) ** 2) / np.sum((Ye - Ye.mean(0)) ** 2))
ok(r2_name < 0.05,
   f"负对照：光靠物体名字预测动作块的 R2 只有 {r2_name:.4f} —— 动作确实依赖当次实例，"
   f"所以「VLM 预训练里没有机器人数据」这件事在探针里是真的")

section(f"断言总数：{N_ASSERT}")
print(f"断言总数 = {N_ASSERT}")
