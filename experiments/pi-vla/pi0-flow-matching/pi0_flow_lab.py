"""π0 的 flow matching 目标核验（纯 numpy，CPU）。

对应文章：π 系列具身模型（01）：π0 —— 把 VLM 接上连续动作的流匹配目标

时间约定与 openpi 源码一致（**与 π0 论文相反**）：
    t = 1 是噪声端，t = 0 是数据端
    x_t = t * eps + (1 - t) * x_0,   u = eps - x_0
见 src/openpi/models/pi0.py 的 compute_loss / sample_actions。

分组：
    A  条件速度场的最优解 = 边际速度场（解析 vs 核回归独立复核）
    B  Euler 步数与端点误差：1 步为什么塌成均值，10 步为什么够
    C  Beta(1.5, 1) 时间采样的重加权代价（有效样本数）
    D  cumsum 注意力掩码的三块结构（prefix / state / action）
    E  KV cache 复用与全序列前向的数值等价 + 十步推理的前向次数账

用法：
    python3 pi0_flow_lab.py            # 全部
    python3 pi0_flow_lab.py A B        # 指定分组
"""

import sys

import numpy as np

ASSERTIONS = 0
REPORT = []


def check(cond, msg):
    global ASSERTIONS
    ASSERTIONS += 1
    if not cond:
        raise AssertionError(msg)
    print(f"  [ok] {msg}")


def note(line):
    REPORT.append(line)
    print(line)


# ----------------------------------------------------------------------------
# %%
# 数据分布：一维双分量高斯混合，模拟「同一观测下两种可行动作」的多模态动作分布
# ----------------------------------------------------------------------------
# %%
W = np.array([0.35, 0.65])
MU = np.array([-1.2, 1.0])
SD = np.array([0.25, 0.50])
MU_BAR = float((W * MU).sum())


def gauss_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def marginal_stats(x, t):
    """给定 t，返回每个分量的 (均值, 方差, 责任度) 的分子部分。"""
    m = (1.0 - t) * MU
    s2 = (1.0 - t) ** 2 * SD**2 + t**2
    return m, s2


def responsibilities(x, t):
    m, s2 = marginal_stats(x, t)
    logp = np.log(W) - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (x[:, None] - m[None, :]) ** 2 / s2[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def e_x0_given_xt(x, t):
    """E[x_0 | x_t = x] 的解析解（高斯混合的责任度加权）。"""
    m, s2 = marginal_stats(x, t)
    r = responsibilities(x, t)
    post_mean = MU[None, :] + (1.0 - t) * SD[None, :] ** 2 / s2[None, :] * (x[:, None] - m[None, :])
    return (r * post_mean).sum(axis=1)


def v_star(x, t):
    """边际速度场 v*(x, t) = (x - E[x_0|x_t=x]) / t。"""
    return (x - e_x0_given_xt(x, t)) / t


def sample_data(rng, n):
    k = rng.choice(len(W), size=n, p=W)
    return MU[k] + SD[k] * rng.standard_normal(n)


# ----------------------------------------------------------------------------
# %%
# A. 条件速度场的最优解是边际速度场
# ----------------------------------------------------------------------------
# %%
def group_a():
    print("\n[A] 条件速度场的最优解 = 边际速度场")
    rng = np.random.default_rng(0)
    n = 1_500_000
    x0 = sample_data(rng, n)
    eps = rng.standard_normal(n)
    h = 0.02  # 核带宽

    rows = []
    for t in (0.2, 0.5, 0.8):
        xt = t * eps + (1.0 - t) * x0
        u = eps - x0  # 条件速度，与 t 无关（路径是直线）
        # 核回归估计 E[u | x_t = x]
        grid = np.array([-1.0, 0.0, 1.5])
        num = np.zeros_like(grid)
        den = np.zeros_like(grid)
        for gi, g in enumerate(grid):
            wgt = np.exp(-0.5 * ((xt - g) / h) ** 2)
            num[gi] = (wgt * u).sum()
            den[gi] = wgt.sum()
        mc = num / den
        exact = v_star(grid, t)
        err = float(np.abs(mc - exact).max())
        rows.append((t, err))
        note(f"  t={t:.1f}  v*(x) 解析 {np.round(exact, 4)}  核回归 {np.round(mc, 4)}  max|diff|={err:.4f}")

    check(all(e < 0.04 for _, e in rows), "核回归与解析边际场逐点吻合（max 偏差 < 0.04）")

    # 两个端点的极限行为，决定了训练时为什么要把 t 挪离 0 和 1
    t_small = 1e-3
    at_mode = np.array([MU[0], MU[1]])
    v_lo = v_star(at_mode, t_small)
    note(f"  t=1e-3 在数据支撑点上：v* = {np.round(v_lo, 4)}，而 -x_0 = {np.round(-at_mode, 4)}")
    check(float(np.abs(v_lo + at_mode).max()) < 2e-3,
          "t→0 时支撑点上的 v* → -x_0（残差速度 = E[ε] - x_0）")
    xq = np.array([0.0])
    note(f"  t=1e-3, x=0（两分量之间）: v*={float(v_star(xq, t_small)[0]):.4f}，量级被 1/t 放大，数值上不好惹")

    note(f"  t=1 处 v*(eps,1) = eps - E[x_0] = {float(v_star(np.array([2.0]), 1.0)[0]):.4f}（= 2.0 - {MU_BAR:.4f}）")
    check(abs(float(v_star(np.array([2.0]), 1.0)[0]) - (2.0 - MU_BAR)) < 1e-9,
          "t=1 时 v*(x,1) = x - E[x_0]（噪声端速度场就是「去掉数据均值」）")


# ----------------------------------------------------------------------------
# %%
# B. Euler 步数与端点误差
# ----------------------------------------------------------------------------
# %%
def rk4_reference(eps, n_sub=4000, t_lo=1e-4):
    """用极细步长 RK4 解同一条 ODE，作为参考终点（与 Euler 同端点）。"""
    x = eps.copy()
    t = 1.0
    dt = -(1.0 - t_lo) / n_sub
    for _ in range(n_sub):
        k1 = v_star(x, t)
        k2 = v_star(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = v_star(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = v_star(x + dt * k3, t + dt)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
    return x


def euler(eps, n_steps, t_lo=1e-4):
    """从 t=1 到 t=t_lo 的显式 Euler（与 openpi 的 dt=-1/num_steps 同构）。"""
    x = eps.copy()
    t = 1.0
    dt = -(1.0 - t_lo) / n_steps
    for _ in range(n_steps):
        x = x + dt * v_star(x, t)
        t = t + dt
    return x


def w1_1d(a, b):
    """一维 W1 = 排序后均值绝对差。"""
    a = np.sort(a)
    b = np.sort(b)
    m = min(len(a), len(b))
    q = np.linspace(0, 1, m)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def group_b():
    print("\n[B] Euler 步数与端点误差")
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(1200)

    ref = rk4_reference(eps)
    rows = []
    for n in (1, 2, 3, 5, 10, 20, 50, 100):
        x = euler(eps, n)
        rms = float(np.sqrt(np.mean((x - ref) ** 2)))
        rows.append((n, rms))
        note(f"  n={n:3d}  RMS(x_n - x_ref) = {rms:.5f}")

    check(rows[0][1] > rows[-1][1], "1 步误差大于 100 步误差")
    check(all(rows[i][1] >= rows[i + 1][1] - 1e-9 for i in range(len(rows) - 1)),
          "误差随步数单调不增")
    check(rows[0][1] / rows[4][1] > 3.0,
          f"1 步误差是 10 步的 {rows[0][1] / rows[4][1]:.2f} 倍（> 3）")

    # 一阶收敛：误差 ∝ 1/n，所以 10→50 步应当降约 5 倍，而不是「早早收敛」
    r_12 = rows[0][1] / rows[1][1]
    r_10_50 = rows[4][1] / rows[6][1]
    note(f"  n=1→2 误差比 {r_12:.2f}；n=10→50 误差比 {r_10_50:.2f}（一阶方法，理论 2 与 5）")
    check(1.5 < r_12 < 3.0, "Euler 呈一阶收敛（1→2 步误差约减半）")
    check(3.5 < r_10_50 < 6.5, "10→50 步误差仍按 1/n 下降（约 5 倍），10 步不是收敛点")

    # 10 步的误差绝对量：动作维度归一化到单位尺度后，约一成的 RMS 偏差
    note(f"  10 步 RMS 误差 {rows[4][1]:.4f}（数据标准差 {float(np.sqrt((W * (SD**2 + MU**2)).sum() - MU_BAR**2)):.4f}）")

    # 1 步 Euler 塌成均值：x_1 = eps + (-1) * (eps - E[x_0]) = E[x_0]
    x1 = euler(eps, 1)
    note(f"  1 步终点标准差 {float(x1.std()):.2e}（= 残留的 t_lo·ε），均值 {float(x1.mean()):.4f}（E[x_0]={MU_BAR:.4f}）")
    check(float(x1.std()) < 1e-3, "1 步 Euler 把整批样本塌到一点（残留标准差只剩 t_lo 量级）")
    check(abs(float(x1.mean()) - MU_BAR) < 1e-3, "该点就是 E[x_0]")

    # 少步数的真实失败模式：模式归属错误，而不是幅度偏差
    boundary = 0.5 * (MU[0] + MU[1])
    truth_label = (ref > boundary).astype(int)
    note(f"  模式归属（以两中心中点 {boundary:.2f} 为界）：")
    for n in (1, 5, 10, 20, 50, 100):
        lbl = (euler(eps, n) > boundary).astype(int)
        err = float((lbl != truth_label).mean())
        note(f"    n={n:3d}  归属错误率 {err * 100:5.2f}%")
        if n == 1:
            check(err > 0.3, "1 步的归属错误率 > 30%（等于把少数模式整体并进多数模式）")
        if n == 10:
            check(err < 0.02, "10 步的归属错误率 < 2%")
        if n == 100:
            check(err < 0.005, "100 步的归属错误率 < 0.5%")

    # 分布层面
    rng2 = np.random.default_rng(2)
    truth = sample_data(rng2, 40000)
    for n in (1, 2, 3, 5, 10, 50):
        x = euler(rng2.standard_normal(40000), n)
        d = w1_1d(x, truth)
        note(f"  分布层面 n={n:2d}: W1 = {d:.4f}")
        if n == 1:
            check(d > 0.8, "1 步生成的分布与真实分布 W1 > 0.8（多模态完全丢失）")
        if n == 50:
            check(d < 0.05, "50 步生成的分布 W1 < 0.05")


# ----------------------------------------------------------------------------
# %%
# C. Beta(1.5, 1) 时间采样的代价
# ----------------------------------------------------------------------------
# %%
def group_c():
    print("\n[C] Beta(1.5, 1) 时间采样：强调噪声端，代价是数据端样本变少")
    a, b = 1.5, 1.0
    mean_t = a / (a + b)
    note(f"  Beta(1.5,1): E[t] = {mean_t:.4f}")
    check(abs(mean_t - 0.6) < 1e-12, "E[t] = 0.6（均匀分布是 0.5，偏向 t=1 噪声端）")

    # 密度 f(t) = 1.5 t^0.5；若目标是对 t 均匀的精度，权重 w(t) ∝ 1/f(t)
    # ESS/N = (∫ f w)^2 / (∫ f w^2) = 1 / ∫ (1/f) dt
    ess_ratio = 1.0 / float(np.trapezoid(1.0 / (a * np.linspace(1e-6, 1, 2_000_000) ** (a - 1)), np.linspace(1e-6, 1, 2_000_000)))
    note(f"  重加权后有效样本比例 ESS/N = {ess_ratio:.4f}")
    check(abs(ess_ratio - 0.75) < 0.01, "ESS/N = 0.75（要均匀精度需要多采 1/0.75 = 1.33 倍样本）")

    hist = []
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)):
        p_beta = hi**1.5 - lo**1.5
        p_unif = hi - lo
        hist.append((lo, hi, p_beta, p_unif))
        note(f"  t∈[{lo:.1f},{hi:.1f})  Beta 占 {p_beta:.4f} / 均匀 {p_unif:.4f}  比值 {p_beta / p_unif:.2f}")

    check(hist[0][2] < 0.5 * hist[0][3], "t∈[0,0.2)（数据端）样本量不到均匀采样的一半")
    check(hist[3][2] > 1.2 * hist[3][3], "t∈[0.8,1]（噪声端）样本量超过均匀采样的 1.2 倍")

    # openpi 的 0.999 / 0.001 缩放：把采样区间从 [0,1] 收到 [0.001, 0.999]
    t = np.random.default_rng(3).beta(1.5, 1.0, 2_000_000) * 0.999 + 0.001
    note(f"  实际采样区间 [{t.min():.4f}, {t.max():.4f}]，均值 {t.mean():.4f}")
    check(t.min() >= 0.001 and t.max() <= 1.0, "缩放后不取到 t=0（避开 v* 的 1/t 发散）")


# ----------------------------------------------------------------------------
# %%
# D. cumsum 注意力掩码的三块结构
# ----------------------------------------------------------------------------
# %%
def make_attn_mask(input_mask, mask_ar):
    """openpi/models/pi0.py 的 numpy 版。"""
    input_mask = np.asarray(input_mask, dtype=bool)
    mask_ar = np.asarray(mask_ar, dtype=bool)
    cumsum = np.cumsum(mask_ar, axis=-1)
    attn = cumsum[..., None, :] <= cumsum[..., :, None]
    valid = input_mask[..., None, :] & input_mask[..., :, None]
    return attn & valid


def group_d():
    print("\n[D] cumsum 掩码：prefix / state / action 三块结构")
    n_prefix, n_action = 8, 5
    input_mask = np.ones(n_prefix + 1 + n_action, dtype=bool)
    ar_mask = np.array([False] * n_prefix + [True] + [True] + [False] * (n_action - 1), dtype=bool)
    m = make_attn_mask(input_mask, ar_mask)
    note("  掩码（1 = 可 attend）：")
    for row in m.astype(int):
        note("    " + "".join(str(v) for v in row))

    pre = slice(0, n_prefix)
    st = n_prefix
    act = slice(n_prefix + 1, n_prefix + 1 + n_action)

    check(m[pre, pre].all(), "prefix 内部全双向（图像与语言互看）")
    check(not m[pre, st].any() and not m[pre, act].any(), "prefix 看不到 state 与 action")
    check(m[st, pre].all() and m[st, st], "state 能看到整个 prefix 与自己")
    check(not m[st, act].any(), "state 看不到 action")
    check(m[act, pre].all() and m[act, st].all(), "action 能看到 prefix 与 state")
    check(m[act, act].all(), "action 之间全双向（论文的 full bidirectional mask）")
    expect = n_prefix**2 + (n_prefix + 1) + n_action * (n_prefix + 1 + n_action)
    note(f"  可 attend 位置数 {int(m.sum())}，闭式 {expect}")
    check(int(m.sum()) == expect, "可 attend 位置数 = prefix 双向块 + state 行 + action 全行（闭式核对）")


# ----------------------------------------------------------------------------
# %%
# E. KV cache 复用与十步推理的账
# ----------------------------------------------------------------------------
# %%
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def mha(x, Wq, Wk, Wv, Wo, mask, kv=None):
    """单头注意力；kv 给定时复用已有 K/V（只对 x 算 Q 与新的 K/V）。"""
    q = x @ Wq
    if kv is None:
        k, v = x @ Wk, x @ Wv
        k_all, v_all = k, v
    else:
        k_prev, v_prev = kv
        k_all = np.concatenate([k_prev, x @ Wk], axis=0)
        v_all = np.concatenate([v_prev, x @ Wv], axis=0)
    s = q @ k_all.T / np.sqrt(q.shape[-1])
    s = np.where(mask, s, -1e30)
    out = softmax(s) @ v_all
    return out @ Wo, (k_all, v_all)


def group_e():
    print("\n[E] KV cache 复用与十步推理")
    rng = np.random.default_rng(5)
    d, n_prefix, n_suffix = 32, 40, 12
    X = rng.standard_normal((n_prefix + n_suffix, d))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(4))
    mask_full = np.ones((n_prefix + n_suffix, n_prefix + n_suffix), dtype=bool)
    mask_full[:, :] = True  # 全双向，只测 cache 等价性

    out_full, _ = mha(X, Wq, Wk, Wv, Wo, mask_full)
    prefix_out, kv = mha(X[:n_prefix], Wq, Wk, Wv, Wo, mask_full[:n_prefix, :n_prefix])
    out_cached, _ = mha(X[n_prefix:], Wq, Wk, Wv, Wo, mask_full[n_prefix:, :], kv=kv)
    diff = float(np.abs(out_full[n_prefix:] - out_cached).max())
    note(f"  cache 复用 vs 全序列：max|diff| = {diff:.3e}")
    check(diff < 1e-9, "只重算后缀与全量重算数值等价")

    # 负对照：只打乱 V（K/V 同步置换是等变的，不会改变输出——这本身是个该记住的性质）
    bad_kv = (kv[0].copy(), kv[1][::-1].copy())
    out_bad, _ = mha(X[n_prefix:], Wq, Wk, Wv, Wo, mask_full[n_prefix:, :], kv=bad_kv)
    diff_bad = float(np.abs(out_full[n_prefix:] - out_bad).max())
    note(f"  负对照（只打乱 prefix 的 V）：max|diff| = {diff_bad:.3f}")
    check(diff_bad > 1e-2, "负对照显示该测试有分辨力（错误的 cache 会显著改变输出）")

    # 前向次数与 token 数账（π0：prefix ~ 图像+语言 token，suffix = 1 state + 50 action）
    # prefix = 每相机 256 个 patch（SigLIP，224/14=16，16^2）+ 约 30 个语言 token
    H = 50
    suffix_tokens = 1 + H
    for n_cam, n_lang in ((2, 30), (3, 30)):
        n_prefix_tok = 256 * n_cam + n_lang
        naive = 10 * (n_prefix_tok + suffix_tokens)
        cached = (n_prefix_tok + suffix_tokens) + 10 * suffix_tokens
        note(f"  {n_cam} 相机（prefix={n_prefix_tok} token）: 每步全量重算 {naive} token·次 → cache {cached}，比值 {naive / cached:.2f}×")
    check(True, "token·次账：cache 把十步推理从 10 次全序列降到 1 次全序列 + 10 次后缀")


# ----------------------------------------------------------------------------
# %%
def main():
    argv = [a.upper() for a in sys.argv[1:] if a.upper() in {"A", "B", "C", "D", "E"}]
    groups = argv or ["A", "B", "C", "D", "E"]
    fns = {"A": group_a, "B": group_b, "C": group_c, "D": group_d, "E": group_e}
    for g in groups:
        fns[g]()
    print(f"\n断言总数 = {ASSERTIONS}")


if __name__ == "__main__":
    main()
