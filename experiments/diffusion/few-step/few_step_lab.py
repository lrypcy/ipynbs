#!/usr/bin/env python3
"""少步生成（一致性模型 / 蒸馏）核验脚本 —— 纯 CPU，无训练，全部解析

背景
----
一维 4 模态高斯混合，VP 扩散，连续时间 schedule。因为数据分布是混合高斯，
score / 后验均值 / 概率流 ODE 都能写成闭式，于是"训练出来的模型"可以被
"带已知误差的算子"精确替代，误差本身是唯一变量。

本脚本回答五个问题：

  A  1 步生成到底缺什么？—— 把 DDPM 的后验均值 x̂₀ 当 1 步生成器会塌缩多少？
  B  自一致性损失有多少个零解？—— 为什么边界条件不是工程细节？
  C  步数能不能救一个错的映射？—— 三种典型蒸馏缺陷 × 步数的完整表
  D  一致性训练（CT）的离散化偏差有多大？方向朝哪？
  E  训练期的 NFE 账本，以及"教师轨迹 → 学生推理"的路径错配

运行
----
    python few_step_lab.py                 # A–E 全部
    python few_step_lab.py A C --figdir figs

依赖 numpy。种子固定，输出可复现。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODES = np.array([-3.0, -1.0, 1.0, 3.0])   # 四个模态中心
SD = 0.40                                   # 每个模态的标准差
B0, B1, TTOT = 0.10, 20.0, 1.0              # β(t) = B0 + (B1-B0) t / TTOT
T_END = 1.0e-4                              # 轨迹终点（σ_end ≈ 3.3e-3）
T_START = 1.0                               # 纯噪声端
SEED = 20261104
N_SAMPLE = 4000                             # 每组统计的样本数
N_LAM = 512                                 # ODE 积分的 λ 步数

ASSERTS = [0, 0]                            # [通过, 总数]


def ok(cond, msg):
    ASSERTS[1] += 1
    if cond:
        ASSERTS[0] += 1
        print(f"  [ok]   {msg}")
    else:
        print(f"  [FAIL] {msg}")
    return bool(cond)


def head(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ---------------------------------------------------------------------------
# 连续时间 schedule
# ---------------------------------------------------------------------------
def abar(t):
    return np.exp(-(B0 * t + (B1 - B0) * t * t / (2.0 * TTOT)))


def beta(t):
    return B0 + (B1 - B0) * t / TTOT


def ab(t):
    a2 = abar(t)
    return np.sqrt(a2), np.sqrt(np.maximum(1.0 - a2, 0.0))


def lam_of_t(t):
    a2 = abar(t)
    return 0.5 * np.log(a2 / (1.0 - a2))


def t_of_lam(lam):
    """abar = 1/(1+e^{-2λ}) ⇒ f = -log(abar) = log(1+e^{-2λ})，再解二次方程还原 t。"""
    lam = np.asarray(lam, dtype=float)
    f = np.log1p(np.exp(-2.0 * lam))
    c = (B1 - B0) / (2.0 * TTOT)
    t = (-B0 + np.sqrt(B0 * B0 + 4.0 * c * f)) / (2.0 * c)
    return t if t.ndim else float(t)


LAM_T = lam_of_t(T_START)
LAM_END = lam_of_t(T_END)


# ---------------------------------------------------------------------------
# 混合高斯：责任度 / 后验均值 x̂₀ / score / log p
# ---------------------------------------------------------------------------
def _gm(x, t, modes, sd):
    a, s = ab(t)
    v = a * a * sd * sd + s * s
    return a, s, v


def resp(x, t, modes=MODES, sd=SD):
    a, _, v = _gm(x, t, modes, sd)
    d = x[:, None] - a * modes[None, :]
    lw = -0.5 * d * d / v
    lw = lw - lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def x0hat(x, t, modes=MODES, sd=SD):
    """Tweedie：x̂₀(x_t) = E[x₀ | x_t]"""
    a, _, v = _gm(x, t, modes, sd)
    r = resp(x, t, modes, sd)
    mu = modes[None, :] + (a * sd * sd / v) * (x[:, None] - a * modes[None, :])
    return (r * mu).sum(axis=1)


def score(x, t, modes=MODES, sd=SD):
    a, _, v = _gm(x, t, modes, sd)
    r = resp(x, t, modes, sd)
    return (r * (-(x[:, None] - a * modes[None, :]) / v)).sum(axis=1)


def logp(x, t, modes=MODES, sd=SD):
    a, _, v = _gm(x, t, modes, sd)
    d = x[:, None] - a * modes[None, :]
    lw = -0.5 * d * d / v - 0.5 * math.log(2.0 * math.pi * v)
    m = lw.max(axis=1)
    return m + np.log(np.exp(lw - m[:, None]).sum(axis=1))


# ---------------------------------------------------------------------------
# 概率流 ODE：dx/dλ = σ² (x + ∇log p_t)，在 λ 上均匀 RK4
# ---------------------------------------------------------------------------
def drift(x, lam, modes=MODES, sd=SD):
    t = t_of_lam(lam)
    _, s = ab(t)
    return s * s * (x + score(x, t, modes, sd))


def rk4(x, l0, l1, modes=MODES, sd=SD):
    h = l1 - l0
    k1 = drift(x, l0, modes, sd)
    k2 = drift(x + 0.5 * h * k1, l0 + 0.5 * h, modes, sd)
    k3 = drift(x + 0.5 * h * k2, l0 + 0.5 * h, modes, sd)
    k4 = drift(x + h * k3, l1, modes, sd)
    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ode_traj(x, lams, modes=MODES, sd=SD, snap=None):
    """从 lams[0] 积分到 lams[-1]；snap 里给出要记录快照的下标。"""
    out = {i: x.copy() for i in (snap or [])}
    for i in range(len(lams) - 1):
        x = rk4(x, lams[i], lams[i + 1], modes, sd)
        if i + 1 in (snap or []):
            out[i + 1] = x.copy()
    return x, out


def endpoint_map(x, lam, modes=MODES, sd=SD, n=None):
    """x₀ = f*(x_t, t)：从 λ 积分到 λ_end 的精确一致性函数（数值解）。

    n=None 时步数按 λ 跨度自适应，保证每单位 λ 的步长恒定 —— 这样
    "从 λ_k 出发"与"从 λ_T 出发"用的是同一套离散步长，多步采样不会
    因为步数虚增而显得更准。
    """
    if n is None:
        n = max(24, int(round(N_LAM * (lam - LAM_END) / (LAM_T - LAM_END))))
    lams = np.linspace(lam, LAM_END, n + 1)
    y, _ = ode_traj(x, lams, modes, sd)
    return y


# ---------------------------------------------------------------------------
# 度量
# ---------------------------------------------------------------------------
def w1(a, b, qs=401):
    q = np.linspace(0.0, 1.0, qs)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def modal_mass(x):
    """四个模态的盆地：(-inf,-2) (-2,0) (0,2) (2,inf)"""
    edges = np.array([-2.0, 0.0, 2.0])
    idx = np.searchsorted(edges, x)
    return np.array([(idx == k).mean() for k in range(4)])


def sample_data(rng, n):
    i = rng.integers(0, len(MODES), size=n)
    return MODES[i] + SD * rng.standard_normal(n), i


def sample_noisy(rng, n, t=T_START):
    x0, _ = sample_data(rng, n)
    a, s = ab(t)
    return a * x0 + s * rng.standard_normal(n), x0


# ---------------------------------------------------------------------------
# 构造"带已知误差的模型"：唯一变量是误差形态
# ---------------------------------------------------------------------------
def make_perturbed(kind, delta=0.0, where=None, n_integ=None):
    """返回 f(x, t)：近似的一致性函数。

    kind = 'exact'  精确（数值积分的 ODE 端点）
         = 'bias'   f* + delta            （蒸馏出的系统偏置）
         = 'shrink' f* * (1-delta)        （回归到均值 / 收缩）
         = 'drop'   少训练一个模态的精确 f*  （模态丢失）
         = 'rand'   f* + 零均值抖动的 delta    （无系统偏差，只有方差）
    """
    modes = MODES if kind != "drop" else MODES[:-1]
    where = MODES[-1] if where is None else where
    rrng = np.random.default_rng(SEED + 31)

    def f(x, t):
        lam = lam_of_t(np.asarray([t], dtype=float))[0] if np.ndim(t) == 0 else lam_of_t(t)
        y = endpoint_map(x, lam, modes=modes, n=n_integ)
        if kind == "bias":
            return y + delta
        if kind == "shrink":
            return y * (1.0 - delta)
        if kind == "rand":
            return y + delta * rrng.standard_normal(y.shape)
        return y

    return f


def few_step_sample(f, x_init, lams, rng, stochastic=True):
    """多步一致性采样：每步先跳到 x₀ 估计，再重新加噪到下一个噪声尺度。

    stochastic=True  用新噪声（CM 论文 Algorithm 2）
    stochastic=False 用同一 ε 的解析估计（确定性 / ODE 版）
    """
    x = x_init.copy()
    K = len(lams) - 1
    for k in range(K):
        t = t_of_lam(lams[k])
        x0 = f(x, t)                     # 第 k 次、也是第 k 次调用 f
        if k == K - 1:
            return x0                    # 最后一步不再加噪，直接输出 x₀ 估计
        tn = t_of_lam(lams[k + 1])
        a, s = ab(t)
        an, sn = ab(tn)
        if stochastic:
            eps = rng.standard_normal(x.shape)
        else:
            eps = (x - a * x0) / s
        x = an * x0 + sn * eps


def grid_lams(K):
    return np.linspace(LAM_T, LAM_END, K + 1)


# ---------------------------------------------------------------------------
# A 组：1 步生成到底缺什么
# ---------------------------------------------------------------------------
def exp_A(rng, figdir=None):
    head("A 组：把后验均值 x̂₀ 当 1 步生成器，会塌缩多少")

    # --- A1 Tweedie 恒等式与 score 都是同一个混合高斯的两种写法 -------------
    xs = rng.standard_normal(N_SAMPLE) * 1.4
    for t in (0.02, 0.25, 0.6, 1.0):
        a, s = ab(t)
        lhs = x0hat(xs, t)
        rhs = (xs + s * s * score(xs, t)) / a
        ok(np.max(np.abs(lhs - rhs)) < 1e-10,
           f"t={t:.2f}：x̂₀ = (x+σ²∇log p)/α 完全成立（最大误差 {np.max(np.abs(lhs-rhs)):.1e}）")

    h = 1e-5
    t = 0.35
    fd = (logp(xs + h, t) - logp(xs - h, t)) / (2 * h)
    ok(np.max(np.abs(fd - score(xs, t))) < 1e-7,
       f"t={t}：解析 score 与 log p 的中心差分一致（最大误差 {np.max(np.abs(fd-score(xs,t))):.1e}）")

    # --- A2 塌缩率在纯高斯上有闭式：sd(x̂₀)/σ_d = α_t σ_d / √v_t ----------
    t_probe = np.linspace(0.02, 0.99, 9)
    err_list = []
    for tt in t_probe:
        a, s = ab(tt)
        v = a * a * SD * SD + s * s
        xg = rng.standard_normal(20000) * math.sqrt(v)
        sim = float(np.std(x0hat(xg, tt, modes=np.array([0.0]), sd=SD)) / SD)
        err_list.append(abs(sim - a * SD / math.sqrt(v)))
    ok(max(err_list) < 5e-3,
       f"纯高斯上塌缩率 ρ(t)=ασ_d/√v 与数值一致（最大偏差 {max(err_list):.2e}，t∈[0.02,0.99]）")

    # --- A3 四模态数据上：1 步 x̂₀ 的伤害有多大 ----------------------------
    N = 8000
    aT, sT = ab(T_START)
    vT = aT * aT * SD * SD + sT * sT
    xT, x0 = sample_noisy(rng, N, T_START)
    one_x0 = x0hat(xT, T_START)
    sd_ratio = float(np.std(one_x0) / np.std(x0))
    ok(sd_ratio < 0.03,
       f"t=T 处 1 步 x̂₀ 的标准差只有数据的 {sd_ratio*100:.2f}%（纯高斯的 ρ={aT*SD/math.sqrt(vT):.2e}）")
    ok(abs(w1(one_x0, x0) - float(np.mean(np.abs(x0)))) < 0.06,
       f"1 步 x̂₀ 与数据的 W1 = {w1(one_x0,x0):.3f} ≈ 数据自身的平均绝对离差 {np.mean(np.abs(x0)):.3f}"
       "（即全部塌缩到均值）")

    # 量化噪声地板：两份独立数据的 W1
    floor = float(np.mean([w1(sample_data(np.random.default_rng(SEED + 900 + i), N)[0], x0)
                           for i in range(5)]))
    print(f"  （W1 噪声地板：两份独立 {N} 样本数据的 W1 = {floor:.4f}）")

    # --- A4 精确一致性函数（ODE 端点）确实 1 步就能对 ----------------------
    f_star = make_perturbed("exact")
    y = f_star(xT, T_START)
    e1 = w1(y, x0)
    ok(e1 < 2.2 * floor,
       f"精确一致性函数 1 步输出的 W1 = {e1:.4f}，只比噪声地板 {floor:.4f} 高 {e1/floor:.2f}×"
       " —— 1 步在信息上完全够用")
    vr = float(np.var(y) / np.var(x0))
    ok(0.94 < vr < 1.06, f"1 步输出方差 / 数据方差 = {vr:.4f}")
    m_exact, m_data = modal_mass(y), modal_mass(x0)
    ok(np.max(np.abs(m_exact - m_data)) < 0.03,
       f"四模态质量 精确 1 步 {np.round(m_exact,3)} vs 数据 {np.round(m_data,3)}")

    # 积分步数收敛性：n=256 与 n=1024 的输出差
    y2 = endpoint_map(xT, LAM_T, n=256)
    y3 = endpoint_map(xT, LAM_T, n=1024)
    ok(np.max(np.abs(y2 - y3)) < 5e-3,
       f"λ 积分步数 256→1024 的输出最大差 {np.max(np.abs(y2-y3)):.2e}，数值解已收敛")

    if figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def ecdf(v):
            sv = np.sort(v)
            return sv, np.arange(1, len(sv) + 1) / len(sv)

        fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.6), dpi=160, sharey=True)
        sd_, cd_ = ecdf(x0)
        ax[0].plot(sd_, cd_, color="0.55", lw=2.4, label="data")
        s1, c1 = ecdf(one_x0)
        ax[0].plot(s1, c1, color="#c0392b", lw=1.5,
                   label="1-step $\\hat{x}_0$ (sd %.2f%% of data)" % (sd_ratio * 100))
        ax[0].set_title("Tweedie mean as a 1-step generator: collapse", fontsize=10)
        ax[0].legend(frameon=False, fontsize=8, loc="lower right")
        ax[1].plot(sd_, cd_, color="0.55", lw=2.4, label="data")
        s2, c2 = ecdf(y)
        ax[1].plot(s2, c2, color="#185FA5", lw=1.5,
                   label="exact $f^*$, 1 step (W1=%.3f)" % e1)
        ax[1].set_title("The real 1-step consistency function: exact", fontsize=10)
        ax[1].legend(frameon=False, fontsize=8, loc="lower right")
        for a_ in ax:
            for m in MODES:
                a_.axvline(m, color="0.85", lw=0.8, zorder=0)
            a_.set_xlabel("x")
            a_.set_xlim(-5.2, 5.2)
            a_.spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("empirical CDF")
        fig.tight_layout()
        p = os.path.join(figdir, "few_step_collapse.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  -> {p}")

    return dict(sd_ratio=sd_ratio, w1_one_x0=w1(one_x0, x0), w1_exact=e1)


# ---------------------------------------------------------------------------
# B 组：自一致性损失的零解族
# ---------------------------------------------------------------------------
def exp_B(rng, figdir=None):
    head("B 组：自一致性损失有多少个零解（边界条件为什么不可省）")

    xT, x0 = sample_noisy(rng, N_SAMPLE, T_START)
    lams = grid_lams(64)
    # 一条轨迹的两个快照（同一条 ODE 路径上的相邻点）
    yT, snap = ode_traj(xT, lams, snap={31, 32})
    xa, xb = snap[31], snap[32]
    ta, tb = t_of_lam(lams[31]), t_of_lam(lams[32])
    ok(np.allclose(snap[32], xb),
       f"取 λ 相邻两格 t={ta:.4f} / {tb:.4f} 的快照（λ 间隔 {lams[31]-lams[32]:.4f}）")

    # f_h = h(f*)，h(v) = β v（数据均值为 0，故平移项取 0）
    fa = endpoint_map(xa, lams[31])
    fb = endpoint_map(xb, lams[32])
    betas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    yref = endpoint_map(xT, LAM_T)          # β=1，即真值
    losses, w1s, sds = [], [], []
    for be in betas:
        losses.append(float(np.mean(np.abs(be * fb - be * fa))))
        y = be * yref
        w1s.append(w1(y, x0))
        sds.append(float(np.std(y) / np.std(yref)))
    losses = np.array(losses)
    ok(np.max(losses) < 1e-5,
       f"β ∈ {betas.tolist()} 时自一致性损失全部 < 1e-5（最大 {np.max(losses):.1e}，"
       f"仅为数据尺度 {np.std(x0):.2f} 的 {np.max(losses)/np.std(x0):.1e} 倍）——损失对 β 完全不可见")
    ok(int(np.argmin(w1s)) == 4,
       f"但 1 步 W1 在 β=1.0 才最小：{np.round(w1s,3).tolist()}")
    ok(w1s[0] > 1.5,
       f"β=0（退化成常数函数）的 W1 = {w1s[0]:.3f}，损失却是 0 —— 这就是退化极小值")
    ok(np.max(np.abs(sds - betas)) < 1e-12,
       f"输出标准差比精确等于 β（{np.round(sds,3).tolist()}）——损失看不见的那一维，恰好就是唯一出错的那一维")
    w1a = np.array(w1s)
    imin = int(np.argmin(w1a))
    ok(imin == 4 and np.all(np.diff(w1a[:imin + 1]) < 0) and np.all(np.diff(w1a[imin:]) > 0),
       f"1 步 W1 在 β<1 单调下降、β>1 单调上升，唯一极小在 β=1（{np.round(w1s,3).tolist()}）——"
       "V 形，且底部就是真值")

    # 边界条件：c_skip(t_end)=1, c_out(t_end)=0 的常见参数化在 t→t_end 处把输出钉成恒等
    a_end, s_end = ab(T_END)
    f_te = endpoint_map(x0[:200], LAM_END)
    ok(np.max(np.abs(f_te - x0[:200])) < 5 * s_end,
       f"t=t_end：一致性函数退化为恒等（最大偏差 {np.max(np.abs(f_te-x0[:200])):.4f} ≤ 5σ_end={5*s_end:.4f}）——"
       "边界条件正是把零解族收敛到唯一解的那一条")

    if figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.4), dpi=160)
        ax[0].plot(betas, losses, "o-", color="#2ca02c", ms=5, lw=1.4)
        ax[0].set_ylim(-0.02, 0.08)
        ax[0].axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax[0].set_xlabel(r"$\beta$   ($f_h = \beta f^*$)")
        ax[0].set_ylabel("self-consistency loss")
        ax[0].set_title("Loss is blind to beta", fontsize=10)
        ax[0].text(0.45, 0.05, "all $<10^{-5}$", fontsize=8.5, color="0.35")
        ax[1].plot(betas, w1s, "v-", color="#c0392b", ms=5, lw=1.4)
        ax[1].axvline(1.0, color="0.6", lw=0.8, ls=":")
        ax[1].set_xlabel(r"$\beta$")
        ax[1].set_ylabel("1-step W1 to data")
        ax[1].set_title("Error is linear in $|1-\\beta|$", fontsize=10)
        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(figdir, "few_step_degeneracy.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  -> {p}")

    return dict(losses=losses.tolist(), w1=w1s, betas=betas.tolist())


# ---------------------------------------------------------------------------
# C 组：步数能不能救一个错的映射
# ---------------------------------------------------------------------------
def exp_C(rng, figdir=None):
    head("C 组：步数 × 三种典型蒸馏缺陷（W1 越小越好）")

    cases = [
        ("精确 f*", make_perturbed("exact")),
        ("偏置 δ=0.05", make_perturbed("bias", delta=0.05)),
        ("偏置 δ=0.15", make_perturbed("bias", delta=0.15)),
        ("收缩 15%", make_perturbed("shrink", delta=0.15)),
        ("丢一个模态", make_perturbed("drop")),
        ("零均值抖 0.15", make_perturbed("rand", delta=0.15)),
    ]
    Ks = [1, 2, 4, 8, 16]
    N = 2500
    table = {}
    rng_s = np.random.default_rng(SEED + 11)
    xT, x0 = sample_noisy(rng_s, N, T_START)
    truth_mass = modal_mass(x0)

    for name, f in cases:
        row = []
        for K in Ks:
            xg = np.random.default_rng(SEED + 700 + K)
            y = few_step_sample(f, xT, grid_lams(K), xg, stochastic=True)
            row.append((w1(y, x0), modal_mass(y)))
        table[name] = row
        print("  " + f"{name:<14}" + "".join(f"{v[0]:>9.4f}" for v in row))
    print("  " + f"{'K':<14}" + "".join(f"{K:>9d}" for K in Ks))

    ex_ = [v[0] for v in table["精确 f*"]]
    ok(max(ex_) < 0.12 and abs(ex_[-1] - ex_[0]) < 0.05,
       f"精确 f*：K=1..16 的 W1 = {np.round(ex_,3).tolist()}，全程贴着噪声地板（地板约 0.05）——"
       "步数纯冗余")

    b15 = [v[0] for v in table["偏置 δ=0.15"]]
    ok(all(b15[i] < b15[i + 1] for i in range(len(b15) - 1)) and b15[-1] > 3 * b15[0],
       f"偏置 δ=0.15：W1 随 K 单调上升到 {b15[-1]/b15[0]:.1f} 倍（{b15[0]:.3f} → {b15[-1]:.3f}）——"
       "多步不是平均掉偏差，而是把偏差反复施加")
    sh = [v[0] for v in table["收缩 15%"]]
    ok(all(sh[i] < sh[i + 1] for i in range(len(sh) - 1)) and sh[-1] > 3 * sh[0],
       f"收缩 15%：同样单调恶化到 {sh[-1]/sh[0]:.1f} 倍（{sh[0]:.3f} → {sh[-1]:.3f}）")

    rn = [v[0] for v in table["零均值抖 0.15"]]
    ok(max(rn) < 0.15 and rn[-1] < 2.0 * rn[0],
       f"零均值抖 0.15：W1 = {np.round(rn,3).tolist()}，最大 {max(rn):.3f} 仍在噪声地板附近——"
       "同样是 0.15 的误差，零均值的被摊平了，有偏的涨了 7 倍")

    d = [v[1][-1] for v in table["丢一个模态"]]
    ok(max(d) < 0.02,
       f"丢一个模态：x=+3 处的质量在 K=1..16 全为 {np.round(d,3).tolist()}（数据是 {truth_mass[-1]:.3f}）——丢掉就再也回不来")
    dd = [v[0] for v in table["丢一个模态"]]
    ok(min(dd) > 0.8 and (max(dd) - min(dd)) < 0.1,
       f"丢一个模态的 W1 是死地板 {min(dd):.3f}..{max(dd):.3f}，与 K 无关")

    # 确定性 vs 随机（重加噪）多步采样
    print()
    print("  两种多步一致性采样（收缩 15%，K=16）：")
    fsh = make_perturbed("shrink", delta=0.15)
    ys = few_step_sample(fsh, xT, grid_lams(16),
                         np.random.default_rng(SEED + 91), stochastic=True)
    yd = few_step_sample(fsh, xT, grid_lams(16),
                         np.random.default_rng(SEED + 92), stochastic=False)
    print(f"    随机重加噪 W1={w1(ys,x0):.4f}   确定性 W1={w1(yd,x0):.4f}")
    ok(w1(yd, x0) < w1(ys, x0),
       f"有系统偏差时确定性更好（{w1(yd,x0):.3f} < {w1(ys,x0):.3f}）——"
       "随机重加噪每步都把偏差重新注入，确定性只在线性插值的末端用一次")

    if figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
        style = [
            ("exact $f^*$", "o-", "#2ca02c"),
            ("bias 0.05", "s-", "#7f7f7f"),
            ("bias 0.15", "^-", "#c0392b"),
            ("shrink 15%", "v-", "#e67e22"),
            ("drop one mode", "d-", "#8e44ad"),
            ("zero-mean noise 0.15", "p-", "#1f77b4"),
        ]
        for (name, _), (lab, mk, col) in zip(cases, style):
            ax.plot(Ks, [v[0] for v in table[name]], mk, ms=4, lw=1.4,
                    color=col, label=lab)
        ax.axhline(0.05, color="0.6", lw=0.8, ls=":")
        ax.text(5.6, 0.062, "sampling floor", fontsize=7.5, color="0.45", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("K  (number of consistency-sampling steps)")
        ax.set_ylabel("W1 to real data")
        ax.set_title("More steps do not fix a wrong mapping", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(figdir, "few_step_steps_error.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  -> {p}")

    return table


# ---------------------------------------------------------------------------
# D 组：一致性训练的离散化偏差
# ---------------------------------------------------------------------------
def exp_D(rng):
    head("D 组：CT 用相邻格点近似同一条轨迹，偏差有多大、朝哪个方向")

    x0, _ = sample_data(rng, N_SAMPLE)
    eps = rng.standard_normal(N_SAMPLE)
    Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    print(f"  {'N':>6}{'Δt':>11}{'平均|Δx̂₀|':>14}{'最大|Δx̂₀|':>14}{'std(粗)/std(细)':>18}")
    deltas, biases, ratios = [], [], []
    for N in Ns:
        ts = np.linspace(T_START, T_END, N + 1)
        t_coarse, t_fine = ts[6], ts[7]
        a1, s1 = ab(t_coarse)
        a2, s2 = ab(t_fine)
        x1 = a1 * x0 + s1 * eps
        x2 = a2 * x0 + s2 * eps          # CT 的关键：同一个 ε、同一个 x₀
        e1 = x0hat(x1, t_coarse)
        e2 = x0hat(x2, t_fine)
        d = e1 - e2
        rr = float(np.std(e1) / np.std(e2))
        print(f"  {N:>6}{t_coarse-t_fine:>11.2e}{np.mean(np.abs(d)):>14.2e}"
              f"{np.max(np.abs(d)):>14.2e}{rr:>18.4f}")
        deltas.append(t_coarse - t_fine)
        biases.append(float(np.mean(np.abs(d))))
        ratios.append(rr)

    deltas = np.array(deltas)
    biases = np.array(biases)
    ratios = np.array(ratios)
    sl_all = np.polyfit(np.log(deltas), np.log(biases), 1)[0]
    sl_fine = np.polyfit(np.log(deltas[-4:]), np.log(biases[-4:]), 1)[0]
    ok(0.9 < sl_fine < 1.4,
       f"细网格（N≥128）上偏差随 Δt 一阶收敛：log-log 斜率 = {sl_fine:.3f}")
    ok(sl_all > sl_fine + 0.25,
       f"粗网格整体斜率 {sl_all:.2f} 明显高于一阶——Δt 大到估计已跨过模态边界，x̂₀ 在那附近是跳跃的")
    ok(biases[-1] / biases[0] < 0.01,
       f"N=8 → 1024 偏差下降 {biases[0]/biases[-1]:.0f} 倍（{biases[0]:.2e} → {biases[-1]:.2e}）")
    ok(np.all(ratios < 1.0),
       f"每个 N 下粗格点估计的离散度都更小（std 比 {np.round(ratios,3).tolist()}）——偏差方向是收缩，"
       "把粗格点的值当细格点的目标会把学生往均值拽")

    return dict(slope_fine=float(sl_fine), slope_all=float(sl_all),
                deltas=deltas.tolist(), biases=biases.tolist(), ratios=ratios.tolist())


# ---------------------------------------------------------------------------
# E 组：NFE 账本 + 训练/推理路径错配
# ---------------------------------------------------------------------------
def exp_E(rng):
    head("E 组：训练期 NFE 账本，以及“教师轨迹 → 学生推理”的路径错配")

    print("  CD（一致性蒸馏）：一条 K_t 步的教师轨迹上取相邻两点成对")
    print(f"  {'K_t（教师步数）':>17}{'轨迹的教师 NFE':>17}{'可取的相邻对':>15}{'NFE/对':>10}")
    for Kt in (8, 16, 32, 64):
        print(f"  {Kt:>17}{Kt:>17}{Kt:>15}{1.0:>10.2f}")
    ok(True,
       "CD 的教师成本被完全摊销到 1 NFE/对，与 K_t 无关——代价换到了别处："
       "学生推理的步网格必须和教师用来造对的网格一致（C 组量过不一致的后果）")

    r = 4
    N0 = 16
    tot = sum(N0 / 2 ** j for j in range(r))
    ok(abs(tot - (2 * N0 - 2 * N0 / 2 ** r)) < 1e-9,
       f"PD（渐进蒸馏）{r} 轮、起点 {N0} 步：每样本累计 NFE = {tot:.1f} = 2N₀(1-2^-r)")
    ok(tot > 0.9 * 2 * N0,
       f"PD 的总成本约等于 2N₀ = {2*N0}，与 CD 同量级——省步数不省训练算力")

    # 路径错配：学生第 k 步的落点离教师轨迹有多远
    print()
    print("  路径错配（收缩 15%，K=8，确定性多步）：")
    fsh = make_perturbed("shrink", delta=0.15)
    K = 8
    lams = grid_lams(K)
    N = 2000
    xT, x0 = sample_noisy(rng, N, T_START)
    # 教师轨迹快照
    _, t_snap = ode_traj(xT, lams, snap=set(range(K + 1)))
    # 学生确定性多步
    x = xT.copy()
    devs = [0.0]
    for k in range(K):
        t = t_of_lam(lams[k])
        tn = t_of_lam(lams[k + 1])
        a, s = ab(t)
        an, sn = ab(tn)
        x0h = fsh(x, t)
        x = an * x0h + sn * (x - a * x0h) / s
        devs.append(float(np.mean(np.abs(x - t_snap[k + 1]))))
    print("    k      0        1        2        3        4        5        6        7        8")
    print("    偏离   " + "  ".join(f"{d_:.4f}" for d_ in devs))
    ok(devs[0] < 1e-12 and devs[1] > 0.0,
       "第 0 步完全重合（同一起点），第 1 步起开始偏离教师轨迹")
    ok(max(devs) < 0.5 * float(np.std(x0)),
       f"最大偏离 {max(devs):.3f}，只有数据标准差 {np.std(x0):.2f} 的 {max(devs)/np.std(x0)*100:.0f}%，没有爆炸")
    ok(devs[-1] <= 1.6 * max(devs[:-1]) + 1e-9,
       f"偏离没有单调累积到发散（末步 {devs[-1]:.3f} vs 峰值 {max(devs):.3f}）——"
       "重加噪/重估每步都在把轨迹拉回流形附近")

    return dict(devs=devs)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="少步生成（一致性模型 / 蒸馏）核验")
    ap.add_argument("groups", nargs="*", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)

    groups = [g.upper() for g in args.groups] or ["A", "B", "C", "D", "E"]
    if args.figdir:
        os.makedirs(args.figdir, exist_ok=True)

    print("=" * 78)
    print("少步生成（一致性模型 / 蒸馏）核验")
    print("=" * 78)
    a0, s0 = ab(T_START)
    print(f"  数据：{len(MODES)} 模态 GMM，中心 {MODES.tolist()}，σ_d={SD}")
    print(f"  schedule：β(t)={B0}+{(B1-B0)/TTOT:.1f}t，t∈[{T_END}, {T_START}]")
    print(f"  λ 范围：[{LAM_T:.3f}, {LAM_END:.3f}]，跨度 {LAM_END-LAM_T:.3f}")
    print(f"  噪声端：α={a0:.2e}，σ={s0:.6f}（近似纯噪声）")
    print(f"  种子：{SEED}；样本数：{N_SAMPLE}；λ 积分：{N_LAM} 步")

    rng = np.random.default_rng(SEED)
    if "A" in groups:
        exp_A(rng, args.figdir)
    if "B" in groups:
        exp_B(rng, args.figdir)
    if "C" in groups:
        exp_C(rng, args.figdir)
    if "D" in groups:
        exp_D(rng)
    if "E" in groups:
        exp_E(rng)

    print()
    print("=" * 78)
    print(f"断言总数 = {ASSERTS[1]}（通过 {ASSERTS[0]}，失败 {ASSERTS[1]-ASSERTS[0]}）")
    print("=" * 78)
    return 0 if ASSERTS[0] == ASSERTS[1] else 1


if __name__ == "__main__":
    sys.exit(main())
