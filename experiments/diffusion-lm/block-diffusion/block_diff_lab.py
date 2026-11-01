#!/usr/bin/env python3
"""块扩散（block diffusion）的两笔账：可表达性与算力

背景
----
离散扩散（掩码扩散）一次性并行解出整条序列，模型最强但也最贵；自回归最省但一次一个。
块扩散把两者接起来：**块内双向、块间因果**，一次解一个块。

这个设计有两个后果，都能精确算出来：

  1. 可表达性：同一批同时解出的 token 之间被强制**条件独立**。
     即使模型完美，能表示的分布族也装不下真实分布；缺的那块恰好是
     「同组 token 之间的条件 total correlation」。
  2. 算力：块间因果让已生成块的 KV 可以缓存，每步只算 B 个 query。
     省多少、代价在哪，是一本可以精确记账的流水。

本脚本在 V=4、L=6 的可枚举空间上把这两笔账都算到小数点后十几位：

  A  可表达性：KL(p‖q*) 与「条件 total correlation」逐位相等
  B  块大小 / 组大小 / 分组方式 三者的影响
  C  与自回归（B=1）和全序列扩散（B=L）的两端对齐
  D  KV 缓存：复用与全量重算数值一致，以及 cache 为什么依赖块间因果
  E  算力账本：全序列扩散 / 块扩散 / 自回归的 query-key 对数与交叉点

运行
----
    python block_diff_lab.py            # A–E 全部，约 5 秒
    python block_diff_lab.py A B        # 只跑指定组

依赖 numpy。固定种子，输出可复现。
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np

V = 4          # 词表大小
L = 6          # 序列长度
COPY = 0.85    # 生成数据时的「复制间隔 2」概率
SEED = 20261105

ASSERTS = [0, 0]


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
# 可枚举的真实分布：二阶马尔可夫链（以 0.85 的概率复制 x_{i-2}）
# ---------------------------------------------------------------------------
def make_joint():
    """返回形状 (V,)*L 的联合分布：强 distance-2 耦合，便于观察块内相关性。"""
    p = np.zeros((V,) * L)
    seqs = np.array(list(itertools.product(range(V), repeat=L)))
    for s in seqs:
        w = 1.0 / V                                   # x_1
        w *= 1.0 / V                                  # x_2
        for i in range(2, L):
            w *= COPY if s[i] == s[i - 2] else (1.0 - COPY) / (V - 1)
        p[tuple(s)] = w
    return p / p.sum()


def marginal(joint, keep):
    """把 keep 里列出的位置留下，其余求和掉；返回 (V,)*len(keep)"""
    axes = tuple(i for i in range(joint.ndim) if i not in keep)
    return joint.sum(axis=axes)


def entropy(p):
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl(p, q):
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    m = p > 0
    return float((p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))).sum())


# ---------------------------------------------------------------------------
# 分组：把位置按「同时解出」切成若干组，组内条件独立、组间按顺序条件依赖
# ---------------------------------------------------------------------------
def groups_for(L_, block, step):
    """块大小 block、组大小 step（step | block）。返回 [[位置...], ...]，已按揭示顺序。"""
    assert block % step == 0
    out = []
    for b0 in range(0, L_, block):
        for g0 in range(0, block, step):
            out.append(list(range(b0 + g0, b0 + g0 + step)))
    return out


def pre_sets(groups):
    """第 j 组的「已揭示前缀」= 之前所有组的并集（不假设组内/组间的下标顺序）。"""
    revealed, pres = [], []
    for g in groups:
        pres.append(sorted(revealed))
        revealed = revealed + list(g)
    return pres


def _full_rank(a, axes, ndim):
    """把 marginal 压出来的瘦数组放回全秩广播形状（轴位置由 axes 指定）。"""
    shape = [1] * ndim
    for k, ax in enumerate(axes):
        shape[ax] = a.shape[k]
    return a.reshape(shape)


def q_from_groups(joint, groups):
    """结构 q* = Π_j Π_{i∈G_j} p(x_i | x_{<G_j})。

    同组 token 是各自独立地从逐位置边缘采出来的（一次 τ-leaping 提交一组），
    所以组内不允许有条件依赖 —— 这正是并行的代价。
    """
    nd = joint.ndim
    q = np.ones_like(joint)
    for g, pre in zip(groups, pre_sets(groups)):
        den = np.maximum(_full_rank(marginal(joint, pre), pre, nd), 1e-300)
        for i in g:
            num = _full_rank(marginal(joint, pre + [i]), pre + [i], nd)
            q = q * (num / den)
    return q


def q_temp(joint, groups, temp):
    """家族内的扰动：把逐位置条件分布做温度缩放后重新归一（仍属于同一结构族）。"""
    nd = joint.ndim
    q = np.ones_like(joint)
    for g, pre in zip(groups, pre_sets(groups)):
        den = np.maximum(_full_rank(marginal(joint, pre), pre, nd), 1e-300)
        for i in g:
            num = _full_rank(marginal(joint, pre + [i]), pre + [i], nd)
            c = (num / den) ** (1.0 / temp)
            c = c / c.sum(axis=i, keepdims=True)
            q = q * c
    return q


def cond_tc(joint, groups):
    """Σ_j TC(p_{G_j | x_{<G_j}}) = Σ_j [Σ_{i∈G_j} H(x_i|x_<) - H(x_{G_j}|x_<)]"""
    total = 0.0
    for g, pre in zip(groups, pre_sets(groups)):
        h_pre = entropy(marginal(joint, pre))
        h_given = entropy(marginal(joint, pre + g)) - h_pre
        per = sum(entropy(marginal(joint, pre + [i])) - h_pre for i in g)
        total += per - h_given
    return float(total)


# ---------------------------------------------------------------------------
# A 组：可表达性下界 = 条件 total correlation
# ---------------------------------------------------------------------------
def exp_A(joint):
    head("A 组：完美模型下，块扩散的 KL 下界就是同组 token 之间的条件互信息")

    print(f"  {'块大小 B':>10}{'组大小 s':>10}{'组数':>6}{'KL(p‖q*)':>14}{'Σ 条件 TC':>14}{'差':>12}")
    rows = []
    for block, step in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (6, 1), (6, 2), (6, 3), (6, 6)]:
        g = groups_for(L, block, step)
        q = q_from_groups(joint, g)
        k = kl(joint, q)
        tc = cond_tc(joint, g)
        rows.append((block, step, len(g), k, tc))
        print(f"  {block:>10}{step:>10}{len(g):>6}{max(k,0.0):>14.6f}{max(tc,0.0):>14.6f}{abs(k-tc):>12.2e}")

    ok(max(abs(k - tc) for _, _, _, k, tc in rows) < 1e-12,
       "KL(p‖q*) 与 Σ 条件 TC 逐位相等（最大差 < 1e-12）——可表达性下界有闭式")

    # 极端情形：一次解完整条序列 ⇒ KL 就是整条序列的 total correlation
    g_all = [[i for i in range(L)]]
    q_all = q_from_groups(joint, g_all)
    tc_all = cond_tc(joint, g_all)
    h_sum = sum(entropy(marginal(joint, [i])) for i in range(L))
    h_joint = entropy(joint)
    ok(abs(tc_all - (h_sum - h_joint)) < 1e-12,
       f"s=L：KL = 整序列 total correlation = ΣH(x_i) - H(x) = {h_sum-h_joint:.6f}（逐位相等）")
    ok(abs(kl(joint, q_all) - (h_sum - h_joint)) < 1e-12,
       f"    数值 KL = {kl(joint,q_all):.6f}，与上式一致")

    # 下界的最优性：在同一个结构族内扰动（条件分布做温度缩放），KL 只会变大
    g3 = groups_for(L, 3, 3)
    k0 = kl(joint, q_from_groups(joint, g3))
    ks = [kl(joint, q_temp(joint, g3, T)) for T in (0.7, 0.85, 1.15, 1.4)]
    ok(all(k > k0 for k in ks) and ks[0] > ks[1] and ks[3] > ks[2],
       f"下界是最小值不是估计值：同一结构族内把条件分布做温度缩放 T=0.7/0.85/1.15/1.4，"
       f"KL 从 {k0:.4f} 升到 {np.round(ks,4).tolist()}，且离 T=1 越远越差")
    return rows, (h_sum, h_joint)


# ---------------------------------------------------------------------------
# B 组：块大小、组大小、分组方式
# ---------------------------------------------------------------------------
def exp_B(joint, figdir=None):
    head("B 组：块大小 / 组大小 / 分组方式各自损失多少")

    print("  （一）固定组大小 s=1（块内逐 token 解出），块大小无关")
    for block in (1, 2, 3, 6):
        g = groups_for(L, block, 1)
        print(f"      B={block}: KL={max(kl(joint, q_from_groups(joint,g)),0.0):.3e}")

    print("  （二）固定块大小 B=L=6，组大小 s 从 1 到 6")
    vals = []
    for step in (1, 2, 3, 6):
        g = groups_for(L, 6, step)
        vals.append((step, kl(joint, q_from_groups(joint, g))))
        print(f"      s={step}: KL={max(vals[-1][1],0.0):.6f}  组数={len(g)}")
    ok(all(vals[i][1] <= vals[i + 1][1] + 1e-12 for i in range(len(vals) - 1)),
       f"同一块内：组越大 KL 不减（{np.round(np.maximum([v[1] for v in vals], 0.0), 4).tolist()}）")
    ok(vals[1][1] < 1e-12,
       "但 s=1→2 一点没损失：相邻配对把「distance-2 复制」这条耦合整个留在了跨组条件里——"
       "丢掉的不是「同时解几个」，而是「同时解的那几个之间有没有相关性」")

    print("  （三）组大小都是 2，但分组方式不同（B=4，位置 1-4）")
    parts = {
        "相邻 (1,2)(3,4)": [[0, 1], [2, 3], [4], [5]],
        "间隔 (1,3)(2,4)": [[0, 2], [1, 3], [4], [5]],
        "交错 (1,4)(2,3)": [[0, 3], [1, 2], [4], [5]],
    }
    res = {}
    for name, g in parts.items():
        res[name] = kl(joint, q_from_groups(joint, g))
        print(f"      {name}: KL={max(res[name],0.0):.6f}")
    ok(res["间隔 (1,3)(2,4)"] > 1.0,
       f"把强耦合的 (1,3) 放进同一组：KL={res['间隔 (1,3)(2,4)']:.3f}，"
       "而相邻配对与交错配对都是 0.000 —— 同样的并行度、同样的组数，只换了配对方式")
    ok(all(v < 1e-3 for k, v in res.items() if k != "间隔 (1,3)(2,4)"),
       "另两种配对一点没损失：这块数据的耦合在距离 2 上，只被 (1,3) 这种配对撞上——"
       "所以并行分组要按数据自身的相关结构切，不能按位置平均切")

    if figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h_joint = entropy(joint)
        tc_all = sum(entropy(marginal(joint, [i])) for i in range(joint.ndim)) - h_joint
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=160)
        xs = [v[0] for v in vals]
        ys = [max(v[1], 0.0) for v in vals]
        ax.bar([str(x) for x in xs], ys, color="#185FA5", width=0.62)
        ax.axhline(tc_all, color="#c0392b", lw=1.2, ls="--")
        ax.text(1.45, tc_all - 0.28, "total correlation", color="#c0392b", fontsize=8)
        for i, y in enumerate(ys):
            ax.text(i, y + 0.06, f"{y:.3f}" if y > 0.01 else "0", ha="center", fontsize=8)
        ax.set_xlabel("group size $s$  (tokens unmasked per step)")
        ax.set_ylabel("KL$(p\\,||\\,q^*)$   (nats)")
        ax.set_title("Parallelism costs correlations, not count", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        pth = os.path.join(figdir, "block_diff_kl.png")
        fig.savefig(pth, dpi=150)
        plt.close(fig)
        print(f"  -> {pth}")
    return vals, res


# ---------------------------------------------------------------------------
# C 组：两端对齐（B=1 是自回归，B=L 是一次性并解）
# ---------------------------------------------------------------------------
def exp_C(joint):
    head("C 组：两端——B=1 精确等于自回归，s=L 精确等于一次性并解")

    g_ar = groups_for(L, 1, 1)
    q_ar = q_from_groups(joint, g_ar)
    ok(kl(joint, q_ar) < 1e-15,
       f"B=1：KL={kl(joint,q_ar):.2e}——块扩散退化成按位置的链式分解，表达力与自回归完全相同")

    logp = np.log(joint)
    cross = float(q_ar.ravel() @ logp.ravel())
    ok(abs(cross + entropy(joint)) < 1e-12,
       f"    且 Σ p·log q* = {cross:.6f} = -H(p)，即 q*=p（交叉熵等于自身熵）")

    g_par = [[i for i in range(L)]]
    q_par = q_from_groups(joint, g_par)
    indep = np.ones((V,) * L)
    for i in range(L):
        shape = [1] * L
        shape[i] = V
        indep = indep * marginal(joint, [i]).reshape(shape)
    ok(np.max(np.abs(q_par - indep)) < 1e-12,
       f"s=L：q* 恰好等于各位置边缘的乘积（最大差 {np.max(np.abs(q_par-indep)):.2e}）——"
       "一次性并解就是「假设所有位置独立」")
    ok(kl(joint, q_par) > 1.0,
       f"    此时 KL={kl(joint,q_par):.3f}，是这个数据上并行解码的天花板")
    return kl(joint, q_par)


# ---------------------------------------------------------------------------
# D 组：KV 缓存
# ---------------------------------------------------------------------------
def attn(q, k, v, mask):
    """单头注意力：mask 为 True 表示可见。"""
    d = q.shape[-1]
    s = q @ k.T / math.sqrt(d)
    s = np.where(mask, s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v


def block_mask(L_, block):
    """块间因果 + 块内双向：位置 i 可见 j 当且仅当 block(j) <= block(i)"""
    bi = np.arange(L_) // block
    return bi[:, None] >= bi[None, :]


def exp_D(joint, rng):
    head("D 组：KV 缓存——复用与全量重算逐位一致，前提是块间因果")

    block = 2
    n = L // block
    d = 8
    Q = rng.standard_normal((L, d))
    K = rng.standard_normal((L, d))
    Vv = rng.standard_normal((L, d))
    m = block_mask(L, block)
    full = attn(Q, K, Vv, m)

    # 逐块增量：第 b 块只用 K/V 的前 (b+1)*block 行
    inc = np.zeros_like(full)
    for b in range(n):
        lo, hi = b * block, (b + 1) * block
        mm = np.zeros((hi - lo, hi), dtype=bool)   # 本块 query 只看 [0, hi)
        mm[:, :hi] = True                          # 块间因果 + 块内双向（前缀全可见）
        inc[lo:hi] = attn(Q[lo:hi], K[:hi], Vv[:hi], mm)
    ok(np.max(np.abs(inc - full)) < 1e-12,
       f"逐块复用前 (b+1)B 行的 K/V 与全量重算完全一致（最大差 {np.max(np.abs(inc-full)):.2e}）")

    # 反例：块内改成「只看本块」（丢掉前文），或者去掉块间因果
    bad = np.zeros_like(full)
    for b in range(n):
        lo, hi = b * block, (b + 1) * block
        mm = np.zeros((hi - lo, hi), dtype=bool)
        mm[:, lo:hi] = True                        # 只看本块
        bad[lo:hi] = attn(Q[lo:hi], K[:hi], Vv[:hi], mm)
    ok(np.max(np.abs(bad - full)) > 1e-3,
       f"若把块内改成「只看本块」，输出差 {np.max(np.abs(bad-full)):.3f}——"
       "缓存能用的唯一原因是块间因果 + 块内双向")

    # 块内双向：同一块里后面的 token 也能看见前面的
    ok(bool(m[1, 0]) and bool(m[0, 1]),
       "同一块内两个位置互相可见（块内双向）；跨块只有后者能看前者（块间因果）")
    m1 = block_mask(L, 1)
    ok(np.array_equal(m1, np.tril(np.ones((L, L), dtype=bool))),
       "B=1 时同一套掩码退化成严格下三角——块扩散在 B=1 处就是标准因果注意力")
    return block


# ---------------------------------------------------------------------------
# E 组：算力账本（以 query-key 对数为单位）
# ---------------------------------------------------------------------------
def exp_E(figdir=None):
    head("E 组：算力账本——固定调用次数后，块扩散的 attention 成本是全序列扩散的 B/(2L)")

    L_ = 1024
    print(f"  口径：只数 attention 的 query-key 对数（不计 MLP 与 KV 投影）")
    print(f"  L={L_}，全序列扩散与块扩散用**同一个总调用次数** C = n·S_b = {L_}")
    print(f"  {'B':>6}{'每块步数':>10}{'块扩散':>14}{'无缓存':>14}{'全序列扩散':>14}{'块/全':>10}{'B(L+B)/2L²':>12}")

    C = L_
    base = C * L_ * L_
    rows = []
    for B in (8, 16, 32, 64, 128, 256):
        n = L_ // B
        S_b = max(1, C // n)
        c_cache = sum(S_b * B * (b + 1) * B for b in range(n))
        c_nocache = sum(S_b * B * L_ for b in range(n))
        form = B * (L_ + B) / (2.0 * L_ * L_)
        rows.append((B, S_b, c_cache, c_nocache, c_cache / base, form))
        print(f"  {B:>6}{S_b:>10}{c_cache:>14.3e}{c_nocache:>14.3e}{base:>14.3e}"
              f"{c_cache/base:>10.4f}{form:>12.4f}")
    ar = L_ * (L_ + 1) // 2
    print(f"  {'自回归':>6}{'—':>10}{ar:>14.3e}{'—':>14}{base:>14.3e}{ar/base:>10.5f}{'—':>12}")

    ok(max(abs(r[4] - r[5]) for r in rows) < 1e-12,
       "数值比与闭式 B(L+B)/(2L²) 逐位相等——账本不用靠跑")
    ok(rows[2][4] < 0.02,
       f"B=32 时只有全序列扩散的 {rows[2][4]*100:.2f}%：每步只算 B 个 query，而不是 L 个")
    ok(rows[0][4] < rows[-1][4] / 4,
       f"块越大越贵：B=8 是 {rows[0][4]*100:.3f}%，B=256 是 {rows[-1][4]*100:.2f}%"
       "（每步要覆盖的长前缀随 B 变长）")
    ok(all(r[3] > r[2] for r in rows),
       f"无缓存时分母变成 L：B=8 从 {rows[0][2]/base*100:.3f}% 涨到 {rows[0][3]/base*100:.3f}%"
       "——缓存不是锦上添花，它把「前缀重算」那一项直接抹掉")
    ok(ar < rows[0][4] * base,
       f"自回归仍然最省（{ar/base*100:.3f}%）：块扩散买的不是「更省」，"
       "是「B 个 token 同时出」")

    if figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Bs = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=160)
        ax.plot(Bs, [r[4] for r in rows], "o-", color="#185FA5", lw=1.4, ms=5,
                label="block diffusion / full")
        ax.plot(Bs, [r[5] for r in rows], "--", color="#c0392b", lw=1.0,
                label="$B(L+B)/2L^2$")
        ax.plot(Bs, [r[3] / base for r in rows], "s:", color="#8e44ad", lw=1.0, ms=4,
                label="same, no KV cache")
        ax.axhline(ar / base, color="#2ca02c", lw=0.9, ls="-.")
        ax.text(Bs[0], ar / base * 1.25, "autoregressive", color="#2ca02c", fontsize=7.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("block size $B$   (L=1024, same number of calls)")
        ax.set_ylabel("attention cost relative to full diffusion")
        ax.set_title("Blocks buy latency, not FLOPs", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        pth = os.path.join(figdir, "block_diff_cost.png")
        fig.savefig(pth, dpi=150)
        plt.close(fig)
        print(f"  -> {pth}")
    return rows


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="块扩散：可表达性与算力")
    ap.add_argument("groups", nargs="*", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)
    if args.figdir:
        import os
        os.makedirs(args.figdir, exist_ok=True)
    groups = [g.upper() for g in args.groups] or ["A", "B", "C", "D", "E"]

    print("=" * 78)
    print("块扩散（block diffusion）：把多少 token 同时解出来，就丢掉多少互信息")
    print("=" * 78)
    joint = make_joint()
    print(f"  词表 V={V}，长度 L={L}，可枚举序列 {V**L} 条")
    print(f"  数据：二阶马尔可夫链，P(x_i = x_{{i-2}}) = {COPY}")
    print(f"  联合熵 H(x) = {entropy(joint):.4f} nats，"
          f"独立时的 ΣH = {sum(entropy(marginal(joint,[i])) for i in range(L)):.4f} nats")
    print(f"  种子：{SEED}")

    rng = np.random.default_rng(SEED)
    info = {}
    if "A" in groups:
        info["A"] = exp_A(joint)
    if "B" in groups:
        info["B"] = exp_B(joint, args.figdir)
    if "C" in groups:
        info["C"] = exp_C(joint)
    if "D" in groups:
        info["D"] = exp_D(joint, rng)
    if "E" in groups:
        info["E"] = exp_E(args.figdir)

    print()
    print("=" * 78)
    print(f"断言总数 = {ASSERTS[1]}（通过 {ASSERTS[0]}，失败 {ASSERTS[1]-ASSERTS[0]}）")
    print("=" * 78)
    return 0 if ASSERTS[0] == ASSERTS[1] else 1


if __name__ == "__main__":
    sys.exit(main())
