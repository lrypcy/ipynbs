#!/usr/bin/env python3
"""视频扩散的 token 预算：把时间轴压下去，训练为什么是四次方的

背景
----
图像扩散的 token 数是 $$n_h n_w$$，视频多出一个时间轴 $$n_t$$，总 token
$$N=n_t n_h n_w$$。attention 的 query-key 对数按 $$N^2$$ 走，于是「帧数翻倍」
在所有量上都翻倍，唯独在训练成本上翻四倍。

这个脚本只做算术，但每一笔都有闭式、都能对账：

  A  token 账本：N 对 (帧数, 分辨率, patch, VAE 压缩) 的依赖；谁是线性、谁是平方
  B  三种注意力的成本与交叉点：全 3D / 时空分解 / 因果时间 + KV 缓存
  C  训练 vs 推理：成本随帧数的指数（实测斜率），以及为什么两者会分叉
  D  3D VAE 的时空压缩：$$f_t$$ 对时间平方项的压缩比，以及空间/时间的兑换率
  E  固定算力下的最优分配（网格搜索 + 与前缀假设的对比）

运行
----
    python video_budget_lab.py          # A–E 全部，约 1 秒
    python video_budget_lab.py A C

依赖 numpy。所有数字都是精确算术，没有采样。
"""
from __future__ import annotations

import argparse
import math
import sys
import unicodedata

import numpy as np

FPS = 24
ASSERTS = [0, 0]


def _dw(s):
    """显示宽度：CJK 全角算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w, right=False):
    """按显示宽度补齐（str.ljust/rjust 对 CJK 会算少一半）。"""
    s = str(s)
    fill = " " * max(0, w - _dw(s))
    return fill + s if right else s + fill


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
# token 账本
# ---------------------------------------------------------------------------
def shape(H, W, T, fs=8, ft=4, p=2):
    """(n_t, n_h, n_w) 与总 token 数。

    fs: VAE 空间下采样；ft: VAE 时间下采样；p: latent 上的 patch 边长
    """
    nh = H // (fs * p)
    nw = W // (fs * p)
    nt = max(1, T // ft)
    return nt, nh, nw, nt * nh * nw


def pairs_full(N):
    """全 3D 注意力：每个 token 看所有 token。"""
    return N * N


def pairs_factorized(nt, nh, nw):
    """时空分解：先帧内空间注意力，再跨帧时间注意力。"""
    nhw = nh * nw
    return nt * (nhw * nhw) + nhw * (nt * nt)


def pairs_causal(nt, nh, nw, dsteps):
    """因果时间 + KV cache：逐帧生成，第 t 帧的 query 只对本帧与之前帧。

    每次去噪要跑 dsteps 步，每步只算本帧 nhw 个 query。
    合计 = dsteps · n_hw² · n_t(n_t+1)/2

    注意：这条式子对「有无 KV cache」是同一个数——cache 剪掉的是重复的
    K/V 投影，不是重复的 query-key 配对。
    """
    nhw = nh * nw
    return dsteps * nhw * sum((t + 1) * nhw for t in range(nt))


def pairs_bidir(N, dsteps):
    """双向（非因果）采样：每步 N 个 query 都对 N 个 key。

    视频扩散默认就是这种——整条 clip 一起去噪。
    """
    return dsteps * N * N


# ---------------------------------------------------------------------------
# A 组
# ---------------------------------------------------------------------------
def exp_A():
    head("A 组：token 账本——谁线性、谁平方")

    base = dict(H=720, W=1280, T=96, fs=8, ft=4, p=2)
    nt, nh, nw, N = shape(**base)
    print(f"  基准：{base['H']}×{base['W']}，{base['T']} 帧（{base['T']/FPS:.1f}s @ {FPS}fps），"
          f"VAE {base['fs']}×/{base['ft']}×，patch {base['p']}")
    print(f"        n_t={nt}，n_h={nh}，n_w={nw}，N={N}")

    rows = []
    for key, mult in [("T", 2), ("H", 2), ("W", 2)]:
        cfg = dict(base)
        cfg[key] *= mult
        nt2, nh2, nw2, N2 = shape(**cfg)
        rows.append((key, N2 / N))
        print(f"  {key} 翻倍 → N×{N2/N:.2f}，训练 attention ×{(N2/N)**2:.2f}")
    ok(all(abs(r[1] - 2.0) < 1e-12 for r in rows),
       "帧数与分辨率翻倍都让 N 精确翻倍——token 数对三个旋钮都是线性的")

    # patch 与 VAE：平方反比
    _, _, _, Np4 = shape(**{**base, "p": 4})
    print(f"  patch {base['p']}→4：N={N}→{Np4}（×{Np4/N:.4f}）")
    print(f"        n_h {720//16}→{720//32}、n_w {1280//16}→{1280//32}——"
          "720 不被 32 整除，整除截断让实际比值略低于 1/4")
    # 整除安全的边长上验证精确的 1/4
    Na = shape(1024, 1024, 96, 8, 4, 2)[3]
    Nb = shape(1024, 1024, 96, 8, 4, 4)[3]
    ok(abs(Nb / Na - 0.25) < 1e-12,
       f"patch 边长翻倍 ⇒ token 数精确降到 1/4（1024² 上 {Na}→{Nb}）——"
       "patch 是平方反比，帧数只是线性")
    ok(abs(Np4 / N - 0.25) < 0.01,
       f"720p 上截断让它变成 {Np4/N:.4f}：分辨率不能被 fs·p 整除时，"
       "账本要先按整除算，别直接用 1/4 估")

    # VAE 时间压缩
    _, _, _, N_ft1 = shape(**{**base, "ft": 1})
    _, _, _, N_ft8 = shape(**{**base, "ft": 8})
    ok(abs(N_ft8 / N_ft1 - 1 / 8) < 1e-12,
       f"VAE 时间压缩 1→8：token 数 {N_ft1}→{N_ft8}，正好 1/8——"
       "这是 3D VAE 相对 2D VAE 唯一真正「省在时间轴」的地方")

    print()
    print("  典型配置对账（H×W，帧数，VAE，patch → N）")
    for H, W, T, fs, ft, p in [(256, 256, 16, 8, 4, 1), (512, 512, 16, 8, 4, 1),
                               (512, 512, 49, 8, 4, 2), (720, 1280, 96, 8, 4, 2),
                               (720, 1280, 96, 8, 4, 1), (1080, 1920, 192, 8, 4, 2)]:
        nt_, nh_, nw_, N_ = shape(H, W, T, fs, ft, p)
        print(f"    {H}×{W}, {T:>3} 帧, VAE {fs}/{ft}, patch {p}"
              f"  →  n_t={nt_:>2} n_h={nh_:>2} n_w={nw_:>2}  N={N_:>6}")
    return N


# ---------------------------------------------------------------------------
# B 组：三种注意力
# ---------------------------------------------------------------------------
def exp_B():
    head("B 组：三种注意力的成本与交叉点")

    nt, nh, nw = 24, 45, 80          # 720p / 96 帧 / patch 2 ⇒ n_t=24
    nhw = nh * nw
    N = nt * nhw
    d = 4                            # 采样步数（示意）
    full = pairs_full(N)
    fac = pairs_factorized(nt, nh, nw)
    bid = pairs_bidir(N, d)
    cau = pairs_causal(nt, nh, nw, d)
    print(f"  n_t={nt}，n_h×n_w={nhw}，N={N}，采样 {d} 步")
    print("  " + pad("方案", 26) + pad("query-key 对数", 18, True) + pad("相对全 3D", 12, True))
    for name, v in [("全 3D（训练，1 遍）", full),
                    ("时空分解（训练，1 遍）", fac),
                    (f"双向采样（{d} 步）", bid),
                    (f"因果+KV 采样（{d} 步）", cau)]:
        print("  " + pad(name, 26) + pad(f"{v:.3e}", 18, True) + pad(f"{v/full:.4f}", 12, True))

    # 闭式：fac = N·n_hw + n_hw·n_t²，full = N² ⇒ 比值 = 1/n_t + 1/n_hw
    ok(abs(fac / full - (1.0 / nt + 1.0 / nhw)) < 1e-12,
       f"分解/全3D = 1/n_t + 1/n_hw = {1/nt + 1/nhw:.6f}"
       f"（实测 {fac/full:.6f}）——空间项给 1/n_t，时间项给 1/n_hw，两个旋钮各管一项")
    ok(fac / full < 0.05,
       f"时空分解的训练成本只有全 3D 的 {fac/full*100:.2f}%——"
       f"n_hw 越大，这个比值越趋近 1/n_t")

    # 因果结构 vs 双向：只省 2×（这是「结构」的红利，不是 KV cache 的功劳）
    ok(abs(cau / bid - (nt + 1) / (2 * nt)) < 1e-12,
       f"因果/双向 = (n_t+1)/(2n_t) = {(nt+1)/(2*nt):.6f}"
       f"（实测 {cau/bid:.6f}）")
    ok(1.90 < bid / cau < 2.00,
       f"因果结构相对双向只省 {bid/cau:.3f}×，上界就是 2×——"
       "总 query 数没变（都是 N），"
       "变的是平均 key 数从 n_t·n_hw 落到 (n_t+1)/2·n_hw")
    print(f"  （n_t=6 时省 {2*6/7:.3f}×，n_t=96 时省 {2*96/97:.3f}×，"
          "再长也压在 2× 以下）")

    # KV cache 省的是另一笔账：K/V 投影的 token 次数，不是 attention 对数
    print()
    print(f"  KV cache 到底省哪一笔（因果采样，d={d}）")
    print("  " + pad("", 18) + pad("query-key 对数", 18, True)
          + pad("K/V 投影 token 次", 20, True))
    kv = [("不缓存", pairs_causal(nt, nh, nw, d), d * nhw * nt * (nt + 1) // 2),
          ("缓存", pairs_causal(nt, nh, nw, d), N)]
    for name, p_, q_ in kv:
        print("  " + pad(name, 18) + pad(f"{p_:.3e}", 18, True) + pad(f"{q_:.0f}", 20, True))
    ok(kv[0][1] == kv[1][1],
       "对数这一列两行完全相同：KV cache 在 attention 对数上一分不省——"
       "剪掉的是重复的 K/V 投影，不是重复的配对")
    ok(abs(kv[0][2] / kv[1][2] - d * (nt + 1) / 2) < 1e-12,
       f"投影这一列缓存省 {kv[0][2]/kv[1][2]:.0f}×，正好 d(n_t+1)/2——"
       "「KV cache 有 n_t 倍加速」这句话的真身在这里，在投影上，不在配对上")

    # 采样一遍 vs 前向一遍
    ok(cau > fac,
       f"{d} 步因果采样（{cau:.3e}）比分解注意力的一次训练前向（{fac:.3e}）贵 "
       f"{cau/fac:.1f} 倍——训练一遍只跑一次，采样要同一 clip 跑 {d} 遍")
    ok(cau > full,
       f"甚至比全 3D 的一次前向（{full:.3e}）还贵 {cau/full:.3f}×，"
       f"而这个比值就是 (n_t+1)/(2n_t)·d ≈ d/2")
    ok(abs(cau / full - d * (nt + 1) / (2 * nt)) < 1e-12,
       f"因果采样/全3D 前向 = d(n_t+1)/(2n_t) = {d*(nt+1)/(2*nt):.6f}（实测 "
       f"{cau/full:.6f}）——与分辨率、与 clip 长度都无关，只与步数 d 成正比")

    # 交叉点：n_t(n_hw-1) > n_hw 时分解更省
    print()
    print("  时空分解 vs 全 3D：相等条件 n_t·n_hw = n_hw + n_t ⇔ n_t = n_hw/(n_hw-1)")
    for nhw_ in (2, 16, 256, 4096):
        print(f"    n_hw={nhw_:>5}：临界 n_t = {nhw_/(nhw_-1):.4f}")
    grid = [(w_, n_) for w_ in (2, 4, 16, 256, 4096) for n_ in range(1, 40)]
    ok(all((pairs_factorized(n_, 1, w_) < pairs_full(n_ * w_)) == (n_ * (w_ - 1) > w_)
           for w_, n_ in grid),
       "分解更省的充要条件是 n_t(n_hw-1) > n_hw，等价于 n_t > n_hw/(n_hw-1) ≤ 2——"
       "只要 n_t ≥ 2（n_hw ≥ 4）就成立；n_t=1 时那项 n_hw·n_t² 纯属白付")

    # 拆成两笔账：空间项 N·n_hw 与时间项 n_hw·n_t²，比值恰好 n_t/n_hw
    print()
    print("  分解后两项各占多少（空间 = N·n_hw，时间 = n_hw·n_t²）")
    print("  " + pad("配置", 30) + pad("空间项", 13, True) + pad("时间项", 13, True)
          + pad("时间/空间", 11, True))
    splits = []
    for H, W, T, fs, ft, p in [(256, 256, 16, 8, 4, 1), (512, 512, 49, 8, 4, 2),
                               (720, 1280, 96, 8, 4, 2), (720, 1280, 384, 8, 4, 2),
                               (1080, 1920, 192, 8, 4, 2)]:
        nt_, nh_, nw_, N_ = shape(H, W, T, fs, ft, p)
        sp = N_ * nh_ * nw_
        tp = nh_ * nw_ * nt_ * nt_
        splits.append((tp / sp, nt_, nh_ * nw_))
        print("  " + pad(f"{H}x{W}, {T} 帧, p={p}", 30) + pad(f"{sp:.3e}", 13, True)
              + pad(f"{tp:.3e}", 13, True) + pad(f"{tp/sp:.5f}", 11, True))
    ok(all(abs(rt - n_ / w_) < 1e-12 for rt, n_, w_ in splits),
       "时间/空间 = n_t/n_hw，精确到 1e-12——时间注意力的标价是 n_hw·n_t²，"
       "只占空间项的 n_t/n_hw")
    ok(all(rt < 0.05 for rt, _, _ in splits),
       f"所有配置下时间项都不到空间项的 5%（720p/96 帧时只有 {splits[2][0]*100:.2f}%）"
       "——视频扩散的注意力账单几乎全在帧内空间，不在跨帧时间")
    return dict(full=full, fac=fac, bid=bid, cau=cau)


# ---------------------------------------------------------------------------
# C 组：训练 vs 推理的指数
# ---------------------------------------------------------------------------
def exp_C():
    head("C 组：帧数翻倍时，训练与采样各自的指数")

    nh, nw = 45, 80
    nhw = nh * nw
    d = 4
    Ts = [24, 48, 96, 192, 384]
    print(f"  n_h×n_w={nhw}，采样 {d} 步")
    print("  " + pad("T（帧）", 8, True) + pad("n_t", 6, True)
          + pad("训练（全 3D 前向）", 20, True) + pad("因果+KV 采样", 18, True)
          + pad("训练/采样", 12, True))
    nts, tr, inf = [], [], []
    for T in Ts:
        nt = T // 4
        f = pairs_full(nt * nhw)
        c = pairs_causal(nt, nh, nw, d)
        nts.append(nt)
        tr.append(f)
        inf.append(c)
        print("  " + pad(T, 8, True) + pad(nt, 6, True) + pad(f"{f:.3e}", 20, True)
              + pad(f"{c:.3e}", 18, True) + pad(f"{f/c:.3f}", 12, True))

    sl_tr = np.polyfit(np.log(nts), np.log(tr), 1)[0]
    sl_inf = np.polyfit(np.log(nts), np.log(inf), 1)[0]
    ok(abs(sl_tr - 2.0) < 1e-9,
       f"训练成本随 n_t 的 log-log 斜率 = {sl_tr:.4f}（平方）——"
       "帧数翻倍，训练算力翻四倍")
    ok(1.94 < sl_inf < 2.0,
       f"因果 + KV 采样的斜率 = {sl_inf:.4f}，略低于 2 正是因为 (n_t+1) 里的那个 +1"
       "（首帧只见自己）")
    ok(abs(sl_tr - sl_inf) < 0.06,
       "两者同为平方：因果结构没有改变指数，只改变常数——"
       "「采样是线性的」是把 token 数 N 当成算力，漏掉了 d 步")

    ratio = [a / b for a, b in zip(tr, inf)]
    ok(all(ratio[i] < ratio[i + 1] for i in range(len(ratio) - 1))
       and abs(ratio[-1] - 2.0 / d) < 0.01,
       f"训练/采样 的比值单调升向 2/d = {2.0/d:.3f}"
       f"（{ratio[0]:.3f} → {ratio[-1]:.3f}）——与 n_t 无关，只由步数定")
    ok(all(2.0 * n / (d * (n + 1)) < 1.0 for n in (1, 3, 24, 96)),
       f"d={d} ≥ 2 时采样一遍严格比训练一遍贵（每 clip 被重跑 d 遍，因果只省回 2×）")
    ok(2.0 * 24 / (1 * 25) > 1.0,
       "d=1 时反过来：采样一遍比训练一遍便宜——"
       "这就是少步生成把整条账翻过来的地方（对应少步生成那篇）")
    return sl_tr, sl_inf


# ---------------------------------------------------------------------------
# D 组：3D VAE 的时空压缩
# ---------------------------------------------------------------------------
def exp_D():
    head("D 组：3D VAE 的时空压缩——时间轴压缩比空间轴贵还是便宜")

    H, W, T = 720, 1280, 96
    print(f"  固定 {H}×{W}、{T} 帧，目标：把 N 压到基准的给定比例")
    print("  " + pad("基准", 26) + pad("N", 10, True) + pad("相对", 9, True))
    base = shape(H, W, T, fs=8, ft=1, p=1)[3]
    cfg = [("fs=8, ft=1, p=1", dict(fs=8, ft=1, p=1)),
           ("fs=8, ft=4, p=1", dict(fs=8, ft=4, p=1)),
           ("fs=8, ft=1, p=2", dict(fs=8, ft=1, p=2)),
           ("fs=8, ft=4, p=2", dict(fs=8, ft=4, p=2)),
           ("fs=16, ft=4, p=2", dict(fs=16, ft=4, p=2))]
    for name, kw in cfg:
        N = shape(H, W, T, **kw)[3]
        print("  " + pad(name, 26) + pad(N, 10, True) + pad(f"{N/base:.4f}", 9, True))

    # 兑换率：时间压缩 4 倍 vs 空间压缩 2 倍（各 4×）
    N_t4 = shape(H, W, T, fs=8, ft=4, p=1)[3]
    N_s2 = shape(H, W, T, fs=16, ft=1, p=1)[3]
    ok(abs(N_t4 / N_s2 - 1.0) < 1e-12,
       f"ft 1→4（4×）与 fs 8→16（空间 4×）压出完全相同的 token 数（{N_t4} = {N_s2}）——"
       "在 token 账本上两者等价，差别只在信息量")

    # 但注意力：时间平方项对 ft 的敏感度
    nt1, nh1, nw1 = shape(H, W, T, fs=8, ft=1, p=1)[:3]
    nt4, nh4, nw4 = shape(H, W, T, fs=8, ft=4, p=1)[:3]
    f1 = pairs_full(nt1 * nh1 * nw1)
    f4 = pairs_full(nt4 * nh4 * nw4)
    ok(abs(f1 / f4 - 16.0) < 1e-9,
       f"ft=1→4 让 N 降到 1/4，attention 成本降到 1/16（{f1:.3e}→{f4:.3e}）——"
       "时间压缩的收益在平方项上被放大一次")

    # 因果 VAE 的代价：首帧 token 少？给出「每帧 token 数随 ft 的变化」
    print()
    print("  每帧平均 token 数（决定 KV cache 的粒度）")
    for ft in (1, 2, 4, 8):
        nt_, nh_, nw_, N_ = shape(H, W, T, fs=8, ft=ft, p=2)
        print(f"    ft={ft}: n_t={nt_:>3}，每帧 {N_/nt_:>6.1f} token，总 N={N_}")
    return N_t4


# ---------------------------------------------------------------------------
# E 组：固定算力下的最优分配
# ---------------------------------------------------------------------------
def exp_E():
    head("E 组：固定算力预算，帧数与分辨率怎么分配")

    budget = 3.0e11              # query-key 对（一次训练前向，全 3D）
    Nmax = math.sqrt(budget)
    print(f"  预算（全 3D 训练的 query-key 对数）= {budget:.1e}，对应 N* = √budget = {Nmax:.0f}")
    print("  闭式：N = n_t·n_h²，N² ≤ budget ⇒ n_h = √(√budget / n_t) ⇒ 边长 ∝ T^(-1/2)")
    print("  " + pad("T（帧）", 8, True) + pad("n_t", 6, True) + pad("边长*（连续）", 18, True)
          + pad("边长（取 16 倍数）", 20, True) + pad("N", 10, True) + pad("每秒 token", 13, True))
    rows = []
    for T in (24, 48, 96, 192, 384, 768):
        nt = T // 4
        nh = math.sqrt(Nmax / nt)
        S_star = 16.0 * nh
        S = (int(S_star) // 16) * 16
        nh_i = S // 16
        N = nt * nh_i * nh_i
        rows.append((T, nt, S_star, S, N, N / (T / FPS)))
        print("  " + pad(T, 8, True) + pad(nt, 6, True) + pad(f"{S_star:.0f}", 18, True)
              + pad(S, 20, True) + pad(N, 10, True) + pad(f"{N/(T/FPS):.0f}", 13, True))

    r = [row[2] / rows[0][2] for row in rows]
    th = [math.sqrt(rows[0][0] / row[0]) for row in rows]
    ok(max(abs(a - b) for a, b in zip(r, th)) < 1e-12,
       f"固定预算下最大边长 ∝ T^(-1/2)（比值 {np.round(r,3).tolist()} vs 理论 "
       f"{np.round(th,3).tolist()}）——帧数翻 4 倍，边长只能砍一半")
    ok(all(abs(row[3] - row[2]) < 16.0 and row[4] * row[4] <= budget for row in rows),
       "取 16 倍数后的整数解仍在预算内，且与连续解相差不到 16 px")
    ok(rows[-1][5] < rows[0][5],
       f"每秒钟的 token 数随帧数下降（{rows[0][5]:.0f} → {rows[-1][5]:.0f}）——"
       "平方预算下，长视频只能靠降分辨率换")

    # 换成固定分辨率：两种注意力各能撑多少秒
    print()
    nhw = 45 * 80                    # 720p / patch 2
    print(f"  固定 720p（n_hw={nhw}）、固定预算，两种注意力各能撑多久：")
    nt_full = int(Nmax // nhw)                       # n_t·n_hw ≤ √budget
    # 分解：n_t·n_hw² + n_hw·n_t² ≤ budget ⇒ n_t² + n_hw·n_t − budget/n_hw = 0
    nt_fac = int((-nhw + math.sqrt(nhw * nhw + 4.0 * budget / nhw)) / 2.0)
    for name, n_ in [("全 3D", nt_full), ("时空分解", nt_fac)]:
        sec = n_ * 4 / FPS
        print("    " + pad(name, 10) + f"n_t={n_:>5}，T={n_*4:>6} 帧，"
              + (f"{sec:>8.2f} 秒" if sec < 90 else f"{sec/60:>8.1f} 分钟"))
    ok(abs(pairs_full(nt_full * nhw) / budget - 1.0) < 0.02,
       f"全 3D：n_t={nt_full} 正好把预算用满（{pairs_full(nt_full*nhw):.3e}）")
    ok(abs(pairs_factorized(nt_fac, 1, nhw) / budget - 1.0) < 0.02,
       f"时空分解：n_t={nt_fac} 把同一笔预算用满（{pairs_factorized(nt_fac,1,nhw):.3e}）")
    ok(nt_fac / nt_full > 40,
       f"同预算、同分辨率，分解注意力能撑 {nt_fac/nt_full:.0f}× 的帧数"
       f"（{nt_full*4/FPS:.0f} 秒 → {nt_fac*4/FPS/60:.0f} 分钟）——"
       "注意力结构一变，瓶颈就从注意力挪到了别处")

    # 全 3D 前向：分辨率每涨一档，能撑的时长按平方反比缩
    print()
    print("  固定分辨率、同一笔预算，全 3D 前向能撑多长：")
    durs = []
    for S in (720, 1080):
        nh_ = S // 16
        nt_ = int(Nmax // (nh_ * nh_))
        durs.append(nt_ * 4 / FPS)
        print("    " + pad(f"{S}p", 8) + f"n_t={nt_:>4}，T={nt_*4:>5} 帧，"
              f"{nt_*4/FPS:>6.1f} 秒")
    ok(abs(durs[1] / durs[0] - (720.0 / 1080.0) ** 2) < 0.01,
       f"720p 能撑 {durs[0]:.0f} 秒、1080p 只有 {durs[1]:.0f} 秒，比值 {durs[0]/durs[1]:.3f} "
       f"就是 (1080/720)² = {(1080/720)**2:.3f}——"
       "在平方预算里，分辨率涨一档就等于时长砍一半")
    return rows


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def make_figs(figdir):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#343A40",
                         "axes.labelcolor": "#212529", "text.color": "#212529",
                         "xtick.color": "#343A40", "ytick.color": "#343A40",
                         "figure.facecolor": "white", "axes.facecolor": "white"})

    nt, nh, nw, d = 24, 45, 80, 4
    nhw = nh * nw
    N = nt * nhw
    full = pairs_full(N)
    fac = pairs_factorized(nt, nh, nw)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # (a) 四种方案相对一次全 3D 前向
    ax = axes[0]
    names = ["full 3D\ntrain", "factorized\ntrain",
             "bidirectional\nsample", "causal+KV\nsample"]
    vals = [1.0, fac / full, pairs_bidir(N, d) / full,
            pairs_causal(nt, nh, nw, d) / full]
    cols = ["#4C6EF5", "#12B886", "#F08C00", "#E8590C"]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 8.0)
    ax.axhline(1.0, color="#868E96", lw=1, ls="--")
    ax.set_ylabel("query-key pairs  /  one full-3D training pass")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"(a) one pass at n_t={nt}, n_hw={nhw}, d={d}", fontsize=10)

    # (b) 因果采样随步数 d：与两条训练线交叉
    ax = axes[1]
    ds = np.arange(1, 17)
    cost = np.array([pairs_causal(nt, nh, nw, int(k)) / full for k in ds])
    ax.plot(ds, cost, "o-", color="#E8590C", ms=4, lw=1.6,
            label=f"causal+KV sampling (n_t={nt})")
    ax.axhline(1.0, color="#4C6EF5", lw=1.4, label="full-3D training pass")
    ax.axhline(fac / full, color="#12B886", lw=1.4, label="factorized training pass")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 30.0)
    ax.set_xlabel("sampling steps per clip  d")
    ax.set_ylabel("cost  /  one full-3D training pass")
    dstar = 2.0 * nt / (nt + 1.0)
    ax.axvline(dstar, color="#868E96", lw=1, ls=":")
    ax.annotate(f"d* = 2n_t/(n_t+1) = {dstar:.2f}\nabove this d, sampling\nthe clip costs more\nthan training it once",
                xy=(dstar, 1.0), xytext=(3.6, 0.075), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#495057", lw=1))
    ax.set_title("(b) sampling repeats the clip d times", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "video_budget_schemes.png"), dpi=190)
    plt.close(fig)

    # (c) 固定预算：最大边长 ∝ T^(-1/2)，以及与 720p 水平线的交点
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    budget = 3.0e11
    Nmax = math.sqrt(budget)
    Ts = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024])
    S_star = 16.0 * np.sqrt(Nmax / (Ts / 4.0))
    S_int = (S_star.astype(int) // 16) * 16
    ax.plot(Ts, S_star, "-", color="#4C6EF5", lw=1.6,
            label=r"continuous  $S^*\propto T^{-1/2}$")
    ax.plot(Ts, S_int, "o", color="#E8590C", ms=4, label="rounded to 16 px")
    ax.axhline(720, color="#12B886", lw=1.2, ls="--", label="720p (side = 720)")
    ax.axhline(1080, color="#9C36B5", lw=1.2, ls="--", label="1080p")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clip length  T  (frames, 24 fps)")
    ax.set_ylabel("largest feasible side  S  (px)")
    ax.set_title("fixed attention budget: frames trade against resolution", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "video_budget_scaling.png"), dpi=190)
    plt.close(fig)

    print(f"  图已写入 {figdir}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="视频扩散的 token 预算")
    ap.add_argument("groups", nargs="*", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)
    groups = [g.upper() for g in args.groups] or list("ABCDE")

    print("=" * 78)
    print("视频扩散的 token 预算：把时间轴压下去，训练为什么是四次方的")
    print("=" * 78)
    print(f"  fps={FPS}；所有量为精确算术，无采样")

    if "A" in groups:
        exp_A()
    if "B" in groups:
        exp_B()
    if "C" in groups:
        exp_C()
    if "D" in groups:
        exp_D()
    if "E" in groups:
        exp_E()
    if args.figdir:
        make_figs(args.figdir)

    print()
    print("=" * 78)
    print(f"断言总数 = {ASSERTS[1]}（通过 {ASSERTS[0]}，失败 {ASSERTS[1]-ASSERTS[0]}）")
    print("=" * 78)
    return 0 if ASSERTS[0] == ASSERTS[1] else 1


if __name__ == "__main__":
    sys.exit(main())
