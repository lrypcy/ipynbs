#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dit_core：DiT（Diffusion Transformer）核心机制的数值验证
=====================================================

对应文章：《扩散 Transformer（01）：Patchify、AdaLN-Zero 与 token 数的账》
（lrypcy.github.io，生成模型分类，DiT 系列第 1 篇）

六组检查，全部纯 numpy / CPU / 固定 seed，每条都做「解析真值 <-> 数值反算」自校验：

    1. FLOPs 记账：闭式公式反算 DiT 论文 Table 4 的 13 个 Gflops 数字
    2. 三种训练参数化（epsilon / x0 / v）的互换恒等式与误差放大因子
    3. AdaLN-Zero 的初始行为：整块恒等、梯度不消失、分支输出非零
    4. 时间步的必要性：贝叶斯最优 MSE 在「知道 t」与「不知道 t」下的差
    5. 条件注入的表达力：全局向量 vs 交叉注意力（闭式最小二乘下界）
    6. 采样步数与求解器阶：Euler 一阶 / Heun 二阶，以及折算到 DiT 的总算力

运行：
    python3 run.py                # 跑全部六组
    python3 run.py 1 3            # 只跑第 1、3 组
    python3 run.py --selftest     # 只跑断言（静默），适合挂 CI
"""

import sys
import numpy as np

SEED = 20261014
QUIET = False


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------

def section(no, title):
    note()
    note("-" * 72)
    note(f"{no}. {title}")
    note("-" * 72)


def note(*args):
    if not QUIET:
        print(*args)


def check(cond, msg):
    if not cond:
        raise AssertionError("自校验失败：" + msg)
    if not QUIET:
        print(f"  自校验通过：{msg}")


# ----------------------------------------------------------------------------
# 1. FLOPs 记账：闭式公式 vs DiT 论文 Table 4
# ----------------------------------------------------------------------------

# DiT 论文 Table 1：(depth, hidden_size, heads)
DIT_CFG = {
    "S":  (12, 384, 6),
    "B":  (12, 768, 12),
    "L":  (24, 1024, 16),
    "XL": (28, 1152, 16),
}

# DiT 论文 Table 4：256x256（latent 32x32x4）下的前向 Gflops
PAPER_256 = {
    "S":  {8: 0.36,  4: 1.41,   2: 6.06},
    "B":  {8: 1.42,  4: 5.56,   2: 23.01},
    "L":  {8: 5.01,  4: 19.70,  2: 80.71},
    "XL": {8: 7.39,  4: 29.05,  2: 118.64},
}

# 512x512（latent 64x64x4）下论文正文只给了一个数：XL/2 = 524.60 Gflops
PAPER_512 = {"XL": {2: 524.60}}

MLP_RATIO = 4.0          # DiT 的 mlp_ratio 默认值
LATENT_C = 4             # SD-VAE latent 通道数
TS_EMB = 256             # frequency embedding 维度


def dit_macs(depth, d, n_tokens, patch, ratio=MLP_RATIO, c=LATENT_C,
             adaln_mult=6, final_mult=2):
    """单个前向的乘加次数（1 MAC = 1 flop，与 DiT 论文 Table 4 的口径一致）。"""
    p2c = patch * patch * c
    # patch embedding / final linear：每 token 一层 (p^2 c) <-> d 的投影
    embed = n_tokens * p2c * d
    final = n_tokens * d * p2c
    # 每个 DiT Block
    qkv = 3 * n_tokens * d * d
    attn = 2 * n_tokens * n_tokens * d           # QK^T 与 AV
    oproj = n_tokens * d * d
    mlp = 2 * ratio * n_tokens * d * d           # d -> r d -> d
    adaln = adaln_mult * d * d                   # 条件向量 -> 6d 的调制参数
    block = qkv + attn + oproj + mlp + adaln
    # 时间步嵌入 MLP 与 FinalLayer 的 adaLN
    ts_mlp = TS_EMB * d + d * d
    final_adaln = final_mult * d * d
    return {
        "total": depth * block + embed + final + ts_mlp + final_adaln,
        "attn": depth * attn,
        "linear": depth * (qkv + oproj + mlp) + embed + final,
        "mod": depth * adaln + ts_mlp + final_adaln,
        "n_tokens": n_tokens,
    }


def attn_dominant_n(d, ratio=MLP_RATIO):
    """注意力二次项追平线性项所需的 token 数：2 N^2 D = N D^2 (4 + 2r)。"""
    return d * (2.0 + ratio)


def exp1_flops():
    section(1, "FLOPs 记账：闭式公式反算 DiT 论文 Table 4 的 Gflops")
    note("口径：1 次乘加 = 1 flop；latent 32x32x4（256x256 图像，VAE 下采样 f=8）")
    note("公式  MACs = L(3ND^2 + 2N^2D + ND^2 + 2rND^2 + 6D^2) + 2Np^2CD + 2D^2 + 256D + D^2")
    note()
    note("  模型   p    token 数 |   闭式 Gflops   论文 Gflops   相对误差 | 注意力占比")
    note("  " + "-" * 68)

    worst = 0.0
    worst_tag = ""
    rows = []
    for size in ("S", "B", "L", "XL"):
        depth, d, _ = DIT_CFG[size]
        for p in (8, 4, 2):
            n = (32 // p) ** 2
            m = dit_macs(depth, d, n, p)
            g = m["total"] / 1e9
            ref = PAPER_256[size][p]
            rel = abs(g - ref) / ref
            share = m["attn"] / m["total"]
            rows.append((size, p, n, g, ref, rel, share))
            if rel > worst:
                worst, worst_tag = rel, f"DiT-{size}/{p}"
            note(f"  DiT-{size:<3} {p:<4} {n:>7} | {g:>12.3f} {ref:>12.2f} "
                 f"{rel * 100:>10.3f}% | {share * 100:>8.2f}%")

    depth, d, _ = DIT_CFG["XL"]
    m = dit_macs(depth, d, (64 // 2) ** 2, 2)
    g512 = m["total"] / 1e9
    ref512 = PAPER_512["XL"][2]
    rel512 = abs(g512 - ref512) / ref512
    note(f"  DiT-XL/2 @512 {m['n_tokens']:>7} | {g512:>12.3f} {ref512:>12.2f} "
         f"{rel512 * 100:>10.3f}% | {m['attn'] / m['total'] * 100:>8.2f}%")
    if rel512 > worst:
        worst, worst_tag = rel512, "DiT-XL/2 @512"

    note()
    note(f"  13 个数字里最大的相对误差 = {worst * 100:.3f}%（{worst_tag}）")
    note("  论文只给两位小数：0.36 这一档的四舍五入颗粒度本身就有 1.4%，")
    note("  所以判断标准是「落在 0.5% 或论文的两位小数颗粒度之内」。")
    worst_abs = 0.0
    for size in ("S", "B", "L", "XL"):
        depth, d, _ = DIT_CFG[size]
        for p in (8, 4, 2):
            n = (32 // p) ** 2
            g = dit_macs(depth, d, n, p)["total"] / 1e9
            ref = PAPER_256[size][p]
            worst_abs = max(worst_abs, abs(g - ref) / max(0.005, 5e-3 * ref))
    depth, d, _ = DIT_CFG["XL"]
    g512 = dit_macs(depth, d, 1024, 2)["total"] / 1e9
    worst_abs = max(worst_abs, abs(g512 - PAPER_512["XL"][2]) / max(0.005, 5e-3 * 524.60))
    check(worst_abs <= 1.0, f"13 个 Gflops 数字全部落在 0.5% 或论文两位小数颗粒度内"
                            f"（最紧的一处用了容忍度的 {worst_abs * 100:.0f}%）")

    # 形状级计数器 <-> 闭式公式（构造真值 vs 反算值）
    depth, d, _ = DIT_CFG["XL"]
    n = 256
    p = 2
    rd = int(MLP_RATIO * d)
    per_block_shape = sum([
        n * d * 3 * d,          # qkv 投影
        n * d * n,              # QK^T
        n * n * d,              # A @ V
        n * d * d,              # 输出投影
        n * d * rd,             # MLP 第一层
        n * rd * d,             # MLP 第二层
        1 * d * 6 * d,          # adaLN 调制
    ])
    closed_per_block = dit_macs(1, d, n, p)["total"] - (n * p * p * LATENT_C * d) - (n * d * p * p * LATENT_C) - (TS_EMB * d + d * d) - 2 * d * d
    note()
    note(f"  形状级逐算子累加（每 Block）= {per_block_shape:,} MACs")
    note(f"  闭式公式（每 Block）        = {closed_per_block:,} MACs")
    check(per_block_shape == closed_per_block, "逐算子累加与闭式公式逐位相等")

    # patch size 减半的放大倍数
    note()
    note("  patch size 减半对总 Gflops 的影响（DiT-XL，256x256）：")
    g8 = PAPER_256["XL"][8]
    g4 = PAPER_256["XL"][4]
    g2 = PAPER_256["XL"][2]
    note(f"    p=8 -> p=4：{g4 / g8:.3f}x    p=4 -> p=2：{g2 / g4:.3f}x")
    note("    论文原话是「halving p will quadruple T, and thus at least quadruple")
    note("    total transformer Gflops」；实测 3.931x / 4.084x —— 二次项严格给 4x，")
    note("    而 adaLN 调制（6D^2/Block）与 embed/final 不随 N 增长，会把比值拉低一点。")
    check(g2 / g4 > 4.0, "p 从 4 减到 2 时总 Gflops 放大超过 4 倍")

    # 注意力主导的临界 token 数与 KV 压缩
    note()
    note("  注意力二次项追平线性项的临界 token 数 N* = (2+r)D：")
    note(f"    DiT-XL（D=1152, r=4）：N* = {attn_dominant_n(d):.0f} tokens")
    note("    KV 空间压缩 s=2（PixArt-Sigma 思路）：N* = "
         f"{attn_dominant_n(d) * 4:.0f} tokens")
    note()
    note("  不同 token 规模下的注意力占比（DiT-XL 形状：28 层 / D=1152 / r=4）：")
    note("    场景                          token 数   总 Gflops   注意力占比   KV 压缩(s=2) 后")
    note("    " + "-" * 66)
    scen = [
        ("256x256 (f=8, p=2)", 32, 2),
        ("512x512 (f=8, p=2)", 64, 2),
        ("1024x1024 (f=8, p=2)", 128, 2),
        ("3840x2160 4K (f=8, p=2)", None, 2),
    ]
    for name, lat, p in scen:
        if lat is None:
            n = 240 * 135           # 480x270 latent / p=2
        else:
            n = (lat // p) ** 2
        m = dit_macs(depth, d, n, p)
        base = m["total"] / 1e9
        share = m["attn"] / m["total"]
        comp = (m["linear"] + m["mod"] + m["attn"] / 4.0) / 1e9
        note(f"    {name:<28}{n:>8} {base:>12.1f} {share * 100:>10.1f}% {comp:>14.1f}")
    note()
    note("    4K 下注意力占 80% 以上，KV 压缩直接砍掉四分之三的注意力开销；")
    note("    256x256 下注意力只占个位数，同样的压缩几乎无感。")
    return worst


# ----------------------------------------------------------------------------
# 2. 三种训练参数化的互换恒等式与误差放大因子
# ----------------------------------------------------------------------------

def path_linear(t):
    """Rectified flow / 线性插值路径：alpha = 1-t, sigma = t。"""
    a, s = 1.0 - t, t
    return a, s, -1.0, 1.0


def path_cosine(t):
    """余弦（方差保持）路径：alpha = cos(pi t/2), sigma = sin(pi t/2)，alpha^2+sigma^2=1。"""
    w = np.pi * t / 2.0
    a, s = np.cos(w), np.sin(w)
    return a, s, -(np.pi / 2.0) * np.sin(w), (np.pi / 2.0) * np.cos(w)


def exp2_parameterization():
    section(2, "训练参数化：epsilon / x0 / v 的互换恒等式与误差放大因子")
    rng = np.random.default_rng(SEED)
    d = 8
    x0 = rng.normal(size=d)
    eps = rng.normal(size=d)

    note("两条路径：")
    note("  线性（rectified flow）  alpha = 1-t,        sigma = t")
    note("  余弦（方差保持 VP）     alpha = cos(pi t/2), sigma = sin(pi t/2)")
    note()
    for pname, path in (("线性", path_linear), ("余弦", path_cosine)):
        note(f"  [{pname}路径] 恒等式残差（float64）")
        note("        t |  x0 反解残差    v 定义残差    v->x0 反解残差    eps 反解残差")
        note("    " + "-" * 64)
        worst_x0 = worst_v = worst_vx = worst_e = 0.0
        for t in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
            a, s, ad, sd = path(t)
            xt = a * x0 + s * eps
            # (b) 由 x_t 与 x0 反解 eps
            e_rec = (xt - a * x0) / s
            # (c) 由 x_t 与 eps 反解 x0
            x_rec = (xt - s * eps) / a
            # (d) v = alpha_dot x0 + sigma_dot eps = (ad/a) x_t + (sd - ad*s/a) eps
            v_true = ad * x0 + sd * eps
            v_alt = (ad / a) * xt + (sd - ad * s / a) * eps
            # (e) 由 (x_t, v) 反解 (x0, eps)：行列式 Delta = alpha*sigma_dot - alpha_dot*sigma
            det = a * sd - ad * s
            x_from_v = (sd * xt - s * v_true) / det
            e_from_v = (a * v_true - ad * xt) / det
            worst_x0 = max(worst_x0, float(np.max(np.abs(x_rec - x0))))
            worst_v = max(worst_v, float(np.max(np.abs(v_alt - v_true))))
            worst_vx = max(worst_vx, float(np.max(np.abs(x_from_v - x0))))
            worst_e = max(worst_e, float(np.max(np.abs(e_rec - eps))),
                          float(np.max(np.abs(e_from_v - eps))))
            note(f"    {t:6.2f} | {np.max(np.abs(x_rec - x0)):.3e} "
                 f"   {np.max(np.abs(v_alt - v_true)):.3e}   "
                 f"{np.max(np.abs(x_from_v - x0)):.3e}     {np.max(np.abs(e_rec - eps)):.3e}")
        check(max(worst_x0, worst_v, worst_vx, worst_e) < 1e-12,
              f"[{pname}] 四条恒等式的最大残差 < 1e-12"
              f"（实测 {max(worst_x0, worst_v, worst_vx, worst_e):.2e}）")

    note()
    note("  预测误差到 x0 空间的放大因子（同一个 delta 的预测误差，落到样本上有多大）：")
    note("    x0 反解：  ||dx0|| = (sigma/alpha) ||d eps||   |   ||dx0|| = (sigma/Delta) ||dv||")
    note()
    note("        t |  线性: eps放大   线性: v放大 |  余弦: eps放大   余弦: v放大")
    note("    " + "-" * 66)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        row = [f"    {t:6.2f} |"]
        for path in (path_linear, path_cosine):
            a, s, ad, sd = path(t)
            det = a * sd - ad * s
            row.append(f" {s / a:>12.3f} {s / det:>12.3f} |")
        note("".join(row))

    # 数值反算：给预测加一个固定扰动，量真正的 ||dx0||
    note()
    note("  数值反算（给预测值加一个固定扰动 delta，量 ||dx0||/||delta||）：")
    note("        t |  线性 eps   线性 v   |  余弦 eps   余弦 v")
    note("    " + "-" * 52)
    for t in (0.1, 0.5, 0.9, 0.99):
        cells = []
        for path in (path_linear, path_cosine):
            a, s, ad, sd = path(t)
            det = a * sd - ad * s
            xt = a * x0 + s * eps
            delta = rng.normal(size=d)
            delta = delta / np.linalg.norm(delta)
            # eps 预测被扰动
            e_hat = eps + delta
            xr = (xt - s * e_hat) / a
            cells.append(np.linalg.norm(xr - x0))
            # v 预测被扰动
            v_true = ad * x0 + sd * eps
            v_hat = v_true + delta
            xr2 = (sd * xt - s * v_hat) / det
            cells.append(np.linalg.norm(xr2 - x0))
        note(f"    {t:6.2f} | {cells[0]:9.3f} {cells[1]:9.3f}   |"
             f" {cells[2]:9.3f} {cells[3]:9.3f}")

    a, s, ad, sd = path_cosine(0.99)
    note()
    note(f"  t=0.99 处：eps 预测的放大因子 sigma/alpha = {s / a:.2f}，v 预测的放大因子")
    note(f"  sigma/Delta = {(s / (a * sd - ad * s)):.3f}。这正是 Salimans & Ho 引入 v")
    note("  参数化的动机：eps 预测在高噪声端会把预测误差无限放大。")
    check(s / a > 10.0, "t=0.99 时 eps 预测的误差放大因子 > 10")
    check(s / (a * sd - ad * s) < 1.0, "余弦路径下 v 预测的误差放大因子恒 < 1")

    # 隐含的 x0 空间损失权重
    note()
    note("  隐含的 x0 空间损失权重（最小化该参数化的均匀-t MSE，等价于最小化")
    note("  加权的 x0 空间 MSE）：")
    note("        t |  线性 w_eps=(a/s)^2  线性 w_v=1/t^2 |  余弦 w_eps   余弦 w_v")
    note("    " + "-" * 66)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        row = [f"    {t:6.2f} |"]
        for path in (path_linear, path_cosine):
            a, s, ad, sd = path(t)
            det = a * sd - ad * s
            row.append(f" {(a / s) ** 2:>16.3f} {(det / s) ** 2:>13.3f} |")
        note("".join(row))
    note()
    note("  两端的行为才是关键：t->0 时两个权重都发散（都在低噪声段过度用力），")
    note("  但 t->1 时 w_eps -> 0（高噪声段几乎不计入损失），w_v 收敛到常数。")
    return None


# ----------------------------------------------------------------------------
# 3. AdaLN-Zero 的初始行为
# ----------------------------------------------------------------------------

def _layernorm(h, gain=None, bias=None, eps=1e-6):
    mu = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    out = (h - mu) / np.sqrt(var + eps)
    if gain is not None:
        out = out * gain
    if bias is not None:
        out = out + bias
    return out


def _modulate(h, shift, scale):
    return (1.0 + scale) * h + shift


class DiTBlock:
    """DiT Block 的 numpy 版：Pre-LN + AdaLN + MHSA + AdaLN + MLP，两个残差门控。"""

    def __init__(self, n, d, heads, ratio, rng, zero_init):
        self.n, self.d, self.heads, self.dh = n, d, heads, d // heads
        self.ratio = ratio
        sc = np.sqrt(2.0 / d)
        self.wqkv = rng.normal(scale=sc, size=(d, 3 * d))
        self.wo = rng.normal(scale=sc, size=(d, d))
        self.w1 = rng.normal(scale=sc, size=(d, int(ratio * d)))
        self.w2 = rng.normal(scale=sc * 0.5, size=(int(ratio * d), d))
        # adaLN 调制：条件向量 -> 6D（gamma1 beta1 alpha1 gamma2 beta2 alpha2）
        self.wmod = rng.normal(scale=sc, size=(d, 6 * d))
        if zero_init:
            self.wmod[:, 2 * d:3 * d] = 0.0      # alpha1
            self.wmod[:, 5 * d:6 * d] = 0.0      # alpha2
            self.wmod[:, 0:d] = 0.0              # gamma1
            self.wmod[:, d:2 * d] = 0.0          # beta1
            self.wmod[:, 3 * d:4 * d] = 0.0      # gamma2
            self.wmod[:, 4 * d:5 * d] = 0.0      # beta2
        self.zero_init = zero_init

    def modulation(self, cond):
        return self.wmod.T @ cond        # (6D,)

    def attn(self, h):
        n, d, heads, dh = self.n, self.d, self.heads, self.dh
        qkv = h @ self.wqkv                                   # (n, 3d)
        q = qkv[:, :d].reshape(n, heads, dh).transpose(1, 0, 2)
        k = qkv[:, d:2 * d].reshape(n, heads, dh).transpose(1, 0, 2)
        v = qkv[:, 2 * d:].reshape(n, heads, dh).transpose(1, 0, 2)
        logits = q @ k.transpose(0, 2, 1) / np.sqrt(dh)       # (h, n, n)
        logits = logits - logits.max(axis=-1, keepdims=True)
        w = np.exp(logits)
        w = w / w.sum(axis=-1, keepdims=True)
        o = (w @ v).transpose(1, 0, 2).reshape(n, d)
        return o @ self.wo

    def mlp(self, h):
        u = h @ self.w1
        g = 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u ** 3)))
        return g @ self.w2

    def forward(self, h, cond):
        m = self.modulation(cond)
        g1, b1, a1, g2, b2, a2 = [m[i * self.d:(i + 1) * self.d] for i in range(6)]
        ha = self.attn(_modulate(_layernorm(h), b1, g1))
        h1 = h + a1 * ha
        hm = self.mlp(_modulate(_layernorm(h1), b2, g2))
        return h1 + a2 * hm, ha, hm


def _stack(blocks, h, cond):
    for b in blocks:
        h, _, _ = b.forward(h, cond)
    return h


class FinalLayer:
    def __init__(self, d, out_dim, rng, zero_init):
        sc = np.sqrt(2.0 / d)
        self.w = rng.normal(scale=sc, size=(d, out_dim))
        self.wmod = rng.normal(scale=sc, size=(d, 2 * d))
        if zero_init:
            self.w = np.zeros_like(self.w)
            self.wmod = np.zeros_like(self.wmod)
        self.d = d

    def forward(self, h, cond):
        m = self.wmod.T @ cond
        shift, scale = m[:self.d], m[self.d:]
        return _modulate(_layernorm(h), shift, scale) @ self.w


def exp3_adaln_zero():
    section(3, "AdaLN-Zero 的初始行为：整块恒等、梯度不消失、分支输出非零")
    n, d, heads, depth = 16, 64, 4, 28
    rng_a = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED + 1)
    rng_c = np.random.default_rng(SEED + 2)

    out_dim = 2 * 2 * LATENT_C            # p=2, C=4 -> 每个 token 输出 16 个数
    x = rng_c.normal(size=(n, d))
    cond = rng_c.normal(size=d)
    target = rng_c.normal(size=(n, d))    # 主损失用的目标（D 维）
    target_out = rng_c.normal(size=(n, out_dim))

    z_blocks = [DiTBlock(n, d, heads, MLP_RATIO, rng_a, True) for _ in range(depth)]
    r_blocks = [DiTBlock(n, d, heads, MLP_RATIO, rng_b, False) for _ in range(depth)]
    z_final = FinalLayer(d, out_dim, rng_a, True)
    r_final = FinalLayer(d, out_dim, rng_b, False)

    xz = _stack(z_blocks, x, cond)
    xr = _stack(r_blocks, x, cond)

    note(f"  形状：{depth} 层 / D={d} / {heads} 头 / N={n} tokens")
    note()
    note("  (a) 初始时刻整块是不是恒等映射")
    dz = float(np.max(np.abs(xz - x)))
    dr = float(np.linalg.norm(xr - x) / np.linalg.norm(x))
    note(f"    AdaLN-Zero  max|out - x|                 = {dz:.3e}")
    note(f"    普通初始化  ||out - x||_F / ||x||_F      = {dr:.4f}")
    check(dz == 0.0, "AdaLN-Zero 下 28 层堆叠的输出与输入逐位相等（恒等）")
    check(dr > 0.5, "普通初始化下 28 层堆叠的输出已经明显偏离输入")

    note()
    note("  (b) 被门控掉的分支本身是不是零（决定门控梯度能不能动起来）")
    _, ha, hm = z_blocks[0].forward(x, cond)
    note(f"    ||Attn(AdaLN(x))||_F / ||x||_F  = "
         f"{np.linalg.norm(ha) / np.linalg.norm(x):.4f}")
    note(f"    ||MLP(AdaLN(x))||_F / ||x||_F   = "
         f"{np.linalg.norm(hm) / np.linalg.norm(x):.4f}")
    check(np.linalg.norm(ha) > 0.1 * np.linalg.norm(x),
          "分支输出非零：门控向量 alpha 的梯度不为零，第一步就能学到东西")

    note()
    note("  (c) 梯度：恒等初始化有没有把梯度也一起归零")
    def loss(out):
        return float(np.mean((out - target) ** 2))

    dout = 2.0 * (xz - target) / (n * d)          # dL/d(out)
    # 中心差分求 dL/dx（对 5 个随机坐标）
    rng_f = np.random.default_rng(SEED + 3)
    idx = [(int(i), int(j)) for i, j in rng_f.integers([n, d], size=(5, 2))]
    h_fd = 1e-5
    fd = np.array([(loss(_stack(z_blocks, _bump(x, i, j, +h_fd), cond))
                    - loss(_stack(z_blocks, _bump(x, i, j, -h_fd), cond))) / (2 * h_fd)
                   for i, j in idx])
    an = np.array([dout[i, j] for i, j in idx])
    rel = float(np.max(np.abs(fd - an) / np.maximum(np.abs(an), 1e-12)))
    note(f"    dL/dx 的中心差分 vs 解析值：最大相对偏差 = {rel:.2e}")
    note(f"    |dL/dx| 均值 = {np.abs(an).mean():.3e}（非零，梯度原样穿过 28 层）")
    check(rel < 1e-5, "恒等初始化下 dL/dx = dL/d(out)，梯度原样穿过整个堆叠")
    check(np.abs(an).mean() > 0, "梯度范数非零（不是死梯度）")

    # alpha 的梯度：解析 vs 中心差分
    b0 = z_blocks[0]
    m = b0.modulation(cond)
    a1 = m[2 * d:3 * d]
    def loss_alpha(vec):
        mm = m.copy()
        mm[2 * d:3 * d] = vec
        h1 = x + mm[2 * d:3 * d] * ha
        out = _stack(z_blocks[1:], h1, cond)
        return loss(out)
    ga = np.array([float(np.sum(dout * np.outer(np.ones(n), _onehot(d, j)) * ha))
                   for j in range(0, d, d // 4)])
    gfd = np.array([(loss_alpha(_bump_vec(a1, j, h_fd)) - loss_alpha(_bump_vec(a1, j, -h_fd)))
                    / (2 * h_fd) for j in range(0, d, d // 4)])
    note(f"    dL/d(alpha1) 解析 vs 中心差分：{[f'{v:.3e}' for v in ga[:3]]} / "
         f"{[f'{v:.3e}' for v in gfd[:3]]}")
    check(np.all(np.abs(ga) > 0), "alpha 的梯度非零：零初始化不是把这块冻住")

    note()
    note("  (d) 初始损失（输出层也零初始化）")

    def loss_out(out):
        return float(np.mean((out - target_out) ** 2))

    lz = loss_out(z_final.forward(xz, cond))
    lr = loss_out(r_final.forward(xr, cond))
    lref = float(np.mean(target_out ** 2))
    note(f"    目标自身的能量 E||y||^2      = {lref:.4f}")
    note(f"    AdaLN-Zero（初始输出 = 0）   = {lz:.4f}")
    note(f"    普通初始化                   = {lr:.4f}")
    note(f"    比值                         = {lr / lz:.2f}x")
    check(abs(lz - lref) < 1e-12, "零初始化下初始损失精确等于目标自身的能量")
    check(lr > lz, "普通初始化的初始损失高于零初始化")
    return None


def _bump(arr, i, j, delta):
    out = arr.copy()
    out[i, j] += delta
    return out


def _bump_vec(vec, j, delta):
    out = vec.copy()
    out[j] += delta
    return out


def _onehot(d, j):
    v = np.zeros(d)
    v[j] = 1.0
    return v


# ----------------------------------------------------------------------------
# 4. 时间步的必要性
# ----------------------------------------------------------------------------

def exp4_timestep():
    section(4, "时间步的必要性：知道 t 与不知道 t 的贝叶斯最优 MSE 之差")
    note("设定：标量数据 x0 ~ N(0, s^2)，eps ~ N(0, 1)，t ~ U[0,1]")
    note("      方差保持（余弦）路径下 alpha^2 + sigma^2 = 1，取 s = 1 时")
    note("      x_t ~ N(0, 1) 对**所有** t 都成立 —— x_t 的边缘分布完全不携带 t 的信息。")
    note()
    s2 = 1.0
    # 网格求积
    ts = np.linspace(1e-4, 1 - 1e-4, 20001)
    xs = np.linspace(-8, 8, 32001)
    dx = xs[1] - xs[0]

    # 已知 t：MMSE(t) = alpha^2 s^2 / (alpha^2 s^2 + sigma^2)
    mmse_t = np.array([_mmse_known(t, s2) for t in ts])
    known = float(np.trapezoid(mmse_t, ts))
    # 未知 t：g(x) = E[eps | x] = [int sigma_t/v_t * N(x;0,v_t) dt / int N(x;0,v_t) dt] * x
    num = np.zeros_like(xs)
    den = np.zeros_like(xs)
    for t in ts[::20]:
        a, sg, _, _ = path_cosine(t)
        v = a * a * s2 + sg * sg
        pdf = np.exp(-0.5 * xs ** 2 / v) / np.sqrt(2 * np.pi * v)
        num += (sg / v) * pdf
        den += pdf
    g = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0) * xs
    px = den / den.sum()
    px = px / px.sum()
    # p(x) 归一化到网格
    px_grid = den / (np.trapezoid(den, xs))
    e_eps2_given_x = 1.0                      # Var(eps|x,t) 与 x 无关？用全方差算
    # MMSE_unknown = E_x[ E[eps^2|x] - g(x)^2 ]
    # E[eps^2|x] = int p(t|x) E[eps^2|x,t] dt；E[eps^2|x,t] = v_eps - cov^2/v = 1 - sigma^2/v
    e2 = np.zeros_like(xs)
    den2 = np.zeros_like(xs)
    for t in ts[::20]:
        a, sg, _, _ = path_cosine(t)
        v = a * a * s2 + sg * sg
        pdf = np.exp(-0.5 * xs ** 2 / v) / np.sqrt(2 * np.pi * v)
        # E[eps^2 | x, t] = Var(eps|x,t) + E[eps|x,t]^2 = (1 - sigma^2/v) + (sigma/v * x)^2
        e2 += ((1.0 - sg * sg / v) + (sg / v * xs) ** 2) * pdf
        den2 += pdf
    e2 = e2 / np.maximum(den2, 1e-300)
    mmse_unk = float(np.trapezoid((e2 - g ** 2) * px_grid, xs))

    note(f"  已知 t 的最优 MSE（对 t 取平均）      = {known:.6f}")
    note(f"  未知 t 的最优 MSE                     = {mmse_unk:.6f}")
    note(f"  绝对差                                = {mmse_unk - known:.6f}")
    note(f"  相对差                                = "
         f"{(mmse_unk - known) / known * 100:.2f}%")
    note()
    note("  解析对照：余弦路径 + s=1 时")
    note("    MMSE_known(t) = alpha^2 = cos^2(pi t/2)，对 t 积分 = 1/2")
    note("    g(x) = (int_0^1 sigma_t dt) x = (2/pi) x = 0.636620 x")
    note("    MMSE_unknown  = 1 - (2/pi)^2 = 0.594715")
    note(f"    数值积分得 {mmse_unk:.6f}，解析值 {1 - (2 / np.pi) ** 2:.6f}")
    check(abs(known - 0.5) < 1e-3, "已知 t 的最优 MSE = 1/2（解析）")
    check(abs(mmse_unk - (1 - (2 / np.pi) ** 2)) < 1e-3,
          "未知 t 的最优 MSE = 1 - (2/pi)^2（解析）")
    check(mmse_unk > known, "丢掉时间步后贝叶斯最优 MSE 严格变差")

    note()
    note("  同一个 x_t、不同 t 下最优输出的差异（构造碰撞）：")
    note("        t |  E[eps | x=1, t] = sigma_t * 1")
    note("    " + "-" * 40)
    for t in (0.05, 0.2, 0.5, 0.8, 0.95):
        _, sg, _, _ = path_cosine(t)
        note(f"    {t:6.2f} |  {sg:>10.4f}")
    _, s8, _, _ = path_cosine(0.8)
    _, s2_, _, _ = path_cosine(0.2)
    note()
    note(f"  输入完全相同（x_t = 1.0），t=0.2 的最优输出 {s2_:.4f}，t=0.8 的最优输出 "
         f"{s8:.4f}，")
    note(f"  两者相差 {s8 - s2_:.4f}。不知道 t 的模型只能输出一个折中值，这就是上表"
         f" {(mmse_unk - known) / known * 100:.1f}% 的来源。")
    return None


def _mmse_known(t, s2):
    a, sg, _, _ = path_cosine(t)
    v = a * a * s2 + sg * sg
    return a * a * s2 / v


# ----------------------------------------------------------------------------
# 5. 条件注入的表达力：全局向量 vs 交叉注意力
# ----------------------------------------------------------------------------

def exp5_conditioning():
    section(5, "条件注入的表达力：全局向量 vs 交叉注意力（闭式最小二乘）")
    rng = np.random.default_rng(SEED)
    for K, d, n_patch in ((8, 32, 64), (2, 32, 64)):
        # 精确正交的 token 嵌入（Gram-Schmidt）
        raw = rng.normal(size=(K, d))
        q, _ = np.linalg.qr(raw.T)
        E = q[:, :K].T if q.shape[1] >= K else None
        if E is None or E.shape[0] != K:
            E, _ = np.linalg.qr(raw)
            E = E[:K].T
        E = E[:K] / np.linalg.norm(E[:K], axis=1, keepdims=True)

        assign = np.repeat(np.arange(K), n_patch // K) if n_patch % K == 0 else \
            rng.integers(0, K, size=n_patch)
        Y = E[assign]                                     # (n_patch, d)

        # (A) 全局向量：整段文本池化成一个向量后，所有 patch 只能共享同一个输出
        pooled = E.mean(axis=0)
        best_global = Y.mean(axis=0)
        # 误差口径：对每个 patch 取平方 L2 范数再对 patch 求平均（目标行范数 = 1）
        mseA = float(np.mean(np.sum((Y - best_global) ** 2, axis=1)))
        # 闭式下界：正交单位范数嵌入 + 均衡分配 -> 1 - 1/K
        bound = 1.0 - 1.0 / K
        # 交叉验证：设计矩阵只有一列全 1（所有 patch 共享同一个输出向量）
        ones = np.ones((n_patch, 1))
        coef, _, _, _ = np.linalg.lstsq(ones, Y, rcond=None)
        mseA_ls = float(np.mean(np.sum((Y - ones @ coef) ** 2, axis=1)))

        # (B) 硬选择交叉注意力
        mseB = float(np.mean(np.sum((Y - E[assign]) ** 2, axis=1)))

        # (C) 可学习的 softmax 交叉注意力（对 logits 做梯度下降）
        A_logit = np.zeros((n_patch, K))
        lr = 100.0
        for _ in range(3000):
            W = np.exp(A_logit)
            W = W / W.sum(axis=1, keepdims=True)
            pred = W @ E
            grad = (pred - Y) @ E.T
            grad = W * (grad - (W * grad).sum(axis=1, keepdims=True))
            A_logit -= lr * grad
        W = np.exp(A_logit)
        W = W / W.sum(axis=1, keepdims=True)
        mseC = float(np.mean(np.sum((Y - W @ E) ** 2, axis=1)))

        note(f"  K={K} 个 token / d={d} / {n_patch} 个 patch，嵌入行范数 = 1")
        note(f"    全局向量（池化成 1 个向量）  MSE = {mseA:.6f}"
             f"   解析下界 1-1/K = {bound:.6f}")
        note(f"    同一结果用 lstsq 反算        MSE = {mseA_ls:.6f}")
        note(f"    交叉注意力（硬选择）         MSE = {mseB:.6f}")
        note(f"    交叉注意力（softmax + GD）   MSE = {mseC:.6e}")
        note(f"    全局向量能解释的能量占比     = {1 - mseA:.4f}（解析 1/K = {1 / K:.4f}）")
        check(abs(mseA - bound) < 1e-8, f"K={K}: 全局向量条件的 MSE 精确等于 1-1/K")
        check(abs(mseA - mseA_ls) < 1e-10, f"K={K}: 闭式解与 lstsq 一致")
        check(mseB < 1e-12, f"K={K}: 硬选择交叉注意力可以精确复现目标")
        check(mseC < 0.05 * mseA, f"K={K}: softmax 交叉注意力的 MSE 低于全局向量的 5%")
        note()
    note("  读法：全局向量条件能解释的能量恰好是 1/K（K 个正交语义里平均分到的一份），")
    note("  剩下 (K-1)/K 必须靠「每个位置各自去读不同的文本 token」才能拿回来。")
    note("  这也是 AdaLN（全局调制）必须和交叉注意力搭配、而不是二选一的原因。")
    return None


# ----------------------------------------------------------------------------
# 6. 采样步数与求解器阶
# ----------------------------------------------------------------------------

def exp6_sampler():
    section(6, "采样步数与求解器阶：Euler 一阶 / Heun 二阶，折算到 DiT 总算力")
    s = 2.0                       # 数据标准差；噪声标准差 1
    s2 = s * s

    def lam(t):
        # v*(x,t) = lambda(t) x，线性（rectified flow）路径 + 高斯数据的解析速度场
        return (t - (1 - t) * s2) / ((1 - t) ** 2 * s2 + t ** 2)

    # 解析自校验：int_0^1 lambda dt 必须等于 ln(1/s)
    grid = np.linspace(0, 1, 2000001)
    L = float(np.trapezoid(lam(grid), grid))
    note(f"  设定：x0 ~ N(0, s^2)，s = {s}；线性路径 alpha=1-t, sigma=t")
    note(f"  解析速度场 v*(x,t) = lambda(t) x，lambda(t) = (t - (1-t)s^2) / "
         f"((1-t)^2 s^2 + t^2)")
    note(f"  int_0^1 lambda(t) dt 数值积分 = {L:.8f}，解析值 ln(1/s) = "
         f"{np.log(1 / s):.8f}")
    check(abs(L - np.log(1 / s)) < 1e-6, "速度场积分 = ln(1/s)（流必须把 N(0,1) 送到 N(0,s^2)）")
    check(abs(np.exp(-2 * L) - s2) / s2 < 1e-6, "exp(-2L) = s^2（终点方差自校验）")

    def euler(S):
        """一阶（Euler）：x_{k+1} = x_k - h v(x_k, t_k)，v = lambda(t) x 时闭式为连乘。"""
        h = 1.0 / S
        t = 1.0 - h * np.arange(S)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.exp(2.0 * np.sum(np.log1p(-h * lam(t)))))

    def heun(S):
        """二阶（Heun / 梯形）：先 Euler 预测，再用两端速度的平均值校正。"""
        h = 1.0 / S
        t = 1.0 - h * np.arange(S)
        l1 = lam(t)
        l2 = lam(t - h)
        f = 1.0 - 0.5 * h * (l1 + l2 * (1.0 - h * l1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.exp(2.0 * np.sum(np.log(f))))

    def min_steps(fn, tol, smax=1 << 20):
        """找满足误差容限的最小步数：先几何放大定位，再二分。"""
        hi = 1
        while hi < smax and abs(fn(hi) / s2 - 1) >= tol:
            hi *= 2
        lo = max(1, hi // 2)
        while lo < hi:
            mid = (lo + hi) // 2
            if abs(fn(mid) / s2 - 1) < tol:
                hi = mid
            else:
                lo = mid + 1
        return hi

    note()
    note("  终点方差的相对误差 |Var_num / s^2 - 1|：")
    note("    步数 S |        Euler 误差   阶 |        Heun 误差   阶")
    note("    " + "-" * 54)
    steps = (4, 8, 16, 32, 64, 128)
    e_err, h_err = [], []
    prev_e = prev_h = None
    for S in steps:
        ve, vh = euler(S), heun(S)
        ee, eh = abs(ve / s2 - 1), abs(vh / s2 - 1)
        e_err.append(ee)
        h_err.append(eh)
        oe = f"{np.log2(prev_e / ee):6.2f}" if prev_e else "     -"
        oh = f"{np.log2(prev_h / eh):6.2f}" if prev_h and eh > 0 else "     -"
        note(f"    {S:>6} | {ee:>16.6e} {oe} | {eh:>16.6e} {oh}")
        prev_e, prev_h = ee, eh

    se = np.polyfit(np.log(steps), np.log(e_err), 1)[0]
    sh = np.polyfit(np.log(steps), np.log(h_err), 1)[0]
    note()
    note(f"  log-log 拟合斜率：Euler = {se:.3f}（理论 -1），Heun = {sh:.3f}（理论 -2）")
    check(se < -0.85, f"Euler 阶数实测 {se:.2f} ≈ -1")
    check(sh < -1.7, f"Heun 阶数实测 {sh:.2f} ≈ -2")

    gxl2 = dit_macs(*DIT_CFG["XL"][:2], 256, 2)["total"] / 1e9
    note()
    note("  折算成 DiT-XL/2（256x256）的总算力：")
    note("    误差容限 |  Euler 需要步数   总 Gflops |  Heun 需要步数   总 Gflops（2 前向/步）")
    note("    " + "-" * 74)
    for tol in (1e-2, 3e-3, 1e-3, 3e-4):
        Se = min_steps(euler, tol)
        Sh = min_steps(heun, tol)
        note(f"    {tol:>8.0e} | {Se:>14} {Se * gxl2:>12.0f} | {Sh:>14} "
             f"{Sh * 2 * gxl2:>12.0f}")
    note("    （Gflops 列按 DiT-XL/2 单次前向 "
         f"{gxl2:.2f} Gflops 折算；Heun 每步 2 次前向）")
    note()
    note("  要点：Heun 每步要 2 次前向，但二阶收敛把步数压到 sqrt 量级。")
    note("  容限 1e-2 时总算力之比 = "
         f"{(295 * gxl2) / (11 * 2 * gxl2):.1f}x；容限 1e-3 时 = "
         f"{(2963 * gxl2) / (32 * 2 * gxl2):.1f}x。")
    note("  推理成本 = 步数 x 单次前向，两个因子不能分开谈：")
    note("  换求解器阶数是在动第一个因子，量化 / 稀疏 / KV 压缩是在动第二个。")
    return None


# ----------------------------------------------------------------------------

EXPS = {
    1: exp1_flops,
    2: exp2_parameterization,
    3: exp3_adaln_zero,
    4: exp4_timestep,
    5: exp5_conditioning,
    6: exp6_sampler,
}


def main(argv):
    global QUIET
    args = [a for a in argv[1:] if not a.startswith("--")]
    QUIET = "--selftest" in argv[1:]
    rng = np.random.default_rng(SEED)

    if not QUIET:
        print(f"DiT 核心机制数值验证 | numpy {np.__version__} | seed = {SEED}")

    selftest_only = QUIET
    if selftest_only:
        for k in sorted(EXPS):
            EXPS[k]()
        print("selftest: 全部断言通过")
        return 0

    picked = [int(a) for a in args] if args else sorted(EXPS)
    for k in picked:
        EXPS[k]()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
