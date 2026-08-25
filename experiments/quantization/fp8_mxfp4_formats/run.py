#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《PTQ（08）：GGUF、FP8 与 MXFP4》配套实验。

纯 numpy 位运算仿真各种数值格式的 round-trip 量化，回答一个问题：
同样 4 bit，为什么 MXFP4 能打；FP8 的 E4M3 与 E5M2 差在哪。

  (A) 通用浮点格式仿真器：给定 (e_bits, m_bits, bias)，用
      frexp + 半偶舍入（IEEE round-to-nearest-even）实现量化，
      自动处理正规数 / 子正规数 / 溢出饱和
  (B) 格式横评：FP16 / BF16 / E5M2 / E4M3 / INT8 在高斯与拉普拉斯
      权重上的 SQNR
  (C) 4-bit 专场：MXFP4（E2M1 + 组内共享指数, group=32）vs
      NF4（常数码本 + 块 absmax, block=64）——展示共享 scale 的作用
  (D) 表示层级密度图：E4M3 与 E5M2 在 (0, 2] 区间的台阶分布

对应文章：https://lrypcy.github.io/2026/08/24/ptq-08-gguf-fp8-mxfp4/
输出图到 results/（相对本文件）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
NS = 100_000
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

NF4_LEVELS = np.array([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7226, 1.0,
])
E2M1_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ----------------------------------------------------------------------
# (A) 浮点格式仿真器
# ----------------------------------------------------------------------
def quant_float(x, e_bits, m_bits, bias, saturate=False):
    """把 x 量化到 (e_bits 指数, m_bits 尾数, bias) 的浮点格式。

    正规数：值域 [(2-2^-m)·2^emin, (2-2^-m)·2^emax]，步长随指数翻倍；
    子正规数：步长固定为 2^(emin-m)；溢出默认 -> inf（saturate=True 则夹到最大值）。
    舍入一律半偶（np.round 即 round-half-to-even，与 IEEE 一致）。
    """
    x = np.asarray(x, dtype=np.float64)
    emax = (2 ** e_bits - 2) - bias          # 最大正规指数（无偏）
    emin = 1 - bias                          # 最小正规指数（无偏）
    maxval = (2.0 - 2.0 ** (-m_bits)) * 2.0 ** emax

    sign = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x)
    out = np.zeros_like(ax)

    nz = ax > 0
    e_unb = np.zeros_like(ax)
    e_unb[nz] = np.floor(np.log2(ax[nz]))
    e_q = np.clip(e_unb, emin - m_bits, emax).astype(np.int64)  # 允许下探到子正规步长
    step = 2.0 ** (e_q - m_bits)
    q = np.round(ax / step) * step                        # 半偶舍入
    # 尾数进位（如 1.999..5 舍入后跨过 2 的幂）：指数 +1 再走一遍
    carry = q >= 2.0 ** (e_q + 1)
    if np.any(carry & nz):
        e_q2 = np.minimum(e_q + 1, emax)
        step2 = 2.0 ** (e_q2 - m_bits)
        q[carry] = np.round(ax[carry] / step2[carry]) * step2[carry]
        q[carry] = np.minimum(q[carry], maxval)
    if saturate:
        q = np.minimum(q, maxval)
    else:
        q = np.where(q > maxval, np.inf, q)
    out[nz] = q[nz]
    return sign * out


def quant_int8_sym(x):
    """逐张量对称 INT8 参照。"""
    s = np.max(np.abs(x)) / 127.0
    return np.round(x / s) * s


def quant_mxfp4(x, group=32):
    """MXFP4：E2M1 码本 + 组内共享指数 scale（OCP MX）。"""
    g = x.reshape(-1, group)
    amax = np.max(np.abs(g), axis=1)
    # 减 2：让组内最大元落进 [4,8)，E2M1 的头部码级(4,6)才用得上
    e_shared = np.floor(np.log2(np.maximum(amax, 1e-30))) - 2.0
    scale = 2.0 ** e_shared
    u = g / scale[:, None]
    # 最近邻查表（对 |u| 查表后补符号；平局取较低码级，统计上足够）
    mid = (E2M1_LEVELS[:-1] + E2M1_LEVELS[1:]) / 2.0
    au = np.abs(u)
    idx = np.searchsorted(mid, au)
    qg = np.sign(u) * E2M1_LEVELS[idx]
    return (qg * scale[:, None]).reshape(x.shape)


def quant_nf4(x, block=64):
    """QLoRA 式 NF4：常数码本 + 块内 absmax 归一。"""
    pad = (-x.size) % block
    xp = np.concatenate([x.ravel(), np.zeros(pad)])
    g = xp.reshape(-1, block)
    amax = np.max(np.abs(g), axis=1)
    scale = np.maximum(amax, 1e-30)
    u = g / scale[:, None]
    mid = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2.0
    idx = np.searchsorted(mid, u.ravel())
    qg = NF4_LEVELS[idx].reshape(g.shape)
    return (qg * scale[:, None]).reshape(-1)[: x.size]


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def sqnr_db(x, xq):
    err = x - xq
    return float(10.0 * np.log10(np.sum(x ** 2) / np.sum(err ** 2)))


def main():
    rng = np.random.default_rng(SEED)
    data = {
        "gauss": rng.standard_normal(NS),
        "laplace": rng.laplace(0.0, 1.0 / np.sqrt(2.0), NS),  # 单位方差
    }

    formats = [
        ("INT8 sym", lambda v: quant_int8_sym(v)),
        ("FP16  e5m10", lambda v: quant_float(v, 5, 10, 15)),
        ("BF16  e8m7", lambda v: quant_float(v, 8, 7, 127)),
        ("E5M2 (FP8)", lambda v: quant_float(v, 5, 2, 15)),
        ("E4M3 (FP8)", lambda v: quant_float(v, 4, 3, 7, saturate=True)),
        ("MXFP4 grp32", quant_mxfp4),
        ("NF4 blk64", quant_nf4),
    ]

    table = {}
    for name, fn in formats:
        row = {}
        for dname, x in data.items():
            row[dname] = sqnr_db(x, fn(x))
        table[name] = row
        print("%-12s gauss %6.2f dB | laplace %6.2f dB" %
              (name, row["gauss"], row["laplace"]))

    res = {"sqnr_table_db": table}

    # ---- 图 1：SQNR 柱状 ----
    names = [n for n, _ in formats]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = ["#4C72B0"] * 5 + ["#DD8452", "#55A868"]
    for ax, dname, ttl in zip(axes, ["gauss", "laplace"],
                              ["Gaussian weights", "Laplace weights"]):
        vals = [table[n][dname] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("SQNR (dB)")
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Format round-trip SQNR (higher is better)", y=1.02)
    fig.tight_layout()
    p1 = os.path.join(RES, "format_sqnr_comparison.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight")

    # ---- 图 2：E4M3 vs E5M2 表示台阶 ----
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.2))
    grid = np.linspace(1e-4, 2.0, 4000)
    for fmt, eb, mb, bs, c in [("E4M3", 4, 3, 7, "#4C72B0"),
                               ("E5M2", 5, 2, 15, "#C44E52")]:
        q = np.abs(quant_float(grid, eb, mb, bs, saturate=True))
        lv = np.unique(q[q > 0])
        ax2.semilogx(lv, np.arange(1, len(lv) + 1), ".", ms=4, color=c, label=fmt)
    ax2.set_xlabel("representable magnitude")
    ax2.set_ylabel("level index")
    ax2.set_title("Representable levels in (0, 2]: E4M3 has denser steps\n"
                  "(extra mantissa bit) but fewer exponent codes")
    ax2.legend()
    ax2.grid(alpha=0.3)
    p2 = os.path.join(RES, "e4m3_vs_e5m2_levels.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=130)

    print(f"[save] {p1}")
    print(f"[save] {p2}")
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
