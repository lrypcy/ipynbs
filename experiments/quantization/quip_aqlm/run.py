#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《PTQ（05）：QuIP# 与 AQLM》配套实验。

纯 numpy + matplotlib，CPU 秒级。两件事：

  (A) incoherence processing 的必要性：带稀疏大值的权重矩阵直接做
      逐张量 INT3 会灾难性失败（step 被离群值绑架）；Hadamard 旋转
      把能量摊平后同一量化器误差降一个量级
  (B) 同等比特预算下 向量量化 vs 标量量化：对每个码字向量做 RMS
      归一化（AQLM/QuIP# 风格）后的 VQ 在旋转基下明显更优

数学骨架：W -> H W 后，层输出 y = xW^T = x(H^T H)W^T，把 x 替换为
x H^T 即可在"旋转域"完成计算，层数学等价。

对应文章：https://lrypcy.github.io/2026/08/24/ptq-05-quip-aqlm/
输出图到 results/（相对本文件）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
K = N = 256                       # 方阵权重，便于 Hadamard 旋转
OUT_FRAC = 0.02                   # 稀疏大值比例
OUT_MAG = 8.0                     # 大值倍数
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def fast_wht(x):
    """沿最后一维归一化 Walsh-Hadamard 变换（n 为 2 的幂）。"""
    n = x.shape[-1]
    h = x.astype(np.float64).copy()
    step = 1
    while step < n:
        h = h.reshape(-1, step, 2, n // (step * 2))
        a = h[:, :, 0, :].copy()
        b = h[:, :, 1, :].copy()
        h[:, :, 0, :] = a + b
        h[:, :, 1, :] = a - b
        h = h.reshape(-1, n)
        step *= 2
    return (h / np.sqrt(n)).reshape(x.shape)


def make_weight():
    """低秩骨架 + 稀疏大值：真实权重的两个关键特征。"""
    rng = np.random.default_rng(SEED)
    rank = 16
    W = rng.standard_normal((K, rank)) @ rng.standard_normal((rank, N)) / np.sqrt(rank)
    mask = rng.random((K, N)) < OUT_FRAC
    W[mask] *= OUT_MAG
    return W


# ----------------------------------------------------------------------
# 标量量化（逐张量对称，mid-rise 网格）
# ----------------------------------------------------------------------
def scalar_quant_tensor(W, bits):
    n_half = 2 ** (bits - 1)
    amax = np.max(np.abs(W))
    if amax == 0:
        return np.zeros_like(W)
    step = amax / n_half
    idx = np.clip(np.round((np.abs(W) - step / 2) / step), 0, n_half - 1)
    return np.sign(W) * (idx + 0.5) * step


# ----------------------------------------------------------------------
# 向量量化：逐块 RMS 归一化 + 手写 kmeans++/Lloyd
# ----------------------------------------------------------------------
def kmeans(Xtr, n_cent, iters=15, seed=SEED + 1):
    rg = np.random.default_rng(seed)
    n = Xtr.shape[0]
    cent = np.empty((n_cent, Xtr.shape[1]))
    cent[0] = Xtr[rg.integers(n)]
    d2 = np.sum((Xtr - cent[0]) ** 2, axis=1)
    for i in range(1, n_cent):
        p = np.maximum(d2, 0)
        cent[i] = Xtr[rg.choice(n, p=p / p.sum())]
        d2 = np.minimum(d2, np.sum((Xtr - cent[i]) ** 2, axis=1))
    assign = None
    for _ in range(iters):
        new_assign = np.empty(n, dtype=np.int64)
        for s in range(0, n, 16384):
            e = min(s + 16384, n)
            d = ((Xtr[s:e, None, :] - cent[None, :, :]) ** 2).sum(-1)
            new_assign[s:e] = d.argmin(1)
        if assign is not None and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for k in range(n_cent):
            m = assign == k
            if m.any():
                cent[k] = Xtr[m].mean(axis=0)
    return cent


def vector_quant_scaled(W, v, n_cent):
    """把列切成 v 维码字；逐块 RMS 归一化后查全局码本（免被大值绑架）。"""
    Nb = W.shape[1]
    flat = W.T.reshape(Nb, -1, v).reshape(-1, v)
    rms = np.sqrt(np.mean(flat ** 2, axis=1, keepdims=True)) + 1e-12
    u = flat / rms
    rg = np.random.default_rng(SEED + 1)
    sub = u[rg.choice(len(u), min(len(u), 40000), replace=False)]
    cent = kmeans(sub, n_cent)
    outu = np.empty_like(u)
    for s in range(0, len(u), 16384):
        e = min(s + 16384, len(u))
        d = ((u[s:e, None, :] - cent[None, :, :]) ** 2).sum(-1)
        outu[s:e] = cent[d.argmin(1)]
    rec_blocks = (outu * rms).reshape(Nb, -1, v)     # (n,b,c)
    return rec_blocks.transpose(1, 2, 0).reshape(K, N)


def rel_fro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    W = make_weight()
    H = fast_wht(np.eye(K))
    Wr = H @ W                                        # 旋转域权重

    print("合成权重：%dx%d，低秩骨架(rank16) + %.0f%% 稀疏大值(x%.0f)"
          % (K, N, OUT_FRAC * 100, OUT_MAG))
    print("=" * 64)

    # ---- (A) 旋转对标量量化的拯救 ----
    rows = []
    for bits in (2, 3, 4):
        e_raw = rel_fro(scalar_quant_tensor(W, bits), W)
        e_rot = relf_rot = rel_fro(H @ scalar_quant_tensor(Wr, bits), W)
        rows.append({"method": f"INT{bits} tensor", "bpw": bits,
                     "err_raw": e_raw, "err_rot": e_rot})
        print("INT%d 标量(逐张量): raw=%.4f | rotated=%.4f  (%.0f%%↓)"
              % (bits, e_raw, relf_rot, 100 * (1 - e_rot / e_raw)))
    print("-" * 64)

    # ---- (B) 同预算 VQ vs 标量（旋转基）----
    e_vq_raw = rel_fro(vector_quant_scaled(W, 8, 256), W)
    e_vq_rot = rel_fro(H @ vector_quant_scaled(Wr, 8, 256), W)
    print("VQ v=8,C=256 (~1bpw 码字 + 每块RMS):")
    print("   raw=%.4f | rotated=%.4f" % (e_vq_raw, e_vq_rot))
    int2 = next(r for r in rows if r["bpw"] == 2)
    print("   同 2-bit 场景对比（旋转基）：VQ %.4f vs INT2 %.4f -> %+.1f%%"
          % (e_vq_rot, int2["err_rot"],
             -100 * (e_vq_rot - int2["err_rot"]) / int2["err_rot"]))
    print("=" * 64)

    res = {"rows": rows,
           "vq_v8c256": {"err_raw": e_vq_raw, "err_rot": e_vq_rot}}

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(rows) + 1)
    labels = [r["method"] for r in rows] + ["VQ v8 C256\n(+blk-RMS)"]
    raw_vals = [r["err_raw"] for r in rows] + [e_vq_raw]
    rot_vals = [r["err_rot"] for r in rows] + [e_vq_rot]
    wd = 0.38
    ax.bar(x - wd / 2, raw_vals, wd, color="#C44E52", label="raw basis")
    ax.bar(x + wd / 2, rot_vals, wd, color="#4C72B0",
           label="Hadamard-rotated basis")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative reconstruction error (log)")
    ax.set_title("Incoherence rescues scalar quant;\n"
                 "block-normalized VQ wins at matched budget (rotated)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    p1 = os.path.join(RES, "vq_vs_scalar_and_incoherence.png")
    fig.savefig(p1, dpi=130)

    print(f"[save] {p1}")
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
