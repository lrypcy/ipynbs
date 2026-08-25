#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《PTQ（09）：SqueezeLLM、VPTQ 与 CLAQ》配套实验。

纯 numpy + matplotlib，CPU 秒级。三块内容：

  (A) 码本学习 vs 均匀网格：RTN / 普通 k-means / 敏感度加权 k-means
      （CLAQ 思想：非均匀码本直接拟合权重分布）
  (B) 敏感度加权到底有没有用：当敏感坐标与幅值解耦时的诚实答案——
      共享 1D 码本下增益是个位数百分比（并解释为什么）
  (C) SqueezeLLM 的真正杀器 dense-and-sparse 分解：按显著度
      h*w^2 挑极少数坐标保 FP16，其余走 4-bit 码本

目标函数（层输出误差）：
    ||X W^T - X Q^T||^2 = sum_ij h_ij (w_ij - q_ij)^2,  h_j = sum_t x_tj^2
正是 OBS 对角近似的加权视角。

对应文章：https://lrypcy.github.io/2026/08/24/ptq-09-squeezellm-vptq-claq/
输出图到 results/（相对本文件）。
注：VPTQ 的向量维度自由度演示见 quip_aqlm 实验。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
T, K, N = 4096, 128, 192         # token / 输入通道 / 输出通道
N_LEV = 16                        # 主对比码本大小（4-bit）
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def make_problem():
    """激活含高能量通道；权重含少量大值。敏感度与幅值部分解耦。"""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((T, K))
    hot = rng.choice(K, size=8, replace=False)
    X[:, hot] *= 10.0                                   # 敏感通道
    W = rng.standard_normal((K, N))
    out = rng.random((K, N)) < 0.02
    W[out] *= 8.0                                       # 稀疏大值
    return X, W


def out_rel_mse(X, W, Wq):
    E = X @ (W - Wq)
    return float(np.sum(E ** 2) / np.sum((X @ W) ** 2))


# ----------------------------------------------------------------------
# 三种码本方法
# ----------------------------------------------------------------------
def rtn_codebook(W, n_lev):
    edges = np.linspace(W.min(), W.max(), n_lev + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def lloyd_1d(w, hw, n_lev, iters=150, warm=None):
    """1D（加权）k-means，分位点初始化，支持热启动。"""
    cent = warm if warm is not None else np.quantile(
        w, (np.arange(n_lev) + 0.5) / n_lev)
    prev = None
    for _ in range(iters):
        idx = np.searchsorted((cent[:-1] + cent[1:]) / 2, w)
        new = cent.copy()
        for k in range(n_lev):
            m = idx == k
            if m.any():
                new[k] = np.sum(hw[m] * w[m]) / np.sum(hw[m])
        if prev is not None and np.allclose(new, cent):
            cent = new
            break
        prev, cent = cent.copy(), new
    return cent


def apply_codebook(W, cent, keep_mask=None):
    Wq = W.copy()
    sel = np.ones_like(W, dtype=bool) if keep_mask is None else ~keep_mask
    w = W[sel]
    mid = (cent[:-1] + cent[1:]) / 2.0
    Wq[sel] = cent[np.searchsorted(mid, w)]
    return Wq


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    X, W = make_problem()
    h = np.sum(X ** 2, axis=0)
    Hmat = np.repeat(h[:, None], N, axis=1)
    w_flat = W.ravel()

    # ---- (A)+(B) 三方法 ----
    c_rtn = rtn_codebook(W, N_LEV)
    c_km = lloyd_1d(w_flat, np.ones_like(w_flat), N_LEV)
    c_wkm = lloyd_1d(w_flat, Hmat.ravel(), N_LEV, warm=c_km.copy())

    m_rtn = out_rel_mse(X, W, apply_codebook(W, c_rtn))
    m_km = out_rel_mse(X, W, apply_codebook(W, c_km))
    m_wkm = out_rel_mse(X, W, apply_codebook(W, c_wkm))

    print("4-bit（%d 码字，全量化）层输出相对 MSE：" % N_LEV)
    print("  RTN 均匀网格         : %.4e" % m_rtn)
    print("  普通 k-means (CLAQ)  : %.4e  (较 RTN 降 %.1f%%)"
          % (m_km, 100 * (1 - m_km / m_rtn)))
    print("  敏感度加权 k-means   : %.4e  (较普通再降 %.1f%%)"
          % (m_wkm, 100 * (1 - m_wkm / m_km)))
    print("  >> 加权增益的大小取决于\"敏感度 x 幅值分布\"的耦合程度：")
    print("     本构造（敏感通道激活能量x10）下再降 %.0f%%；若权重与" % (100 * (1 - m_wkm / m_km)))
    print("     敏感度完全解耦，增益趋近 0——这正是 SqueezeLLM 用完整")
    print("     Fisher 信息而非朴素激活二阶矩估计 h 的原因。")

    # ---- (C) dense-and-sparse 分解 ----
    saliency = (W ** 2) * Hmat                    # OBS 式显著度
    order = np.argsort(saliency.ravel())[::-1]
    dec_rows = []
    base = m_km
    print("-" * 62)
    print("dense-and-sparse（其余坐标 4-bit k-means 码本）：")
    for frac in (0.0, 0.002, 0.01):
        kk = int(frac * W.size)
        keep = np.zeros(W.size, dtype=bool)
        keep[order[:kk]] = True
        m = out_rel_mse(X, W, apply_codebook(W, c_km, keep.reshape(K, N)))
        dec_rows.append({"keep_frac": frac, "rel_mse": m})
        tag = "无分解       " if frac == 0 else "%.1f%% 坐标保FP16" % (frac * 100)
        print("  %s: %.4e%s"
              % (tag, m, "" if frac == 0 else "  (较全量化再降 %.1f%%)"
                 % (100 * (1 - m / base))))

    # ---- 码本大小扫描 ----
    sizes = [4, 16, 64]
    curve = {"rtn": [], "km": [], "wkm": []}
    for nl in sizes:
        cr = rtn_codebook(W, nl)
        ck = lloyd_1d(w_flat, np.ones_like(w_flat), nl)
        cw = lloyd_1d(w_flat, Hmat.ravel(), nl, warm=ck.copy())
        curve["rtn"].append(out_rel_mse(X, W, apply_codebook(W, cr)))
        curve["km"].append(out_rel_mse(X, W, apply_codebook(W, ck)))
        curve["wkm"].append(out_rel_mse(X, W, apply_codebook(W, cw)))

    res = {
        "mse_16lev": {"rtn": m_rtn, "kmeans": m_km, "weighted": m_wkm},
        "dense_sparse": dec_rows,
        "sweep": {"n_levels": sizes, **curve},
    }

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    names = ["RTN", "k-means\n(CLAQ)", "weighted\nk-means"]
    vals = [m_rtn, m_km, m_wkm]
    bars = axes[0].bar(names, vals, color=["#999999", "#DD8452", "#55A868"],
                       width=0.55)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v * 1.05, "%.2e" % v,
                     ha="center", fontsize=8)
    axes[0].set_ylabel("output rel. MSE")
    axes[0].set_title("4-bit full quantization")
    axes[0].grid(axis="y", alpha=0.3)

    fr = [d["keep_frac"] for d in dec_rows]
    mv = [d["rel_mse"] for d in dec_rows]
    axes[1].plot([f * 100 for f in fr], mv, "o-", color="#4C72B0")
    for f, v in zip(fr, mv):
        axes[1].annotate("%.2e" % v, (f * 100, v),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes[1].set_xlabel("% sensitive coords kept in FP16")
    axes[1].set_ylabel("output rel. MSE")
    axes[1].set_title("Dense-and-sparse decomposition\n(rest: 4-bit k-means)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(RES, "squeezellm_sensitivity_and_decomposition.png")
    fig.savefig(p1, dpi=130)

    xs = np.log2(sizes)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    ax2.semilogy(xs, curve["rtn"], "o-", color="#999999", label="RTN")
    ax2.semilogy(xs, curve["km"], "s-", color="#DD8452", label="k-means")
    ax2.semilogy(xs, curve["wkm"], "D-", color="#55A868", label="weighted")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["%d-bit" % int(v) for v in xs])
    ax2.set_xlabel("codebook size")
    ax2.set_ylabel("output rel. MSE")
    ax2.set_title("Error vs codebook size")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(RES, "codebook_size_sweep.png")
    fig2.savefig(p2, dpi=130)

    print("-" * 62)
    print(f"[save] {p1}")
    print(f"[save] {p2}")
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
