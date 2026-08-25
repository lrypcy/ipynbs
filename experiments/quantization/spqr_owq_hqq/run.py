#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《LLM PTQ 深度解析（04）：SpQR、OWQ 与 HQQ》配套实验。

纯 numpy + matplotlib(Agg)，CPU 数秒跑完。
对应文章：https://lrypcy.github.io/2026/08/24/ptq-04-spqr-owq-hqq/

复现三件事：
  (A) OWQ 弱列感知混合精度：按校准激活统计 mu_j = E|x_j| 找出 outlier 通道（弱列），
      整列保留高精度（fp16 代理），其余列做 INT4 分组量化；
      与全矩阵 INT4 RTN 对比激活加权输出相对误差。
  (B) SpQR 元素级敏感度拆分对照：s_ij = |W_ij| * sqrt(H_jj)（OBQ 敏感度的对角近似），
      top-alpha 元素保高精度、其余 INT4 —— 与 OWQ 的"显微镜 vs 放大镜"粒度对照；
      并按文章 §2.3 公式计算两者的有效位宽 b_eff（fp16 开销 + 稀疏索引开销）。
  (C) HQQ 半二次分裂：Y 步 Y=(W+lam*Q)/(1+lam) 与 (s,z) 步最小二乘法方程均有闭式解，
      交替迭代至自洽；演示迭代收敛性（第 1~2 步吃掉绝大部分收益）、
      以及 2/3/4bit 下对 min-max RTN 的权重重构 MSE 优势（位宽越低差距越大）。

输出图到本目录 results/（相对本文件，从任意 cwd 运行均可）：
  owq_mixed_precision.png   OWQ/SpQR 混合精度误差对比 + 保护比例扫描
  hqq_half_quadratic.png    HQQ 迭代收敛曲线 + 各位宽 RTN/HQQ MSE 对比
另将全部关键数字写入 results/results.json。
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 与文章一致的固定超参 ----
SEED = 0
D_OUT, D_IN = 512, 512        # 权重矩阵形状 (d_out, d_in)
N_OUT_COLS = 6                # 种植的 outlier 通道数（6/512 ≈ 1.2%，文章口径 0.1%~1%）
GROUP_SIZE = 128              # 分组量化组大小（沿输入通道方向），与文章代码一致
BITS = 4                      # 混合精度主实验中低比特部分的位宽
HQ_LAM = 1e-4                 # HQQ 论文默认 lambda
PROTECT_RATIO = 0.01          # 主对比用的保护比例（OWQ 按通道 / SpQR 按元素）


# ----------------------------------------------------------------------
# 合成数据：带 outlier 通道的线性层 + 校准激活
# ----------------------------------------------------------------------
def make_synth(rng):
    """生成合成层：少数输入通道的权重与激活幅值同时放大（复刻 LLM 观察）。"""
    out_cols = rng.choice(D_IN, size=N_OUT_COLS, replace=False)
    W = rng.normal(0.0, 0.05, (D_OUT, D_IN))
    W[:, out_cols] *= 20.0                    # outlier 通道的权重幅值放大
    X = rng.normal(0.0, 1.0, (64, D_IN))      # 校准集：64 个 token
    X[:, out_cols] *= 8.0                     # 同一批通道的激活放大 -> 长尾
    return W, X, out_cols


# ----------------------------------------------------------------------
# 基线：分组 RTN（min-max 仿射，逐组 scale/zero）
# ----------------------------------------------------------------------
def rtn_group_quantize(W, bits=BITS, group_size=GROUP_SIZE, eps=1e-12):
    """分组 RTN 仿射量化（min-max -> 整数 -> 反量化），沿输入通道分组。"""
    W = W.astype(np.float64)
    out = np.empty_like(W)
    n_in = W.shape[1]
    qmax = 2 ** bits - 1
    for start in range(0, n_in, group_size):
        col = slice(start, min(start + group_size, n_in))
        wg = W[:, col]
        wmin, wmax = wg.min(), wg.max()
        scale = (wmax - wmin) / (qmax + eps)
        zero = wmin
        q = np.clip(np.round((wg - zero) / (scale + eps)), 0, qmax)
        out[:, col] = q * scale + zero
    return out


# ----------------------------------------------------------------------
# (A) OWQ：通道级混合精度（outlier 通道整列保高精度）
# ----------------------------------------------------------------------
def owq_quantize(W, X_calib, ratio=PROTECT_RATIO, bits=BITS, group_size=GROUP_SIZE):
    """
    OWQ 简化实现（文章 §3）：
      1) 校准激活幅度均值 mu_j = mean|x_j|
      2) 取 mu 最高的一小部分通道作为 outlier 通道（弱列）
      3) 这些整列保留原值（论文为 fp16），其余列 INT4 分组量化
    返回 (反量化矩阵 Wq, 被保护的通道索引)。
    """
    mu = np.abs(X_calib).mean(axis=0)                 # (d_in,) 激活幅度统计
    k = max(0, int(round(ratio * D_IN)))
    weak = np.sort(np.argsort(mu)[-k:]) if k > 0 else np.array([], dtype=int)
    Wq = rtn_group_quantize(W, bits, group_size)
    Wq[:, weak] = W[:, weak]                          # 弱列整列保留（fp16 代理 = 原值）
    return Wq, weak


# ----------------------------------------------------------------------
# (B) SpQR：元素级敏感度拆分（对照）
# ----------------------------------------------------------------------
def spqr_quantize(W, X_calib, ratio=PROTECT_RATIO, bits=BITS, group_size=GROUP_SIZE):
    """
    SpQR 简化实现（文章 §2）：
      1) Hessian 对角近似 H_jj = E[x_j^2]
      2) 敏感度 s_ij = |W_ij| * sqrt(H_jj)   （s_ij = |W_ij|/sqrt([H^-1]_jj) 的对角近似）
      3) top-alpha 元素拆出保原值（论文为 fp16 + CSR 索引）
      4) 其余元素 INT4 分组量化
    返回 (反量化矩阵 Wq, outlier 坐标)。
    """
    h = (X_calib.astype(np.float64) ** 2).mean(axis=0)   # (d_in,)
    S = np.abs(W) * np.sqrt(h)[None, :]                  # 元素级敏感度
    k = max(0, int(round(ratio * W.size)))
    if k > 0:
        flat = np.argpartition(S.ravel(), -k)[-k:]
        rows, cols = np.unravel_index(flat, W.shape)
    else:
        rows = cols = np.array([], dtype=int)
    Wq = rtn_group_quantize(W, bits, group_size)
    Wq[rows, cols] = W[rows, cols]
    return Wq, (rows, cols)


# 有效位宽账（文章 §2.3）：b_eff = (1-a)*b_dense + a*16 + 索引开销
def b_eff_owq(alpha):
    """OWQ：稠密混合精度列，无稀疏坐标开销。"""
    return (1 - alpha) * BITS + alpha * 16.0


def b_eff_spqr(alpha):
    """SpQR：元素级 fp16 + 行列坐标索引开销 alpha*(ceil(log2 d_out)+ceil(log2 d_in))。"""
    idx_bits = int(np.ceil(np.log2(D_OUT))) + int(np.ceil(np.log2(D_IN)))
    return (1 - alpha) * BITS + alpha * 16.0 + alpha * idx_bits


# ----------------------------------------------------------------------
# (C) HQQ：半二次分裂，两个子问题均为闭式解
# ----------------------------------------------------------------------
def hqq_group(wg, bits=BITS, n_iter=2, lam=HQ_LAM, eps=1e-12):
    """
    对单个权重组做 HQQ 迭代（文章 §4.4 伪代码），返回反量化结果。
      q 步:     q = clip(round((W-z)/s), 0, qmax)
      Y 步:     Y = (W + lam*Q) / (1 + lam)          <- 闭式解一
      (s,z) 步: s = Cov(Y,q)/Var(q), z = mean(Y)-s*mean(q)   <- 最小二乘法方程闭式解二
    n_iter=0 时退化为 min-max 初始化的一次性 RTN（即基线）。
    """
    qmax = 2 ** bits - 1
    wmin, wmax = wg.min(), wg.max()
    s = (wmax - wmin) / (qmax + eps)
    z = wmin
    for _ in range(int(n_iter)):
        q = np.clip(np.round((wg - z) / (s + eps)), 0, qmax)
        Q = s * q + z
        Y = (wg + lam * Q) / (1.0 + lam)              # Y 步闭式解
        ym, qm = Y.mean(), q.mean()
        s = np.sum((Y - ym) * (q - qm)) / (np.sum((q - qm) ** 2) + eps)  # (s,z) 步闭式解
        z = ym - s * qm
    q = np.clip(np.round((wg - z) / (s + eps)), 0, qmax)
    return s * q + z


def hqq_quantize(W, bits=BITS, group_size=GROUP_SIZE, n_iter=2, lam=HQ_LAM):
    """分组 HQQ 量化（每组独立估计 s/z，data-free：只碰权重本身）。"""
    W = W.astype(np.float64)
    n_in = W.shape[1]
    out = np.empty_like(W)
    for start in range(0, n_in, group_size):
        col = slice(start, min(start + group_size, n_in))
        out[:, col] = hqq_group(W[:, col], bits, n_iter, lam)
    return out


# ----------------------------------------------------------------------
# 误差度量
# ----------------------------------------------------------------------
def wmse(W, What):
    """权重空间重构 MSE。"""
    return float(np.mean((W - What) ** 2))


def act_rel_err(W, What, X):
    """激活加权输出相对误差 ||(W-Wq) X^T||_F / ||W X^T||_F（outlier 通道被放大的口径）。"""
    return float(np.linalg.norm((W - What) @ X.T) / (np.linalg.norm(W @ X.T) + 1e-12))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(here, "results")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    W, X, out_cols = make_synth(rng)

    w_rtn = rtn_group_quantize(W, BITS, GROUP_SIZE)

    # ---- 混合精度主对比（保护比例 1%）----
    ratio = PROTECT_RATIO
    w_owq, weak = owq_quantize(W, X, ratio, BITS, GROUP_SIZE)
    w_spqr, (sr, sc) = spqr_quantize(W, X, ratio, BITS, GROUP_SIZE)
    hit_owq = len(set(weak.tolist()) & set(out_cols.tolist()))
    # SpQR 元素命中统计：拆出的元素中落在真实 outlier 列上的个数
    mask_true = np.zeros(W.shape, dtype=bool)
    mask_true[:, out_cols] = True
    hit_spqr = int(mask_true[sr, sc].sum())

    # ---- 保护比例扫描（验证幂律/饱和观察）----
    ratios = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    sweep = {"ratios": ratios, "owq": [], "spqr": [],
             "owq_beff": [], "spqr_beff": []}
    for r in ratios:
        wq_o, _ = owq_quantize(W, X, r, BITS, GROUP_SIZE)
        wq_s, _ = spqr_quantize(W, X, r, BITS, GROUP_SIZE)
        sweep["owq"].append(act_rel_err(W, wq_o, X))
        sweep["spqr"].append(act_rel_err(W, wq_s, X))
        sweep["owq_beff"].append(b_eff_owq(r))
        sweep["spqr_beff"].append(b_eff_spqr(r))

    # ---- HQQ：位宽对比 + 迭代次数消融 ----
    bits_list = [2, 3, 4]
    hqq_bits = {}
    for b in bits_list:
        w_r = rtn_group_quantize(W, b, GROUP_SIZE)
        w_h = hqq_quantize(W, b, GROUP_SIZE, n_iter=2)
        hqq_bits[b] = (wmse(W, w_r), wmse(W, w_h))

    iter_list = [0, 1, 2, 4, 8]
    iter_curve = {}
    for b in (2, 4):
        iter_curve[b] = [wmse(W, hqq_quantize(W, b, GROUP_SIZE, n_iter=n)) for n in iter_list]

    # ============================================================
    # 打印真实数字
    # ============================================================
    print("=" * 66)
    print("合成数据统计（复刻 LLM outlier 通道观察）")
    print("=" * 66)
    print(f"权重形状              : W = {D_OUT} x {D_IN}")
    print(f"outlier 通道数/占比   : {N_OUT_COLS}/{D_IN} = {N_OUT_COLS/D_IN*100:.2f}%")
    print(f"outlier 通道权重 sigma: {W[:, out_cols].std():.3f}  vs 其余 {np.delete(W, out_cols, axis=1).std():.3f}")

    print("\n" + "=" * 66)
    print(f"[主对比] INT{BITS} 低比特 + {ratio*100:.0f}% 高精度保护（激活加权相对误差）")
    print("=" * 66)
    e_rtn = act_rel_err(W, w_rtn, X)
    e_owq = act_rel_err(W, w_owq, X)
    e_spqr = act_rel_err(W, w_spqr, X)
    print(f"{'RTN INT4 全量化'          :<28}{e_rtn:>12.4f}")
    print(f"{'SpQR 元素级 1% fp16'      :<28}{e_spqr:>12.4f}   b_eff={b_eff_spqr(ratio):.2f} bit")
    print(f"{'OWQ 通道级 1% 列 fp16'    :<28}{e_owq:>12.4f}   b_eff={b_eff_owq(ratio):.2f} bit")
    print(f"OWQ 相对 RTN 误差降幅      : {(1 - e_owq/e_rtn)*100:.1f}%")
    print(f"SpQR 相对 RTN 误差降幅     : {(1 - e_spqr/e_rtn)*100:.1f}%")
    print(f"OWQ 命中的真实 outlier 通道: {hit_owq}/{N_OUT_COLS}"
          f"（保护 {len(weak)} 列）")
    print(f"SpQR 拆出元素落在 outlier 列: {hit_spqr}/{int(round(ratio*W.size))}")

    print("\n" + "=" * 66)
    print("[扫描] 保护比例 -> 激活加权相对误差 / 有效位宽")
    print("=" * 66)
    print(f"{'ratio':>7} {'OWQ err':>10} {'OWQ b_eff':>10} {'SpQR err':>10} {'SpQR b_eff':>11}")
    for i, r in enumerate(ratios):
        print(f"{r:>7.3f} {sweep['owq'][i]:>10.4f} {sweep['owq_beff'][i]:>10.2f}"
              f" {sweep['spqr'][i]:>10.4f} {sweep['spqr_beff'][i]:>11.2f}")

    print("\n" + "=" * 66)
    print("[HQQ vs RTN] 权重重构 MSE（半二次分裂闭式解，n_iter=2）")
    print("=" * 66)
    for b in bits_list:
        m_r, m_h = hqq_bits[b]
        print(f"bits={b}:  RTN={m_r:.4e}   HQQ={m_h:.4e}   降低 {(1-m_h/m_r)*100:.1f}%")

    print("\n[HQQ 迭代次数消融] 权重 MSE（验证 1~2 步收敛）")
    for b in (2, 4):
        row = "  ".join(f"n={n}:{m:.3e}" for n, m in zip(iter_list, iter_curve[b]))
        print(f"  bits={b}: {row}")

    # ============================================================
    # 图 A：OWQ/SpQR 混合精度
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["RTN INT4", f"SpQR elem\n{ratio*100:.0f}% fp16", f"OWQ col\n{ratio*100:.0f}% fp16"]
    vals = [e_rtn, e_spqr, e_owq]
    ax[0].bar(names, vals, color=["#999999", "#4C72B0", "#55A868"])
    ax[0].set_ylabel("activation-weighted rel. error")
    ax[0].set_title(f"Mixed precision vs plain INT{BITS}")
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.02, f"{v:.4f}", ha="center", fontsize=9)
    ax[0].set_ylim(0, max(vals) * 1.18)

    ax[1].plot(ratios, sweep["owq"], "o-", color="#55A868", label="OWQ (channel)")
    ax[1].plot(ratios, sweep["spqr"], "s-", color="#4C72B0", label="SpQR (element)")
    ax[1].axhline(e_rtn, color="#C44E52", linestyle="--", label=f"RTN INT{BITS} baseline")
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("protected ratio")
    ax[1].set_ylabel("activation-weighted rel. error")
    ax[1].set_title("Protection ratio sweep (power-law saturation)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    pa = os.path.join(img_dir, "owq_mixed_precision.png")
    fig.savefig(pa, dpi=130)
    print(f"\n[save] {pa}")

    # ============================================================
    # 图 B：HQQ 半二次分裂
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for b, c in zip((2, 4), ("#C44E52", "#4C72B0")):
        ax[0].plot(iter_list, iter_curve[b], "o-", color=c, label=f"HQQ {b}-bit")
        ax[0].axhline(hqq_bits[b][0], color=c, linestyle="--", alpha=0.6,
                      label=f"RTN {b}-bit ref")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("HQQ iterations n_iter")
    ax[0].set_ylabel("weight reconstruction MSE")
    ax[0].set_title("Half-quadratic splitting convergence")
    ax[0].legend(fontsize=8)

    x = np.arange(len(bits_list))
    wd = 0.36
    ax[1].bar(x - wd/2, [hqq_bits[b][0] for b in bits_list], wd,
              label="RTN", color="#C44E52")
    ax[1].bar(x + wd/2, [hqq_bits[b][1] for b in bits_list], wd,
              label="HQQ (n_iter=2)", color="#55A868")
    for i, b in enumerate(bits_list):
        red = (1 - hqq_bits[b][1] / hqq_bits[b][0]) * 100
        top = max(hqq_bits[b][0], hqq_bits[b][1])
        ax[1].text(i, top * 1.03, f"-{red:.0f}%", ha="center", fontsize=9, color="#333333")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x, [f"{b}-bit" for b in bits_list])
    ax[1].set_ylabel("weight reconstruction MSE")
    ax[1].set_title("Closed-form (s,z) beats min-max grid")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    pb = os.path.join(img_dir, "hqq_half_quadratic.png")
    fig.savefig(pb, dpi=130)
    print(f"[save] {pb}")

    # ---- 全部关键数字落盘 ----
    payload = {
        "config": {"SEED": SEED, "D_OUT": D_OUT, "D_IN": D_IN,
                   "N_OUT_COLS": N_OUT_COLS, "BITS": BITS,
                   "GROUP_SIZE": GROUP_SIZE, "HQ_LAM": HQ_LAM},
        "mixed_precision_INT4_ratio1pct": {
            "rtn_act_rel_err": e_rtn,
            "spqr_act_rel_err": e_spqr,
            "owq_act_rel_err": e_owq,
            "owq_reduction_pct": (1 - e_owq / e_rtn) * 100,
            "spqr_reduction_pct": (1 - e_spqr / e_rtn) * 100,
            "owq_b_eff_bit": b_eff_owq(ratio),
            "spqr_b_eff_bit": b_eff_spqr(ratio),
            "owq_hit_planted_cols": hit_owq},
        "ratio_sweep": sweep,
        "hqq_vs_rtn_mse": {str(b): {"rtn": hqq_bits[b][0],
                                    "hqq_n2": hqq_bits[b][1],
                                    "reduction_pct": (1 - hqq_bits[b][1] / hqq_bits[b][0]) * 100}
                           for b in bits_list},
        "hqq_iter_curve": {str(b): dict(zip(iter_list, iter_curve[b])) for b in (2, 4)},
    }
    pj = os.path.join(img_dir, "results.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {pj}")
    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
