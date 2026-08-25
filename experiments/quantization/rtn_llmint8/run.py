#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- PTQ 系列《01: RTN 基线与 LLM.int8()》配套实验。

移植自博客仓库 experiments_rtn_llmint8.py，算法逻辑与超参不变；
随机种子按本仓库约定固定为 SEED=0。纯 numpy + matplotlib，CPU 秒级跑完。
复现五件事：
  (a) RTN 量化器（per-tensor / per-channel）             —— 文章 §1.2, §1.3
  (b) MSE 最优 scale 的暴力搜索                          —— 文章 §1.3
  (c) 合成带 outlier 的激活数据（对齐 LLM.int8() 论文 §4）—— 文章 §2.2
  (d) 简化版 LLM.int8()（vector-wise + mixed-precision） —— 文章 §3.1-§3.3
  (e) 误差与内存占用对比                                 —— 文章 §4.3, §4.4

输出图到本目录 results/（相对本文件，从任意 cwd 运行均可）：
  synthetic_activation_outliers.png   合成激活的 outlier 形态
  method_error_memory.png             三种方法的误差与内存对比
另将全部关键数字写入 results/results.json。
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")   # 输出到本目录 results/
os.makedirs(OUT, exist_ok=True)

SEED = 0                              # 本仓库约定：固定随机种子
Q8 = 2**7 - 1                         # int8 对称量化上限 127


# ----------------------------------------------------------------------
# (a) RTN 量化器
# ----------------------------------------------------------------------
def rtn_quantize_per_tensor(x, bits=8):
    """RTN 对称 absmax 量化（per-tensor 单一 scale）。

    公式（文章 §1.2）:
        s = max|x| / 127
        q = round(clip(x / s, -127, 127))
        x_hat = q * s
    量化误差上界: |x - x_hat| <= s/2（未 clip 时）。
    """
    s = np.abs(x).max() / Q8
    if s == 0:
        s = 1e-9
    q = np.clip(np.round(x / s), -Q8, Q8)
    return q.astype(np.int8), np.float32(s)


def rtn_quantize_per_channel(w):
    """权重 per-output-channel scale（RTN 的工程改良，文章 §2.1）。"""
    s = np.abs(w).max(axis=0, keepdims=True) / Q8
    s = np.where(s == 0, 1e-9, s)
    q = np.clip(np.round(w / s), -Q8, Q8)
    return q.astype(np.int8), s


def mse_optimal_scale(x, bits=8, n_grid=512):
    """暴力搜索 MSE 最优 clip 点 alpha*（等价于最优 scale s = alpha* / 127）。

    文章 §1.3: min_s  E[(x - s*round(x/s))^2] 在对称量化下等价于
    选择 clip 点 alpha = 127*s。对高斯分布存在一个有限的最优 alpha*，
    但对含 outlier 的长尾分布，alpha* 会被 outlier 拉大。
    """
    amax = float(np.abs(x).max())
    alphas = np.linspace(amax / n_grid, amax, n_grid)
    best_a, best_err = None, np.inf
    for a in alphas:
        s = a / Q8
        xq = np.clip(np.round(x / s), -Q8, Q8) * s
        err = float(np.mean((x - xq) ** 2))
        if err < best_err:
            best_err, best_a = err, a
    return best_a


# ----------------------------------------------------------------------
# (c) 合成数据：对齐论文 §4 的实证发现
#     - 正常维度范围约 [-3.5, 3.5]（论文原文）
#     - outlier 通道数量 |O| <= 7（论文：13B 以内 |O| <= 7）
#     - outlier 幅度为正常维度的 3~20x（论文原文），这里取 25~45
# ----------------------------------------------------------------------
def build_synthetic(tokens=128, d=256, n=256, n_out=6, amp=(25.0, 45.0),
                    active_prob=0.75):
    """合成数据：对齐论文 §4 的实证发现。

    - 正常维度范围约 [-3.5, 3.5]（论文原文）
    - outlier 通道数量 |O| <= 7（论文：13B 以内 |O| <= 7）
    - outlier 幅度为正常维度的 3~20x（论文原文），这里取 25~45
    - outlier 只在约 75% 的 token（序列维度）上出现（论文：6.7B 时
      约 75% 序列维度受影响）——所以逐 token 的动态范围差异很大，
      这正是 token-wise 动态量化的动机
    """
    X = np.random.randn(tokens, d).astype(np.float32) * 0.75     # 正常维度
    W = (np.random.randn(d, n) * 0.08).astype(np.float32)        # 权重
    out_cols = np.linspace(5, d - 5, n_out).astype(int)          # 固定的 outlier 特征维度
    for j in out_cols:
        active = np.random.rand(tokens) < active_prob            # 该维度只在部分 token 激活
        sign = np.random.choice([-1.0, 1.0], tokens)             # 论文: outlier 通常单侧/非对称
        X[active, j] = sign[active] * np.random.uniform(*amp, active.sum())
    return X, W, out_cols


# ----------------------------------------------------------------------
# (d) 简化版 LLM.int8()（文章 §3.1-§3.3 的公式逐行对应）
# ----------------------------------------------------------------------
def llm_int8_matmul(X, W, alpha=6.0):
    """简化版 LLM.int8() 矩阵乘。

    Step 1  判定 outlier 通道集合（论文 §3.2, alpha=6.0）:
        O = { j | max_t |X[t,j]| > alpha }

    Step 2  mixed-precision 分解（论文式(2)-(4)）:
        X = X_L + X_H,  X_L 在 O 列置零, X_H 只在 O 列非零
        W = W_L + W_H,  W_L 在 O 行置零, W_H 只在 O 行非零
        由于 X_H 支撑集 = O 列、W_H 支撑集 = O 行，交叉项为零:
        XW = X_L W_L + X_H W_H   （分解本身零近似误差）

    Step 3  int8 路径: vector-wise quantization（论文 §3.1）
        X_L 按行(token)取 scale:  s_x[t] = max_i |X_L[t,i]| / 127
        W_L 按列取 scale:         s_w[o] = max_i |W_L[i,o]| / 127
        Y_int8[t,o] = s_x[t] * s_w[o] * (X_Lq @ W_Lq)[t,o]   （外积反量化）

    Step 4  fp16 路径: Y_out = X[:,O] @ W[O,:]（精确，不量化）

    Step 5  Y = Y_int8 + Y_out
    """
    col_max = np.abs(X).max(axis=0)
    O = np.where(col_max > alpha)[0]

    XL = X.copy(); XL[:, O] = 0.0
    WL = W.copy(); WL[O, :] = 0.0

    sx = np.abs(XL).max(axis=1, keepdims=True) / Q8    # (T, 1) token-wise
    sw = np.abs(WL).max(axis=0, keepdims=True) / Q8    # (1, n) column-wise
    sx = np.where(sx == 0, 1e-9, sx)
    sw = np.where(sw == 0, 1e-9, sw)

    Xq = np.clip(np.round(XL / sx), -Q8, Q8)
    Wq = np.clip(np.round(WL / sw), -Q8, Q8)
    Y_int8 = (sx * sw) * (Xq @ Wq)                     # 外积 scale 反量化

    Y_out = X[:, O].astype(np.float32) @ W[O, :].astype(np.float32)
    return Y_int8 + Y_out, O


# ----------------------------------------------------------------------
# (e) 指标与内存
# ----------------------------------------------------------------------
def rel_err(Yh, Y):
    """相对 Frobenius 误差 ||Yh - Y||_F / ||Y||_F"""
    return float(np.linalg.norm(Yh - Y) / np.linalg.norm(Y))


def weight_memory_bytes(W, O):
    """权重持久化内存（字节）: fp16 / int8-RTN / LLM.int8()。"""
    k, n = W.shape
    fp16 = 2 * k * n
    rtn8 = k * n + 4                      # int8 主体 + 1 个 per-tensor scale (fp32)
    llm = k * n + 4 * n + 2 * len(O) * n  # int8 主体 + 逐列 scale + outlier 行 fp16 副本
    return fp16, rtn8, llm


def main():
    np.random.seed(SEED)
    X, W, out_cols = build_synthetic()
    Y = X @ W
    T, d, n = X.shape[0], X.shape[1], W.shape[1]
    X_reg = X.copy(); X_reg[:, out_cols] = 0.0      # 正常部分（信息载体）
    Y_reg_true = X_reg @ W                          # 正常部分的理想输出

    print("=" * 78)
    print("合成数据（对齐 LLM.int8() 论文 §4 的实证发现）")
    print("=" * 78)
    print(f"  激活 X: ({T}, {d})   权重 W: ({d}, {n})   类型: float32")
    print(f"  outlier 通道数: {len(out_cols)} ({100*len(out_cols)/d:.2f}% of {d} 维)  位置: {out_cols.tolist()}")
    rowmax = np.abs(X).max(axis=1)
    x_reg_min, x_reg_max = float(X_reg.min()), float(X_reg.max())
    out_abs = np.abs(X[:, out_cols])
    out_absmin = float(out_abs[out_abs > 1].min())
    out_absmax = float(out_abs.max())
    mag_ratio = float(np.abs(X[:, out_cols]).max() / np.abs(X_reg).max())
    rowmax_min, rowmax_med, rowmax_max = float(rowmax.min()), float(np.median(rowmax)), float(rowmax.max())
    print(f"  正常维度范围: [{x_reg_min:.2f}, {x_reg_max:.2f}]   (论文: 正常维度约 [-3.5, 3.5])")
    print(f"  outlier 幅度: [{out_absmin:.1f}, "
          f"{out_absmax:.1f}]  "
          f"(最大约为正常维度的 {mag_ratio:.1f}x)")
    print(f"  逐 token 行 max: min {rowmax_min:.2f} / 中位数 {rowmax_med:.2f} / "
          f"max {rowmax_max:.2f}（outlier 稀疏 → token-wise 动态范围差异大）")

    print("\n" + "=" * 78)
    print("方法对比（相对 Frobenius 误差）")
    print("=" * 78)
    # 1) 权重 per-channel RTN，激活不动 —— 论文: 权重好量化
    Wq, sw = rtn_quantize_per_channel(W)
    Y_w = X @ (Wq.astype(np.float32) * sw)
    # 2) 双张量 per-tensor RTN —— 论文: absmax 在 6.7B+ 上 ppl 崩坏
    Xq, sx = rtn_quantize_per_tensor(X)
    Wq2, sw2 = rtn_quantize_per_tensor(W)
    Y_rtn = (Xq.astype(np.float32) * sx) @ (Wq2.astype(np.float32) * sw2)
    # 3) LLM.int8()
    Y_llm, O = llm_int8_matmul(X, W, alpha=6.0)

    # 常规部分的误差：把 outlier 通道的贡献从输出中剥离后，量化误差
    # 对"信息载体"（正常通道）的相对破坏程度 —— 论文 §4.3 的核心论点：
    # 全局 scale 被 outlier 拉大后，正常值被量化为 0（信息湮灭）
    err_reg_rtn = rel_err((Xq.astype(np.float32) * sx) @ (Wq2.astype(np.float32) * sw2)
                          - Y_reg_true, Y_reg_true)

    e_w = rel_err(Y_w, Y)
    e_rtn = rel_err(Y_rtn, Y)
    e_llm = rel_err(Y_llm, Y)
    rows = [
        ("FP16 基线", Y, 0.0),
        ("① 权重 per-channel RTN（激活 fp16）", Y_w, e_w),
        ("② 双张量 per-tensor RTN", Y_rtn, e_rtn),
        ("③ 简化版 LLM.int8()", Y_llm, e_llm),
    ]
    for name, _, e in rows:
        print(f"  {name:<36s} rel-err = {e:.5f}")
    print(f"  ├─ 其中 ② 对'正常部分输出'的相对误差: {err_reg_rtn:.4f}"
          f"（信息载体被破坏，论文: ppl +600~1000%）")
    print(f"  └─ 其中 ③ 的误差全部来自 int8 路径，fp16 outlier 路径精确"
          f"（|O| = {len(O)} ≤ 7，与论文一致）")

    # ② 为什么崩：量化级别被 outlier 吃掉
    print("\n" + "=" * 78)
    print("per-tensor RTN 的崩溃机制（文章 §2.4）")
    print("=" * 78)
    s_x = np.abs(X).max() / Q8
    n_levels_normal = 2 * int(3.5 / s_x)          # 正常值域 ±3.5 能占用的量化级别数
    frac_zero = float(np.mean(np.abs(X) < s_x / 2))  # 被量化到 0 的比例
    print(f"  per-tensor scale s = {float(s_x):.4f}（由 outlier {np.abs(X).max():.1f} 决定）")
    print(f"  正常值域 ±3.5 只占用约 {n_levels_normal} 个量化级别"
          f"（{np.log2(max(n_levels_normal,1)):.1f} bit 等效精度）")
    print(f"  幅度 < s/2 而被直接量化为 0 的元素占比: {frac_zero*100:.1f}%")

    # MSE 最优 scale：高斯 vs 含 outlier 的激活
    print("\n" + "=" * 78)
    print("MSE 最优 clip 点 alpha*（文章 §1.3）")
    print("=" * 78)
    gauss = np.random.randn(1_000_000).astype(np.float32)
    a_g = mse_optimal_scale(gauss)
    a_x = mse_optimal_scale(X)
    print(f"  标准正态 N(0,1):  alpha* = {a_g:.3f}  (≈ {a_g:.2f} sigma)")
    print(f"  含 outlier 的激活: alpha* = {a_x:.2f}  （被 outlier 拉大，"
          f"正常质量区间的分辨率被牺牲）")

    # 内存对比
    print("\n" + "=" * 78)
    print("权重持久化内存对比（本实验维度）")
    print("=" * 78)
    m_fp16, m_rtn, m_llm = weight_memory_bytes(W, O)
    print(f"  FP16 基线       : {m_fp16:>10,} B  ({m_fp16/m_fp16:5.2f}x)")
    print(f"  int8 RTN        : {m_rtn:>10,} B  ({m_fp16/m_rtn:5.2f}x 缩减)")
    print(f"  LLM.int8()      : {m_llm:>10,} B  ({m_fp16/m_llm:5.2f}x 缩减,"
          f" outlier fp16 副本 {2*len(O)*n:,} B)")
    print(f"  检测到的 outlier 通道: {O.tolist()}  (|O| = {len(O)})")

    # ============================================================
    # 图 A：合成激活的 outlier 形态
    # ============================================================
    col_max = np.abs(X).max(axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(np.arange(d), col_max, lw=0.9, color="#4C72B0", label="per-col max|X|")
    ax[0].scatter(out_cols, col_max[out_cols], color="#C44E52", zorder=5,
                  label=f"outlier cols ({len(out_cols)})")
    ax[0].axhline(6.0, color="gray", ls="--", lw=1, label="LLM.int8() threshold $\\alpha$=6.0")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("input feature j")
    ax[0].set_ylabel("max |activation| (log)")
    ax[0].set_title("Synthetic activation: persistent outlier columns")
    ax[0].legend(fontsize=8)
    ax[1].hist(rowmax, bins=40, color="#55A868")
    ax[1].axvline(rowmax_med, color="gray", ls="--", lw=1,
                  label=f"median={rowmax_med:.2f}")
    ax[1].axvline(rowmax_max, color="#C44E52", ls=":", lw=1.4,
                  label=f"max={rowmax_max:.2f} (outlier token)")
    ax[1].set_xlabel("per-token row max |X|")
    ax[1].set_ylabel("count of tokens")
    ax[1].set_title("Token-wise dynamic range spread (sparse outliers)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    pa = os.path.join(OUT, "synthetic_activation_outliers.png")
    fig.savefig(pa, dpi=130)
    plt.close(fig)
    print(f"\n[save] {pa}")

    # ============================================================
    # 图 B：三种方法的误差与内存对比
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["FP16\n(baseline)", "W per-ch\nRTN", "dual per-tensor\nRTN", "LLM.int8()\nmixed"]
    errs = [1e-6, e_w, e_rtn, e_llm]           # FP16 取下限便于 log 轴展示
    colors = ["#999999", "#4C72B0", "#C44E52", "#55A868"]
    bars = ax[0].bar(names, errs, color=colors)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("rel error vs FP16 (log)")
    ax[0].set_title("GEMM output relative error")
    labels_txt = ["0", f"{e_w:.4f}", f"{e_rtn:.4f}", f"{e_llm:.5f}"]
    for b, t in zip(bars, labels_txt):
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() * 1.25, t,
                   ha="center", fontsize=9)
    cfgs = ["FP16", "int8 RTN", "LLM.int8()"]
    mems = [m_fp16, m_rtn, m_llm]
    ax[1].bar(cfgs, mems, color=["#999999", "#4C72B0", "#55A868"])
    ax[1].set_ylabel("weight memory (bytes)")
    ax[1].set_title(f"Weight memory ({d}x{n} layer, |O|={len(O)})")
    for i, v in enumerate(mems):
        ax[1].text(i, v * 1.01, f"{v/1024:.0f}KB\n{v/m_fp16:.2f}x",
                   ha="center", fontsize=9)
    fig.tight_layout()
    pb = os.path.join(OUT, "method_error_memory.png")
    fig.savefig(pb, dpi=130)
    plt.close(fig)
    print(f"[save] {pb}")

    # ---- 全部关键数字落盘 results.json（与 stdout 打印一致）----
    payload = {
        "config": {"SEED": SEED, "tokens": T, "d_in": d, "d_out": n,
                   "n_outlier_channels": int(len(out_cols)),
                   "alpha_threshold": 6.0},
        "synth": {"normal_range": [round(x_reg_min, 2), round(x_reg_max, 2)],
                  "outlier_amp_range": [round(out_absmin, 1), round(out_absmax, 1)],
                  "outlier_over_normal_x": round(mag_ratio, 1),
                  "rowmax_min_med_max": [round(rowmax_min, 2),
                                         round(float(rowmax_med), 2),
                                         round(rowmax_max, 2)]},
        "rel_err_vs_fp16": {"w_perchannel_rtn_act_fp16": round(e_w, 5),
                            "dual_per_tensor_rtn": round(e_rtn, 5),
                            "llmint8_mixed": round(e_llm, 5),
                            "pertensor_err_on_normal_part": round(err_reg_rtn, 4)},
        "collapse_mechanism": {"per_tensor_scale": round(float(s_x), 4),
                               "levels_for_normal_pm3.5": int(n_levels_normal),
                               "equiv_bits_normal": round(float(np.log2(max(n_levels_normal, 1))), 1),
                               "frac_quantized_to_zero_pct": round(frac_zero * 100, 1)},
        "mse_optimal_clip_alpha": {"gauss_n01": round(float(a_g), 3),
                                   "activation_with_outliers": round(float(a_x), 2)},
        "memory_bytes": {"fp16": m_fp16, "int8_rtn": m_rtn,
                         "llmint8": m_llm,
                         "llmint8_compression_vs_fp16": round(m_fp16 / m_llm, 2)},
        "detected_outlier_cols": [int(o) for o in O.tolist()],
    }
    pj = os.path.join(OUT, "results.json")
    with open(pj, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {pj}")


if __name__ == "__main__":
    main()
