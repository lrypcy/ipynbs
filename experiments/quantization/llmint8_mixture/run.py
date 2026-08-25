#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《大模型量化算法（01）：LLM.int8()》配套实验。

纯 numpy + matplotlib，CPU 几秒跑完。复现三件事：
  (A) 激活 outlier 长尾的实证刻画（列级 max-abs 分布 + 范数占比）
  (B) 混合精度分解 Y = X_int8 W_int8 + X_out W_out 的双路径数值验证
  (C) INT8 GEMM 的显存带宽账（访存字节 + 正常值有效位宽坍塌）

输出图到本目录 results/（相对本文件，从任意 cwd 运行均可），文件名用有含义的英文：
  activation_outlier_tail.png      长尾实证
  mixed_decomposition_relmse.png   双路径数值验证（含阈值 τ 扫描）
  gemm_bandwidth_breakdown.png     带宽/有效位宽账
另将全部关键数字写入 results/results.json。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 与文章一致的固定超参 ----
SEED = 0
T, K, N = 256, 1024, 128          # token 数 / 特征(输入通道) / 输出通道
QMAX = 127                         # INT8 正半轴对称网格 2^(b-1)-1, b=8
TAU = 6.0                          # outlier 绝对幅值阈值（论文记作 alpha=6.0）


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def sym_scale_per_row(x, qmax=QMAX):
    """token-wise（逐行）对称动态 scale：每个 token 一行一个 scale。"""
    s = np.max(np.abs(x), axis=1, keepdims=True) / qmax
    return np.where(s == 0, 1.0, s)


def sym_scale_per_col(w, qmax=QMAX):
    """per-channel（逐列）对称静态 scale：权重每个输出通道一套。"""
    s = np.max(np.abs(w), axis=0, keepdims=True) / qmax
    return np.where(s == 0, 1.0, s)


def quant_int8(x, s):
    """对称 RTN 量化到 int8（返回 int8 码；s 形状需可广播）。"""
    return np.clip(np.round(x / s), -QMAX - 1, QMAX).astype(np.int8)


def rel_mse(a, b):
    return float(np.mean((a - b) ** 2) / (np.mean(b ** 2) + 1e-12))


# ----------------------------------------------------------------------
# 合成数据：复刻论文观察（约 0.4% 通道贡献约 30% 范数、幅值约 100x）
# ----------------------------------------------------------------------
def make_synth(rng):
    x = rng.normal(0.0, 1.0, (T, K)).astype(np.float64)
    n_out = 4                                   # 4 个持久 outlier 通道 (4/1024=0.39%)
    out_cols = rng.choice(K, n_out, replace=False)
    x[:, out_cols] = rng.normal(100.0, 10.0, (T, n_out)).astype(np.float64)
    w = rng.normal(0.0, 1.0, (K, N)).astype(np.float64)
    return x, w, out_cols


# ----------------------------------------------------------------------
# 核心：混合精度分解
# ----------------------------------------------------------------------
def llm_int8_matmul(x, w, tau=TAU):
    """简化版 LLM.int8() 混合精度矩阵乘。

    返回 (y, stats)：y = y_int8 + y_out（混合精度结果）；stats 含统计。
    分解本身零近似；近似只发生在 int8 路径的 RTN 上。
    """
    # 1) outlier 列：列内任一 token |x| >= tau（论文 alpha=6.0）
    col_max = np.max(np.abs(x), axis=0)                 # (K,)
    out_cols = np.where(col_max >= tau)[0]
    in_cols = np.where(col_max < tau)[0]

    # 2) 精确分解 X = X_int8 + X_out（按列切分，支撑集互斥且并为全集）
    x_out = np.zeros_like(x)
    x_out[:, out_cols] = x[:, out_cols]
    x_int8 = x - x_out

    # 3) int8 路径：激活 token-wise scale，权重 per-channel scale
    sx = sym_scale_per_row(x_int8)
    xq = quant_int8(x_int8, sx)
    sw = sym_scale_per_col(w)
    wq = quant_int8(w, sw)

    # 4) int8 GEMM：整数累加 + 外积 scale 反量化
    cq = xq.astype(np.int32) @ wq.astype(np.int32)      # (T, N) 整数累加
    y_int8 = cq.astype(np.float64) * sx * sw           # (T,1)*(1,N) 外积

    # 5) fp16 outlier 路径：仅 outlier 列
    y_out = x_out @ w

    if len(out_cols) > 0:
        mag_ratio = float(np.max(np.abs(x[:, out_cols])) / np.median(np.abs(x)))
    else:
        mag_ratio = 0.0  # 无 outlier 列时（阈值过高）退化为纯 int8 路径
    stats = {
        "n_out": len(out_cols),
        "frac_out": len(out_cols) / K,
        "frac_norm": float(np.sum(np.abs(x_out)) / np.sum(np.abs(x))),
        "mag_ratio": mag_ratio,
        "tau": tau,
    }
    return y_int8 + y_out, stats


# ----------------------------------------------------------------------
# 对照方案
# ----------------------------------------------------------------------
def naive_rtn(x, w):
    """朴素 per-tensor 激活 RTN（权重 per-channel）。"""
    sx = np.max(np.abs(x)) / QMAX
    xq = quant_int8(x, sx)
    sw = sym_scale_per_col(w)
    wq = quant_int8(w, sw)
    return (xq.astype(np.float64) @ wq.astype(np.float64)) * sx * sw


def token_rtn_no_split(x, w):
    """token-wise 动态激活量化，但不拆 outlier（scale 仍被 outlier 绑架）。"""
    sx = sym_scale_per_row(x)
    xq = quant_int8(x, sx)
    sw = sym_scale_per_col(w)
    wq = quant_int8(w, sw)
    return (xq.astype(np.float64) @ wq.astype(np.float64)) * sx * sw


# ----------------------------------------------------------------------
# 有效位宽坍塌计算
# ----------------------------------------------------------------------
def effective_bits(x, tau=TAU):
    """正常值在 int8 路径中可用的有效位宽（受 outlier 绑架时坍塌）。"""
    col_max = np.max(np.abs(x), axis=0)
    out_cols = np.where(col_max >= tau)[0]
    x_int8 = x.copy()
    x_int8[:, out_cols] = 0.0
    # 不带分解：全局 token-wise scale
    s_full = sym_scale_per_row(x)
    # 带分解：干净部分的 token-wise scale
    s_split = sym_scale_per_row(x_int8)
    # 典型正常值绝对值取整张量 3σ 处的参考 3.0（与合成 σ=1 一致）
    ref = 3.0
    lvl_full = ref / s_full                          # (T,1) 每 token 可用正电平数
    lvl_split = ref / s_split
    bits_full = np.log2(2 * lvl_full + 1)
    bits_split = np.log2(2 * lvl_split + 1)
    return float(bits_full.mean()), float(bits_split.mean())


# ----------------------------------------------------------------------
# 带宽账
# ----------------------------------------------------------------------
def bandwidth_bytes(x, w):
    """三种配置的 GEMM 输入访存字节（不含 scale 元数据）。"""
    b_fp16 = (x.size + w.size) * 2
    b_int8 = (x.size + w.size) * 1                     # 朴素 INT8
    col_max = np.max(np.abs(x), axis=0)
    out_cols = np.where(col_max >= TAU)[0]
    x_out = x[:, out_cols]
    w_out = w[out_cols, :]
    b_mix = (x.size + w.size) * 1 + (x_out.size + w_out.size) * 2
    return b_fp16, b_int8, b_mix


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(here, "results")   # 输出到本目录 results/
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x, w, out_cols = make_synth(rng)

    y_ref = x @ w
    stats = llm_int8_matmul(x, w)[1]
    y_naive = naive_rtn(x, w)
    y_token = token_rtn_no_split(x, w)
    y_mix, _ = llm_int8_matmul(x, w)

    mse_naive = rel_mse(y_naive, y_ref)
    mse_token = rel_mse(y_token, y_ref)
    mse_mix = rel_mse(y_mix, y_ref)

    bits_full, bits_split = effective_bits(x)
    b_fp16, b_int8, b_mix = bandwidth_bytes(x, w)

    # ---- 阈值 τ 扫描：验证分解对阈值在合理区间内的鲁棒性 ----
    taus = np.array([2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0])
    mse_tau = []
    nout_tau = []
    for tau in taus:
        y_t, st = llm_int8_matmul(x, w, tau=tau)
        mse_tau.append(rel_mse(y_t, y_ref))
        nout_tau.append(st["n_out"])
    mse_tau = np.array(mse_tau)

    # ============================================================
    # 打印真实数字（文章直接引用）
    # ============================================================
    print("=" * 60)
    print("合成数据统计（复刻 LLM.int8() 论文观察）")
    print("=" * 60)
    print(f"激活形状           : X = {T} x {K},  W = {K} x {N}")
    print(f"outlier 列数/占比  : {stats['n_out']} / {K} = {stats['frac_out']*100:.3f}%")
    print(f"outlier 范数占比   : {stats['frac_norm']*100:.1f}%  (论文观察约 30%)")
    print(f"outlier 幅值倍数   : {stats['mag_ratio']:.1f}x  (论文报告 >=20x)")
    print(f"阈值 tau           : {TAU}  (论文记作 alpha=6.0)")

    print("\n" + "=" * 60)
    print("GEMM 输出相对 MSE（参考 = FP16/fp32 全精度）")
    print("=" * 60)
    print(f"{'FP16 基线'                    :<28}{0.0:>12.6f}")
    print(f"{'朴素 RTN (per-tensor 激活)'   :<28}{mse_naive:>12.6f}")
    print(f"{'token-wise (不拆 outlier)'    :<28}{mse_token:>12.6f}")
    print(f"{'LLM.int8() 混合精度'          :<28}{mse_mix:>12.6f}")

    print("\n" + "=" * 60)
    print("有效位宽坍塌（正常值可用的 int8 电平数）")
    print("=" * 60)
    print(f"不拆 outlier (scale 被绑架) : {bits_full:.2f} bit")
    print(f"混合分解 (clean scale)      : {bits_split:.2f} bit")
    print(f"恢复倍数                     : {bits_split/bits_full:.2f}x")

    print("\n" + "=" * 60)
    print("显存带宽账（GEMM 输入访存字节，不含 scale 元数据）")
    print("=" * 60)
    print(f"{'FP16 全精度'            :<28}{b_fp16:>12}  (1.00x)")
    print(f"{'朴素 INT8'              :<28}{b_int8:>12}  ({b_int8/b_fp16:.3f}x)")
    print(f"{'LLM.int8() 混合(GEMM层)':<28}{b_mix:>12}  ({b_mix/b_fp16:.3f}x)")
    print("注: 真实部署需保留完整 fp16 权重副本供 outlier/反量化，")
    print("    实际显存约 2.0x FP16（见文章 §5 批判）。")

    print("\n" + "=" * 60)
    print("阈值 tau 扫描（鲁棒性 / 脆弱性）")
    print("=" * 60)
    for tau, m, n in zip(taus, mse_tau, nout_tau):
        print(f"  tau={tau:>6.1f}  n_out={n:>2}  relMSE={m:.6f}")

    # ============================================================
    # 图 A：激活 outlier 长尾实证
    # ============================================================
    col_max = np.max(np.abs(x), axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(col_max, bins=60, color="#4C72B0")
    ax[0].axvline(TAU, color="#C44E52", linestyle="--", label=f"tau={TAU}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("per-column max |activation| (log scale)")
    ax[0].set_ylabel("count of features")
    ax[0].set_title("Activation outlier long tail")
    ax[0].legend()
    # 右图：按范数占比排序，展示 0.1% 通道占 ~30% 范数
    col_norm = np.sum(np.abs(x), axis=0)
    order = np.argsort(col_norm)[::-1]
    cum = np.cumsum(col_norm[order]) / np.sum(col_norm)
    frac_cols = (np.arange(K) + 1) / K
    ax[1].plot(frac_cols * 100, cum * 100, color="#55A868")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("top features (% of columns, log)")
    ax[1].set_ylabel("cumulative norm (%)")
    ax[1].set_title("Norm concentration (top cols)")
    ax[1].axhline(stats["frac_norm"] * 100, color="#C44E52", linestyle=":",
                  label=f"top {stats['frac_out']*100:.2f}% cols = {stats['frac_norm']*100:.1f}% norm")
    ax[1].legend()
    fig.tight_layout()
    pa = os.path.join(img_dir, "activation_outlier_tail.png")
    fig.savefig(pa, dpi=130)
    print(f"\n[save] {pa}")

    # ============================================================
    # 图 B：双路径数值验证 + tau 扫描
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    methods = ["FP16", "naive RTN", "token-wise\n(no split)", "LLM.int8\nmixed"]
    vals = [0.0, mse_naive, mse_token, mse_mix]
    colors = ["#999999", "#C44E52", "#DD8452", "#55A868"]
    ax[0].bar(methods, vals, color=colors)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("relative MSE vs FP16")
    ax[0].set_title("Mixed-precision decomposition error")
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.3, f"{v:.2e}", ha="center", fontsize=8)
    ax[1].plot(taus, mse_tau, "o-", color="#4C72B0")
    ax[1].axvline(TAU, color="#C44E52", linestyle="--", label=f"tau={TAU}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("threshold tau (abs magnitude)")
    ax[1].set_ylabel("mixed relMSE")
    ax[1].set_title("Robustness vs threshold tau")
    ax[1].legend()
    fig.tight_layout()
    pb = os.path.join(img_dir, "mixed_decomposition_relmse.png")
    fig.savefig(pb, dpi=130)
    print(f"[save] {pb}")

    # ============================================================
    # 图 C：带宽账 + 有效位宽
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cfgs = ["FP16", "INT8 naive", "LLM.int8 mixed"]
    bytes_cfg = [b_fp16, b_int8, b_mix]
    ax[0].bar(cfgs, bytes_cfg, color=["#999999", "#4C72B0", "#55A868"])
    ax[0].set_ylabel("GEMM input bytes")
    ax[0].set_title("Memory traffic (GEMM inputs)")
    for i, v in enumerate(bytes_cfg):
        ax[0].text(i, v * 1.01, f"{v/1024:.0f}KB\n{v/b_fp16:.2f}x", ha="center", fontsize=8)
    ax[1].bar(["no split", "mixed split"], [bits_full, bits_split],
              color=["#C44E52", "#55A868"])
    ax[1].set_ylabel("effective bits for normal activations")
    ax[1].set_title("Effective bit-width of normal values")
    for i, v in enumerate([bits_full, bits_split]):
        ax[1].text(i, v + 0.1, f"{v:.2f} bit", ha="center", fontsize=9)
    fig.tight_layout()
    pc = os.path.join(img_dir, "gemm_bandwidth_breakdown.png")
    fig.savefig(pc, dpi=130)
    print(f"[save] {pc}")

    # ---- 全部关键数字落盘 results.json（与 stdout 打印一致）----
    payload = {
        "config": {"SEED": SEED, "T": T, "K": K, "N": N, "QMAX": QMAX, "TAU": TAU},
        "synth": {"n_outlier_cols": int(stats["n_out"]),
                  "frac_outlier_cols_pct": round(stats["frac_out"] * 100, 3),
                  "frac_norm_pct": round(stats["frac_norm"] * 100, 1),
                  "mag_ratio_x": round(stats["mag_ratio"], 1)},
        "rel_mse_vs_fp16": {"naive_rtn_per_tensor": mse_naive,
                            "tokenwise_no_split": mse_token,
                            "llmint8_mixed": mse_mix,
                            "improvement_naive_over_mixed_x": mse_naive / mse_mix},
        "effective_bits_normal": {"no_split": round(bits_full, 2),
                                  "mixed_split": round(bits_split, 2),
                                  "recovery_x": round(bits_split / bits_full, 2)},
        "gemm_input_bytes": {"fp16": b_fp16, "int8_naive": b_int8,
                             "llmint8_mixed": b_mix},
        "tau_sweep": {"taus": taus.tolist(),
                      "n_out": [int(n) for n in nout_tau],
                      "rel_mse": [float(m) for m in mse_tau]},
    }
    pj = os.path.join(img_dir, "results.json")
    with open(pj, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {pj}")


if __name__ == "__main__":
    main()
