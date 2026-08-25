#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- PTQ 系列《10: Outlier Suppression / OS+》配套实验。

移植自博客仓库 experiments_os_plus.py，算法逻辑与超参不变
（SEED=0 与源脚本一致）。纯 numpy + matplotlib，CPU 秒级跑完。
复现两个 Demo：
  A. γ 是离群值放大器 + Gamma Migration 等效变换（LN 参数迁移）
  B. OS+ 的 channel-wise shift + scale 对比 SmoothQuant 型 scale-only

输出图到本目录 results/（相对本文件，从任意 cwd 运行均可）：
  gamma_migration.png            Demo A：γ 放大效应与迁移前后误差对比
  shift_vs_scale_w8a8.png        Demo B：shift+scale vs 仅 scale
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

SEED = 0


# ---------- 公共构件 ----------
def per_tensor_sym_quant(A, bits=8):
    """对称 per-tensor 量化（激活量化的最坏基线）"""
    s = np.abs(A).max() / (2 ** (bits - 1) - 1)
    return np.round(A / s).clip(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s


def rel_err(Y, Y_ref):
    return float(np.linalg.norm(Y - Y_ref) / np.linalg.norm(Y_ref))


def make_block(rng, d=512, n_tokens=4096, n_outlier=3):
    """合成 LN→Linear 片段：归一化后隐状态 xhat、LN 参数 γ/β、下游权重 W"""
    xhat = rng.normal(0, 1.0, (n_tokens, d))
    oc = rng.choice(d, n_outlier, replace=False)
    xhat[:, oc] *= 6                      # 归一化后的常驻离群通道（真实 LLM 中存在）
    gamma = np.ones(d)
    gamma[oc] *= 5                        # 训练后的 γ 进一步放大这些通道
    beta = rng.normal(0, 0.1, d)
    W = rng.normal(0, 0.02, (d, d))       # 下游线性层
    return xhat, gamma, beta, W


# ---------- Demo A: γ 放大效应 与 Gamma Migration ----------
def demo_a(rng):
    xhat, gamma, beta, W = make_block(rng)
    a_ln = gamma * xhat + beta              # LN 输出（fp16 路径的真实值）
    Y_ref = a_ln @ W.T

    res = {"absmax_xhat": float(np.abs(xhat).max()),
           "absmax_ln_out": float(np.abs(a_ln).max())}
    print("[DemoA] γ 迁移前后的动态范围与量化误差")
    print(f"  xhat        per-tensor absmax = {res['absmax_xhat']:7.2f}")
    print(f"  LN 输出     per-tensor absmax = {res['absmax_ln_out']:7.2f}   "
          f"（γ 放大 {(res['absmax_ln_out']/res['absmax_xhat']):.1f}x）")

    # 路径 1：直接量化 LN 输出
    Y_q_direct = per_tensor_sym_quant(a_ln) @ W.T
    # 路径 2：Gamma Migration——γ 折进权重，β 折进 bias，量化对象变成 xhat
    W_mig = W * gamma[None, :]
    b_mig = beta @ W.T                       # 常数项，fp16 计算，不参与激活量化
    Y_q_mig = per_tensor_sym_quant(xhat) @ W_mig.T + b_mig

    e_direct = rel_err(Y_q_direct, Y_ref)
    e_mig = rel_err(Y_q_mig, Y_ref)
    res["e_direct"] = e_direct
    res["e_mig"] = e_mig
    print(f"  直接量化 LN 输出   : 输出相对误差 = {e_direct:.4f}")
    print(f"  Gamma Migration 后 : 输出相对误差 = {e_mig:.4f}   （降低 {(1-e_mig/e_direct)*100:.1f}%）")

    # ---- 图：γ 放大效应 + 迁移前后误差 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = ax[0].bar(["$\\hat{x}$ (pre-LN)", "LN output $\\gamma\\hat{x}+\\beta$"],
                     [res["absmax_xhat"], res["absmax_ln_out"]],
                     color=["#4C72B0", "#C44E52"])
    for b, v in zip(bars, [res["absmax_xhat"], res["absmax_ln_out"]]):
        ax[0].text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.1f}",
                   ha="center", fontsize=9)
    ax[0].set_ylabel("per-tensor absmax")
    ax[0].set_title("DemoA: $\\gamma$ amplifies emergent outliers")
    ax[0].grid(alpha=0.3, axis="y")
    bars = ax[1].bar(["quantize LN output\ndirectly", "Gamma Migration\n($\\gamma\\to W$, $\\beta\\to b$)"],
                     [e_direct, e_mig], color=["#C44E52", "#55A868"])
    for b, v in zip(bars, [e_direct, e_mig]):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.4f}",
                   ha="center", fontsize=9)
    ax[1].set_ylabel("output rel error (W8A8 per-tensor)")
    ax[1].set_title("Gamma Migration reduces quantization error")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    pa = os.path.join(OUT, "gamma_migration.png")
    fig.savefig(pa, dpi=130)
    plt.close(fig)
    print(f"[save] {pa}")
    return res


# ---------- Demo B: OS+ 的 shift+scale vs 仅 scale（SmoothQuant 型） ----------
def make_activation(rng, n_tokens=4096, d=512, n_outlier=16):
    """带不对称离群通道的激活：离群值有大的正均值(直流分量)+小幅波动,
       形状跨 token 稳定——OS+ 论文观察到的'通道间不对称'形态"""
    X = rng.normal(0, 1.0, (n_tokens, d))
    oc = rng.choice(d, n_outlier, replace=False)
    mus = rng.uniform(20.0, 50.0, n_outlier)      # 每个离群通道自己的偏移量
    sigmas = rng.uniform(0.4, 1.0, n_outlier)     # 波动反而很小
    for k, j in enumerate(oc):
        X[:, j] = mus[k] + sigmas[k] * rng.normal(0, 1.0, n_tokens)
    return X, oc, mus, sigmas


def demo_b(rng):
    X, oc, mus, sigmas = make_activation(rng)
    W2 = rng.normal(0, 0.02, (256, X.shape[1]))
    Y_ref2 = X @ W2.T

    mu = X.mean(0)                            # 校准集通道均值 → shift 向量
    s_smooth = np.abs(X).max(0)               # SmoothQuant 口径的逐通道 scale
    s_os = np.abs(X - mu).max(0)              # OS+ 口径：先去均值再取逐通道 absmax

    X_smooth = X / s_smooth                   # 等效变换 1：只缩放
    X_os_raw = (X - mu) / s_os                # 等效变换 2：先移再缩

    res = {"absmax_raw": float(np.abs(X).max()),
           "absmax_shifted": float(np.abs(X - mu).max())}
    print("\n[DemoB] 激活侧 W8A8（per-tensor int8）三种处理对比")
    print(f"  原始激活       per-tensor absmax = {res['absmax_raw']:8.2f}")
    print(f"  shift 后       per-tensor absmax = {res['absmax_shifted']:8.2f}"
          f"   （去均值使动态范围缩小 {res['absmax_raw']/res['absmax_shifted']:.1f}x）")

    # 变体 1：直接量化
    Y_v1 = per_tensor_sym_quant(X) @ W2.T
    # 变体 2：SmoothQuant 型仅 scale：quantize(X/s) @ (diag(s)W)^T
    Y_v2 = per_tensor_sym_quant(X_smooth) @ (W2 * s_smooth[None, :]).T
    # 变体 3：OS+ shift+scale：quantize((X-mu)/s) @ (diag(s)W)^T + mu@W^T（常数项走 fp16）
    const = mu @ W2.T
    Y_v3 = per_tensor_sym_quant(X_os_raw) @ (W2 * s_os[None, :]).T + const

    errs = {}
    for name, Y in [("direct", Y_v1), ("scale_only", Y_v2), ("shift_scale", Y_v3)]:
        errs[name] = rel_err(Y, Y_ref2)
    names_cn = [("直接量化           ", "direct"),
                ("仅 scale(Smooth型) ", "scale_only"),
                ("shift+scale(OS+)   ", "shift_scale")]
    for name_cn, key in names_cn:
        print(f"  {name_cn}: 输出相对误差 = {errs[key]:.4f}")

    # 附：离群通道波动项在三种方案下的有效级数（σ ÷ 该通道的原始单位网格步长）
    j = oc[np.argmax(mus)]
    k = list(oc).index(j)
    step_direct = np.abs(X).max() / 127        # 直接量化：全局 absmax 定步长
    step_smooth = s_smooth[j] / 127            # 仅 scale：该通道自己的 absmax 决定其步长
    step_os     = s_os[j] / 127                # shift+scale：去均值后 scale 大幅缩小
    levels = (sigmas[k] / step_direct, sigmas[k] / step_smooth, sigmas[k] / step_os)
    print(f"\n  离群通道 {j} (mu={mu[j]:.1f}, 波动σ≈{sigmas[k]:.2f}) 的波动项拿到的 int8 有效级数:")
    print(f"    直接量化    : {levels[0]:6.1f} 级   (该通道步长 {step_direct:.4f})")
    print(f"    仅 scale    : {levels[1]:6.1f} 级   (步长几乎不变! scale 救的是其他通道)")
    print(f"    shift+scale : {levels[2]:6.1f} 级   (去均值后步长缩小 {step_smooth/step_os:.0f}x)")

    # ---- 图：三种方案的误差 + 离群通道波动项有效级数 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["direct\nquantize", "scale only\n(SmoothQuant-type)", "shift+scale\n(OS+)"]
    vals = [errs["direct"], errs["scale_only"], errs["shift_scale"]]
    bars = ax[0].bar(labels, vals, color=["#999999", "#DD8452", "#55A868"])
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.4f}",
                   ha="center", fontsize=9)
    ax[0].set_ylabel("output rel error (per-tensor int8)")
    ax[0].set_title("DemoB: asymmetric outliers need shift, not just scale")
    ax[0].grid(alpha=0.3, axis="y")
    lv = [float(x) for x in levels]
    bars = ax[1].bar(["direct", "scale only", "shift+scale"], lv,
                     color=["#999999", "#DD8452", "#55A868"])
    for b, v in zip(bars, lv):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.1f}",
                   ha="center", fontsize=9)
    ax[1].set_ylabel(f"effective int8 levels for fluctuation of ch.{j}")
    ax[1].set_title("Outlier-channel fluctuation: usable code levels")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    pb = os.path.join(OUT, "shift_vs_scale_w8a8.png")
    fig.savefig(pb, dpi=130)
    plt.close(fig)
    print(f"[save] {pb}")

    res.update({"rel_err": errs,
                "outlier_ch": int(j),
                "outlier_mu": float(mu[j]),
                "outlier_sigma": float(sigmas[k]),
                "effective_levels": {"direct": lv[0], "scale_only": lv[1],
                                     "shift_scale": lv[2]},
                "step_shrink_x": float(step_smooth / step_os)})
    return res


def main():
    rng = np.random.default_rng(SEED)   # 与源脚本一致：default_rng(0)，调用顺序不变
    ra = demo_a(rng)
    rb = demo_b(rng)

    payload = {"config": {"SEED": SEED},
               "demo_a_gamma_migration": {
                   "absmax_xhat": round(ra["absmax_xhat"], 2),
                   "absmax_ln_output": round(ra["absmax_ln_out"], 2),
                   "gamma_amplify_x": round(ra["absmax_ln_out"] / ra["absmax_xhat"], 1),
                   "rel_err_direct": round(ra["e_direct"], 4),
                   "rel_err_gamma_migration": round(ra["e_mig"], 4),
                   "reduction_pct": round((1 - ra["e_mig"] / ra["e_direct"]) * 100, 1)},
               "demo_b_shift_vs_scale": {
                   "absmax_raw": round(rb["absmax_raw"], 2),
                   "absmax_after_shift": round(rb["absmax_shifted"], 2),
                   "range_shrink_x": round(rb["absmax_raw"] / rb["absmax_shifted"], 1),
                   "rel_err": {k: round(v, 4) for k, v in rb["rel_err"].items()},
                   "outlier_channel": rb["outlier_ch"],
                   "outlier_mu_sigma": [round(rb["outlier_mu"], 1),
                                        round(rb["outlier_sigma"], 2)],
                   "effective_levels_fluctuation":
                       {k: round(v, 1) for k, v in rb["effective_levels"].items()},
                   "step_shrink_x_by_shift": round(rb["step_shrink_x"])},
               }
    pj = os.path.join(OUT, "results.json")
    with open(pj, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {pj}")


if __name__ == "__main__":
    main()
