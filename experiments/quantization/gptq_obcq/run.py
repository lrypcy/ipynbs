#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《LLM PTQ 深度解析（02）：GPTQ 与 OBQC》配套实验。

纯 numpy + matplotlib(Agg)，CPU 数秒跑完。
对应文章：https://lrypcy.github.io/2026/08/24/ptq-02-gptq/

复现三件事（对应文章 §2 OBQ 推导、§3.3 Cholesky、§3.4 dampening、§6 数值对比）：
  (A) OBQC 核心循环：手写 Hessian H = 2XX^T 与其逆，逐列量化 -> 误差按
      w[j>q] += err / [H^-1]_qq * [H^-1]_{j,q} 反传到未量化列（截断补偿更新）；
      在 bits ∈ {4,3,2} 上对比 RTN vs GPTQ 的输出相对 MSE 与 SNR，
      并检查校准集/留出集误差是否持平（文章 §6.3 观察一）。
  (B) group size 扫描：{32,64,128,256}，验证"粒度越细误差越低、GPTQ 增益稳定存在"。
  (C) dampening 对 Cholesky 数值稳定性的影响：构造低样本校准集（n_cal < d_col），
      使 H = 2XX^T 秩亏 -> damp=0 时 Cholesky 直接失败；扫描 damp_frac 展示
      合适的 lambda 既救活分解又能改善精度（文章 §3.4"保险"叙述）。

输出图到本目录 results/（相对本文件，从任意 cwd 运行均可）：
  gptq_vs_rtn.png         RTN vs GPTQ 位宽对比 + group size 扫描
  dampening_cholesky.png  dampening 对 H 条件数 / Cholesky 可解性 / 最终误差的影响
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
D_ROW, D_COL = 64, 256        # 权重矩阵形状 (d_out, d_in)，同文章 §6
N_CAL = 4096                  # 校准样本数（文章口径 4096）
N_EVAL = 2048                 # 留出评估样本数
GROUP_SIZE = 128              # 主实验分组大小（文章 §4.1 的甜蜜点）
BITS_LIST = (4, 3, 2)
DAMP_FRAC = 0.01              # 默认 dampening：对角均值的 1%（文章 §6 代码同款）

N_OUT_COLS = 5                # 种植的权重 outlier 列数（模拟 LLM 重尾结构）


# ----------------------------------------------------------------------
# 合成数据：病态协方差激活 + 带 outlier 列的权重（复刻文章 §6 数据构造）
# ----------------------------------------------------------------------
def make_data(rng):
    """校准/留出激活共享同一病态协方差（特征值跨 3 个数量级），权重含 outlier 列。"""
    u, _ = np.linalg.qr(rng.normal(size=(D_COL, D_COL)))
    lam = np.geomspace(100.0, 0.1, D_COL)          # 条件数 ~1e3，GPTQ 优势显现的前提
    cov = (u * lam) @ u.T
    X_cal = rng.multivariate_normal(np.zeros(D_COL), cov, size=N_CAL).T   # (d_col, n)
    X_eval = rng.multivariate_normal(np.zeros(D_COL), cov, size=N_EVAL).T
    W = rng.normal(size=(D_ROW, D_COL)) * 0.02
    out_cols = rng.choice(D_COL, N_OUT_COLS, replace=False)
    W[:, out_cols] *= 15.0                          # 少数权重 outlier 列
    return W, X_cal, X_eval


# ----------------------------------------------------------------------
# 基线：分组对称 RTN（每行每组一个 scale，整数码范围 [-qmax, qmax]）
# ----------------------------------------------------------------------
def rtn_sym_group(W, bits, group_size):
    """分组对称 RTN：scale = max|w_group|/qmax，round-to-nearest 后反量化。"""
    qmax = 2 ** (bits - 1) - 1
    Wq = np.empty_like(W)
    d_in = W.shape[1]
    for s in range(0, d_in, group_size):
        sl = slice(s, min(s + group_size, d_in))
        sc = np.abs(W[:, sl]).max(axis=1, keepdims=True) / qmax     # 用原始组算 scale
        sc = np.maximum(sc, 1e-12)
        Wq[:, sl] = np.clip(np.round(W[:, sl] / sc), -qmax, qmax) * sc
    return Wq


def group_scales_row(w, group_size, qmax):
    """单行权重的逐组 scale（固定用原始 W 计算，不随补偿漂移——文章 §6 同款简化）。"""
    d_in = w.shape[0]
    sc = np.empty(d_in)
    for s in range(0, d_in, group_size):
        sl = slice(s, min(s + group_size, d_in))
        sc[sl] = max(np.abs(w[sl]).max() / qmax, 1e-12)
    return sc


# ----------------------------------------------------------------------
# 核心：OBQC / GPTQ 逐列量化 + 截断补偿更新（Cholesky 版，文章 §3.3 + §6）
# ----------------------------------------------------------------------
def gptq_layer(X, W, bits, group_size, damp_frac=DAMP_FRAC):
    """
    对单个 Linear 层做 GPTQ 量化。
    X: 校准激活 (d_col x n)；W: 权重 (d_row x d_col)。
    返回 (反量化矩阵 Wq, info dict)。
    数学骨架（arXiv:2210.17323 Algorithm 1 的固定顺序形式）：
      H = 2XX^T -> dampening -> Cholesky 得 H^-1 -> 逐列量化并把误差
      按 [H^-1] 的列比例反传给所有未量化列。
    """
    d_row, d_col = W.shape
    qmax = 2 ** (bits - 1) - 1

    # 1) 手写 Hessian + dampening（Alg.1 第 1-2 行）
    H = 2.0 * (X @ X.T)                              # H = 2XX^T, (d_col, d_col)
    lam = damp_frac * np.mean(np.diag(H))
    Hd = H + lam * np.eye(d_col)

    # 2) Cholesky 求 H^-1：Hd = L L^T => Hd^-1 = L^-T L^-1
    try:
        L = np.linalg.cholesky(Hd)
    except np.linalg.LinAlgError:
        return None, {"cholesky_ok": False}
    Li = np.linalg.inv(L)
    M = Li.T @ Li                                    # 即 Hd^{-1}
    info = {"cholesky_ok": True,
            "cond_H": float(np.linalg.cond(H)),
            "cond_H_damped": float(np.linalg.cond(Hd))}

    # 3) 逐行逐列：quantize -> 误差反传到未量化列（Alg.1 第 5 行的截断补偿更新）
    scales = np.empty_like(W)
    for i in range(d_row):
        scales[i] = group_scales_row(W[i], group_size, qmax)   # scale 固定自原始 W

    Wq = W.copy()
    for i in range(d_row):
        w = Wq[i]
        sc = scales[i]
        for q in range(d_col):
            w_hat_q = np.clip(np.round(w[q] / sc[q]), -qmax, qmax) * sc[q]
            err = w_hat_q - w[q]                     # c = ŵ_q - w_q
            w[q] = w_hat_q                           # 先落格点
            if q + 1 < d_col:
                # 截断补偿更新：w[j>q] += err / [H^-1]_qq * [H^-1]_{j,q}
                w[q + 1:] += (err / M[q, q]) * M[q + 1:, q]
    return Wq, info


# ----------------------------------------------------------------------
# 误差度量：输出相对 MSE 与 SNR
# ----------------------------------------------------------------------
def rel_mse(W, Wq, X):
    """||WX - WqX||_F^2 / ||WX||_F^2（layer-wise 目标函数本身）。"""
    num = np.linalg.norm((W - Wq) @ X) ** 2
    den = np.linalg.norm(W @ X) ** 2
    return float(num / den)


def snr_db(rel):
    """以 dB 计的输出信噪比 = -10*log10(rel_mse)。"""
    return -10.0 * np.log10(max(rel, 1e-300))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(here, "results")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    W, X_cal, X_eval = make_data(rng)

    # ============================================================
    # (A) 主对比：RTN vs GPTQ，bits ∈ {4,3,2}
    # ============================================================
    table_a = []                                     # (bits, method, mse_cal, mse_eval, snr)
    for bits in BITS_LIST:
        Wr = rtn_sym_group(W, bits, GROUP_SIZE)
        Wg, _ = gptq_layer(X_cal, W, bits, GROUP_SIZE)
        for name, Wq in (("RTN", Wr), ("GPTQ", Wg)):
            mc, me = rel_mse(W, Wq, X_cal), rel_mse(W, Wq, X_eval)
            table_a.append({"bits": bits, "method": name,
                            "mse_cal": mc, "mse_eval": me, "snr_db": snr_db(me)})
    gain = {}
    for bits in BITS_LIST:
        r = next(t for t in table_a if t["bits"] == bits and t["method"] == "RTN")
        g = next(t for t in table_a if t["bits"] == bits and t["method"] == "GPTQ")
        gain[bits] = r["mse_eval"] / g["mse_eval"]

    # ============================================================
    # (B) group size 扫描（bits=3）
    # ============================================================
    gs_list = [32, 64, 128, 256]
    sweep_b = {"gs": gs_list, "rtn": [], "gptq": []}
    for gs in gs_list:
        Wr = rtn_sym_group(W, 3, gs)
        Wg, _ = gptq_layer(X_cal, W, 3, gs)
        sweep_b["rtn"].append(rel_mse(W, Wr, X_eval))
        sweep_b["gptq"].append(rel_mse(W, Wg, X_eval))

    # ============================================================
    # (C) dampening vs Cholesky 稳定性：低样本校准集使 H 秩亏
    # ============================================================
    n_small = 48                                     # << d_col=256 => H 秩 <= 48
    X_small = X_cal[:, :n_small]
    fracs = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    sweep_c = {"frac": fracs, "ok": [], "cond": [], "mse_eval": [], "mse_rtn_ref": None}
    Wr_ref = rtn_sym_group(W, 3, GROUP_SIZE)
    sweep_c["mse_rtn_ref"] = rel_mse(W, Wr_ref, X_eval)
    for f in fracs:
        Wq, info = gptq_layer(X_small, W, 3, GROUP_SIZE, damp_frac=f)
        if not info["cholesky_ok"]:
            sweep_c["ok"].append(False)              # damp=0: Cholesky 直接失败
            sweep_c["cond"].append(None)
            sweep_c["mse_eval"].append(None)
            continue
        sweep_c["ok"].append(True)
        sweep_c["cond"].append(info["cond_H_damped"])
        sweep_c["mse_eval"].append(rel_mse(W, Wq, X_eval))

    # ============================================================
    # 打印真实数字
    # ============================================================
    print("=" * 66)
    print("合成数据统计")
    print("=" * 66)
    ev = np.linalg.eigvalsh(2.0 * (X_cal @ X_cal.T))
    print(f"层形状                : W = {D_ROW} x {D_COL}, group={GROUP_SIZE}")
    print(f"Hessian 条件数(全样本): {ev[-1]/ev[0]:.2e}")
    print(f"outlier 权重列        : {N_OUT_COLS}/{D_COL} 列 x15 幅值")

    print("\n" + "=" * 66)
    print("[主对比 A] RTN vs GPTQ 输出相对 MSE / SNR（group=128）")
    print("=" * 66)
    print(f"{'bits':>4} {'method':>6} {'MSE@cal':>11} {'MSE@eval':>11} {'SNR@eval(dB)':>13}")
    for t in table_a:
        print(f"{t['bits']:>4} {t['method']:>6} {t['mse_cal']:>11.3e} "
              f"{t['mse_eval']:>11.3e} {t['snr_db']:>13.2f}")
    for bits in BITS_LIST:
        print(f"bits={bits}: GPTQ 相对 RTN 的 eval MSE 降低 "
              f"{(1 - 1/gain[bits])*100:.1f}%  ({gain[bits]:.2f}x)")

    print("\n" + "=" * 66)
    print("[扫描 B] group size（bits=3, eval 相对 MSE）")
    print("=" * 66)
    for i, gs in enumerate(gs_list):
        print(f"group={gs:>4}: RTN={sweep_b['rtn'][i]:.3e}  GPTQ={sweep_b['gptq'][i]:.3e}")

    print("\n" + "=" * 66)
    print(f"[扫描 C] dampening vs Cholesky 稳定性（bits=3, 低样本 n_cal={n_small}<{D_COL} => H 秩亏）")
    print("=" * 66)
    for i, f in enumerate(fracs):
        if not sweep_c["ok"][i]:
            print(f"damp_frac={f:>7.0e}: Cholesky 分解失败（H 非正定），GPTQ 无法运行")
        else:
            print(f"damp_frac={f:>7.0e}: cond(H+lamI)={sweep_c['cond'][i]:.2e}  "
                  f"GPTQ eval MSE={sweep_c['mse_eval'][i]:.3e}"
                  f"  (RTN 参照 {sweep_c['mse_rtn_ref']:.3e})")

    # ============================================================
    # 图 A：RTN vs GPTQ（位宽对比 + group size 扫描）
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(BITS_LIST))
    wd = 0.36
    rtn_vals = [next(t["mse_eval"] for t in table_a if t["bits"] == b and t["method"] == "RTN")
                for b in BITS_LIST]
    gptq_vals = [next(t["mse_eval"] for t in table_a if t["bits"] == b and t["method"] == "GPTQ")
                 for b in BITS_LIST]
    ax[0].bar(x - wd/2, rtn_vals, wd, label="RTN", color="#C44E52")
    ax[0].bar(x + wd/2, gptq_vals, wd, label="GPTQ (OBQC)", color="#4C72B0")
    for xi, (vr, vg) in enumerate(zip(rtn_vals, gptq_vals)):
        ax[0].text(xi - wd/2, vr * 1.03, f"{vr:.2e}", ha="center", fontsize=8)
        ax[0].text(xi + wd/2, vg * 1.03, f"{vg:.2e}", ha="center", fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, [f"INT{b}" for b in BITS_LIST])
    ax[0].set_ylabel("output rel. MSE (held-out)")
    ax[0].set_title("RTN vs GPTQ across bit-widths")
    ax[0].legend(fontsize=9)

    ax[1].plot(gs_list, sweep_b["rtn"], "o-", color="#C44E52", label="RTN")
    ax[1].plot(gs_list, sweep_b["gptq"], "s-", color="#4C72B0", label="GPTQ")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(gs_list, [str(g) for g in gs_list])
    ax[1].set_xlabel("group size (input channels)")
    ax[1].set_ylabel("output rel. MSE (held-out)")
    ax[1].set_title("Group-size sweep @ INT3")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    pa = os.path.join(img_dir, "gptq_vs_rtn.png")
    fig.savefig(pa, dpi=130)
    print(f"\n[save] {pa}")

    # ============================================================
    # 图 B：dampening 对数值稳定性的影响
    # ============================================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    xs = [f for f, ok in zip(fracs, sweep_c["ok"]) if ok]
    ys = [c for c, ok in zip(sweep_c["cond"], sweep_c["ok"]) if ok]
    ax[0].plot(xs, ys, "o-", color="#55A868")
    if len(xs) < len(fracs):                         # damp=0 失败标注在轴左端
        ax[0].annotate("damp=0:\nCholesky fails\n(H rank-deficient)",
                       xy=(xs[0], ys[0]), xytext=(xs[0]*3, ys[0]*0.4),
                       fontsize=9, color="#C44E52",
                       arrowprops=dict(arrowstyle="->", color="#C44E52"))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("damp_frac $\\lambda$ / mean(diag H)")
    ax[0].set_ylabel("cond($H+\\lambda I$)")
    ax[0].set_title(f"Damping tames the Hessian (n_cal={n_small} < d_col={D_COL})")

    ms = [m for m, ok in zip(sweep_c["mse_eval"], sweep_c["ok"]) if ok]
    ax[1].plot(xs, ms, "o-", color="#4C72B0", label="GPTQ (low-sample calib)")
    ax[1].axhline(sweep_c["mse_rtn_ref"], color="#999999", linestyle="--",
                  label=f"RTN ref (full calib) = {sweep_c['mse_rtn_ref']:.2e}")
    best_i = int(np.argmin(ms))
    ax[1].annotate(f"best damp={xs[best_i]:.0e}\nMSE={ms[best_i]:.2e}",
                   xy=(xs[best_i], ms[best_i]),
                   xytext=(xs[best_i]*2, ms[best_i]*1.6), fontsize=9,
                   arrowprops=dict(arrowstyle="->"))
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("damp_frac $\\lambda$ / mean(diag H)")
    ax[1].set_ylabel("output rel. MSE (held-out)")
    ax[1].set_title("Suitable damping stabilizes AND helps accuracy")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    pb = os.path.join(img_dir, "dampening_cholesky.png")
    fig.savefig(pb, dpi=130)
    print(f"[save] {pb}")

    # ---- 全部关键数字落盘 ----
    payload = {
        "config": {"SEED": SEED, "D_ROW": D_ROW, "D_COL": D_COL,
                   "N_CAL": N_CAL, "N_EVAL": N_EVAL,
                   "GROUP_SIZE": GROUP_SIZE, "DAMP_FRAC": DAMP_FRAC},
        "main_bits_sweep": table_a,
        "gptq_gain_x_vs_rtn": {str(b): gain[b] for b in BITS_LIST},
        "group_size_sweep_INT3": sweep_b,
        "dampening_low_sample": sweep_c,
    }
    pj = os.path.join(img_dir, "results.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {pj}")
    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
