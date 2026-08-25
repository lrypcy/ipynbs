#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《PTQ（03）：AWQ 与 OmniQuant》配套实验（OmniQuant 侧）。

纯 numpy + matplotlib，CPU 秒级。复现 OmniQuant 的核心直觉：
可学习的量化区间（learnable clipping）显著优于 MinMax。

  (A) 全局 clip 系数 alpha 扫描：L(alpha) 曲线呈"澡盆"形，
      alpha=1（MinMax）不在盆底
  (B) 可学习裁剪：对每列独立做黄金分割坐标下降（目标不可导，
      一阶 STE 梯度从 MinMax 出发近似零信号——本实验同时把这个
      失败模式作为对照打印出来），收敛到与逐列网格一致的盆底
  (C) 四种方法对比：MinMax / 全局网格 / 朴素STE-GD / 逐列可学习

对应文章：https://lrypcy.github.io/2026/08/24/ptq-03-awq-omniq/
输出图到 results/（相对本文件），文件名用有含义的英文。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
T, K, N = 512, 512, 128          # token 数 / 输入通道 / 输出通道
QMAX = 7                          # INT4 对称网格 2^(b-1)-1
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


# ----------------------------------------------------------------------
# 数据与量化器
# ----------------------------------------------------------------------
def make_layer():
    """构造带异质列尺度 + 少量权重离群列的线性层 Y = X @ W。"""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((T, K))
    col_scale = rng.uniform(0.3, 1.0, size=(1, N))       # 异质列尺度
    W = rng.standard_normal((K, N)) * col_scale
    out_idx = rng.choice(N, size=6, replace=False)
    W[:, out_idx] *= 12.0                                 # 权重离群列
    return X, W


def quant_W(W, s_vec):
    """按列对称均匀量化：q = s * clamp(round(w/s), +-qmax)。"""
    return np.clip(np.round(W / s_vec[None, :]), -QMAX, QMAX) * s_vec[None, :]


def out_rel_mse(X, W, Wq):
    """层输出相对 MSE：||X(W-Wq)||_F^2 / ||XW||_F^2。"""
    E = X @ (W - Wq)
    return float(np.sum(E ** 2) / np.sum((X @ W) ** 2))


def minmax_scale(W):
    return np.max(np.abs(W), axis=0) / QMAX


# ----------------------------------------------------------------------
# (B-1) 朴素 STE 梯度（对照组：预期失效）
# ----------------------------------------------------------------------
def ste_gd_from_minmax(X, W, iters=300, lr=0.10):
    """LSQ 风格 scale 梯度 dL/ds = -2*sum_t e_t*qhat_t，从 MinMax 出发。

    教学点：内部元素的舍入残差 e 与 r=round(w/s) 近似不相关，
    一阶信号是二阶小量 -> GD 几乎不移动。OmniQuant 实践因此采用
    分位数/网格初始化而非 MinMax 初始化。
    """
    s = minmax_scale(W).copy()
    traj = []
    for _ in range(iters):
        qhat = np.clip(np.round(W / s[None, :]), -QMAX, QMAX)
        q = qhat * s[None, :]
        E = X @ (W - q)
        ref = np.maximum(np.sum((X @ W) ** 2, axis=0), 1e-12)
        grad_log_s = (-2.0 * np.sum(E * qhat, axis=0)) * s / ref
        s = s * np.exp(-lr * grad_log_s)
        if len(traj) < 5:
            traj.append(float(np.mean(s / minmax_scale(W))))
    return s


# ----------------------------------------------------------------------
# (B-2) 可学习裁剪：逐列黄金分割坐标下降（免梯度、对精确目标优化）
# ----------------------------------------------------------------------
def golden_section_col(x_col, w_col, s_mm, lo=0.30, hi=1.00, iters=40):
    """单列：min_a ||x(w - Q(w, a*s_mm))||^2，a∈[lo,hi]，黄金分割搜索。"""
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def f(a):
        s = a * s_mm
        q = np.clip(np.round(w_col / s), -QMAX, QMAX) * s
        e = x_col @ (w_col - q)
        return float(e @ e)

    b = hi - gr * (hi - lo)
    c = lo + gr * (hi - lo)
    fb, fc = f(b), f(c)
    for _ in range(iters):
        if fb < fc:
            # 最小值在 [lo, c]：右界移到 c，c 继承担旧 b
            hi, c, fc = c, b, fb
            b = hi - gr * (hi - lo)
            fb = f(b)
        else:
            # 最小值在 [b, hi]：左界移到 b，b 继承担旧 c
            lo, b, fb = b, c, fc
            c = lo + gr * (hi - lo)
            fc = f(c)
    return (lo + hi) / 2.0


def learnable_clip_percol(X, W):
    """逐列独立的可学习裁剪（坐标下降版 OmniQuant clip）。"""
    s_mm = minmax_scale(W)
    a_star = np.empty(W.shape[1])
    for j in range(W.shape[1]):
        a_star[j] = golden_section_col(X[:, :], W[:, j], s_mm[j])
    return a_star * s_mm


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    X, W = make_layer()

    # ---- (A) 全局 alpha 扫描：s = alpha * minmax_scale ----
    alphas = np.linspace(0.30, 1.00, 71)
    losses = np.array([
        out_rel_mse(X, W, quant_W(W, a * minmax_scale(W))) for a in alphas
    ])
    a_grid_best = float(alphas[int(np.argmin(losses))])

    # ---- (B) 可学习 ----
    s_ste = ste_gd_from_minmax(X, W)
    loss_ste = out_rel_mse(X, W, quant_W(W, s_ste))
    a_ste_mean = float(np.mean(s_ste / minmax_scale(W)))

    s_learn = learnable_clip_percol(X, W)
    a_cols = s_learn / minmax_scale(W)
    loss_learn = out_rel_mse(X, W, quant_W(W, s_learn))

    # ---- (C) 四方法对比 ----
    res = {
        "rel_mse_minmax": out_rel_mse(X, W, quant_W(W, minmax_scale(W))),
        "rel_mse_global_grid": float(losses.min()),
        "alpha_global_grid": a_grid_best,
        "rel_mse_ste_gd": loss_ste,
        "alpha_ste_mean": a_ste_mean,
        "rel_mse_learnable": loss_learn,
        "alpha_learnable_min": float(a_cols.min()),
        "alpha_learnable_median": float(np.median(a_cols)),
        "alpha_learnable_max": float(a_cols.max()),
    }
    base = res["rel_mse_minmax"]
    res["mse_reduction_pct"] = {
        "global_grid": 100 * (1 - res["rel_mse_global_grid"] / base),
        "ste_gd": 100 * (1 - res["rel_mse_ste_gd"] / base),
        "learnable": 100 * (1 - res["rel_mse_learnable"] / base),
    }

    # ---- 图 1：澡盆曲线 + 方法标记 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(alphas, losses, "-", color="#4C72B0", lw=2)
    ax[0].axvline(1.0, color="#C44E52", ls="--", lw=1, label="alpha=1 (MinMax)")
    ax[0].axvline(a_grid_best, color="#DD8452", ls=":", lw=1.5,
                  label=f"global grid best={a_grid_best:.2f}")
    ax[0].axvline(float(np.median(a_cols)), color="#55A868", ls="-.", lw=1.5,
                  label=f"per-col learnable median={np.median(a_cols):.2f}")
    ax[0].set_xlabel("global clip ratio alpha")
    ax[0].set_ylabel("output rel. MSE")
    ax[0].set_yscale("log")
    ax[0].set_title("Clipping sweep: MinMax sits off the basin floor")
    ax[0].legend(fontsize=8)

    # ---- 图 2：逐列 alpha 分布（可学习 vs MinMax=1.0）----
    ax[1].hist(a_cols, bins=40, color="#4C72B0", alpha=0.85)
    ax[1].axvline(1.0, color="#C44E52", ls="--", label="MinMax (all cols = 1.0)")
    ax[1].set_xlabel("per-column learned alpha = s*/s_minmax")
    ax[1].set_ylabel("#columns")
    ax[1].set_title("Learnable ranges spread across columns\n(outlier columns need smaller alpha)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(RES, "omniq_clip_basin_and_alpha_hist.png")
    fig.savefig(p1, dpi=130)

    # ---- 打印结论 ----
    print("=" * 62)
    print("[A] 全局 alpha 扫描：盆底 alpha*=%.2f, rel.MSE=%.4e"
          % (a_grid_best, losses.min()))
    print("    MinMax(alpha=1)             rel.MSE=%.4e" % base)
    print("-" * 62)
    print("[B-1] 对照组：朴素 STE 梯度从 MinMax 出发（预期失效）：")
    print("      rel.MSE=%.4e (平均 alpha 漂移至 %.2f，仅降 %.1f%%)"
          % (loss_ste, a_ste_mean, res["mse_reduction_pct"]["ste_gd"]))
    print("      -> 内部舍入残差与 r 不相关，一阶信号为二阶小量；")
    print("         这正是 OmniQuant 用分位数/网格初始化的原因")
    print("-" * 62)
    print("[B-2] 可学习裁剪（逐列黄金分割坐标下降）：")
    print("      rel.MSE=%.4e" % loss_learn)
    print("      学到的逐列 alpha：min=%.2f / median=%.2f / max=%.2f"
          % (res["alpha_learnable_min"], res["alpha_learnable_median"],
             res["alpha_learnable_max"]))
    print("=" * 62)
    print("[C] 相对 MinMax 的输出 MSE 降低：")
    print("    全局网格搜索   : %.1f%%" % res["mse_reduction_pct"]["global_grid"])
    print("    朴素 STE-GD   : %.1f%%（失效对照）" % res["mse_reduction_pct"]["ste_gd"])
    print("    逐列可学习    : %.1f%%  <- OmniQuant 核心收益" % res["mse_reduction_pct"]["learnable"])
    print(f"[save] {p1}")

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
