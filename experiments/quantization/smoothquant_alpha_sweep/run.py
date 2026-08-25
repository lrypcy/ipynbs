# -*- coding: utf-8 -*-
"""
02 篇配套实验：SmoothQuant 的等效缩放迁移 + alpha 扫描（纯 numpy，可复现）。

对应正文章节：
  - 实验 1（对应 §3 等效缩放推导）：naive W8A8 vs SmoothQuant W8A8 的逐层 MSE
  - 实验 2（对应 §3.3 / §5）：alpha 扫描，验证 U 形曲线与“难度守恒”
  - 实验 3（对应 §4 激活统计量估计）：max 估计随校准集漂移 + 分位数估计更稳

约定（与正文符号字典一致）：
  X : 激活 (T, d_in)            W : 权重 (d_in, d_out)
  s_j = max|X_j|^alpha / max|W_j|^(1-alpha)   (j 遍历输入通道)
  X' = X @ diag(s)^{-1}         W' = diag(s) @ W        X'W' = XW 严格成立
  激活量化：per-token 对称；权重量化：per-channel(输出通道) 对称

运行：python run.py   （依赖 numpy + matplotlib，无 GPU）
输出：本目录 results/ 下的 3 张 PNG + results/results.json + stdout 的 markdown 表格
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

QMAX = 2**7 - 1  # INT8 有符号网格上界 127


# ----------------------------------------------------------------------
# 核心算子
# ----------------------------------------------------------------------
def quant_per_token(X, b=8):
    """对称 per-token 量化（激活）：每个 token 一个 scale，覆盖全部输入通道。"""
    qmax = 2 ** (b - 1) - 1
    s = np.max(np.abs(X), axis=1, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(X / s), -qmax, qmax) * s


def quant_per_outchannel(W, b=8):
    """对称 per-channel(输出通道) 量化（权重）：每个输出通道一个 scale。"""
    qmax = 2 ** (b - 1) - 1
    s = np.max(np.abs(W), axis=0, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(W / s), -qmax, qmax) * s


def smooth_scale(X, W, alpha):
    """s_j = max|X_j|^alpha / max|W_j|^(1-alpha)，j 为输入通道。"""
    ax = np.max(np.abs(X), axis=0)          # (d_in,)
    aw = np.max(np.abs(W), axis=1)          # (d_in,)
    return ax**alpha / (aw ** (1 - alpha) + 1e-8)


def w8a8_linear(X, W, alpha=0.5, smooth=True):
    if smooth:
        s = smooth_scale(X, W, alpha)
        Xp = X / s
        Wp = W * s[:, None]
    else:
        Xp, Wp = X, W
    Xq = quant_per_token(Xp)
    Wq = quant_per_outchannel(Wp)
    return Xq @ Wq


def make_data(T=1024, d_in=256, d_out=512, rng=None):
    """合成重尾激活 + 高斯权重。"""
    x = rng.normal(0.0, 0.3, size=(T, d_in))              # 正常激活，约 N(0, 0.09)
    n_out = 12
    oid = rng.choice(d_in, n_out, replace=False)
    x[:, oid] *= rng.uniform(20.0, 80.0, size=n_out)     # 常驻 outlier 通道，放大 20~80x
    w = rng.normal(0.0, 0.04, size=(d_in, d_out))        # 高斯权重
    return x, w, oid


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------
# 实验 1 + 2：naive vs SmoothQuant，以及 alpha 扫描
# ----------------------------------------------------------------------
def exp_alpha_sweep(rng):
    T, d_in, d_out = 1024, 256, 512
    X, W, oid = make_data(T=T, d_in=d_in, d_out=d_out, rng=rng)
    y_ref = X @ W

    y_naive = w8a8_linear(X, W, alpha=0.5, smooth=False)
    y_smooth = w8a8_linear(X, W, alpha=0.5, smooth=True)
    mse_naive = mse(y_naive, y_ref)
    mse_smooth = mse(y_smooth, y_ref)
    rel_naive = rel_err(y_naive, y_ref)
    rel_smooth = rel_err(y_smooth, y_ref)

    # 难度守恒数值核验：alpha=0.5 时每通道 range 乘积应不变
    ax = np.max(np.abs(X), axis=0)
    aw = np.max(np.abs(W), axis=1)
    s = smooth_scale(X, W, 0.5)
    axp = np.max(np.abs(X / s), axis=0)
    awp = np.max(np.abs(W * s[:, None]), axis=1)
    prod_before = ax * aw
    prod_after = axp * awp
    consv_dev = float(np.max(np.abs(prod_after - prod_before) / prod_before))

    # alpha 扫描
    alphas = np.linspace(0.0, 1.0, 21)
    mses = np.array([mse(w8a8_linear(X, W, a, True), y_ref) for a in alphas])
    i_opt = int(np.argmin(mses))
    alpha_opt = float(alphas[i_opt])

    # 动态范围 seesaw：每通道平均范围 vs alpha
    act_ranges, wgt_ranges = [], []
    for a in alphas:
        sa = smooth_scale(X, W, a)
        Xp = X / sa
        Wp = W * sa[:, None]
        act_ranges.append(float(np.mean(np.max(np.abs(Xp), axis=0))))
        wgt_ranges.append(float(np.mean(np.max(np.abs(Wp), axis=1))))
    act_ranges = np.array(act_ranges)
    wgt_ranges = np.array(wgt_ranges)

    res = {
        "shape": {"T": T, "d_in": d_in, "d_out": d_out, "n_outlier": int(len(oid))},
        "mse_naive": mse_naive,
        "mse_smooth": mse_smooth,
        "rel_naive": rel_naive,
        "rel_smooth": rel_smooth,
        "speedup_ratio": mse_naive / mse_smooth,
        "conservation_max_rel_dev": consv_dev,
        "alpha_opt": alpha_opt,
        "mse_at_opt": float(mses[i_opt]),
        "alphas": alphas.tolist(),
        "mses": mses.tolist(),
        "act_ranges": act_ranges.tolist(),
        "wgt_ranges": wgt_ranges.tolist(),
    }
    return res


# ----------------------------------------------------------------------
# 实验 3：激活统计量估计（max vs 分位数，随校准集漂移）
# ----------------------------------------------------------------------
def exp_activation_stat(rng):
    d_in = 256
    n_out = 12
    oid = rng.choice(d_in, n_out, replace=False)
    # 构造一个“无限”校准池：1e6 token 的激活
    pool = rng.normal(0.0, 0.3, size=(1_000_000, d_in))
    pool[:, oid] *= rng.uniform(20.0, 80.0, size=n_out)

    calib_sizes = [256, 1024, 4096, 16384, 65536]
    # 选一个代表性 outlier 通道与一个正常通道
    och = int(oid[0])
    nch = 0
    while nch in oid:
        nch += 1

    max_est, pct_est = {}, {}
    # max 估计随校准集漂移
    drift_max = []
    for N in calib_sizes:
        vals = []
        for _ in range(8):
            idx = rng.choice(pool.shape[0], N, replace=False)
            sub = pool[idx]
            vals.append(float(np.max(np.abs(sub[:, och]))))
        drift_max.append(float(np.mean(vals)))
    # 同一 N，比较 max 与 99.9 分位数估计的方差（稳定性）
    N = 4096
    max_var, pct_var = [], []
    for _ in range(30):
        idx = rng.choice(pool.shape[0], N, replace=False)
        sub = pool[idx]
        max_var.append(np.max(np.abs(sub[:, och])))
        pct_var.append(np.percentile(np.abs(sub[:, och]), 99.9))
    max_std = float(np.std(max_var))
    pct_std = float(np.std(pct_var))

    # 正常通道用 max 估计（应稳定，因为无极端尾）
    norm_max_std = float(np.std([np.max(np.abs(pool[rng.choice(pool.shape[0], N, replace=False)][:, nch]))
                                 for _ in range(30)]))

    res = {
        "calib_sizes": calib_sizes,
        "outlier_channel_max_vs_N": drift_max,
        "max_est_std_outlier": max_std,
        "pct99_est_std_outlier": pct_std,
        "max_est_std_normal": norm_max_std,
        "stability_ratio_max_over_pct": max_std / pct_std,
    }
    return res


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot_alpha_sweep(res):
    alphas = np.array(res["alphas"])
    mses = np.array(res["mses"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

    ax = axes[0]
    ax.plot(alphas, mses, "o-", color="#C44E52", label="measured MSE(Y_ref, Y_hat)")
    i_opt = int(np.argmin(mses))
    ax.scatter([alphas[i_opt]], [mses[i_opt]], color="black", zorder=5, s=30)
    ax.annotate(f"alpha*={alphas[i_opt]:.2f}", (alphas[i_opt], mses[i_opt]),
                textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax.set_xlabel("migration strength alpha")
    ax.set_ylabel("layer output MSE (log)")
    ax.set_yscale("log")
    ax.set_title("Alpha sweep: U-shaped MSE curve (W8A8)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    act = np.array(res["act_ranges"])
    wgt = np.array(res["wgt_ranges"])
    ax.plot(alphas, act, "s-", label="avg activation range max|X'_j|")
    ax.plot(alphas, wgt, "^-", label="avg weight range max|W'_j|")
    ax.set_xlabel("migration strength alpha")
    ax.set_ylabel("avg per-channel dynamic range (log)")
    ax.set_yscale("log")
    ax.set_title("Dynamic-range seesaw: activation vs weight")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(OUT, "smoothquant_alpha_sweep_mse.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_difficulty_conservation(rng, res):
    T, d_in, d_out = 1024, 256, 512
    X, W, oid = make_data(T=T, d_in=d_in, d_out=d_out, rng=rng)
    ax = np.max(np.abs(X), axis=0)
    aw = np.max(np.abs(W), axis=1)
    s = smooth_scale(X, W, 0.5)
    axp = np.max(np.abs(X / s), axis=0)
    awp = np.max(np.abs(W * s[:, None]), axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), dpi=150)
    # 上：难度守恒（每通道 range 乘积 invariance）
    ax0 = axes[0]
    prod = axp * awp
    baseline = ax * aw
    xs = np.arange(d_in)
    ax0.plot(xs, prod, lw=0.8, color="#4C72B0", label="max|X'_j| * max|W'_j| (after)")
    ax0.plot(xs, baseline, "--", lw=0.8, color="gray", label="max|X_j| * max|W_j| (before)")
    ax0.set_xlabel("input channel j")
    ax0.set_ylabel("per-channel range product")
    ax0.set_title("Difficulty conservation: range product invariant under smoothing")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # 下：激活各通道范围，平滑前(重尾) vs 平滑后(削平)
    ax1 = axes[1]
    ax1.plot(xs, ax, lw=0.8, color="#C44E52", label="before smoothing max|X_j|")
    ax1.plot(xs, axp, lw=0.8, color="#55A868", label="after smoothing max|X'_j|")
    ax1.set_yscale("log")
    ax1.set_xlabel("input channel j")
    ax1.set_ylabel("activation channel range (log)")
    ax1.set_title("Activation smoothing: outlier channels flattened, normal channels lifted")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = os.path.join(OUT, "smoothquant_difficulty_conservation.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_activation_stat(res):
    calib_sizes = res["calib_sizes"]
    drift = res["outlier_channel_max_vs_N"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

    ax = axes[0]
    ax.plot(calib_sizes, drift, "o-", color="#C44E52")
    ax.set_xscale("log")
    ax.set_xlabel("calibration set size N (tokens)")
    ax.set_ylabel("estimated max|X_outlier| (log)")
    ax.set_yscale("log")
    ax.set_title("Max-statistic drift: estimate grows with calibration N")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    labels = ["max estimator", "99.9-pct estimator"]
    stds = [res["max_est_std_outlier"], res["pct99_est_std_outlier"]]
    bars = ax.bar(labels, stds, color=["#C44E52", "#55A868"])
    ax.set_ylabel("std of estimate across 30 subsets (N=4096)")
    ax.set_title("Stability: percentile estimator less sensitive to sampling")
    for b, v in zip(bars, stds):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(OUT, "smoothquant_activation_stat_estimation.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    rng = np.random.default_rng(0)
    print("=== 实验 1+2: naive vs SmoothQuant + alpha sweep ===")
    r1 = exp_alpha_sweep(rng)
    print(f"naive  W8A8  MSE = {r1['mse_naive']:.6e}  rel_err = {r1['rel_naive']*100:.2f}%")
    print(f"smooth W8A8  MSE = {r1['mse_smooth']:.6e}  rel_err = {r1['rel_smooth']*100:.2f}%")
    print(f"speedup ratio (naive/smooth) MSE = {r1['speedup_ratio']:.1f}x")
    print(f"difficulty conservation max rel dev = {r1['conservation_max_rel_dev']:.2e}")
    print(f"alpha* = {r1['alpha_opt']:.2f}  mse@opt = {r1['mse_at_opt']:.6e}")
    print("alpha sweep (every 0.1):")
    for a, m in zip(r1["alphas"], r1["mses"]):
        if abs(a - round(a, 1)) < 1e-9:
            print(f"  alpha={a:.1f}  MSE={m:.4e}")

    print("\n=== 实验 3: activation statistic estimation ===")
    r3 = exp_activation_stat(rng)
    print("outlier-channel max|X| vs calibration N:")
    for N, v in zip(r3["calib_sizes"], r3["outlier_channel_max_vs_N"]):
        print(f"  N={N:>6}  max|X|={v:.2f}")
    print(f"max estimator std(outlier) = {r3['max_est_std_outlier']:.3f}")
    print(f"99.9pct estimator std(outlier) = {r3['pct99_est_std_outlier']:.3f}")
    print(f"stability ratio max/pct = {r3['stability_ratio_max_over_pct']:.1f}x")
    print(f"max estimator std(normal channel) = {r3['max_est_std_normal']:.4f}")

    p1 = plot_alpha_sweep(r1)
    p2 = plot_difficulty_conservation(rng, r1)
    p3 = plot_activation_stat(r3)
    print("\nfigures:", p1, p2, p3)

    out = {"exp_alpha_sweep": r1, "exp_activation_stat": r3}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("results.json saved.")


if __name__ == "__main__":
    main()
