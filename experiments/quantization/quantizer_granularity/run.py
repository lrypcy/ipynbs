# -*- coding: utf-8 -*-
"""
00 篇配套实验：量化器数学地基的最小可复现研究。

三个 Demo（与文章章节一一对应）：
  A. SNR vs 位宽      —— 验证 6.02b dB 斜率、+1.76 dB 正弦增益、高斯裁剪惩罚   (对应 §2/§5)
  B. 粒度对比         —— per-tensor / per-channel / per-group 在含 outlier 权重上的表现 (对应 §4)
  C. MSE 最优 clipping —— 裁剪比例 alpha 扫描，folklore max-scale 并非最优          (对应 §3.3)

约定（与正文符号字典一致）：
  q = clip(round(x / s) + zp, qmin, qmax)      反量化  x_hat = s * (q - zp)
  对称量化 zp = 0；带符号网格 qmin=-2^(b-1), qmax=2^(b-1)-1

运行：python run.py   （依赖 numpy + matplotlib，无 GPU 需求）
输出：results/ 下的 3 张 PNG + results.json + stdout 的 markdown 表格
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

QMIN = lambda b: -(1 << (b - 1))          # -2^{b-1}
QMAX = lambda b: (1 << (b - 1)) - 1       #  2^{b-1}-1


# ----------------------------------------------------------------------
# 核心算子：仿射量化器（正文 §2 的代码化）
# ----------------------------------------------------------------------
def affine_quantize(x, s, zp, b):
    """x: 浮点张量；s: 广播兼容的步长；zp: 广播兼容的整数零点。返回反量化结果。"""
    q = np.clip(np.round(x / s) + zp, QMIN(b), QMAX(b))
    return s * (q - zp)


def sym_scale_minmax(block_max, b):
    """对称 min-max 步长：s = max|x| / qmax。block_max 为广播形状的逐块最大绝对值。"""
    return block_max / QMAX(b)


def asym_params(x, b, axis=None, keepdims=True):
    """非对称 min-max 参数：s 与整数零点。"""
    lo = x.min(axis=axis, keepdims=keepdims)
    hi = x.max(axis=axis, keepdims=keepdims)
    s = (hi - lo) / (QMAX(b) - QMIN(b))
    zp = np.round(QMIN(b) - lo / s)
    return s, zp


def mse_clipping_search(x, b, alphas, axis=None):
    """对每个 alpha（裁剪比例）计算对称量化的 MSE，返回最优 alpha、最优 MSE、整条曲线。

    scale(alpha) = alpha * max|x| / qmax，alpha<1 时牺牲极值换更细的格距。
    """
    m = np.max(np.abs(x), axis=axis, keepdims=True)
    base = m / QMAX(b)
    curve = []
    for a in alphas:
        xh = affine_quantize(x, a * base, 0.0, b)
        curve.append(float(np.mean((x - xh) ** 2)))
    curve = np.asarray(curve)
    i = int(np.argmin(curve))
    return float(alphas[i]), float(curve[i]), curve


# ----------------------------------------------------------------------
# Demo A：SNR vs 位宽
# ----------------------------------------------------------------------
def demo_a(n=400_000, rng=None):
    bits = np.arange(2, 9)
    res = {"bits": bits.tolist(), "uniform": [], "sine": [], "gauss_folklore": [],
           "gauss_mse_opt": []}
    for b in bits:
        # 1) 均匀满幅 U(-1,1)，无裁剪 -> 理论 6.02b
        x = rng.uniform(-1.0, 1.0, n)
        s = 2.0 / (QMAX(b) - QMIN(b))
        xh = affine_quantize(x, s, 0.0, b)
        snr_u = 10 * np.log10(np.var(x) / np.mean((x - xh) ** 2))
        # 2) 满幅正弦，峰值 1 -> 理论 6.02b + 1.76
        t = rng.uniform(0, 2 * np.pi, n)
        x = np.sin(t)
        xh = affine_quantize(x, s, 0.0, b)
        snr_s = 10 * np.log10(np.var(x) / np.mean((x - xh) ** 2))
        # 3) 高斯 sigma=0.25，folklore scale（样本 max 裁剪）
        x = rng.normal(0, 0.25, n)
        s_folk = np.max(np.abs(x)) / QMAX(b)
        xh = affine_quantize(x, s_folk, 0.0, b)
        snr_gf = 10 * np.log10(np.var(x) / np.mean((x - xh) ** 2))
        # 4) 高斯 + MSE 最优裁剪
        _, _, _ = (None, None, None)
        a_opt, mse_opt, _ = mse_clipping_search(x, b, np.linspace(0.3, 1.0, 36))
        s_opt = a_opt * np.max(np.abs(x)) / QMAX(b)
        xh = affine_quantize(x, s_opt, 0.0, b)
        snr_go = 10 * np.log10(np.var(x) / np.mean((x - xh) ** 2))

        res["uniform"].append(round(float(snr_u), 2))
        res["sine"].append(round(float(snr_s), 2))
        res["gauss_folklore"].append(round(float(snr_gf), 2))
        res["gauss_mse_opt"].append(round(float(snr_go), 2))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.plot(bits, 6.02 * bits, "--", color="gray", lw=1, label=r"theory $6.02b$ (uniform)")
    ax.plot(bits, 6.02 * bits + 1.76, ":", color="gray", lw=1, label=r"theory $6.02b+1.76$ (sine)")
    ax.plot(bits, res["uniform"], "o-", label="measured: uniform full-scale")
    ax.plot(bits, res["sine"], "s-", label="measured: sine full-scale")
    ax.plot(bits, res["gauss_folklore"], "^-", label="measured: Gaussian, max-scale (clipping)")
    ax.plot(bits, res["gauss_mse_opt"], "v-", label="measured: Gaussian, MSE-optimal scale")
    ax.set_xlabel("bit width b")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Quantization SNR vs bit width (n=400k samples)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "snr_vs_bitwidth.png"))
    plt.close(fig)
    return res


# ----------------------------------------------------------------------
# Demo B：粒度对比（权重含通道级 + 点级 outlier）
# ----------------------------------------------------------------------
def make_weight_matrix(out_f=2048, in_f=2048, rng=None):
    """合成权重矩阵：基底 N(0,0.02^2)。
    注入两类真实 LLM 权重中常见的 outlier：
      - 通道级：16 个输出通道整体 x8（模拟系统性大动态范围通道）
      - 点级：约 0.05% 的散点 x30（模拟局部极端权重）
    """
    W = rng.normal(0, 0.02, (out_f, in_f))
    hot_rows = rng.choice(out_f, 16, replace=False)
    W[hot_rows] *= 8.0
    mask = rng.random(W.shape) < 5e-4
    W[mask] *= 30.0
    return W, hot_rows, mask


def demo_b(rng=None):
    W, hot_rows, hot_mask = make_weight_matrix(rng=rng)
    b = 4
    X = rng.normal(0, 1.0, (W.shape[1], 128))     # 输出误差的输入代理
    Y_ref = W @ X

    def report(name, What, extra=""):
        w_mse = float(np.mean((W - What) ** 2))
        w_snr = 10 * np.log10(np.var(W) / w_mse)
        y_err = float(np.linalg.norm(Y_ref - What @ X) / np.linalg.norm(Y_ref))
        return dict(method=name, w_snr_db=round(w_snr, 2),
                    rel_output_err_pct=round(y_err * 100, 3), note=extra)

    rows = []
    # 1) per-tensor 对称
    s = sym_scale_minmax(np.max(np.abs(W)), b)
    rows.append(report("per-tensor sym", affine_quantize(W, s, 0.0, b)))
    # 2) per-tensor 非对称
    s_a, zp_a = asym_params(W, b)
    rows.append(report("per-tensor asym", affine_quantize(W, s_a, zp_a, b)))
    # 3) per-channel 对称（沿 in_features，每个输出通道一套参数）
    s_c = sym_scale_minmax(np.max(np.abs(W), axis=1, keepdims=True), b)
    rows.append(report("per-channel sym", affine_quantize(W, s_c, 0.0, b)))
    # 4) per-channel 对称 + MSE 最优裁剪
    a_opt, _, _ = mse_clipping_search(W, b, np.linspace(0.4, 1.0, 31), axis=1)
    s_cc = a_opt * np.max(np.abs(W), axis=1, keepdims=True) / QMAX(b)
    rows.append(report("per-channel sym + MSE clip",
                       affine_quantize(W, s_cc, 0.0, b), extra=f"alpha*={a_opt:.2f}"))
    # 5) per-group(g=128) 对称
    g = 128
    Wg = W.reshape(W.shape[0], W.shape[1] // g, g)
    s_g = sym_scale_minmax(np.max(np.abs(Wg), axis=2, keepdims=True), b)
    rows.append(report("per-group(128) sym",
                       affine_quantize(Wg, s_g, 0.0, b).reshape(W.shape)))
    # 6) per-channel 非对称
    s_ca, zp_ca = asym_params(W, b, axis=1)
    rows.append(report("per-channel asym", affine_quantize(W, s_ca, zp_ca, b)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    names = [r["method"] for r in rows]
    xs = np.arange(len(names))
    axes[0].bar(xs, [r["w_snr_db"] for r in rows], color="#4C72B0")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("weight SNR (dB)")
    axes[0].set_title("Weight reconstruction SNR (INT4)")
    axes[1].bar(xs, [r["rel_output_err_pct"] for r in rows], color="#C44E52")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("relative output error (%)")
    axes[1].set_title(r"Downstream error of $\hat{W}X$ vs $WX$")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "granularity_comparison.png"))
    plt.close(fig)

    meta = {"shape": list(W.shape), "hot_rows": int(len(hot_rows)),
            "point_outlier_ratio": round(float(hot_mask.mean()), 6),
            "dynamic_range_ratio": round(float(
                np.percentile(np.max(np.abs(W), axis=1), 99) /
                np.median(np.max(np.abs(W), axis=1))), 2)}
    return rows, meta


# ----------------------------------------------------------------------
# Demo C：MSE 最优 clipping
# ----------------------------------------------------------------------
def demo_c(n=500_000, rng=None):
    b = 4
    alphas = np.linspace(0.30, 1.0, 71)
    dists = {
        "Gaussian N(0,1)": rng.normal(0, 1.0, n),
        "Laplace(0,1)": rng.laplace(0, 1.0, n),
    }
    res = {}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for name, x in dists.items():
        sig2 = float(np.var(x))
        xmax = float(np.max(np.abs(x)))
        a_star, mse_star, curve = mse_clipping_search(x, b, alphas)
        u_star = a_star * xmax                       # 最优裁剪点 M*/sigma（sigma=1 或 sqrt(2)）
        mse_full = None
        # folklore 点：alpha=1
        i_full = int(np.where(np.isclose(alphas, 1.0))[0][0])
        res[name] = {
            "sample_max_over_sigma": round(xmax / np.sqrt(sig2), 2),
            "optimal_clip_point_over_sigma": round(u_star / np.sqrt(sig2), 2),
            "mse_at_opt_over_var": round(mse_star / sig2, 6),
            "mse_at_full_over_var": round(float(curve[i_full]) / sig2, 6),
            "snr_gain_db": round(10 * np.log10(curve[i_full] / mse_star), 2),
        }
        ax.plot(alphas, curve / sig2, label=name)
        ax.scatter([a_star], [mse_star / sig2], zorder=5, color="black", s=28)
        ax.annotate(f"$\\alpha^*$={a_star:.2f}", (a_star, mse_star / sig2),
                    textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="folklore scale (no clipping)")
    ax.set_xlabel(r"clipping ratio $\alpha$  ($s=\alpha\cdot\max|x|/q_{\max}$)")
    ax.set_ylabel("normalized MSE  (MSE / Var[x])")
    ax.set_title(f"MSE vs clipping ratio, symmetric INT{b} (n={n:,})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mse_vs_clipping_ratio.png"))
    plt.close(fig)
    return res


def main():
    rng = np.random.default_rng(0)
    print("Running Demo A: SNR vs bitwidth ...")
    ra = demo_a(rng=rng)
    print("Running Demo B: granularity comparison ...")
    rb, meta_b = demo_b(rng=rng)
    print("Running Demo C: MSE-optimal clipping ...")
    rc = demo_c(rng=rng)

    payload = {"demo_a": ra, "demo_b": {"rows": rb, "meta": meta_b}, "demo_c": rc}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n## Demo A: SNR (dB)\n")
    print("| b | uniform (theory 6.02b) | sine (6.02b+1.76) | Gauss max-scale | Gauss MSE-opt |")
    print("|---|---|---|---|---|")
    for i, b in enumerate(ra["bits"]):
        print(f"| {b} | {ra['uniform'][i]} ({6.02*b:.1f}) | {ra['sine'][i]} "
              f"({6.02*b+1.76:.1f}) | {ra['gauss_folklore'][i]} | {ra['gauss_mse_opt'][i]} |")

    print("\n## Demo B: INT4 granularity comparison\n")
    print(f"W {meta_b['shape']}, hot rows={meta_b['hot_rows']}, "
          f"point outliers={meta_b['point_outlier_ratio']:.4%}, "
          f"P99/median row-range={meta_b['dynamic_range_ratio']}x\n")
    print("| method | weight SNR (dB) | rel output err (%) | note |")
    print("|---|---|---|---|")
    for r in rb:
        print(f"| {r['method']} | {r['w_snr_db']} | {r['rel_output_err_pct']} | {r['note']} |")

    print("\n## Demo C: optimal clipping (INT4)\n")
    for k, v in rc.items():
        print(f"- {k}: max|.|/sigma={v['sample_max_over_sigma']}, "
              f"M*/sigma={v['optimal_clip_point_over_sigma']}, "
              f"MSE*/var={v['mse_at_opt_over_var']}, MSE(full)/var={v['mse_at_full_over_var']}, "
              f"gain={v['snr_gain_db']} dB")

    # 理论对照：高斯 b=4 解析最优裁剪点 u* ~= 2.5（正文 §3.3 表格）
    print("\nTheory check: Gaussian b=4 analytic u* ~= 2.5 sigma "
          "(old-draft §1.3 case B); see results.json for measured value.")
    print("\nAll outputs saved to", OUT)


if __name__ == "__main__":
    main()
