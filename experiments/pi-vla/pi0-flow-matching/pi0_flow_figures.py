"""π0 流匹配实验的配图（4 张 SVG）。

所有数值都从 `pi0_flow_lab.py` 的函数重算（直接 import），保证与
`results/stdout.txt`、正文数字一致。输出写在 `figures/`，正文用的是
`assets/img/pi-vla/pi0_*.svg`（博客仓库里）。

运行：python3 pi0_flow_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pi0_flow_lab as L

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

C_MAIN = "#2176ae"
C_ALT = "#c0392b"
C_DASH = "#8a8a8a"


def marginal_pdf(x, t):
    """p(x_t)：同一高斯混合在直线路径上的边际密度。"""
    m, s2 = L.marginal_stats(np.asarray(x), t)
    s = np.sqrt(s2)
    return float((L.W * L.gauss_pdf(np.asarray(x), m, s)).sum())


# ---------------------------------------------------------------- 图 1
def fig_marginal_velocity():
    x = np.linspace(-3.2, 4.2, 600)
    ts = (0.2, 0.5, 0.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax1.fill_between(x, [marginal_pdf(xi, 0.0) for xi in x], color="#bcbcbc", alpha=0.65,
                     label="$t=0$（数据分布，双峰）")
    for t, c in zip(ts, ("#f39c12", C_MAIN, "#8e44ad")):
        ax1.plot(x, [marginal_pdf(xi, t) for xi in x], color=c, lw=2.0, label=f"$t={t}$")
    ax1.set_xlabel("动作取值 $x$")
    ax1.set_ylabel("密度")
    ax1.set_title("同一条直线概率路径上的边际密度：双峰随 $t$ 被噪声摊平", fontsize=12)
    ax1.legend(frameon=False)

    for t, c in zip(ts, ("#f39c12", C_MAIN, "#8e44ad")):
        ax2.plot(x, L.v_star(x, t), color=c, lw=2.0, label=f"$t={t}$")
    for mu in L.MU:
        ax2.axvline(mu, color=C_DASH, ls=":", lw=1.0)
    ax2.annotate("两个数据中心", (L.MU[1], -6.2), textcoords="offset points",
                 xytext=(-70, 0), color=C_DASH, fontsize=10)
    ax2.set_xlabel("当前点 $x_t$")
    ax2.set_ylabel("边际速度场 $v^*(x,t)$")
    ax2.set_title("最优解 $v^*=(x-\\mathbb{E}[x_0\\mid x_t])/t$：$t$ 越小越陡", fontsize=12)
    ax2.legend(frameon=False)
    ax2.set_ylim(-12, 6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pi0_marginal_velocity.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 2
def fig_euler_steps():
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(1200)
    ref = L.rk4_reference(eps)
    steps = (1, 2, 3, 5, 10, 20, 50, 100)
    rms = [float(np.sqrt(np.mean((L.euler(eps, n) - ref) ** 2))) for n in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax1.plot(steps, rms, "o-", color=C_MAIN, lw=2.2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Euler 步数 $N$")
    ax1.set_ylabel("终点 RMS 误差（对比 RK4 参考解）")
    ax1.set_title("一阶收敛：$N$ 翻倍误差减半，10 步仍在曲线上", fontsize=12)
    for n, e in ((1, rms[0]), (10, rms[4]), (50, rms[6])):
        ax1.annotate(f"{e:.4f}", (n, e), textcoords="offset points",
                     xytext=(8, -4), color=C_MAIN, fontsize=10)
    ax1.annotate("1→10 步降 9.68×，10→50 步再降 4.22×", (2, 0.05),
                 color=C_DASH, fontsize=10)

    x1 = L.euler(eps, 1)
    x10 = L.euler(eps, 10)
    bins = np.linspace(-2.6, 2.6, 70)
    ax2.plot(bins, [marginal_pdf(b, 0.0) for b in bins], color="#bcbcbc", lw=2.4,
             label="真实数据分布（双峰）")
    ax2.hist(x10, bins=bins, density=True, color=C_MAIN, alpha=0.55, label="10 步 Euler")
    ax2.hist(x1, bins=bins, density=True, color=C_ALT, alpha=0.75, label="1 步 Euler")
    ax2.axvline(L.MU_BAR, color=C_ALT, ls="--", lw=1.2)
    ax2.annotate(f"全部塌在 $E[x_0]={L.MU_BAR:.2f}$", (L.MU_BAR, 4.2),
                 textcoords="offset points", xytext=(-40, 10), color=C_ALT, fontsize=10)
    ax2.set_xlabel("生成的动作取值")
    ax2.set_ylabel("密度")
    ax2.set_title("1 步把整批样本塌成单点，10 步把双峰找回来", fontsize=12)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pi0_euler_steps.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 3
def fig_time_sampling():
    a, b = 1.5, 1.0
    t = np.linspace(0.0, 1.0, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax1.plot(t, a * t ** (a - 1.0), color=C_MAIN, lw=2.4,
             label="$\\mathrm{Beta}(1.5,1)$：$f(t)=1.5\\sqrt{t}$")
    ax1.plot(t, np.ones_like(t), color=C_DASH, ls="--", lw=1.6, label="均匀")
    ax1.fill_between(t, a * t ** (a - 1.0), color=C_MAIN, alpha=0.12)
    ax1.axvline(0.6, color=C_MAIN, ls=":", lw=1.2)
    ax1.annotate("$E[t]=0.6$", (0.6, 1.35), textcoords="offset points",
                 xytext=(-52, 0), color=C_MAIN, fontsize=10)
    ax1.set_xlabel("时间 $t$（openpi 约定：$t=1$ 是噪声端）")
    ax1.set_ylabel("采样密度")
    ax1.set_title("时间采样偏向噪声端，代价是数据端样本变少", fontsize=12)
    ax1.legend(frameon=False)

    edges = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0))
    labels = ["[0,0.2)\n数据端", "[0.2,0.5)", "[0.5,0.8)", "[0.8,1.0]\n噪声端"]
    ratios = [(hi**1.5 - lo**1.5) / (hi - lo) for lo, hi in edges]
    colors = [C_ALT if r < 1 else C_MAIN for r in ratios]
    ax2.bar(labels, ratios, color=colors, alpha=0.85)
    ax2.axhline(1.0, color=C_DASH, ls="--", lw=1.2)
    for i, r in enumerate(ratios):
        ax2.text(i, r + 0.05, f"{r:.2f}×", ha="center", fontsize=10)
    ax2.set_ylabel("该区间的样本量 / 均匀采样")
    ax2.set_title("重加权后 ESS/N = 0.7507（要多采 1.33 倍样本）", fontsize=12)
    ax2.set_ylim(0, max(ratios) * 1.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pi0_time_sampling.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 4
def fig_kv_cache():
    n_prefix, n_action = 8, 5
    input_mask = np.ones(n_prefix + 1 + n_action, dtype=bool)
    ar_mask = np.array([False] * n_prefix + [True] + [True] + [False] * (n_action - 1), dtype=bool)
    m = L.make_attn_mask(input_mask, ar_mask)

    H = 50
    suffix_tokens = 1 + H
    rows = []
    for n_cam, n_lang in ((2, 30), (3, 30)):
        p = 256 * n_cam + n_lang
        rows.append((f"{n_cam} 相机\nprefix={p}", 10 * (p + suffix_tokens),
                     (p + suffix_tokens) + 10 * suffix_tokens))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax1.imshow(m.astype(int), cmap="Greys", interpolation="nearest")
    ax1.set_xlabel("被 attend 的位置（key）")
    ax1.set_ylabel("查询位置（query）")
    ax1.set_title(f"cumsum 掩码的三块结构（{int(m.sum())} 个可 attend 位置）", fontsize=12)
    for pos, lab in ((n_prefix - 0.5, "prefix"), (n_prefix, "state"), (n_prefix + 3, "action")):
        ax1.text(-1.2, pos + 0.3, lab, fontsize=10, ha="right")

    idx = np.arange(len(rows))
    w = 0.36
    ax2.bar(idx - w / 2, [r[1] for r in rows], w, color=C_ALT, alpha=0.85, label="每步全量重算")
    ax2.bar(idx + w / 2, [r[2] for r in rows], w, color=C_MAIN, alpha=0.85, label="复用 prefix KV 缓存")
    for i, r in enumerate(rows):
        ax2.text(i - w / 2, r[1] + 90, f"{r[1]}", ha="center", fontsize=10)
        ax2.text(i + w / 2, r[2] + 90, f"{r[2]}", ha="center", fontsize=10)
        ax2.text(i, max(r[1], r[2]) * 0.62, f"{r[1] / r[2]:.2f}×", ha="center",
                 color="#333333", fontsize=11)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([r[0] for r in rows])
    ax2.set_ylabel("十步推理的 token·次")
    ax2.set_title("十步推理的前向账：10 次全序列降到 1 次 + 10 次后缀", fontsize=12)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pi0_kv_cache.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_marginal_velocity()
    fig_euler_steps()
    fig_time_sampling()
    fig_kv_cache()
    for f in sorted(os.listdir(OUT)):
        print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) / 1024:.1f} KB")
