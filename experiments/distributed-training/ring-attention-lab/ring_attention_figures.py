"""序列并行一文的配图（3 张 SVG，一张结构图 + 两张数据图）。

数据图上的数字全部来自 `ring_attention_lab.ipynb` / `results/stdout.txt`，
结构图由 `ring_attention_lib.py` 的 `token_owner` 直接算出归属，不写死。

输出写在 `figures/`，正文用的是博客仓库的 `assets/img/dist-train/*.svg`。
同时写一份 PNG 方便人眼核对（PNG 只是校对用，不进博客）。

运行：~/Software/miniconda3/bin/python ring_attention_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

from ring_attention_lib import (
    allgather_comm_bytes,
    ring_comm_bytes,
    causal_work_matrix,
    ring_rounds,
    simulate_wallclock,
    ulysses_comm_bytes,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

ROUND_COLORS = ["#185FA5", "#3B8F6B", "#E0A030", "#C0504D"]
INK = "#1A1A1A"
MUTED = "#6B6B6B"


def save(fig, name):
    svg = os.path.join(OUT, name + ".svg")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", svg)


# ------------------------------------------------------------------ 图 1：切分


def _work_grid(S, P, scheme):
    """每卡每轮的 work（有效 token 对数）矩阵，形状 (P 卡, P 轮)。"""
    rounds, owner = ring_rounds(S, P, scheme)
    C = causal_work_matrix(S)
    M = np.zeros((P, P), dtype=int)
    for t, rnd in enumerate(rounds):
        for r in range(P):
            q = np.nonzero(owner == r)[0]
            M[r, t] = C[np.ix_(q, rnd[r])].sum()
    return M


def fig_partition():
    S, P = 16, 4
    panels = [
        ("contiguous", "连续切分", "Ring Attention 原始做法"),
        ("zigzag", "zigzag 镜像切分", "Megatron 做法"),
        ("striped", "striped 交错切分", "Striped Attention 做法"),
    ]
    grids = {k: _work_grid(S, P, k) for k, _, _ in panels}
    ideal = causal_work_matrix(S).sum() / P
    vmax = max(g.max() for g in grids.values())

    fig = plt.figure(figsize=(11.4, 4.0))
    fig.patch.set_facecolor("white")

    for idx, (scheme, label, sub) in enumerate(panels):
        M = grids[scheme]
        wall = int(M.max(axis=0).sum())
        left = 0.055 + idx * 0.325
        ax = fig.add_axes([left, 0.29, 0.20, 0.49])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax * 1.05,
                       interpolation="nearest")
        for r in range(P):
            for t in range(P):
                v = M[r, t]
                ax.text(t, r, "—" if v == 0 else str(v), ha="center", va="center",
                        fontsize=10, color=("white" if v > vmax * 0.55 else INK))
        ax.set_xticks(range(P))
        ax.set_xticklabels([f"第 {t} 轮" for t in range(P)], fontsize=8, color=MUTED)
        ax.set_yticks(range(P))
        ax.set_yticklabels([f"卡 {r}" for r in range(P)], fontsize=8, color=MUTED)
        ax.set_xlabel("")
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_color(MUTED)
            s_.set_linewidth(0.6)
        ax.set_title(f"{label}（{sub}）", fontsize=8.5, color=INK, pad=8)
        fig.text(left + 0.102, 0.225,
                 f"每轮取该列最大值 → 墙钟 {wall}（理想 {ideal:.0f}，利用率 {ideal / wall:.0%}）",
                 fontsize=7.5, color=MUTED, ha="center")

    fig.text(0.055, 0.935,
             "16 个 token、4 张卡：每格是这一轮这张卡真正要算的 token 对数（空白 = 整块被掩码）",
             fontsize=10.5, color=INK)
    fig.text(0.055, 0.075,
             "连续切法每轮都有一张卡在算整块（16），全环等它；zigzag 把前段短块与后段长块配给同一张卡，每轮负载齐平；"
             "striped 把 token 按序号 mod P 打散，阶梯变缓但仍有残余。",
             fontsize=7.8, color=MUTED)
    save(fig, "sp_partition")


# ------------------------------------------------------------------ 图 2：利用率


def fig_utilization():
    Ps = [2, 4, 8, 16, 32]
    schemes = [("contiguous", "连续切分", "#C0504D", "o"),
               ("striped", "striped 交错切分", "#E0A030", "s"),
               ("zigzag", "zigzag 镜像切分", "#185FA5", "D")]
    data = {}
    for scheme, _, _, _ in schemes:
        util = []
        for P in Ps:
            _, _, ideal = simulate_wallclock(4096, P, scheme)
            wall, _, _ = simulate_wallclock(4096, P, scheme)
            util.append(ideal / wall)
        data[scheme] = util

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor("white")
    for scheme, label, color, marker in schemes:
        ax.plot(Ps, np.array(data[scheme]) * 100, marker=marker, color=color,
                linewidth=1.8, markersize=5, label=label)
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(2.05, 51.5, "50%：每轮都有一台在算整块", fontsize=8, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ps)
    ax.set_xticklabels([str(p) for p in Ps])
    ax.set_xlabel("上下文并行度 P（序列长度 4096）", fontsize=9)
    ax.set_ylabel("注意力算力利用率", fontsize=9)
    ax.set_ylim(40, 106)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("因果掩码下的负载不均衡：连续切法随 P 恶化，另两种切法接近满负荷",
                 fontsize=10, color=INK, pad=8)
    save(fig, "sp_utilization")


# ------------------------------------------------------------------ 图 3：通信量


def fig_comm():
    S, h = 131072, 8192
    Ps = [2, 4, 8, 16, 32, 64]
    ring = [ring_comm_bytes(S, h, P) / 1e9 for P in Ps]
    ag = [allgather_comm_bytes(S, h, P) / 1e9 for P in Ps]
    ul = [ulysses_comm_bytes(S, h, P) / 1e9 for P in Ps]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor("white")
    ax.plot(Ps, ring, marker="o", color="#C0504D", linewidth=1.8, markersize=5,
            label="ring（每卡转发 KV 块）")
    ax.plot(Ps, ag, marker="x", color="#185FA5", linewidth=1.4, markersize=7,
            linestyle="--", label="all-gather（先收齐 KV）")
    ax.plot(Ps, ul, marker="s", color="#3B8F6B", linewidth=1.8, markersize=5,
            label="Ulysses（all-to-all 切 head）")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ps)
    ax.set_xticklabels([str(p) for p in Ps])
    ax.set_xlabel("并行度 P（序列 128K、KV 全宽 8192、bf16、每层每卡）", fontsize=9)
    ax.set_ylabel("每层每卡发送量（GB）", fontsize=9)
    ax.annotate("两者完全相同：\nring 省的不是带宽，是可重叠性",
                xy=(8, ring[2]), xytext=(12.5, 3.05), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    ax.annotate("P=3 交叉", xy=(3.0, 2.0), xytext=(3.3, 1.35), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.7))
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("三种序列并行的通信量：ring 与 all-gather 恒定，Ulysses 随 P 线性下降",
                 fontsize=10, color=INK, pad=8)
    save(fig, "sp_comm")


if __name__ == "__main__":
    fig_partition()
    fig_utilization()
    fig_comm()
