# -*- coding: utf-8 -*-
"""Cosmos 3 Nano 篇的配图：1 张架构图 + 5 张数据图，全部手工排版成 SVG。

字体 PingFang SC（Hiragino Sans GB 会静默缺字），坐标以英寸为单位。
数据图上的每个数字都取自同目录 `results/stdout.txt`，与 lab 输出逐位对齐。

运行：~/Software/miniconda3/bin/python cosmos3_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

BLUE = dict(fc="#E6F1FB", ec="#185FA5", tc="#042C53")
BLUE_D = dict(fc="#B5D4F4", ec="#185FA5", tc="#042C53")
PURPLE = dict(fc="#EEEDFE", ec="#534AB7", tc="#26215C")
PURPLE_D = dict(fc="#CECBF6", ec="#534AB7", tc="#26215C")
TEAL = dict(fc="#E1F5EE", ec="#0F6E56", tc="#04342C")
RED = dict(fc="#FCEBEB", ec="#A32D2D", tc="#501313")
AMBER = dict(fc="#FAEEDA", ec="#854F0B", tc="#412402")
GRAY = dict(fc="#F8F7F4", ec="#888780", tc="#2C2C2A")
GRAY_D = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")

INK = "#2C2C2A"
MUTE = "#5F5E5A"
HINT = "#888780"
PT = 1.0 / 72.0

C_AR = "#185FA5"
C_DM = "#534AB7"
C_ACC = "#A32D2D"
C_OK = "#0F6E56"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.5, fs_sub=10.5, lw=0.9, r=0.12, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(r, max(min(w, h) / 2 - 0.01, 0.01))}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=2,
        **({"linestyle": (0, (4, 2.5))} if dashed else {})))
    n = len(lines)
    if n == 0:
        return
    if n == 1:
        ax.text(x + w / 2, y + h / 2, lines[0], ha="center", va="center",
                fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
        return
    hs = [(fs if i == 0 else fs_sub) * 1.42 * PT for i in range(n)]
    total = sum(hs)
    avail = h - 0.14
    if total > avail:
        hs = [v * avail / total for v in hs]
        total = avail
    cy = y + h / 2 + total / 2
    for i, s in enumerate(lines):
        cy -= hs[i] / 2
        ax.text(x + w / 2, cy, s, ha="center", va="center",
                fontsize=(fs if i == 0 else fs_sub),
                fontweight=("500" if i == 0 else "400"),
                color=(style["tc"] if i == 0 else MUTE), zorder=3)
        cy -= hs[i] / 2


def arrow(ax, p1, p2, color=MUTE, lw=1.5, rad=0.0, dash=None, head="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=head, mutation_scale=ms, linewidth=lw,
        color=color, zorder=3, connectionstyle=f"arc3,rad={rad}",
        **({"linestyle": dash} if dash else {})))


def label(ax, x, y, text, fs=10.5, color=HINT, ha="center", va="center", weight="400"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, zorder=4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax, fs=10.5):
    ax.tick_params(labelsize=fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBB6")
        ax.spines[s].set_linewidth(0.8)


# ================================================================ 图 1 架构
def fig_arch():
    W, H = 12.9, 8.0
    fig, ax = canvas(W, H)

    # ---------------- 顶部：统一序列
    label(ax, 0.42, 7.62, "一条序列，两段子序列（图示为 720p / 189 帧的一次 T2V 请求）",
          fs=12.2, color=INK, ha="left", weight="500")

    y0, hh = 6.72, 0.58
    segs = [
        (0.42, 1.20, ["语言", "tokenizer"], BLUE, C_AR),
        (1.70, 1.20, ["视觉（理解）", "ViT 16x16"], BLUE, C_AR),
        (2.98, 0.56, ["EOS", "BOG"], GRAY, MUTE),
        (3.62, 1.10, ["干净条件", "VAE，可缺省"], PURPLE, C_DM),
        (4.80, 3.30, ["带噪目标", "VAE，参与去噪 44,160 token"], PURPLE_D, C_DM),
        (8.18, 0.86, ["音频", "25 tok/s"], TEAL, "#0F6E56"),
        (9.12, 0.98, ["动作", "每域一套投影"], TEAL, "#0F6E56"),
    ]
    for x, w, lines, st, ec in segs:
        ax.add_patch(FancyBboxPatch((x, y0), w, hh, boxstyle="round,pad=0,rounding_size=0.08",
                                    linewidth=0.9, edgecolor=ec, facecolor=st["fc"], zorder=2))
        ax.text(x + w / 2, y0 + hh * 0.66, lines[0], ha="center", va="center",
                fontsize=10.0, color=st["tc"], fontweight="500", zorder=3)
        ax.text(x + w / 2, y0 + hh * 0.27, lines[1], ha="center", va="center",
                fontsize=9.0, color=MUTE, zorder=3)

    ax.plot([0.42, 2.98], [6.56, 6.56], color=C_AR, lw=1.1)
    label(ax, 1.70, 6.38, "AR 子序列：走推理塔，因果注意力", fs=9.8, color=C_AR)
    ax.plot([3.62, 10.10], [6.56, 6.56], color=C_DM, lw=1.1)
    label(ax, 6.86, 6.38, "DM 子序列：走生成塔，双向注意力；干净条件在前，带噪目标在后",
          fs=9.8, color=C_DM)

    # ---------------- 层的容器
    ax.add_patch(FancyBboxPatch((0.42, 1.34), 12.06, 4.68,
                                boxstyle="round,pad=0,rounding_size=0.20",
                                linewidth=1.0, edgecolor="#BBBBB6", facecolor="#FCFCFA", zorder=1))
    label(ax, 0.64, 5.72, "同一个 decoder layer　Nano 共 36 层，每层两套独立参数",
          fs=11.4, color=INK, ha="left", weight="500")

    yA, yB, bh = 4.42, 2.28, 0.72
    for y, tag, st, tint, name_out, sub_out in (
        (yA, "AR", BLUE, BLUE_D, "下一 token 预测", "词表 151,936"),
        (yB, "DM", PURPLE, PURPLE_D, "速度场 $v_\\theta$", "每模态独立 $\\sigma$"),
    ):
        box(ax, 0.62, y, 1.30, bh, [f"{tag} token", f"$h^{{\\rm {tag}}}$"], st, fs=10.0, fs_sub=9.2)
        box(ax, 2.16, y, 1.24, bh, ["LN + QKV", "本塔参数"], tint, fs=10.0, fs_sub=9.2)
        box(ax, 7.22, y, 1.60, bh, ["LN + MLP", "FFN 12,288"], tint, fs=10.0, fs_sub=9.2)
        box(ax, 9.12, y, 1.86, bh, [name_out, sub_out], GRAY, fs=10.0, fs_sub=9.2)
        arrow(ax, (1.92, y + bh / 2), (2.16, y + bh / 2), color=st["ec"])
        arrow(ax, (3.40, y + bh / 2), (3.66, y + bh / 2), color=st["ec"])
        arrow(ax, (6.96, y + bh / 2), (7.22, y + bh / 2), color=st["ec"])
        arrow(ax, (8.82, y + bh / 2), (9.12, y + bh / 2), color=st["ec"])

    # 共享注意力（跨两条泳道的大框）
    ax.add_patch(FancyBboxPatch((3.66, 1.94), 3.30, 3.40,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                linewidth=1.0, edgecolor="#A32D2D", facecolor="#FCEBEB", zorder=2))
    ax.text(5.31, 4.92, "共享多头注意力", ha="center", va="center",
            fontsize=11.0, color="#501313", fontweight="500", zorder=3)
    ax.text(5.31, 4.52, "Q/K/V 各塔自己投影", ha="center", va="center",
            fontsize=9.4, color=MUTE, zorder=3)
    ax.text(5.31, 3.86, "Q$_{\\rm AR}$ 只看 K$_{\\rm AR}$", ha="center", va="center",
            fontsize=10.2, color=C_AR, zorder=3)
    ax.text(5.31, 3.58, "（因果，token 只读前面的）", ha="center", va="center",
            fontsize=9.0, color=MUTE, zorder=3)
    ax.text(5.31, 3.04, "Q$_{\\rm DM}$ 看 [ K$_{\\rm AR}$ ; K$_{\\rm DM}$ ]", ha="center", va="center",
            fontsize=10.2, color=C_DM, zorder=3)
    ax.text(5.31, 2.76, "（全向，可读提示词与全部条件）", ha="center", va="center",
            fontsize=9.0, color=MUTE, zorder=3)
    ax.plot([4.30, 6.32], [3.34, 3.34], color="#DDD9CE", lw=0.9)

    # ---------------- mask 小图
    mx, my, cell = 11.28, 2.30, 0.20
    label(ax, mx + 3 * cell, 4.28, "注意力掩码", fs=9.8, color=INK, weight="500")
    for i in range(6):
        for j in range(6):
            if i < 3 and j < 3:
                fc = "#B5D4F4" if j <= i else "#FFFFFF"
                ec = "#185FA5"
            elif i < 3 and j >= 3:
                fc, ec = "#F1EFE8", "#BBBBB6"
            else:
                fc, ec = "#CECBF6", "#534AB7"
            ax.add_patch(plt.Rectangle((mx + j * cell, my + (5 - i) * cell), cell, cell,
                                       facecolor=fc, edgecolor=ec, lw=0.5, zorder=2))
    label(ax, mx + 1.5 * cell, my - 0.20, "AR 列", fs=8.4, color=C_AR)
    label(ax, mx + 4.5 * cell, my - 0.20, "DM 列", fs=8.4, color=C_DM)
    label(ax, mx - 0.12, my + 4.5 * cell, "AR", fs=8.4, color=C_AR, ha="right")
    label(ax, mx - 0.12, my + 1.5 * cell, "DM", fs=8.4, color=C_DM, ha="right")
    label(ax, mx + 3 * cell, my + 1.55, "", fs=8)
    label(ax, mx + 3 * cell, my - 0.52, "灰格：AR 从不读 DM", fs=8.6, color=HINT)

    # ---------------- 底部两笔账
    ax.add_patch(FancyBboxPatch((0.42, 0.34), 6.02, 0.76, boxstyle="round,pad=0,rounding_size=0.14",
                                linewidth=0.9, edgecolor="#0F6E56", facecolor="#E1F5EE", zorder=2))
    ax.text(0.64, 0.86, "单层 0.193B  x  36 层  =  单塔 6.946B",
            ha="left", va="center", fontsize=10.4, color="#04342C", fontweight="500", zorder=3)
    ax.text(0.64, 0.53, "两塔 13.892B + 共享词表 0.622B + lm_head 0.622B = 15.137B",
            ha="left", va="center", fontsize=9.4, color=MUTE, zorder=3)

    ax.add_patch(FancyBboxPatch((6.62, 0.34), 5.86, 0.76, boxstyle="round,pad=0,rounding_size=0.14",
                                linewidth=0.9, edgecolor="#854F0B", facecolor="#FAEEDA", zorder=2))
    ax.text(6.84, 0.86, "序列预算 74,000 token（训练与推理共用）",
            ha="left", va="center", fontsize=10.4, color="#412402", fontweight="500", zorder=3)
    ax.text(6.84, 0.53, "720p 一份就占 44,160，所以 720p 只能 B=1",
            ha="left", va="center", fontsize=9.4, color=MUTE, zorder=3)

    save(fig, "cosmos3_mot_architecture")


# ================================================================ 图 2 参数瀑布
def fig_param_waterfall():
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    steps = [
        ("单塔层参数", 6.946, "abs"),
        ("x 2 塔", 6.946, "abs"),
        ("共享词表", 0.622, "abs"),
        ("lm_head", 0.622, "abs"),
        ("ViT + VAE\n+ 动作投影", 0.863, "abs"),
    ]
    xs = np.arange(len(steps))
    bottom = 0.0
    for i, (name, v, _) in enumerate(steps):
        ax.bar(xs[i], v, bottom=bottom, width=0.62,
               color=["#185FA5", "#185FA5", "#534AB7", "#534AB7", "#0F6E56"][i],
               alpha=0.85, edgecolor="white", linewidth=0.8)
        if i > 0:
            ax.plot([xs[i - 1] + 0.31, xs[i] - 0.31], [bottom, bottom],
                    color="#BBBBB6", lw=0.9, linestyle=(0, (3, 2)), zorder=1)
        ax.text(xs[i], bottom + v / 2, f"{v:.3f}", ha="center", va="center",
                fontsize=10.5, color="white", fontweight="500")
        bottom += v

    ax.axhline(bottom, color="#A32D2D", lw=1.2, linestyle=(0, (5, 3)), zorder=4)
    ax.text(xs[-1] + 0.36, bottom + 0.38, f"合计 {bottom:.2f}B", ha="left", va="bottom",
            fontsize=10.8, color="#A32D2D", fontweight="500")
    ax.text(xs[-1] + 0.36, bottom - 0.42, "报告口径 16B", ha="left", va="top",
            fontsize=9.6, color=HINT)

    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps], fontsize=10.2)
    ax.set_ylabel("参数量（十亿）", fontsize=10.8)
    ax.set_ylim(0, 18.2)
    ax.set_xlim(-0.6, 5.55)
    clean_axes(ax)
    ax.text(-0.55, 17.5, "Nano 的 16B 由五个可加项构成；两个塔各占 6.946B",
            fontsize=10.4, color=INK, ha="left")
    fig.tight_layout()
    save(fig, "cosmos3_param_waterfall")


# ================================================================ 图 3 token 预算
def fig_token_budget():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    data = [
        ("256p  320x192", 2880, 25, 6),
        ("480p  832x480", 18720, 3, 3),
        ("720p  1280x720", 44160, 1, 1),
    ]
    ys = np.arange(len(data))[::-1]
    for y, (name, tok, b_calc, b_doc) in zip(ys, data):
        ax.barh(y, tok, height=0.42, color="#B5D4F4", edgecolor="#185FA5", linewidth=0.9)
        ax.text(tok * 1.08, y + 0.10, f"单样本 {tok:,} token", va="center",
                fontsize=10.0, color="#042C53")
        ax.text(tok * 1.08, y - 0.18,
                f"算得 B={b_calc}，报告 B={b_doc}" + ("（对不上）" if b_calc != b_doc else "（一致）"),
                va="center", fontsize=9.2,
                color=("#A32D2D" if b_calc != b_doc else "#0F6E56"))

    ax.axvline(74000, color="#A32D2D", lw=1.3, linestyle=(0, (5, 3)))
    ax.text(80000, 2.60, "74,000 上下文预算", ha="left", va="center",
            fontsize=10.2, color="#A32D2D")
    ax.set_xscale("log")
    ax.set_xlim(1500, 900000)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=10.6)
    ax.set_xlabel("单个 189 帧样本的扩散 token 数（对数轴）", fontsize=10.6)
    ax.set_ylim(-0.55, 2.82)
    clean_axes(ax)
    ax.set_title("上下文预算只在长序列一侧紧：480p 与 720p 严丝合缝，256p 远未用满",
                 fontsize=11.4, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, "cosmos3_token_budget")


# ================================================================ 图 4 FPS 位置对齐
def fig_fps():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    # (a) 帧索引 -> 位置：三种帧率斜率不同
    ax = axes[0]
    n = np.linspace(0, 40, 200)
    for fps, color, ls in ((16, "#534AB7", (0, (5, 2))), (24, "#185FA5", "-"),
                           (30, "#0F6E56", (0, (1, 1.6)))):
        ax.plot(n, n * (24.0 / fps), color=color, lw=1.9, linestyle=ls, label=f"{fps} FPS")
    ax.text(40.6, 40 * 1.5, "1.5", color="#534AB7", fontsize=9.6, va="center")
    ax.text(40.6, 40 * 1.0, "1.0", color="#185FA5", fontsize=9.6, va="center")
    ax.text(40.6, 40 * 0.8, "0.8", color="#0F6E56", fontsize=9.6, va="center")
    ax.set_xlabel("latent 帧序号", fontsize=10.4)
    ax.set_ylabel("时间坐标累计", fontsize=10.4)
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 66)
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=9.6, loc="upper left")
    ax.set_title("(a) 同样 40 帧，位置跨度不同：步长 = 24 / FPS", fontsize=10.4,
                 color=INK, loc="left")

    # (b) 真实时间 -> 位置：三种帧率重合
    ax = axes[1]
    t = np.linspace(0, 6, 200)
    ax.plot(t, 6 * t, color="#A32D2D", lw=3.2, alpha=0.35, solid_capstyle="round")
    for fps, color, ls in ((16, "#534AB7", (0, (5, 2))), (24, "#185FA5", "-"),
                           (30, "#0F6E56", (0, (1, 1.6)))):
        ax.plot(t, t * fps / 4.0 * (24.0 / fps), color=color, lw=1.4, linestyle=ls)
    ax.plot([5.0], [30.0], "o", color="#A32D2D", ms=7, zorder=5)
    ax.annotate("5 秒 -> 30\n三条线完全重合", xy=(5.0, 30.0), xytext=(2.0, 8.0),
                fontsize=10.0, color="#A32D2D",
                arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=1.1))
    ax.set_xlabel("真实时间（秒）", fontsize=10.4)
    ax.set_ylabel("时间坐标累计", fontsize=10.4)
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 40)
    clean_axes(ax)
    ax.set_title("(b) 换成真实时间，三种帧率落到同一条直线", fontsize=10.4,
                 color=INK, loc="left")

    fig.tight_layout()
    save(fig, "cosmos3_fps_alignment")


# ================================================================ 图 5 迁移
def fig_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax = axes[0]
    it = np.array([500, 1000, 1500, 2000])
    pt = np.array([0.0, 73.8, 93.4, 95.2])
    mt = np.array([24.6, 91.4, 95.8, 97.4])
    ax.plot(it, pt, "o-", color="#888780", lw=1.9, ms=6, label="从预训练初始化")
    ax.plot(it, mt, "o-", color="#0F6E56", lw=1.9, ms=6, label="从 mid-training 初始化")
    ax.annotate("500 步：0.0% vs 24.6%", xy=(500, 24.6), xytext=(700, 40),
                fontsize=10.2, color="#A32D2D",
                arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=1.1))
    ax.scatter([500], [0.0], s=46, facecolor="white", edgecolor="#888780", zorder=5)
    ax.set_xlabel("LIBERO-10 后训练迭代", fontsize=10.6)
    ax.set_ylabel("闭环成功率（%）", fontsize=10.6)
    ax.set_ylim(-4, 104)
    ax.set_xlim(380, 2120)
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=9.8, loc="lower right")
    ax.set_title("(a) 冷启动差 24.6 个百分点", fontsize=10.8, color=INK, loc="left")

    ax = axes[1]
    names = ["自驾\nRRE", "自驾\nATE", "相机\nATE", "第一人称\nPSNR", "机器人\nPSNR"]
    rel = [15.26, 18.33, 22.98, 5.91, 9.81]
    colors = ["#185FA5", "#185FA5", "#185FA5", "#0F6E56", "#0F6E56"]
    bars = ax.bar(np.arange(5), rel, color=colors, alpha=0.85, width=0.58)
    for i, v in enumerate(rel):
        ax.text(i, v + 0.7, f"{v:.1f}", ha="center", fontsize=10.2, color=INK)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(names, fontsize=9.6)
    ax.set_ylabel("相对改善（%）", fontsize=10.6)
    ax.set_ylim(0, 28)
    clean_axes(ax)
    ax.set_title("(b) 四个下游域、五项指标的一致改善", fontsize=10.8, color=INK, loc="left")
    ax.text(-0.45, 25.4, "蓝：误差类指标下降   绿：PSNR 上升", fontsize=9.4, color=HINT)

    fig.tight_layout()
    save(fig, "cosmos3_transfer")


# ================================================================ 图 6 协同矩阵
def fig_synergy():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    names = ["自驾", "相机", "GR", "WX", "F1", "F2"]
    fd = np.array([
        [0.00, -0.05, -0.16, -0.19, 0.06, -0.05],
        [0.86, 0.00, 0.79, 0.19, 0.73, 0.48],
        [0.91, 0.94, 0.00, 0.39, 0.81, 0.89],
        [0.90, 0.45, 1.39, 0.00, 0.78, 0.78],
        [0.00, -0.01, 0.79, -0.02, 0.00, 0.58],
        [0.00, -0.05, 0.51, -0.03, 0.15, 0.00],
    ])
    idd = np.array([
        [0.000, -0.010, -0.011, -0.010, -0.010, -0.008],
        [-0.012, 0.000, -0.043, -0.043, -0.047, -0.032],
        [-0.005, -0.005, 0.000, -0.004, -0.005, -0.007],
        [-0.005, -0.003, -0.006, 0.000, -0.005, -0.006],
        [-0.003, -0.003, -0.004, -0.003, 0.000, -0.004],
        [-0.001, -0.001, -0.001, -0.001, -0.001, 0.000],
    ])

    for ax, M, title, unit, fmt in (
        (axes[0], fd, "(a) 前向动力学：PSNR 变化（dB）", "dB", "{:+.2f}"),
        (axes[1], idd, "(b) 逆动力学：MSE 变化", "", "{:+.3f}"),
    ):
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(names, fontsize=9.6)
        ax.set_yticklabels(names, fontsize=9.6)
        for i in range(6):
            for j in range(6):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor="#E8E6DE", edgecolor="white", lw=1.2))
                    continue
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=8.6, color=("white" if abs(M[i, j]) > 0.6 * vmax else INK))
        ax.set_title(title, fontsize=10.6, color=INK, loc="left")
        ax.set_xlabel("加入的辅助域", fontsize=10.0)
        ax.set_ylabel("目标域", fontsize=10.0)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.text(0.012, 0.965, "行 = 目标域，列 = 混入 50% 的辅助域；灰格是单域基线",
             fontsize=10.4, color=INK)
    fig.text(0.012, 0.918,
             "自驾那一行几乎全为负（行均值 -0.078），但自驾那一列是强辅助（平均 +0.53 dB）",
             fontsize=9.8, color=MUTE)
    fig.text(0.012, 0.878,
             "缩写：GR = Google Robot，WX = WidowX，F1 / F2 = 两类第一人称数据",
             fontsize=9.4, color=HINT)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    save(fig, "cosmos3_action_synergy")


# ================================================================ 图 7 shift 曲线
def fig_shift():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    t = np.linspace(0, 1, 400)
    for sh, color, ls in ((1.0, "#185FA5", "-"), (3.0, "#534AB7", (0, (5, 2))),
                          (5.0, "#0F6E56", (0, (3, 2))), (10.0, "#A32D2D", (0, (1, 1.6)))):
        sig = sh * (1 - t) / (1 + (sh - 1) * (1 - t))
        ax.plot(t, sig, color=color, lw=1.9, linestyle=ls, label=f"s = {sh:g}")

    ax.axvline(0.75, color="#BBBBB6", lw=0.9, linestyle=(0, (3, 2)))
    for sh, color in ((1.0, "#185FA5"), (5.0, "#0F6E56")):
        sig = sh * 0.25 / (1 + (sh - 1) * 0.25)
        ax.plot([0.75], [sig], "o", color=color, ms=6, zorder=5)
    ax.annotate("t=0.75：0.250 → 0.625\n（同一个 t，噪声水平差 2.5 倍）",
                xy=(0.75, 0.625), xytext=(0.40, 0.20), fontsize=10.0, color="#A32D2D",
                arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=1.1))

    ax.set_xlabel("插值系数 t（0 = 纯噪声，1 = 干净样本）", fontsize=10.6)
    ax.set_ylabel("实际噪声水平 sigma", fontsize=10.6)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=10.0, loc="upper right")
    ax.set_title("shift 重参数化把采样预算挪向高噪声段：sigma 被抬高，但不是等比缩放",
                 fontsize=11.2, color=INK, loc="left", pad=10)
    fig.tight_layout()
    save(fig, "cosmos3_shift")


if __name__ == "__main__":
    fig_arch()
    fig_param_waterfall()
    fig_token_budget()
    fig_fps()
    fig_transfer()
    fig_synergy()
    fig_shift()
    print("figures written to", OUT)
