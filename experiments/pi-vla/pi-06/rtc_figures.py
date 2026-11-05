# -*- coding: utf-8 -*-
"""06 篇（RTC）的配图：3 张结构图 + 4 张数据图，全部手工排版成 SVG。

与 01–05 篇一致：字体 PingFang SC，无 Mermaid、无论文截图。
RTC 虽是 CC BY 4.0，本系列仍统一自画，结构图按论文 §2–§3 / Alg.1
与附录 A.2 / A.4 的文字重画。数据图的数字逐位来自同目录
`results/stdout.txt`（与 rtc_lab.py 的输出对齐）。

画完必须渲染 PNG 肉眼过一遍：`bbox_inches="tight"` 会把越界元素悄悄撑大画布。

运行：~/Software/miniconda3/bin/python rtc_figures.py    约 20 秒
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from series_common import soft_mask  # noqa: E402

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
TEAL_D = dict(fc="#BCE3D3", ec="#0F6E56", tc="#04342C")
RED = dict(fc="#FCEBEB", ec="#A32D2D", tc="#501313")
AMBER = dict(fc="#FAEEDA", ec="#854F0B", tc="#412402")
GRAY = dict(fc="#F8F7F4", ec="#888780", tc="#2C2C2A")
GRAY_D = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")

INK = "#2C2C2A"
MUTE = "#5F5E5A"
HINT = "#888780"
C_NAIVE = "#A32D2D"
C_HARD = "#854F0B"
C_EXP = "#185FA5"
C_LIN = "#0F6E56"
C_TE = "#534AB7"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.0, lw=0.9, r=0.12, dashed=False,
        align="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(r, max(min(w, h) / 2 - 0.01, 0.01))}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=2,
        **({"linestyle": (0, (4, 2.5))} if dashed else {})))
    if not lines:
        return
    n = len(lines)
    for k, line in enumerate(lines):
        yy = y + h * (n - k - 0.5) / n
        xx = x + w / 2 if align == "center" else x + 0.16
        ax.text(xx, yy, line, ha=align, va="center", fontsize=fs,
                fontweight="500", color=style["tc"], zorder=3)


def arrow(ax, p0, p1, color=MUTE, lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, linestyle=ls, zorder=4,
        connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, txt, fs=10.5, color=INK, ha="left", va="center", weight="400"):
    ax.text(x, y, txt, fontsize=fs, color=color, ha=ha, va=va,
            fontweight=weight, zorder=5)


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), format=ext, dpi=145,
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.svg / .png")


def std_axes(ax, xlabel, ylabel, title=None, grid=True):
    ax.set_xlabel(xlabel, fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    if title:
        ax.set_title(title, fontsize=12.5, color=INK, pad=12, fontweight="600")
    if grid:
        ax.grid(axis="y", color="#E8E6E0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9C6BE")
    ax.tick_params(colors=MUTE, labelsize=10)


# ------------------------------------------------------------------ 图 1
def fig_timeline():
    """三种执行范式的时刻轴：同步 / 朴素异步 / RTC（三段式补全）。"""
    fig, ax = canvas(13.0, 7.0)
    label(ax, 0.5, 6.55, "推理延迟 d 下的三种执行范式", fs=15, weight="600")
    label(ax, 0.5, 6.18, "H=50，s=25，Δt=20 ms；一个动作块 1 秒，只执行前 s 步再切下一个块",
          fs=10.5, color=MUTE)

    x0, xw = 2.75, 9.35
    cut = x0 + 4.20              # 切换点：旧块已执行 s 步

    def rect(x, y, w, h, fc, ec, text="", tc=INK, fs=9.5, hatch=None):
        ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=fc,
                               edgecolor=ec, linewidth=1.0, zorder=2,
                               hatch=hatch))
        if text:
            label(ax, x + w / 2, y, text, fs=fs, color=tc, ha="center")

    # ---- 行 1：synchronous
    y = 5.05
    box(ax, 0.4, y - 0.30, 2.15, 0.60, ["Synchronous"], GRAY, fs=10.2)
    rect(x0, y, 4.20, 0.50, "#DCE9F7", "#185FA5",
         "执行 chunk A₀ 的前 s=25 步", tc="#042C53")
    rect(cut, y, 1.65, 0.50, "#F7D7D7", "#A32D2D",
         "停顿 d 步", tc="#501313", fs=9.2)
    rect(cut + 1.65, y, 3.50, 0.50, "#E6F1FB", "#185FA5",
         "新 chunk A₁ 开始执行", tc="#042C53")
    label(ax, cut, y - 0.40, "s", fs=9.5, color=MUTE, ha="center")
    label(ax, cut + 0.82, y + 0.42, "等推理算完", fs=9.3, color="#A32D2D", ha="center")
    label(ax, x0 + 2.10, y + 0.42, "Δt 连续执行", fs=9.3, color=MUTE, ha="center")

    # ---- 行 2：naive async
    y = 3.35
    box(ax, 0.4, y - 0.30, 2.15, 0.60, ["Naive async"], GRAY, fs=10.2)
    rect(x0, y, 4.20, 0.50, "#DCE9F7", "#185FA5",
         "执行 A₀ 的前 s−d 步 … 再按 A₀ 走完 d 步", tc="#042C53")
    rect(cut, y, 5.15, 0.50, "#FCEBEB", "#A32D2D",
         "新 chunk A₁ 就绪即整体替换", tc="#501313")
    arrow(ax, (cut - 1.85, y + 0.40), (cut, y + 0.40), color="#854F0B", lw=1.4)
    label(ax, cut - 0.92, y + 0.56, "在 s−d 启动推理，耗时 δ", fs=9.2,
          color="#854F0B", ha="center")
    # 切换点断裂标记
    ax.plot([cut, cut], [y - 0.34, y + 0.34], color="#A32D2D", linewidth=2.0, zorder=5)
    ax.plot([cut - 0.09, cut + 0.09], [y + 0.34, y - 0.34],
            color="#FFFFFF", linewidth=2.4, zorder=6)
    label(ax, cut + 0.20, y - 0.43, "切换点：可能跳到另一个策略模式（OOD 加速度）",
          fs=9.2, color="#A32D2D")

    # ---- 行 3：RTC
    y = 1.55
    box(ax, 0.4, y - 0.30, 2.15, 0.60, ["RTC (inpainting)"], TEAL_D, fs=10.2)
    rect(x0, y, 4.20, 0.50, "#DCE9F7", "#185FA5",
         "执行 A₀ 的前 s 步（含推理期间）", tc="#042C53")
    # 新 chunk 三段
    rect(cut, y, 1.05, 0.50, "#FAEEDA", "#854F0B", "冻结 d", tc="#412402", fs=9.2)
    # 软引导：5 个逐渐变浅的小矩形表示权重递减
    guide_x = cut + 1.05
    for k in range(5):
        alpha = 1.0 - 0.16 * k
        ax.add_patch(Rectangle((guide_x + 0.46 * k, y - 0.25), 0.46, 0.50,
                               facecolor=(0.71 + 0.05 * k, 0.83 + 0.03 * k,
                                          0.96 - 0.02 * k, alpha),
                               edgecolor="#185FA5", linewidth=0.8, zorder=2))
    label(ax, guide_x + 1.15, y, "软引导重叠区", fs=9.2, color="#042C53", ha="center")
    fresh_x = guide_x + 2.30
    rect(fresh_x, y, 1.80, 0.50, "#E1F5EE", "#0F6E56",
         "全新生成 s 步", tc="#04342C", fs=9.2)
    label(ax, cut + 0.52, y + 0.42, "W(i)=1", fs=9.2, color="#854F0B", ha="center")
    label(ax, guide_x + 1.15, y + 0.42, "W(i) 指数衰减 1→0", fs=9.2,
          color="#185FA5", ha="center")
    label(ax, fresh_x + 0.90, y + 0.42, "W(i)=0", fs=9.2,
          color="#0F6E56", ha="center")
    label(ax, cut, y - 0.40, "s", fs=9.5, color=MUTE, ha="center")
    label(ax, x0 + 2.10, y + 0.42, "推理在后台线程进行", fs=9.2,
          color=MUTE, ha="center")

    label(ax, 0.5, 0.42,
          "同步用停顿换连续性；朴素异步用连续性换 OOD 跳变；"
          "RTC 把「已经过去」和「仍然重叠」的两段写成约束，只让模型补真正没见过的部分",
          fs=10.0, color=MUTE)
    save(fig, "rtc_timeline")


# ------------------------------------------------------------------ 图 2
def fig_mask():
    """Eq.(5) 软掩码三段式：曲线 + 语义色带。"""
    H, S, d = 50, 25, 11
    W_exp = soft_mask(H, S, d, "exp")
    W_lin = soft_mask(H, S, d, "linear")
    W_hard = soft_mask(H, S, d, "hard")

    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.42,
                          left=0.09, right=0.97, top=0.88, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    i = np.arange(H)
    ax.fill_between([d, H - S], [1.05, 1.05], color="#FAEEDA", alpha=0.55, zorder=0)
    ax.fill_between([H - S, H], [1.05, 1.05], color="#F1EFE8", alpha=0.8, zorder=0)
    ax.plot(i, W_exp, color=C_EXP, linewidth=2.6, label="指数衰减（论文 Eq.5）", zorder=3)
    ax.plot(i, W_lin, color=C_LIN, linewidth=2.0, linestyle=(0, (5, 3)),
            label="线性衰减（附录 A.4 消融）", zorder=3)
    ax.plot(i, W_hard, color=C_HARD, linewidth=1.8, linestyle=(0, (2, 2)),
            label="hard masking（只冻结）", zorder=3)
    ax.axvline(d - 0.5, color=MUTE, linewidth=1.0, linestyle=":")
    ax.axvline(H - S - 0.5, color=MUTE, linewidth=1.0, linestyle=":")
    ax.set_xlim(0, H - 1)
    ax.set_ylim(-0.05, 1.08)
    std_axes(ax, "新 chunk 内的动作索引 i", "guidance 权重 W(i)")
    ax.set_title("Eq.(5)：前 d 个冻结、中间指数衰减、末 s 个全新生成",
                 fontsize=13, color=INK, pad=12, fontweight="600")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.text(4.0, 0.60, "冻结区\nW(i)=1", fontsize=10, color="#854F0B", ha="center")
    ax.text(18.5, 0.86, "软引导区\n0.8381 → 0.0027", fontsize=10,
            color="#185FA5", ha="center")
    ax.text(37.5, 0.60, "全新生成区\nW(i)=0", fontsize=10, color=MUTE, ha="center")

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    segs = [
        (0.00, 0.205, "A_cur 已执行 / 冻结到已知值", AMBER),
        (0.205, 0.295, "与前一 chunk 重叠：软引导", BLUE),
        (0.50, 0.50, "超出前一 chunk：必须自己生成", GRAY),
    ]
    for x, w, txt, st in segs:
        box(ax2, x, 0.18, w - 0.006, 0.56, [txt], st, fs=9.6)
    label(ax2, 0.0, 0.92,
          "Alg.1：A_prev = A_cur[s..H−1] 右填充到 H；W 调节对每个重叠动作的关注度",
          fs=9.8, color=MUTE)
    save(fig, "rtc_mask")


# ------------------------------------------------------------------ 图 3
def fig_guidance():
    """ΠGDM guidance 与 β 裁剪：公式框 + n=5 权重序列。"""
    fig, ax = canvas(12.4, 6.3)
    label(ax, 0.5, 5.95, "ΠGDM guidance：把「补全」写进每一步去噪", fs=15, weight="600")

    box(ax, 0.5, 3.35, 6.0, 2.25, [
        "v_ΠGDM = v(A(τ), o, τ) + min(β, (1−τ)/(τ r²(τ))) · g",
        "",
        "A_hat = A(τ) + (1−τ)·v(A(τ), o, τ)                 (Eq.3)",
        "r²(τ) = (1−τ)² / (τ² + (1−τ)²)                     (Eq.4)",
        "g = (A_prev − A_hat) diag(W) · ∂A_hat/∂A(τ)     (VJP)",
    ], BLUE, fs=10.6, align="left")

    box(ax, 6.85, 3.35, 5.05, 2.25, [
        "τ=0 时 (1−τ)/(τ r²(τ)) → ∞，必须裁剪到 β",
        "去噪步数 n=5（控制任务很少）",
        "β=5 是保守值：仿真消融中 β>5 无边际收益",
        "代价：每个去噪步要一次反向传播",
    ], AMBER, fs=10.4, align="left")

    # 权重序列
    xs = np.array([1.15, 3.15, 5.15, 7.15, 9.15])
    ws = np.array([5.0, 4.25, 2.1667, 2.1667, 4.25])
    y0, hmax = 0.75, 1.75
    for k, (x, w) in enumerate(zip(xs, ws)):
        hh = hmax * w / 5.0
        st = RED if k == 0 else BLUE_D
        ax.add_patch(Rectangle((x, y0), 1.05, hh, facecolor=st["fc"],
                               edgecolor=st["ec"], linewidth=1.1, zorder=3))
        label(ax, x + 0.525, y0 + hh + 0.13, f"{w:.4g}", fs=10.2,
              color=st["tc"], ha="center", weight="600")
        label(ax, x + 0.525, y0 - 0.20, f"τ={k/5:.1f}", fs=9.6,
              color=MUTE, ha="center")
    ax.plot([0.95, 10.5], [y0, y0], color=MUTE, linewidth=1.0)
    label(ax, 0.5, 3.02, "n=5、β=5 的实际 guidance 权重序列", fs=11.2, weight="600")
    label(ax, 0.5, 2.80,
          "红色柱是唯一被 β 裁剪的一步；τ=0.2 的解析值恰为 4.25",
          fs=9.8, color="#501313")
    label(ax, 0.5, 0.22,
          "软掩码 W 决定「补哪里」，裁剪后的权重决定「补多狠」；两者都不需要改训练配方",
          fs=10.0, color=MUTE)
    save(fig, "rtc_guidance")


# ------------------------------------------------------------------ 图 4
def fig_jerk():
    """B 组：切换点最大加速度 vs 延迟。"""
    ds = np.array([0, 6, 11, 16])
    series = {
        "naive async": (C_NAIVE, [0.2592, 0.9149, 0.6647, 0.6425], "-o"),
        "hard masking": (C_HARD, [0.2851, 0.2907, 0.4860, 0.4548], "-s"),
        "soft (exp)": (C_EXP, [0.2002, 0.2127, 0.1838, 0.1966], "-o"),
        "soft (linear)": (C_LIN, [0.1997, 0.2148, 0.1938, 0.1746], "--^"),
        "TE": (C_TE, [0.6479, 1.2929, 0.6865, 0.5587], "-D"),
    }
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for name, (c, ys, mk) in series.items():
        ax.plot(ds, ys, mk, color=c, linewidth=2.1 if "soft (exp)" in name else 1.5,
                markersize=6.5, label=name, zorder=3)
    ax.axhline(0.42, color="#A32D2D", linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(1.2, 0.465, "成功判定阈值 0.42", fontsize=9.4, color="#A32D2D")
    std_axes(ax, "inference delay d（controller steps）", "切换点最大加速度（OOD 代理）",
             "注入延迟把 naive 抬到 0.64–0.91；soft masking 稳定在 0.17–0.21")
    ax.set_xticks(ds)
    ax.set_ylim(0, 1.42)
    ax.legend(frameon=False, fontsize=9.8, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "rtc_jerk_vs_delay")


# ------------------------------------------------------------------ 图 5
def fig_success():
    """B/D 组：成功率与 throughput。"""
    ds = np.array([0, 6, 11, 16])
    succ = {
        "soft (exp)": (C_EXP, [1.000, 0.988, 0.988, 0.988]),
        "hard": (C_HARD, [0.825, 0.863, 0.338, 0.475]),
        "naive": (C_NAIVE, [0.938, 0.000, 0.438, 0.487]),
        "TE": (C_TE, [0.000, 0.000, 0.000, 0.000]),
    }
    tp_sync = [0.9375, 0.7560, 0.6510, 0.5716]
    tp_rtc = [1.0000, 0.9875, 0.9875, 0.9875]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    a = axes[0]
    for name, (c, ys) in succ.items():
        a.plot(ds, ys, "-o", color=c, linewidth=2.0, markersize=6,
               label=name, zorder=3)
    std_axes(a, "inference delay d", "成功率（加速度 + 幅值同时达标）",
             "soft 对注入延迟几乎不退化")
    a.set_xticks(ds)
    a.set_ylim(-0.05, 1.10)
    a.legend(frameon=False, fontsize=9.5, loc="lower left")

    b = axes[1]
    b.plot(ds, tp_sync, "-s", color=C_NAIVE, linewidth=2.1, markersize=6,
           label="synchronous（有推理暂停）", zorder=3)
    b.plot(ds, tp_rtc, "-o", color=C_EXP, linewidth=2.1, markersize=6,
           label="RTC（异步，无暂停）", zorder=3)
    b.fill_between(ds, tp_rtc, tp_sync, color="#FCEBEB", alpha=0.65, zorder=1)
    std_axes(b, "inference delay d", "throughput（归一化）",
             "sync 的退化完全来自 s/(s+d)")
    b.set_xticks(ds)
    b.set_ylim(0.5, 1.08)
    b.legend(frameon=False, fontsize=9.5, loc="lower left")
    b.text(11.0, 0.83, "d=16：0.5716 vs 0.9875", fontsize=9.6, color=MUTE)
    fig.tight_layout()
    save(fig, "rtc_success_throughput")


# ------------------------------------------------------------------ 图 6
def fig_beta():
    """C 组：β 裁剪与去噪步数。"""
    betas = np.array([5, 20, 50, 150])
    gain5 = [0.10, 3.10, 9.10, 29.10]
    gain100 = [0.9450, 0.7950, 0.4950, 0.5050]
    jerk5 = [0.2647, 1.3172, 3.6284, 11.3321]
    norm5 = [0.5624, 2.1333, 5.2888, 15.8118]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    a = axes[0]
    x = np.arange(len(betas))
    a.bar(x - 0.19, gain5, width=0.36, color="#FCEBEB", edgecolor=C_NAIVE,
          linewidth=1.1, label="n=5（论文设定）", zorder=3)
    a.bar(x + 0.19, gain100, width=0.36, color="#E6F1FB", edgecolor=C_EXP,
          linewidth=1.1, label="n=100", zorder=3)
    a.axhline(1.0, color="#A32D2D", linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    a.set_yscale("log")
    a.set_xticks(x, [str(b) for b in betas])
    for k, v in enumerate(gain5):
        a.text(k - 0.19, v * 1.12, f"{v:.2f}", ha="center", fontsize=9.2,
               color="#501313")
    std_axes(a, "β（guidance 权重上界）", "第一步 |gain|（>1 即超调发散）",
             "β=5 是 n=5 下唯一不超调的设定")
    a.legend(frameon=False, fontsize=9.5)

    b = axes[1]
    b.plot(betas, jerk5, "-o", color=C_NAIVE, linewidth=2.1, markersize=6.5,
           label="首步最大加速度", zorder=3)
    b.plot(betas, norm5, "-s", color=C_HARD, linewidth=2.1, markersize=6.5,
           label="首步终态范数", zorder=3)
    b.annotate(f"{jerk5[0]:.4f}", (betas[0], jerk5[0]), xytext=(14, 6),
               textcoords="offset points", ha="left", fontsize=9.0,
               color="#501313")
    k = len(betas) - 1
    b.annotate(f"{jerk5[k]:.4f}", (betas[k], jerk5[k]), xytext=(0, 10),
               textcoords="offset points", ha="center", fontsize=9.0,
               color="#501313")
    b.annotate(f"{norm5[k]:.4f}", (betas[k], norm5[k]), xytext=(0, -16),
               textcoords="offset points", ha="center", fontsize=9.0,
               color="#854F0B")
    b.set_ylim(0, 17.5)
    std_axes(b, "β", "发散程度（n=5）", "β 越大，chunk 越抖（Fig.7 的 jerkiness）")
    b.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, "rtc_beta")


# ------------------------------------------------------------------ 图 7
def fig_boundary():
    """D 组：execution horizon 与「guidance 抑制合法换 mode」的代价。"""
    ss = np.array([10, 15, 20, 25])
    soft_j = [0.1740, 0.2056, 0.2324, 0.2127]
    naive_j = [0.4581, 0.3694, 1.2532, 0.9149]
    soft_s = [1.000, 0.988, 0.975, 0.988]
    names = ["naive", "hard", "soft (exp)"]
    rmses = [0.1564, 0.3925, 1.1159]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    a = axes[0]
    a.plot(ss, soft_j, "-o", color=C_EXP, linewidth=2.2, markersize=6.5,
           label="RTC 切换点加速度", zorder=3)
    a.plot(ss, naive_j, "-s", color=C_NAIVE, linewidth=1.8, markersize=6,
           label="naive 切换点加速度", zorder=3)
    a2 = a.twinx()
    a2.plot(ss, soft_s, "--^", color=C_LIN, linewidth=1.5, markersize=5.5,
            label="RTC 成功率（右轴）", zorder=3)
    a2.set_ylim(0.90, 1.02)
    a2.set_ylabel("成功率", fontsize=10.5, color=C_LIN)
    a2.tick_params(colors=C_LIN, labelsize=9.5)
    a2.spines[["top", "left"]].set_visible(False)
    std_axes(a, "execution horizon s", "切换点最大加速度",
             "s 越小，重叠引导越多；压到最小也救不了 naive")
    a.set_xticks(ss)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9.2, loc="upper left")

    b = axes[1]
    cols = [C_NAIVE, C_HARD, C_EXP]
    bars = b.bar(names, rmses, color=["#FCEBEB", "#FAEEDA", "#E6F1FB"],
                 edgecolor=cols, linewidth=1.4, width=0.55, zorder=3)
    for bar, v in zip(bars, rmses):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f"{v:.4f}",
               ha="center", fontsize=10.2, color=INK, weight="600")
    std_axes(b, "新观测要求切换到另一个策略模式", "与理想动作块的 RMSE",
             "guidance 的代价：合法切换也会被压回去")
    b.set_ylim(0, 1.30)
    b.text(1.0, 0.72, "软掩码中间段越强，越难改主意", fontsize=9.6,
           color=MUTE, ha="center")
    fig.tight_layout()
    save(fig, "rtc_boundary")


if __name__ == "__main__":
    fig_timeline()
    fig_mask()
    fig_guidance()
    fig_jerk()
    fig_success()
    fig_beta()
    fig_boundary()
    print("done:", len([f for f in os.listdir(OUT) if f.endswith(".svg")]), "SVG")
