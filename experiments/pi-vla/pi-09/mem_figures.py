# -*- coding: utf-8 -*-
"""09 篇（MEM）配图：1 张结构图（按论文文字重画）+ 5 张数据图，全部手工排版成 SVG。

数据图的数字**逐位解析自同目录 results/stdout.txt 的 [MACHINE-BACKFILL-*] 块**
（与「正文回填表」同源），配图不硬编码任何合成数字。结构图标注的是论文写明的
常量（§III-A/§III-B/§III-C/§III-D：6 帧/1 s → 18 帧/54 s、token 数不增、
每 4 层 temporal attention、15 min、π0.6 = Gemma3-4B + 860M），并明确标注
「非论文原图」；不下载、不复用任何论文 Figure（MEM 虽为 CC BY 4.0，本系列统一自画）。

字体 PingFang SC；无 Mermaid。画完必须渲染 PNG 用 ViewImage 逐张肉眼过一遍：
bbox_inches="tight" 会把越界元素悄悄撑大画布。每张 SVG 保留固有 width/height。

运行：~/Software/miniconda3/bin/python mem_figures.py
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
STDOUT = os.path.join(HERE, "results", "stdout.txt")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

# 结构图配色（沿用 05–08 篇）
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
PT = 1.0 / 72.0

C_MAIN = "#185FA5"
C_RED = "#A32D2D"
C_TEAL = "#0F6E56"
C_AMBER = "#854F0B"
C_PURPLE = "#534AB7"


# ---------------------------------------------------------------- 回填解析
def load_backfill(path):
    data = {}
    inside = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[MACHINE-BACKFILL-BEGIN]":
                inside = True
                continue
            if line == "[MACHINE-BACKFILL-END]":
                inside = False
                continue
            if inside and line.startswith("@@"):
                key, val = line[2:].split("=", 1)
                data[key] = np.array([float(x) for x in val.split(",")])
    assert data, f"未从 {path} 解析到回填块"
    return data


D = load_backfill(STDOUT)
with open(STDOUT, encoding="utf-8") as _f:
    STDOUT_TEXT = _f.read()


def grab(pattern, cast=float, group=1):
    """从固化的 stdout 文本逐位抽取标注里的数字，不在配图里硬编码结果值。"""
    m = re.search(pattern, STDOUT_TEXT)
    assert m, f"stdout 中找不到：{pattern}"
    return cast(m.group(group))


def val(key, i=0):
    """取回填块里的单个标量，保证逐位来自 stdout.txt。"""
    return float(D[key][i])


# 标注里用到的世界常量，也逐位取自 stdout
A_B = grab(r"\[世界\] B=(\d+)", cast=int)
A_NPATCH = grab(r"n_patch=(\d+)", cast=int)
A_TAU2 = grab(r"τ²=([\d.]+)")
A_L = grab(r"L_draw=(\d+)", cast=int)
A_N_STAGE = grab(r"N_stage=(\d+)", cast=int)
A_RHO = grab(r"ρ=(\d+)", cast=int)
A_ALPHA_STAR = val("ALPHA_STAR")


# ---------------------------------------------------------------- 画图工具
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.0, fs_sub=9.6, lw=0.9, r=0.12,
        dashed=False):
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
    avail = h - 0.16
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


def container(ax, x, y, w, h, title, style, fs=12.0, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(0.25, min(w, h) / 2 - 0.01)}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=1))
    ax.text(x + 0.28, y + h - 0.30, title, ha="left", va="center",
            fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
    return y + h - 0.60


def vstack(ax, x, y_top, w, items, gap=0.16):
    cy = y_top
    for h, lines, st in items:
        cy -= h
        box(ax, x, cy, w, h, lines, st)
        cy -= gap
    return cy + gap


def arrow(ax, p1, p2, color=MUTE, lw=1.5, rad=0.0, dash=None, head="-|>", ms=15):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=head, mutation_scale=ms, linewidth=lw,
        color=color, zorder=3, connectionstyle=f"arc3,rad={rad}",
        **({"linestyle": dash} if dash else {})))


def label(ax, x, y, text, fs=10.5, color=HINT, ha="center", va="center",
          weight="400"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, zorder=4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax, fs=9.4):
    ax.tick_params(labelsize=fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBB6")
        ax.spines[s].set_linewidth(0.8)


def note(ax, text, xy, xytext, color, fs=8.8, ha="left", va="center"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=color,
                ha=ha, va=va, zorder=6,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor=color, linewidth=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                                shrinkA=4, shrinkB=3))


# ================================================================ 图 1（结构图）
def fig_architecture():
    W, H = 13.8, 10.4
    fig, ax = canvas(W, H)
    label(ax, W / 2, 10.08, "MEM：高层文本长时记忆 + 低层视频短时记忆，接进同一个 π0.6 VLA",
          fs=14, color=INK, weight="500")
    label(ax, W / 2, 9.74, "按论文 §III-A / §III-B / §III-C / §III-D 文字重画 · 标注为论文写明的常量 · 非论文原图",
          fs=9.6, color=HINT)

    # ---- 输入带
    for x, w, lines, st in [
        (0.40, 4.10, ["多相机图像流（最多 4 路，448×448 px）", r"$o_{t-T:t}$"], BLUE),
        (4.85, 3.90, ["任务目标 / 开放式指令", r"$g$"], AMBER),
        (9.10, 4.30, ["本体状态（连续 embedding）", r"$q_t$ → K 个 proprioceptive state token"], GRAY_D),
    ]:
        box(ax, x, 9.02, w, 0.60, lines, st, fs=10.2, fs_sub=9.2)
    arrow(ax, (2.45, 8.98), (2.45, 8.66), color=BLUE["ec"], lw=1.4)
    arrow(ax, (6.80, 8.98), (6.80, 8.66), color=AMBER["ec"], lw=1.4)
    arrow(ax, (11.25, 8.98), (11.25, 8.66), color=GRAY_D["ec"], lw=1.4)

    # ---- 左：低层 π_LL 与视频短时记忆
    lx, lw_, ly, lh = 0.40, 6.35, 3.35, 5.25
    top = container(ax, lx, ly, lw_, lh,
                    "① 低层 π_LL：视频短时记忆（§III-C / §III-D，数秒尺度）", BLUE)
    vstack(ax, lx + 0.22, top, lw_ - 0.44, [
        (0.66, [r"输入 $o_{t-K:t}$：K 帧观察各帧分别 patchify",
                "ViT 扩展成 video encoder，保持单图 VLM 性能"], BLUE_D),
        (0.86, ["每帧内双向 spatial attention",
                r"每第 4 层叠加「同 patch 跨 timestep 的 causal temporal attention」（因果 mask）"], BLUE),
        (0.86, ["上层丢弃过去 timestep 的 observation token",
                "只把当前 timestep 往后传 → 传入 backbone 的 token 数不增"], GRAY_D),
        (0.80, ["不引入新可学习参数：固定正弦 temporal position encoding e(t)",
                r"$e(0)=0$，K=1 时与单图 VLM 完全一致"], TEAL_D),
        (0.80, [r"复杂度 $O(Kn^2+nK^2)$ vs 全连接 $O(n^2K^2)$",
                "预训练 6 帧 / stride 1 s（≈5 s）→ 后训练最多 18 帧 / 54 s"], AMBER),
    ], gap=0.15)
    label(ax, lx + lw_ / 2, 3.08,
          r"输出：连续动作块 $a_{t:t+H}$（低层看短时观察 + 高层给的子任务 $l_{t+1}$）",
          fs=9.2, color=BLUE["ec"])

    # ---- 右：高层 π_HL 与文本长时记忆
    rx, rw, ry, rh = 7.05, 6.35, 3.35, 5.25
    top2 = container(ax, rx, ry, rw, rh,
                     "② 高层 π_HL：文本长时记忆（§III-A / §III-B，分钟尺度）", PURPLE)
    vstack(ax, rx + 0.22, top2, rw - 0.44, [
        (0.66, [r"输入：目标 $g$、语言记忆 $m_t$、当前观察 $o_t$",
                r"输出：子任务指令 $l_{t+1}$ 与更新记忆 $m_{t+1}$"], PURPLE_D),
        (0.86, [r"记忆自回归：$\pi_{HL}$ 基于自己上一步预测的 $m_t$ 预测 $m_{t+1}$",
                "模型自主决定何时、如何更新记忆"], PURPLE),
        (0.86, [r"训练标签：off-the-shelf LLM 把 $l_{0:t}$ + 成功/失败指示",
                "总结成对后续仍相关的信息；明确要求压缩去冗余"], GRAY_D),
        (0.80, ["例：不记「浅绿/深蓝/亮黄碗」，只记「三个碗放在右上柜」",
                r"失败的子任务不更新记忆（$m$ 保持不变）"], TEAL_D),
        (0.80, ["naive 对照：拼接所有历史子任务指令（不压缩）",
                "同一子任务反复失败、重复出现 → train-inference 分布漂移，显著更差"], RED),
    ], gap=0.15)
    label(ax, rx + rw / 2, 3.20,
          r"输出 $l_{t+1}$ 给低层 $\pi_{LL}$ ｜ 最长任务 15 分钟（引用）",
          fs=9.2, color=PURPLE["ec"])

    # ---- 高层 → 低层：子任务指令（标签压在容器间的窄缝里，只放短符号）
    arrow(ax, (rx, 5.55), (lx + lw_, 5.55), color=TEAL["ec"], lw=2.0)
    ax.text((lx + lw_ + rx) / 2, 5.84, r"$l_{t+1}$",
            ha="center", va="center", fontsize=10.0, color=TEAL["ec"],
            fontweight="500", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    # ---- 记忆自回归回灌（虚线 U 形贴着容器下沿，标签带白底压在虚线上）
    arrow(ax, (12.90, 3.35), (12.90, 2.80), color=PURPLE["ec"], lw=1.3,
          dash=(0, (4, 2)), head="-")
    arrow(ax, (12.90, 2.80), (7.60, 2.80), color=PURPLE["ec"], lw=1.3,
          dash=(0, (4, 2)), head="-")
    arrow(ax, (7.60, 2.80), (7.60, 3.35), color=PURPLE["ec"], lw=1.3,
          dash=(0, (4, 2)))
    ax.text(10.25, 2.80, r"记忆自回归：$m_{t+1}\to m_t$（虚线）",
            ha="center", va="center", fontsize=8.8, color=PURPLE["ec"], zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # ---- 底部：因式分解 + 引用常量带
    box(ax, 0.40, 1.60, 13.00, 1.05, [
        r"$\pi(a_{t:t+H},l_{t+1},m_{t+1}\mid o_{t-T:t},m_t,g)\approx"
        r"\pi_{LL}(a_{t:t+H}\mid o_{t-K:t},l_{t+1},g)\cdot"
        r"\pi_{HL}(l_{t+1},m_{t+1}\mid o_t,m_t,g),\quad K\ll T$",
        r"低层只看 K 帧短时观察 + 高层子任务；高层只看当前观察 + 语言记忆（§III-A）",
    ], GRAY, fs=11.4, fs_sub=9.6)
    box(ax, 0.40, 0.28, 13.00, 1.18, [
        "论文引用常量：基座 π0.6 = Gemma3-4B VLM + 860M action expert ｜ 每 4 层 temporal attention ｜ token 数不增 ｜ 最长 15 分钟",
        "论文无数值表（ltx_table=0），Fig.5/7/8/9 柱高数值未给出 ｜ 未提及 π0.7 ｜ MEM 虽为 CC BY 4.0，本系列统一自画、不复用论文 Figure",
    ], GRAY_D, fs=10.0, fs_sub=9.2)
    save(fig, "fig1_mem_architecture")


# ================================================================ 图 2（A 组 信息保持）
def fig_span_budget():
    K = D["K_GRID"].astype(int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.1))

    a1.plot(K, D["SEP_MSE"], "o--", color=C_MAIN, lw=1.7, ms=6,
            label="总重建 MSE（sep = full，完全重合）")
    a1.plot(K, D["SEP_MAXMSE"], "o-", color=C_TEAL, lw=2.2, ms=6,
            label="sep 最差帧 MSE")
    a1.plot(K, D["FULL_MAXMSE"], "s-", color=C_RED, lw=2.2, ms=6,
            label="full 最差帧 MSE")
    a1.axvline(A_B, color=C_AMBER, lw=1.6, ls="--")
    a1.axvspan(K[0] - 0.5, A_B + 0.5, color=C_TEAL, alpha=0.05)
    a1.set_xticks(K)
    a1.set_xlabel("帧数 K（固定 token 预算 B=12）")
    a1.set_ylabel("每状态重建 MSE（越低越好）")
    a1.set_ylim(-0.03, 1.10)
    a1.set_title("反直觉：总误差相同，差别在最差帧", fontsize=11, color=INK, pad=10)
    a1.grid(alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=8.8, loc="upper left")
    a1.text(8.4, 0.14, "K≤B：全部可恢复", fontsize=8.6, color=C_TEAL,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.4, alpha=0.92))
    a1.annotate(f"K=18：sep {val('SEP_MAXMSE', -1):.4f}\nfull {val('FULL_MAXMSE', -1):.4f}",
                xy=(18, val("FULL_MAXMSE", -1)), xytext=(12.2, 0.80),
                fontsize=8.8, color=C_RED,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_RED, lw=0.7, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))
    clean_axes(a1)

    labels = ["sep\n平均", "full\n平均", "sep\n最差帧", "full\n最差帧", "sep\n近端", "full\n近端"]
    keys = ["SEP_MEANACC", "FULL_MEANACC", "SEP_MINACC", "FULL_MINACC",
            "SEP_MAXACC", "FULL_MAXACC"]
    vals = [val(k, -1) for k in keys]
    cols = [C_TEAL, C_AMBER, C_TEAL, C_RED, C_TEAL, C_RED]
    x = np.arange(len(labels))
    bars = a2.bar(x, vals, 0.60, color=cols, edgecolor="white", linewidth=1.1)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=8.6, color=INK)
    a2.axhline(val("CFG_NONE"), color=C_RED, lw=1.5, ls="--")
    a2.text(5.45, 0.512, "随机 0.5", ha="right", fontsize=8.8, color=C_RED,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92))
    a2.set_xticks(x, labels, fontsize=9.0)
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("符号判别准确率（K=18）")
    a2.set_title("平均指标分不开，最差帧/近端才分开", fontsize=11, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    clean_axes(a2)
    fig.suptitle("合成探针 A 组：固定 token 预算下的信息保持（数字逐位来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_span_budget")


# ================================================================ 图 3（A 组 复杂度 / 跨度）
def fig_complexity_interval():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # 左：K=18 的两种注意力开销；同算力下的 K_max
    names = [f"全连接\nO(n²K²)", f"时空分离\nO(Kn²+nK²)"]
    costs = [val("COST_FULL18"), val("COST_SEP18")]
    bars = a1.bar(names, costs, 0.52, color=[C_RED, C_TEAL],
                  edgecolor="white", linewidth=1.1)
    for b, v in zip(bars, costs):
        a1.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:.0f}",
                ha="center", fontsize=9.4, color=INK)
    a1.set_yscale("log")
    a1.set_ylim(3000, 300000)
    a1.set_ylabel("attention 开销（对数轴，n=16 / K=18）")
    a1.set_title(f"K=18：省 {val('COST_RATIO'):.2f}× 算力", fontsize=11, color=INK, pad=10)
    a1.grid(axis="y", alpha=0.25, linewidth=0.7, which="both")
    a1.set_axisbelow(True)
    note(a1, f"同预算 C={val('C_BUDGET'):.0f}\nK_max {val('KMAX_FULL'):.0f} → {val('KMAX_SEP'):.0f} 帧",
         xy=(1, costs[1]), xytext=(0.15, 26000), color=C_TEAL, fs=8.8)
    clean_axes(a1)

    # 右：Δ 扫描 —— 跨度 ↑ 与可恢复事件比例 ↓
    delta = D["DELTA_GRID"].astype(int)
    span = D["SPAN_DELTA"]
    prec = D["PRECOVER"]
    prec_mc = D["PRECOVER_MC"]
    ax2 = a2.twinx()
    l1, = a2.plot(delta, span, "o-", color=C_MAIN, lw=2.2, ms=7, label="跨度 (K−1)Δ")
    l2, = ax2.plot(delta, prec, "s-", color=C_RED, lw=2.2, ms=7,
                   label="帧间事件可恢复比例 2w/Δ")
    ax2.plot(delta, prec_mc, "x", color=C_RED, ms=7, alpha=0.65)
    for d, s in zip(delta, span):
        a2.annotate(f"{s:.0f}s", (d, s), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9.0, color=C_MAIN)
    for d, p in zip(delta, prec):
        ax2.annotate(f"{p:.4f}", (d, p), xytext=(0, -16), textcoords="offset points",
                     ha="center", fontsize=9.0, color=C_RED)
    a2.set_xticks(delta, [f"{d} s" for d in delta])
    a2.set_xlabel("采样间隔 Δ（K=18，token 数不增）")
    a2.set_ylabel("跨度（秒）", color=C_MAIN)
    ax2.set_ylabel("帧间事件可恢复比例", color=C_RED)
    a2.set_ylim(0, 60)
    ax2.set_ylim(0, 1.15)
    a2.set_title("跨度不是免费的：Δ 越大、时间分辨率越低", fontsize=11,
                 color=INK, pad=10)
    a2.grid(alpha=0.25, linewidth=0.7)
    a2.legend([l1, l2], ["跨度 (K−1)Δ", "可恢复比例 2w/Δ（× 为 MC）"],
              loc="center right", frameon=False, fontsize=8.8)
    clean_axes(a2)
    clean_axes(ax2)
    ax2.spines["top"].set_visible(False)
    fig.suptitle("合成探针 A 组：复杂度与跨度/采样间隔权衡（数字逐位来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig3_complexity_interval")


# ================================================================ 图 4（B 组 遮挡）
def fig_occlusion():
    d = D["D_GRID"].astype(int)
    v6, v18 = D["VIDEO_K6"], D["VIDEO_K18"]
    chance, weak, clear = val("ACC_CHANCE"), val("ACC_WEAK"), val("ACC_CLEAR")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.1))

    a1.plot(d, v18, "o-", color=C_TEAL, lw=2.2, ms=7, label="有视频记忆 K=18")
    a1.plot(d, v6, "s-", color=C_AMBER, lw=2.2, ms=7, label="有视频记忆 K=6")
    a1.axhline(chance, color=C_RED, lw=1.5, ls="--")
    a1.axhline(weak, color=GRAY_D["ec"], lw=1.4, ls=":")
    a1.axhline(clear, color=C_TEAL, lw=1.2, ls="--", alpha=0.55)
    a1.axvline(6, color=C_AMBER, lw=1.4, ls="--", alpha=0.7)
    a1.axvline(18, color=C_TEAL, lw=1.4, ls="--", alpha=0.7)
    _wbox = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92)
    a1.text(1.6, chance + 0.012, f"无记忆 {chance:.4f}", ha="left", fontsize=8.4,
            color=C_RED, zorder=6, bbox=_wbox)
    a1.text(1.6, weak + 0.012, f"只当前帧 {weak:.4f}", ha="left", fontsize=8.4,
            color=MUTE, zorder=6, bbox=_wbox)
    a1.set_xticks(d, [str(t) for t in d])
    a1.set_xlabel("距最后一次清晰可见的帧数 d")
    a1.set_ylabel("遮挡补偿（抽屉判别准确率）")
    a1.set_ylim(0, 0.95)
    a1.set_title("记忆跨度不够 → 退化到「只当前帧」，不是塌到随机",
                 fontsize=10.8, color=INK, pad=10)
    a1.grid(alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=9.0, loc="center right")
    note(a1, "K=6 在 d=7 出窗", xy=(7, v6[3]), xytext=(9.0, 0.60),
         color=C_AMBER, fs=8.8)
    note(a1, "K=18 直到 d=18 才出窗", xy=(15, v18[5]), xytext=(10.5, 0.88),
         color=C_TEAL, fs=8.8)
    clean_axes(a1)

    labels = ["无记忆", "只当前帧", "有视频记忆\n(窗内 d=3)", "K=6\n(d=7 出窗)", "K=18\n(d=7 窗内)"]
    vals = [chance, weak, v18[1], v6[3], v18[3]]
    cols = [GRAY_D["ec"], C_AMBER, C_TEAL, C_RED, C_TEAL]
    bars = a2.bar(labels, vals, 0.58, color=cols, edgecolor="white", linewidth=1.1)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.4f}",
                ha="center", fontsize=8.8, color=INK)
    a2.axhline(chance, color=C_RED, lw=1.4, ls="--")
    a2.set_ylim(0, 0.98)
    a2.set_ylabel("抽屉判别准确率")
    a2.set_title("反直觉：K 6→18 不提高窗内精度，只买覆盖范围",
                 fontsize=10.8, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    clean_axes(a2, 8.8)
    fig.suptitle("合成探针 B 组：短时视频记忆补遮挡（数字逐位来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig4_occlusion")


# ================================================================ 图 5（C 组 长时文本）
def fig_text_compression():
    a = D["A_GRID"]
    acc, acc_mc, cov = D["ACC_ALPHA"], D["ACC_ALPHA_MC"], D["COVER"]
    star = A_ALPHA_STAR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.1))

    a1.plot(a, acc, "o-", color=C_MAIN, lw=2.2, ms=7, label="阶段判别准确率（解析）")
    a1.plot(a, acc_mc, "x", color=C_PURPLE, ms=7, label="MC 反算（40 seed）")
    a1.plot(a, cov, "s--", color=C_AMBER, lw=1.6, ms=5, label="阶段标记覆盖率")
    a1.axvline(star, color=C_TEAL, lw=1.8, ls="--")
    a1.axhline(val("CFG_NONE"), color=C_RED, lw=1.4, ls="--")
    for x_, y_ in zip((a[0], star, a[-1]), (acc[0], acc[3], acc[-1])):
        a1.annotate(f"{y_:.4f}", (x_, y_), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.8, color=INK)
    a1.set_xticks(a, [f"{t:.3f}" for t in a], fontsize=8.6)
    a1.set_xlabel("压缩强度 α（优先删冗余、再删旧阶段标记）")
    a1.set_ylabel("准确率 / 覆盖率")
    a1.set_ylim(0, 1.12)
    a1.set_title(f"倒 U 形：峰值 α*={star:.3f}=ρ/(1+ρ)", fontsize=11, color=INK, pad=10)
    a1.grid(alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=8.8, loc="lower left")
    note(a1, f"naive {acc[0]:.4f}\n（冗余挤掉旧阶段）", xy=(a[0], acc[0]),
         xytext=(0.05, 0.60), color=C_RED, fs=8.6)
    note(a1, f"压缩过头 {acc[-1]:.4f}\n（=随机）", xy=(a[-1], acc[-1]),
         xytext=(0.62, 0.24), color=C_AMBER, fs=8.6)
    clean_axes(a1)

    names = ["都没有", "只视频", "只文本", "两者都有"]
    cfg = D["CFG_STAGE"]
    bars = a2.bar(names, cfg, 0.56,
                  color=[GRAY_D["ec"], C_AMBER, C_TEAL, C_MAIN],
                  edgecolor="white", linewidth=1.1)
    for b, v in zip(bars, cfg):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.4f}",
                ha="center", fontsize=9.0, color=INK)
    a2.axhline(val("CFG_NONE"), color=C_RED, lw=1.4, ls="--")
    a2.text(3.45, 0.512, "随机 0.5", ha="right", fontsize=8.8, color=C_RED,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92))
    a2.set_ylim(0, 1.12)
    a2.set_ylabel("长程阶段判别准确率")
    a2.set_title("反直觉：两者都有 = 只文本（短时视频对长程无效）",
                 fontsize=10.6, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    clean_axes(a2)
    fig.suptitle("合成探针 C 组：文本记忆压缩与长程阶段判别（数字逐位来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_text_compression")


# ================================================================ 图 6（D 组 分工边界）
def fig_division():
    occ, stage = D["CFG_OCC"], D["CFG_STAGE_D"]
    names = ["都没有", "只视频", "只文本", "两者都有"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.1))

    x = np.arange(4)
    w = 0.36
    b1 = a1.bar(x - w / 2, occ, w, color=C_TEAL, edgecolor="white",
                linewidth=1.0, label="遮挡补偿（短时任务）")
    b2 = a1.bar(x + w / 2, stage, w, color=C_PURPLE, edgecolor="white",
                linewidth=1.0, label="长程阶段判别（长时任务）")
    for b, v in zip(b1, occ):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.4f}",
                ha="center", fontsize=8.4, color=INK)
    for b, v in zip(b2, stage):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.4f}",
                ha="center", fontsize=8.4, color=INK)
    a1.set_xticks(x, names, fontsize=9.4)
    a1.set_ylabel("任务准确率")
    a1.set_ylim(0, 1.16)
    a1.set_title("四种记忆配置 × 两类任务", fontsize=11, color=INK, pad=10)
    a1.grid(axis="y", alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=8.8, loc="upper left")
    clean_axes(a1)

    # 交叉矩阵（2×4 热力图，数字逐位来自 stdout）
    M = np.array([occ, stage])
    im = a2.imshow(M, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    a2.set_xticks(range(4), names, fontsize=9.4)
    a2.set_yticks(range(2), ["遮挡补偿\n(短时)", "阶段判别\n(长时)"], fontsize=9.4)
    for i in range(2):
        for j in range(4):
            a2.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    fontsize=9.6, color=(INK if M[i, j] < 0.62 else "white"),
                    fontweight="500")
    a2.add_patch(Rectangle((2.5, -0.5), 1.0, 2.0, fill=False,
                           edgecolor=C_TEAL, linewidth=2.4, zorder=4))
    a2.set_title("互补而非叠加：只有「两者都有」两行都取到最佳",
                 fontsize=10.6, color=INK, pad=10)
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    fig.suptitle("合成探针 D 组：两类记忆的分工边界（数字逐位来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig6_division")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in (("fig1_mem_architecture", fig_architecture),
                     ("fig2_span_budget", fig_span_budget),
                     ("fig3_complexity_interval", fig_complexity_interval),
                     ("fig4_occlusion", fig_occlusion),
                     ("fig5_text_compression", fig_text_compression),
                     ("fig6_division", fig_division)):
        if only and only not in name:
            continue
        fn()
        print("wrote", name)
