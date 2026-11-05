# -*- coding: utf-8 -*-
"""10 篇（π0.7）配图：1 张结构图（按论文文字重画）+ 5 张数据图。

- 数据图的数字**逐位解析自同目录 results/stdout.txt 的 [MACHINE-BACKFILL-*] 块**
  （与「正文回填表」同源），配图不硬编码任何合成数字；结构图标注的是论文写明
  的常量（引用，§V / §VI-B / §VII / §IV），并明确标注「按论文文字重画，非论文
  原图」。
- v2 / abs 页为 arXiv.org perpetual non-exclusive license，官方图不复用；
  不下载、不截图、不外链任何论文 Figure。字体 PingFang SC、无 Mermaid。
- 画完必须渲染 PNG 用 ViewImage 逐张肉眼过一遍：bbox_inches="tight" 会把越界
  元素悄悄撑大画布，柱顶/点注 label 也可能被压住。每张 SVG 保留固有 width/height。

运行：~/Software/miniconda3/bin/python pi07_figures.py
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
STDOUT = os.path.join(HERE, "results", "stdout.txt")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
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
PT = 1.0 / 72.0

C_MAIN = "#185FA5"
C_RED = "#A32D2D"
C_TEAL = "#0F6E56"
C_AMBER = "#854F0B"
C_PURPLE = "#534AB7"
C_GRAY = "#5F5E5A"

SRC = "合成数字逐位来自 results/stdout.txt 正文回填表；结构图常量为论文引用"


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
    m = re.search(pattern, STDOUT_TEXT)
    assert m, f"stdout 中找不到：{pattern}"
    return cast(m.group(group))


def val(key, i=0):
    return float(D[key][i])


def arr(key):
    return D[key]


# 论文引用常量（逐位从 stdout 的「论文常量（引用）」区抽取）
P_BACKBONE = grab(r"backbone=(\w+)", cast=str)
P_EXPERT = grab(r"action_expert=(\w+)", cast=str)
P_BETAS = grab(r"beta=\(([^)]*)\)", cast=str)
P_DROPOUT = grab(r"dropout=([\d/]+%)", cast=str)


# ---------------------------------------------------------------- 画图工具
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.0, fs_sub=9.6, lw=0.9, r=0.10,
        dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(r, max(min(w, h)/2-0.01, 0.01))}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=2,
        linestyle=("--" if dashed else "-")))
    if not lines:
        return
    if len(lines) == 1:
        ax.text(x + w/2, y + h/2, lines[0], ha="center", va="center",
                fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
        return
    hs = [(fs if i == 0 else fs_sub) * 1.42 * PT for i in range(len(lines))]
    total = sum(hs)
    avail = h - 0.12
    if total > avail:
        hs = [v * avail / total for v in hs]
        total = avail
    cy = y + h/2 + total/2
    for i, s in enumerate(lines):
        cy -= hs[i]/2
        ax.text(x + w/2, cy, s, ha="center", va="center",
                fontsize=(fs if i == 0 else fs_sub),
                fontweight=("500" if i == 0 else "400"),
                color=(style["tc"] if i == 0 else MUTE), zorder=3)
        cy -= hs[i]/2


def container(ax, x, y, w, h, title, style, fs=12.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(0.20, min(w, h)/2-0.01)}",
        linewidth=1.0, edgecolor=style["ec"], facecolor=style["fc"], zorder=1))
    ax.text(x + 0.26, y + h - 0.30, title, ha="left", va="center",
            fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
    return y + h - 0.62


def vstack(ax, x, y_top, w, items, gap=0.16):
    cy = y_top
    for h, lines, st in items:
        cy -= h
        box(ax, x, cy, w, h, lines, st)
        cy -= gap
    return cy + gap


def arrow(ax, p1, p2, color=MUTE, lw=1.6, rad=0.0, head="-|>", ms=15):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=head, mutation_scale=ms, linewidth=lw,
        color=color, zorder=3, connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, fs=10.0, color=HINT, ha="center", va="center",
          weight="400"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, zorder=4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax, fs=9.6):
    ax.tick_params(labelsize=fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBB6")
        ax.spines[s].set_linewidth(0.8)


def footnote(fig, y=0.012):
    fig.text(0.5, y, SRC, ha="center", va="bottom", fontsize=8.4, color=HINT)


# ===================================================================== 图 1
def fig_pipeline():
    """结构图：按 §V / §VI-B / §VII / §IV 文字重画的完整 prompt 与训练/推理通路。"""
    W, H = 14.4, 9.4
    fig, ax = canvas(W, H)
    label(ax, W/2, 9.05,
          "π0.7：diverse context conditioning 的完整 prompt C_t 与训练 / 推理通路",
          fs=14, color=INK, weight="500")
    label(ax, W/2, 8.68,
          "按论文文字重画，非论文原图 · 建在 π0.6-MEM 一脉上，作者明确不提出新架构（§II / §IV）",
          fs=10.0, color=MUTE)
    label(ax, W/2, 8.30,
          "① 构造带 dropout 的 C_t  →  ② 训练  →  ③ 测试时给任意子集 C_t，并用 CFG 在 metadata 上引导",
          fs=9.6, color=MUTE)

    cw, ch, cy = 4.40, 5.55, 2.62
    xs = (0.35, 5.00, 9.65)

    # ① 完整 prompt：5 类分量
    t = container(ax, xs[0], cy, cw, ch, "①  完整 prompt C_t（§V / §V-E）", BLUE)
    vstack(ax, xs[0]+0.22, t, cw-0.44, [
        (0.80, ["总体任务语言 ℓ_t", "文本 token · 'Task: peel vegetables'"], BLUE_D),
        (0.80, ["子任务语言 ℓ̂_t", "文本 token · 跟随 π0.5，可随时间变化"], BLUE),
        (0.80, ["片段级元数据 m", "speed(500步分箱) / quality(1–5) / mistake"], BLUE),
        (0.80, ["控制模态标签 c", "文本标识 · c∈{joint, ee}，测试时按任务选"], BLUE_D),
        (0.80, ["视觉子目标图像 g_t", "图像 token · 共享 vision encoder，448×448"], BLUE),
    ], gap=0.13)

    # ② 训练：dropout + RTC
    t = container(ax, xs[1], cy, cw, ch, "②  训练：随机 dropout + RTC（§V-E / §VI-B）", TEAL)
    vstack(ax, xs[1]+0.22, t, cw-0.44, [
        (0.94, ["各分量按不同概率 dropout",
                f"引用：{P_DROPOUT}（子目标图只加到 25% 样本）"], TEAL_D),
        (0.80, ["控制模态 c 不做 dropout（0%）",
                "使模型测试时能处理任意子集 prompt"], TEAL),
        (0.94, ["训练期 RTC：模拟 0–12 步延迟",
                "50 Hz 上对应最大 240 ms 推理延迟"], TEAL_D),
        (0.80, ["本体状态 q_t：线性投影（跟随 MEM）",
                "每个 history state 一个独立 token"], TEAL),
    ], gap=0.13)

    # ③ 推理：CFG steering
    t = container(ax, xs[2], cy, cw, ch, "③  推理：CFG 引导（§VII）", PURPLE)
    vstack(ax, xs[2]+0.22, t, cw-0.44, [
        (0.94, ["只对 episode metadata 用 CFG",
                "score(a|C) + β(score(a|C) − score(a|C_uncond))"], PURPLE_D),
        (0.80, [f"guidance scale β∈{{{P_BETAS}}}",
                "原文称 moderate values"], PURPLE),
        (0.94, ["推理设定：speed = 第 15 百分位",
                "quality ≡ 5（最高）、mistake ≡ false"], PURPLE_D),
        (0.80, ["5 步去噪 → 50 步 action chunk",
                "实际执行 Ĥ∈{15, 25} 步"], PURPLE),
    ], gap=0.13)

    ymid = cy + ch/2
    arrow(ax, (xs[0]+cw, ymid), (xs[1], ymid), color=C_MAIN, lw=2.0, ms=13)
    arrow(ax, (xs[1]+cw, ymid), (xs[2], ymid), color=C_PURPLE, lw=2.0, ms=13)

    # 底部：架构常量 + 核正 + 许可
    box(ax, 0.35, 0.32, W-0.70, 1.92, [
        f"架构（引用）：约 {P_BACKBONE} backbone / {P_EXPERT} action expert / Gemma3 4B VLM · "
        "flow matching 预测连续动作",
        "核正①：论文正文与 112 条参考文献零引用 RLT / RL Token；"
        "核正②：离散文本本体状态属 π0.6，π0.7 跟随 MEM 改线性投影",
        "核正③：RECAP 一词只见官方博客，论文正文中对应实体是 π*0.6（rollout 被蒸馏进 π0.7）",
        "许可：v2 / abs 页为 arXiv.org perpetual non-exclusive license（v1 曾标 CC BY 4.0）；"
        "官方图不复用，本图按文字重画",
    ], GRAY, fs=10.4, fs_sub=9.4)
    save(fig, "fig1_context_pipeline")


# ===================================================================== 图 2
def fig_dropout_tradeoff():
    """A 组：dropout 尺度 t 的三条误差曲线 + 相对交叉项（反直觉）。"""
    T = arr("T_GRID")
    ef, eu, es = arr("ERR_FULL"), arr("ERR_UNCOND"), arr("ERR_SUB")
    off = arr("OFFDIAG")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    a1.plot(T, ef, "o-", color=C_MAIN, lw=2.1, ms=6, label="满上下文 err_full（↑ 代价）")
    a1.plot(T, eu, "s-", color=C_AMBER, lw=2.1, ms=6,
            label="仅任务语言 err_uncond（↓ 无条件分支被训练）")
    a1.plot(T, es, "^-", color=C_TEAL, lw=2.3, ms=7,
            label="任意子集平均 err_sub（U 形）")
    for x, v in zip(T, es):
        a1.annotate(f"{v:.4f}", (x, v), xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", fontsize=8.2, color=C_TEAL)
    _ib = int(np.argmin(es))
    a1.annotate(f"子集鲁棒性最优 t={T[_ib]:.1f}", xy=(T[_ib], es[_ib]),
                xytext=(T[_ib]+0.35, es[_ib]+0.55), fontsize=9.4, color=C_TEAL,
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1))
    # 直接贴线标注，避免图例压住曲线
    a1.text(T[-1]+0.06, ef[-1], "err_full", fontsize=8.8, color=C_MAIN,
            ha="left", va="center")
    a1.text(T[-1]+0.06, eu[-1], "err_uncond", fontsize=8.8, color=C_AMBER,
            ha="left", va="center")
    a1.text(T[-1]+0.06, es[-1], "err_sub", fontsize=8.8, color=C_TEAL,
            ha="left", va="center")
    a1.set_xticks(T, [f"{v:g}" for v in T])
    a1.set_xlim(-0.15, 3.62)
    a1.set_xlabel("dropout 全局尺度 t（t=1 即论文 25/30/15/5% 量级）", fontsize=9.6)
    a1.set_ylabel("动作系数期望平方误差（越低越好）", fontsize=9.6)
    a1.set_ylim(0.45, 2.55)
    a1.set_title("dropout 的三重权衡：满上下文变差 / 无条件分支变好 / 子集鲁棒性 U 形",
                 fontsize=10.6, color=INK, pad=10)
    a1.grid(alpha=0.22, linewidth=0.7)
    clean_axes(a1, 9.2)

    a2.plot(T, off, "o-", color=C_RED, lw=2.2, ms=7)
    for x, v in zip(T, off):
        a2.annotate(f"{v:.4f}", (x, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.4, color=C_RED)
    a2.set_xticks(T, [f"{v:g}" for v in T])
    a2.set_xlabel("dropout 全局尺度 t", fontsize=9.6)
    a2.set_ylabel("相对交叉项 offdiag / diag", fontsize=9.6)
    a2.set_ylim(0.42, 1.32)
    a2.set_title("反直觉：相对交叉项随 dropout 上升", fontsize=10.8, color=INK, pad=10)
    a2.text(0.03, 0.95,
            "对角增益被压得比非对角更快，\n模型更依赖恒在的分量",
            transform=a2.transAxes, ha="left", va="top", fontsize=9.0, color=MUTE)
    a2.grid(alpha=0.22, linewidth=0.7)
    clean_axes(a2, 9.2)

    fig.suptitle("合成探针 A 组：多条件分量 + 训练期 dropout（§V 引言 / §V-E）",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    footnote(fig)
    save(fig, "fig2_dropout_tradeoff")


# ===================================================================== 图 3
def fig_adherence():
    """A 组：t=0 与 t=1 各分量遵从度（dropout 概率越高遵从度越低）。"""
    adh0, adh1 = arr("ADH_T0"), arr("ADH_T1")
    p = arr("P_BASE")
    names = ["任务语言 ℓ", "子任务 ℓ̂", "片段元数据 m", "控制模态 c", "视觉子目标 g"]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    b0 = ax.bar(x - w/2, adh0, w, color="#C9C7C0", edgecolor="white",
                linewidth=1.0, label="无 dropout（t=0）")
    b1 = ax.bar(x + w/2, adh1, w, color=C_MAIN, edgecolor="white",
                linewidth=1.0, label="论文量级 dropout（t=1）")
    for b, v in zip(list(b0) + list(b1), list(adh0) + list(adh1)):
        ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.4f}",
                ha="center", fontsize=8.6, color=INK)
    for xi, pv in zip(x, p):
        ax.text(xi, -0.11, f"p={pv:.2f}", ha="center", va="top",
                fontsize=9.0, color=C_AMBER)
    ax.set_xticks(x, names, fontsize=9.6)
    ax.set_ylim(-0.02, 0.80)
    ax.set_ylabel("分量遵从度 adh_k = W*_kk / B_k", fontsize=9.8)
    ax.set_title("缺了 dropout 没有分量被单独忽略；dropout 概率越高，该分量遵从度越低",
                 fontsize=10.8, color=INK, pad=12)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    clean_axes(ax, 9.4)
    fig.suptitle("合成探针 A 组：dropout 概率与条件遵从度的直接权衡",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.text(0.5, 0.055,
             "底部橙色为各分量训练期 dropout 概率（引用量级 25/30/15/5%，控制模态 0%）",
             ha="center", va="bottom", fontsize=8.8, color=MUTE)
    footnote(fig)
    save(fig, "fig3_adherence")


# ===================================================================== 图 4
def fig_cfg_inverted_u():
    """B 组：CFG 质量损失倒 U（左）+ 有效增益 / 多样性（右）。"""
    bf, lf = arr("BETA_FINE"), arr("LOSS_FINE")
    bg, lossg = arr("BETA_GRID"), arr("LOSS_GRID")
    alpha_g, div_g = arr("ALPHA_GRID"), arr("DIV_GRID")
    b_ana, b_q, b_emp = val("BETA_ANA"), val("BETA_Q"), val("BETA_EMP")
    kappa, r, sc2 = val("KAPPA"), val("R_CFG"), val("SC2")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.3))

    a1.plot(bf, lf, color=C_MAIN, lw=2.4)
    for b, c in zip(bg, (C_AMBER, C_TEAL, C_RED)):
        a1.axvline(b, color=c, lw=1.2, ls=":", alpha=0.85)
    for b, v in zip(bg, lossg):
        a1.plot(b, v, "o", color=C_MAIN, ms=8, zorder=5)
    # 所有数字集中放在两个不压曲线的文字框里，避免逐点标注互相重叠
    _info = (f"论文网格 loss：1.3→{lossg[0]:.6f}  1.7→{lossg[1]:.6f}  "
             f"2.2→{lossg[2]:.6f}\n"
             f"解析 β*={b_ana:.4f}   β_Q={b_q:.3f}   MC 实测 β={b_emp:.2f}")
    a1.text(0.97, 0.95, _info, transform=a1.transAxes, ha="right", va="top",
            fontsize=8.7, color=INK,
            bbox=dict(facecolor="white", edgecolor="#BBBBB6", linewidth=0.7,
                      boxstyle="round,pad=0.4", alpha=0.94))
    a1.text(0.02, 0.05, f"κ={kappa:.3f}  r={r:.2f}  σ_c²={sc2:.4f}  w_div=0.02（自设）",
            transform=a1.transAxes, ha="left", va="bottom", fontsize=8.8, color=MUTE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.5))
    a1.set_xlabel("guidance scale β", fontsize=9.8)
    a1.set_ylabel("复合质量损失 (α−1)² + w_div·多样性损失", fontsize=9.4)
    a1.set_ylim(-0.004, 0.112)
    a1.set_xlim(-0.1, 4.1)
    a1.set_title("倒 U：β 太小欠引导、太大过引导 + 丢多样性", fontsize=10.6,
                 color=INK, pad=10)
    a1.grid(alpha=0.22, linewidth=0.7)
    clean_axes(a1, 9.2)

    # 右：精确高斯 score-CFG 的解析曲线（逐位用回填块里的常量）
    bb = np.linspace(0.0, 4.0, 201)
    gamma = (1.0 + bb) / (1.0 + bb * (1.0 - r))
    aa = kappa * gamma
    sdiv = 1.0 / (1.0 + bb * (1.0 - r))
    a2.plot(bb, aa, color=C_PURPLE, lw=2.2, label="有效增益 α(β)=κ·γ(β)")
    a2.plot(bb, sdiv, color=C_TEAL, lw=2.2, label="多样性 Σ_β/σ_c²")
    a2.axhline(1.0, color=HINT, lw=1.3, ls="--", label="α=1（无欠/过引导）")
    for b, c in zip(bg, (C_AMBER, C_TEAL, C_RED)):
        a2.axvline(b, color=c, lw=1.2, ls=":", alpha=0.85)
    for b, av, dv in zip(bg, alpha_g, div_g):
        a2.plot(b, av, "o", color=C_PURPLE, ms=7, zorder=5)
        a2.plot(b, dv, "s", color=C_TEAL, ms=7, zorder=5)
        a2.annotate(f"α={av:.4f}", (b, av), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8.4,
                    color=C_PURPLE,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
        a2.annotate(f"{dv:.4f}", (b, dv), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.4,
                    color=C_TEAL,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
    a2.set_xlabel("guidance scale β（虚线为论文 {1.3, 1.7, 2.2}）", fontsize=9.4)
    a2.set_ylabel("归一化读数", fontsize=9.6)
    a2.set_ylim(0.38, 1.52)
    a2.set_xlim(-0.1, 4.1)
    a2.set_title("β↑：α 从欠引导走向过引导，多样性单调下降", fontsize=10.6,
                 color=INK, pad=10)
    a2.grid(alpha=0.22, linewidth=0.7)
    a2.legend(frameon=False, fontsize=8.6, loc="upper right")
    clean_axes(a2, 9.2)

    fig.suptitle("合成探针 B 组：CFG 引导强度 β∈{1.3, 1.7, 2.2} 的倒 U（§VII）",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    footnote(fig)
    save(fig, "fig4_cfg_inverted_u")


# ===================================================================== 图 5
def fig_metadata_steering():
    """C 组：三档元数据的轨迹（左）+ 副作用条形（右）。"""
    trj = [arr("C_TRJ0"), arr("C_TRJ1"), arr("C_TRJ2")]
    x2 = arr("C_X2")
    jk, tk, me = arr("C_JERK"), arr("C_TRACK"), arr("C_MODELERR")
    steer, msteer = arr("C_STEER"), arr("C_MODELSTEER")
    adh2 = val("ADH2")
    colors = [C_GRAY, C_RED, C_AMBER]
    labs = [f"稳 (x₂={x2[0]:.1f})", f"快 (x₂={x2[1]:.1f})", f"标记失误 (x₂={x2[2]:.1f})"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))
    tt = np.arange(len(trj[0]))
    for c, l, a in zip(colors, labs, trj):
        a1.plot(tt, a, lw=2.2, color=c, label=l)
    a1.axhline(0, color=HINT, lw=0.9)
    a1.set_xlabel("action chunk 时间步 t（H=24）", fontsize=9.6)
    a1.set_ylabel("动作幅值（合成轨迹）", fontsize=9.6)
    a1.set_title("同一任务、三档元数据 → 三条不同轨迹", fontsize=10.8,
                 color=INK, pad=10)
    a1.grid(alpha=0.22, linewidth=0.7)
    a1.legend(frameon=False, fontsize=8.8, loc="upper right")
    clean_axes(a1, 9.2)

    x = np.arange(3)
    w = 0.26
    b1 = a2.bar(x - w, jk, w, color="#9FB6CC", edgecolor="white",
                linewidth=1.0, label="jerk（动作失真代理）")
    b2 = a2.bar(x, tk, w, color="#BCE3D3", edgecolor="white",
                linewidth=1.0, label="跟踪误差（40 seed）")
    b3 = a2.bar(x + w, me, w, color="#E3B873", edgecolor="white",
                linewidth=1.0, label="模型动作误差")
    for bs in (b1, b2, b3):
        for b in bs:
            a2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                    f"{b.get_height():.4f}", ha="center", fontsize=7.9, color=INK)
    a2.set_xticks(x, ["稳", "快", "标记失误"], fontsize=9.8)
    a2.set_ylabel("三种读数（量纲不同，仅看组内高低）", fontsize=9.2)
    a2.set_ylim(0, 1.32)
    a2.set_title("副作用：可引导性最强的「快」档误差也最大", fontsize=10.6,
                 color=INK, pad=10)
    a2.grid(axis="y", alpha=0.22, linewidth=0.7)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=8.4, loc="upper left")
    clean_axes(a2, 9.0)

    fig.suptitle("合成探针 C 组：元数据 steering（§V-C / §VII）",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.095, 1, 0.94))
    fig.text(0.5, 0.058,
             f"轨迹偏离 |a-a稳|：快 {steer[1]:.4f} / 失误 {steer[2]:.4f}；"
             f"模型偏离 |ahat-a稳|：快 {msteer[1]:.4f} / 失误 {msteer[2]:.4f}；"
             f"元数据遵从度被 15% dropout 衰减到 adh2={adh2:.4f}",
             ha="center", va="bottom", fontsize=8.6, color=MUTE)
    footnote(fig)
    save(fig, "fig5_metadata_steering")


# ===================================================================== 图 6
def fig_delay_conditioning():
    """D 组：0–12 步 / 240 ms 延迟下的三条误差曲线 + 分布偏移 / 软信任度。"""
    dg, dms = arr("D_GRID"), arr("D_MS")
    unc, no, rtc = arr("D_UNCOND"), arr("D_NORTC"), arr("D_RTC")
    shift, phi, ssoft, shat = arr("D_SHIFT"), arr("D_PHI"), arr("D_S_SOFT"), arr("D_S_HAT")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    a1.plot(dg, unc, "s--", color=C_GRAY, lw=1.8, ms=6,
            label="无条件（只用任务语言）")
    a1.plot(dg, no, "o-", color=C_RED, lw=2.3, ms=7,
            label="no_rtc（只在 d=0 训练）")
    a1.plot(dg, rtc, "^-", color=C_TEAL, lw=2.3, ms=7,
            label="rtc（训练期模拟 0–12 步）")
    _lb = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6)
    for x, v in zip(dg, no):
        a1.annotate(f"{v:.4f}", (x, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.2, color=C_RED, bbox=_lb)
    # d=0 处 no_rtc 与 rtc 同为 0.4500，上方红色已标注；跳过避免压住 x 轴
    for x, v in zip(dg[1:], rtc[1:]):
        a1.annotate(f"{v:.4f}", (x, v), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=8.2, color=C_TEAL, bbox=_lb)
    _cross = dg[no > unc[0]][0]
    a1.annotate("反直觉：条件化\n反而比不条件化更差",
                xy=(_cross, no[list(dg).index(_cross)]),
                xytext=(0.6, 12.2), ha="left", va="center",
                fontsize=9.2, color=C_RED,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1))
    a1.set_xticks(dg, [f"{int(v)}" for v in dg])
    a1.set_xlabel("推理延迟 d（步；×20 ms = 0 / 60 / 120 / 180 / 240 ms）", fontsize=9.2)
    a1.set_ylabel("动作系数期望平方误差（越低越好）", fontsize=9.6)
    a1.set_ylim(-0.3, 17.0)
    a1.set_title("条件化放大陈旧观测的分布偏移；训练期 RTC 缓解", fontsize=10.6,
                 color=INK, pad=10)
    a1.grid(alpha=0.22, linewidth=0.7)
    a1.legend(frameon=False, fontsize=8.8, loc="upper left")
    clean_axes(a1, 9.2)

    a2.plot(dg, shift, "o-", color=C_AMBER, lw=2.2, ms=7, label="分布偏移 shift(d)")
    a2.plot(dg, phi, "s-", color=C_PURPLE, lw=2.2, ms=6,
            label="Bayes 最优信任度 φ^d")
    a2.plot(dg, ssoft, "D--", color=C_MAIN, lw=1.6, ms=6,
            label="软衰减信任度 s_soft(d)")
    a2.plot(dg, shat, "^:", color=C_TEAL, lw=1.5, ms=6,
            label="训练期 RTC 的 MC 反算 s_hat(d)")
    for x, v in zip(dg, phi):
        a2.annotate(f"{v:.4f}", (x, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.2, color=C_PURPLE)
    a2.set_xticks(dg, [f"{int(v)}" for v in dms], fontsize=9.4)
    a2.set_xlabel("推理延迟（ms，0–240）", fontsize=9.6)
    a2.set_ylabel("shift（左）/ 信任度（右，0–1）", fontsize=9.4)
    a2.set_ylim(-0.3, 8.0)
    a2.set_title("软衰减信任度 = φ^d = MC 反算增益（构造真值与反算值一致）",
                 fontsize=10.4, color=INK, pad=10)
    a2.grid(alpha=0.22, linewidth=0.7)
    a2.legend(frameon=False, fontsize=8.4, loc="upper left")
    clean_axes(a2, 9.0)

    fig.suptitle("合成探针 D 组：0–12 步 RTC / 240 ms 延迟下的条件化（§VI-B）",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    footnote(fig)
    save(fig, "fig6_delay_conditioning")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in (("fig1_context_pipeline", fig_pipeline),
                     ("fig2_dropout_tradeoff", fig_dropout_tradeoff),
                     ("fig3_adherence", fig_adherence),
                     ("fig4_cfg_inverted_u", fig_cfg_inverted_u),
                     ("fig5_metadata_steering", fig_metadata_steering),
                     ("fig6_delay_conditioning", fig_delay_conditioning)):
        if only and only not in name:
            continue
        fn()
        print("wrote", name)
