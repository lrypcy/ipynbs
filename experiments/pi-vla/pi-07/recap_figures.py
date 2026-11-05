# -*- coding: utf-8 -*-
"""07 篇（RECAP / π*0.6）配图：1 张结构图 + 5 张数据图，全部手工排版成 SVG/PNG。

- 结构图按论文 §IV / Algorithm 1 / §V-B / §V-D 的**文字**重画，不复用任何官方图
  （RECAP 是 arXiv.org perpetual non-exclusive license，与 RTC 的 CC BY 4.0 不同）。
- 数据图的**每一个数字**逐位来自同目录 results/stdout.txt 的「正文回填表」区，
  集中抄在下方的常量里，grep 可核对；不写任何派生数。
- 字体 PingFang SC；无 Mermaid；SVG 保留 matplotlib 默认的固有 width/height。
- 画完必须渲染 PNG 用 ViewImage 逐张肉眼过一遍：bbox_inches="tight" 会把越界元素
  悄悄撑大画布，柱顶/点注 label 也可能被压住。

运行：~/Software/miniconda3/bin/python recap_figures.py
"""
from __future__ import annotations

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
TEAL_D = dict(fc="#BCE3D3", ec="#0F6E56", tc="#04342C")
RED = dict(fc="#FCEBEB", ec="#A32D2D", tc="#501313")
AMBER = dict(fc="#FAEEDA", ec="#854F0B", tc="#412402")
GRAY = dict(fc="#F8F7F4", ec="#888780", tc="#2C2C2A")
GRAY_D = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")

INK = "#2C2C2A"
MUTE = "#5F5E5A"
HINT = "#888780"
C_JOINT = "#185FA5"
C_FLAT = "#A32D2D"
C_CTRL = "#0F6E56"
C_AMBER = "#854F0B"
C_PURPLE = "#534AB7"
PT = 1.0 / 72.0

# =====================================================================
# 所有数字：逐位抄自 results/stdout.txt「正文回填表」区（grep 可核对）
# =====================================================================
D = dict(
    # [世界] / [二值化] / [论文常量]
    tau=0.9520, q=0.3003, eps_l=0.0342, V_ref=-0.0433,
    mu_g=0.6, mu_b=2.4, sig=1.0, w_bad=0.55, B=201, N=50,
    # [A组]
    J_ref=-0.0433, J_cond=-0.0030, J_rej=-0.0030, J_cont_best=-0.0034,
    R_keep=-0.0030, R_drop=-0.0606, mean_ref=1.5900, mean_cond=0.2065,
    H_ref=7.9222, H_cond=6.7834, KL_cond=1.2029,
    # [A组] 连续加权扫描 t:J:H:KL
    T=[0.15, 0.30, 0.60, 1.00, 2.00, 4.00, 10.00],
    J_cont=[-0.0034, -0.0067, -0.0125, -0.0183, -0.0267, -0.0335, -0.0389],
    H_cont=[7.0142, 7.2920, 7.5296, 7.6648, 7.7870, 7.8539, 7.8950],
    KL_cont=[0.9672, 0.6491, 0.3641, 0.2047, 0.0768, 0.0242, 0.0045],
    # [B组]
    J_off=[-0.0425] * 7,
    J_auto=[-0.0425, -0.0108, -0.0052, -0.0038, -0.0032, -0.0030, -0.0028],
    J_corr=[-0.0425, -0.0026, -0.0025, -0.0025, -0.0025, -0.0025, -0.0025],
    floor=-0.0025, gain1=0.0399,
    # [C组]
    N_gr=[1, 2, 3, 4, 5, 8, 16, 64, 256],
    J_bon=[-0.0433, -0.0189, -0.0108, -0.0070, -0.0049, -0.0022, -0.0006,
           -0.0000, -0.0000],
    N_star=8, J_bon_star=-0.0022, H_u_cond=6.7834, H_u_bon=6.7946,
    tail_cond=0.2558, tail_bon=0.1322,
    # [D组]
    info_loss=0.6901,
    delta=[-0.050, -0.010, 0.000, 0.010, 0.050],
    J_affine=[-0.0030] * 5,
    sig_cal=[0.000, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200],
    J_cal=[-0.0030, -0.0031, -0.0037, -0.0053, -0.0084, -0.0159,
           -0.0239, -0.0318],
    c_bad=[0.0, 1.0, 2.0, 3.0, 4.0],
    J_bad=[-0.0025, -0.0062, -0.0154, -0.0327, -0.0652],
    baseline=-0.0425,
)

SRC = "数字逐位来自 results/stdout.txt 正文回填表"


# ---------------------------------------------------------------- 画图纸 helper
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.0, fs_sub=9.6, lw=0.9, r=0.10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(r, max(min(w, h)/2-0.01, 0.01))}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=2))
    if not lines:
        return
    if len(lines) == 1:
        ax.text(x + w/2, y + h/2, lines[0], ha="center", va="center",
                fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
        return
    hs = [(fs if i == 0 else fs_sub) * 1.42 * PT for i in range(len(lines))]
    total = sum(hs)
    avail = h - 0.14
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
    return y + h - 0.60


def vstack(ax, x, y_top, w, items, gap=0.16):
    cy = y_top
    for h, lines, st in items:
        cy -= h
        box(ax, x, cy, w, h, lines, st)
        cy -= gap
    return cy + gap


def arrow(ax, p1, p2, color=MUTE, lw=1.5, rad=0.0, head="-|>", ms=15):
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
    """结构图：按 §IV / Algorithm 1 / §V-B / §V-D 的文字重画的三阶段循环。"""
    W, H = 13.6, 9.4
    fig, ax = canvas(W, H)
    label(ax, W/2, 9.05, "RECAP：从含 advantage conditioning 的 VLA 出发，用真机经验反复自我改进",
          fs=14, color=INK, weight="500")
    label(ax, W/2, 8.66,
          "① 部署收数据  →  ② 用全部数据训 value function  →  ③ 用 binarized advantage 条件化训 VLA（§IV / Algorithm 1）",
          fs=10.2, color=MUTE)

    cw, ch, cy = 3.92, 4.55, 3.75
    xs = (0.35, 4.84, 9.33)
    # 容器 ①
    t = container(ax, xs[0], cy, cw, ch, "①  Data collection（§IV step 1）", BLUE)
    vstack(ax, xs[0]+0.24, t, cw-0.48, [
        (0.86, ["部署上一轮策略 π 收集 rollouts", "autonomous / on-policy 数据"], BLUE_D),
        (0.86, ["给每个 episode 打 outcome label", "决定 reward：成功 0 / 失败大负值"], BLUE),
        (0.86, ["可选 expert interventions", "human-gated DAgger；强制 I=True"], RED),
    ], gap=0.18)
    # 容器 ②
    t = container(ax, xs[1], cy, cw, ch, "②  Value function training（Eq.1）", TEAL)
    vstack(ax, xs[1]+0.24, t, cw-0.48, [
        (0.86, ["用迄今全部数据训练", "B=201 bins 上的分布 p(V | 观测, 指令)"], TEAL_D),
        (0.86, ["预测「距成功剩余步数的负值」", "归一化到 (−1,0)，V=0 为成功"], TEAL),
        (0.86, ["A：N=50 lookahead", "A = 50 步 reward 之和 + V(t+50) − V(t)"], TEAL),
    ], gap=0.18)
    # 容器 ③
    t = container(ax, xs[2], cy, cw, ch, "③  Advantage conditioned training（Eq.3）", PURPLE)
    vstack(ax, xs[2]+0.24, t, cw-0.48, [
        (0.86, ["I = True 当 A > 阈值 ε", "ε 取 value 预测的 30% 分位数"], PURPLE_D),
        (0.86, ["文本前缀：'Advantage: positive'", "或 'Advantage: negative'（§V-B）"], PURPLE),
        (0.86, ["在全部数据上训无条件项+条件项", "不像 AWR 丢数据；纠正强制 I=True"], PURPLE),
    ], gap=0.18)

    # 阶段间箭头（贴着容器边缘中点）
    ymid = cy + ch/2
    arrow(ax, (xs[0]+cw, ymid), (xs[1], ymid), color=C_JOINT, lw=2.0)
    arrow(ax, (xs[1]+cw, ymid), (xs[2], ymid), color=C_PURPLE, lw=2.0)

    # 部署框（③ 下方）
    dx, dy, dw, dh = xs[2], 2.05, cw, 1.02
    box(ax, dx, dy, dw, dh,
        ["部署本轮策略 π：默认 β=1", "直接条件化 'Advantage: positive' 采样（§VI-B）"],
        TEAL_D, fs=11.2, fs_sub=9.8)
    arrow(ax, (xs[2]+cw/2, cy), (xs[2]+cw/2, dy+dh), color=C_CTRL, lw=1.8)

    # 回流箭头：部署 → ①（折线，端点贴框）
    ax.plot([dx, 1.00], [dy+dh/2, dy+dh/2], color=C_AMBER, lw=1.8, zorder=3)
    ax.plot([1.00, 1.00], [dy+dh/2, cy], color=C_AMBER, lw=1.8, zorder=3)
    arrow(ax, (1.00, cy), (xs[0]+cw/2, cy), color=C_AMBER, lw=1.8)
    label(ax, 6.6, dy+dh/2-0.30,
          "把新 rollouts + 专家纠正加入 D_ℓ，重复 K 轮；"
          "policy 与 value 每轮都从 pre-trained checkpoint 微调（§V-D，避免 drift）",
          fs=9.6, color=C_AMBER)

    # 底部常量条
    box(ax, 0.35, 0.30, W-0.70, 1.42, [
        "论文写明的常量（引用，不是从图上量出来的）",
        "B=201 bins · N=50 lookahead · q≈30% 正 advantage · β=1 部署 · CFG β 取 1.5~2.5"
        " · advantage dropout 30% · ε 取 30% 分位",
        "value 670M / Gemma 3 4B / action expert 860M · 双 6-DoF 臂 50 Hz · 与 π*0.6 同架构",
    ], GRAY, fs=11.0, fs_sub=9.8)
    save(fig, "fig1_recap_pipeline")


# ===================================================================== 图 2
def fig_conditioning():
    """A 组：β=1 条件化 = 把行为策略截断到正 advantage 区间 / 对数据做硬筛选。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # 左：π_ref 形状示意（参数来自 [世界] 行），阴影 = 正 advantage 区间
    a = np.linspace(-2.5, 6.0, 600)
    f = ((1-D["w_bad"]) * np.exp(-0.5*((a-D["mu_g"])/D["sig"])**2)
         + D["w_bad"] * np.exp(-0.5*((a-D["mu_b"])/D["sig"])**2))
    a1.plot(a, f, color=C_JOINT, lw=2.2, label="行为策略 π_ref（形状示意）")
    tau = D["tau"]
    m = np.abs(a) < tau
    a1.fill_between(a[m], 0, f[m], color="#BCE3D3", alpha=0.9, zorder=1)
    fcond = np.where(m, f, 0.0)
    fcond = fcond / fcond.max() * f.max()   # 截断并放大的形状示意，不标注数值
    a1.plot(a, fcond, color=C_CTRL, lw=2.0, ls="--",
            label="β=1 条件化 π_ref(a|I)（示意）")
    a1.axvline(0, color=HINT, lw=1.0, ls=":")
    a1.axvline(-tau, color=C_CTRL, lw=1.2, ls="--")
    a1.axvline(tau, color=C_CTRL, lw=1.2, ls="--")
    a1.set_yticks([])
    a1.set_xlim(-2.5, 6.0)
    a1.set_ylim(0, max(f)*1.28)
    a1.set_xlabel("动作 a（θ=0 为成功点；曲线为形状示意）", fontsize=9.8)
    a1.set_title("正 advantage 区间 |a−θ|<τ 就是成功邻域", fontsize=11, color=INK, pad=10)
    a1.annotate(f"τ={D['tau']:.4f}", xy=(tau, max(f)*1.12), xytext=(tau+0.35, max(f)*1.12),
                fontsize=9.6, color=C_CTRL, va="center",
                arrowprops=dict(arrowstyle="->", color=C_CTRL, lw=1.0))
    a1.annotate(f"q={D['q']:.4f} 的数据为正", xy=(0, max(f)*0.55),
                xytext=(1.5, max(f)*0.92), fontsize=9.6, color="#04342C",
                arrowprops=dict(arrowstyle="->", color=C_CTRL, lw=1.0))
    a1.text(0.02, 0.96, f"阈值 ε={D['eps_l']:.4f} · V_ref={D['V_ref']:.4f}",
            transform=a1.transAxes, ha="left", va="top", fontsize=9.4, color=MUTE)
    a1.legend(frameon=False, fontsize=9.0, loc="upper right")
    clean_axes(a1, 9.2)

    # 右：硬筛选读数（水平条，负值向左，label 放条端左侧）
    a2.set_title("β=1 条件化对数据做了什么：按 I 硬筛选 + 重新归一化",
                 fontsize=11, color=INK, pad=10)
    names = ["被丢弃样本\n平均回报", "保留样本\n平均回报", "J(π_ref)", "J(β=1 条件化)"]
    vals = [D["R_drop"], D["R_keep"], D["J_ref"], D["J_cond"]]
    cols = ["#C9C7C0", "#9FB6CC", "#C9C7C0", C_JOINT]
    y = np.arange(len(names))[::-1]
    a2.barh(y, vals, 0.55, color=cols, edgecolor="white", linewidth=1.1)
    for yi, v in zip(y, vals):
        a2.text(v - 0.0015, yi, f"{v:.4f}", ha="right", va="center",
                fontsize=9.8, color=INK)
    a2.axvline(0, color=HINT, lw=0.9)
    a2.set_yticks(y, names, fontsize=9.6)
    a2.set_xlim(min(vals) - 0.016, 0.004)
    a2.set_xlabel("回报 / 策略质量 J（越高越好，上界 0）", fontsize=9.8)
    a2.grid(axis="x", alpha=0.22, linewidth=0.7)
    a2.set_axisbelow(True)
    clean_axes(a2, 9.2)

    fig.suptitle("合成探针 A 组：advantage 二值化与 β=1 条件化", fontsize=12.2,
                 color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.text(0.5, 0.05,
             f"行为均值 {D['mean_ref']:.4f} → 条件均值 {D['mean_cond']:.4f}"
             f" · 保留比例 q={D['q']:.4f} · KL={D['KL_cond']:.4f}",
             ha="center", va="bottom", fontsize=8.8, color=MUTE)
    footnote(fig)
    save(fig, "fig2_conditioning")


# ===================================================================== 图 3
def fig_binary_vs_continuous():
    """A 组：连续 advantage 加权扫描（温度 t）vs 二值条件化；副轴看激进度。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    T = np.array(D["T"])

    a1.plot(T, D["J_cont"], "o-", color=C_JOINT, lw=2.1, ms=7,
            label="连续加权：π_ref·exp(A/β)")
    a1.axhline(D["J_cond"], color=C_CTRL, lw=1.9, ls="--",
               label=f"二值条件化 β=1：{D['J_cond']:.4f}")
    a1.axhline(D["J_ref"], color=C_FLAT, lw=1.5, ls=":",
               label=f"J(π_ref)：{D['J_ref']:.4f}")
    for x, v in zip(T, D["J_cont"]):
        a1.annotate(f"{v:.4f}", (x, v), xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=8.6, color=INK)
    a1.set_xscale("log")
    a1.set_xticks(T, [str(t) for t in T])
    a1.set_xlabel("温度 t（β = t·std(A)，对数轴）", fontsize=9.8)
    a1.set_ylabel("策略质量 J（越高越好）", fontsize=9.8)
    a1.set_ylim(-0.046, 0.006)
    a1.set_title("连续加权扫温度才逼近二值；二值无需调 β", fontsize=11, color=INK, pad=10)
    a1.grid(alpha=0.22, linewidth=0.7)
    a1.legend(frameon=False, fontsize=8.8, loc="lower left",
              bbox_to_anchor=(0.0, 0.08))
    clean_axes(a1, 9.2)

    a2.plot(T, D["H_cont"], "s-", color=C_PURPLE, lw=2.0, ms=6, label="熵 H(π)")
    a2.plot(T, D["KL_cont"], "^-", color=C_AMBER, lw=2.0, ms=6,
            label="KL(π‖π_ref)")
    a2.axhline(D["H_cond"], color=C_CTRL, lw=1.5, ls="--",
               label=f"二值条件化熵 {D['H_cond']:.4f}")
    for x, h, k in zip(T, D["H_cont"], D["KL_cont"]):
        a2.annotate(f"{h:.3f}", (x, h), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8.2, color=C_PURPLE)
        a2.annotate(f"{k:.3f}", (x, k), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.2, color=C_AMBER)
    a2.set_xscale("log")
    a2.set_xticks(T, [str(t) for t in T])
    a2.set_xlabel("温度 t（对数轴）", fontsize=9.8)
    a2.set_ylabel("熵 / KL（激进度代理）", fontsize=9.8)
    a2.set_ylim(-0.1, 8.6)
    a2.set_title("温度越低越激进（熵降、KL 升）", fontsize=11, color=INK, pad=10)
    a2.grid(alpha=0.22, linewidth=0.7)
    a2.legend(frameon=False, fontsize=8.8, loc="center right")
    clean_axes(a2, 9.2)

    fig.suptitle("合成探针 A 组：连续 advantage（保留大小）vs 二值 I（只留排序）",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    footnote(fig)
    save(fig, "fig3_binary_vs_continuous")


# ===================================================================== 图 4
def fig_iterations():
    """B 组：只加离线 / +autonomous / +corrections 的迭代曲线。"""
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    K = np.arange(7)
    ax.plot(K, D["J_off"], "s--", color="#9B9992", lw=1.8, ms=6,
            label="只加离线数据（offline-only）")
    ax.plot(K, D["J_auto"], "^-", color=C_AMBER, lw=2.1, ms=7,
            label="只加 on-policy autonomous rollouts")
    ax.plot(K, D["J_corr"], "o-", color=C_JOINT, lw=2.3, ms=7,
            label="再加 expert corrections（强制 I=True）")
    ax.axhline(D["floor"], color=C_CTRL, lw=1.6, ls=":",
               label=f"解析最优下界 J1(μ1=θ1)={D['floor']:.4f}")
    for k, v in zip(K, D["J_corr"]):
        if k == 0:
            ax.annotate(f"{v:.4f}", (k, v), xytext=(0, -13),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8.6, color=C_JOINT)
        elif k in (1, 6):
            ax.annotate(f"{v:.4f}", (k, v), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=8.6, color=C_JOINT)
    ax.annotate(f"首轮增益 {D['gain1']:.4f}（一轮吃到大头）",
                xy=(1, D["J_corr"][1]), xytext=(2.2, -0.016),
                fontsize=9.6, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", color=C_JOINT, lw=1.1))
    ax.set_xticks(K, [str(k) for k in K])
    ax.set_xlabel("RECAP 迭代轮 k（Algorithm 1，K=6，40 seed 平均）", fontsize=9.8)
    ax.set_ylabel("部署状态回报 J1（越高越好）", fontsize=9.8)
    ax.set_ylim(-0.048, 0.008)
    ax.set_title("离线数据没覆盖的 OOD 状态，只有 on-policy 数据能补",
                 fontsize=11.5, color=INK, pad=12)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9.2, loc="center right")
    clean_axes(ax, 9.4)
    fig.suptitle("合成探针 B 组：on-policy 数据 / expert interventions / 迭代收敛",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    footnote(fig)
    save(fig, "fig4_iterations")


# ===================================================================== 图 5
def fig_bestofN():
    """C 组：best-of-N 与条件化的等价性边界。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.1),
                                 gridspec_kw=dict(width_ratios=[1.45, 1]))
    N = np.array(D["N_gr"])
    a1.plot(N, D["J_bon"], "o-", color=C_PURPLE, lw=2.2, ms=7,
            label="best-of-N：采 N 次取最大回报")
    a1.axhline(D["J_cond"], color=C_CTRL, lw=1.9, ls="--",
               label=f"β=1 条件化 / 拒绝采样：{D['J_cond']:.4f}")
    a1.axhline(D["J_ref"], color=C_FLAT, lw=1.4, ls=":",
               label=f"J(π_ref)：{D['J_ref']:.4f}")
    a1.annotate(f"N*={D['N_star']}\nJ={D['J_bon_star']:.4f}",
                xy=(D["N_star"], D["J_bon_star"]), xytext=(D["N_star"]*2.2, -0.013),
                fontsize=9.4, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.1))
    a1.set_xscale("log")
    a1.set_xticks(N, [str(n) for n in N])
    a1.set_xlabel("采样次数 N（对数轴）", fontsize=9.8)
    a1.set_ylabel("期望回报 J（越高越好，上界 0）", fontsize=9.8)
    a1.set_ylim(-0.048, 0.005)
    a1.set_title("N 要 8 才在 J 上追平条件化，代价是 N 倍采样",
                 fontsize=10.8, color=INK, pad=10)
    a1.grid(alpha=0.22, linewidth=0.7)
    a1.legend(frameon=False, fontsize=8.7, loc="lower right",
              bbox_to_anchor=(1.0, 0.16))
    clean_axes(a1, 9.2)

    # 右：两个「分不开 / 分得开」的对照
    x = np.arange(2)
    w = 0.34
    ent = [D["H_u_cond"], D["H_u_bon"]]
    tail = [D["tail_cond"], D["tail_bon"]]
    b1 = a2.bar(x - w/2, ent, w, color="#9FB6CC", edgecolor="white",
                linewidth=1.0, label="|a−θ| 熵")
    b2 = a2.bar(x + w/2, tail, w, color="#E3B873", edgecolor="white",
                linewidth=1.0, label="尾部质量 Pr(|a−θ|>0.7)")
    for b, v in zip(list(b1)+list(b2), ent+tail):
        a2.text(b.get_x()+b.get_width()/2, v+0.13, f"{v:.4f}",
                ha="center", fontsize=8.8, color=INK)
    a2.set_xticks(x, ["β=1 条件化", f"best-of-{D['N_star']}"], fontsize=9.6)
    a2.set_ylim(0, 8.0)
    a2.set_ylabel("读数（两组量纲不同，仅看组内高低）", fontsize=9.4)
    a2.set_title("反直觉：J 和熵都分不开，只有尾部质量能分开",
                 fontsize=10.6, color=INK, pad=10)
    a2.legend(frameon=False, fontsize=8.6, loc="upper left")
    a2.grid(axis="y", alpha=0.22, linewidth=0.7)
    a2.set_axisbelow(True)
    clean_axes(a2, 9.2)

    fig.suptitle("合成探针 C 组：条件化 positive ≠ 采样多次取最好",
                 fontsize=12.2, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    footnote(fig)
    save(fig, "fig5_bestofN")


# ===================================================================== 图 6
def fig_calibration_cost():
    """D 组：value 标定误差传导（左）、坏 corrections 让迭代变差（右）。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.1))

    # 左：标定噪声 σ 扫描 + 仿射平移免疫
    xs = np.arange(len(D["sig_cal"]))
    a1.plot(xs, D["J_cal"], "o-", color=C_JOINT, lw=2.2, ms=7,
            label="非线性标定噪声：V 估计 = V + ξ(a)")
    a1.axhline(D["J_ref"], color=C_FLAT, lw=1.5, ls=":",
               label=f"J(π_ref)={D['J_ref']:.4f}")
    for xi, v in zip(xs, D["J_cal"]):
        a1.annotate(f"{v:.4f}", (xi, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.2, color=C_JOINT)
    a1.set_xticks(xs, [f"{s:.3f}" for s in D["sig_cal"]], rotation=30, ha="right")
    a1.set_xlabel("标定噪声 σ（24 seed 平均）", fontsize=9.6)
    a1.set_ylabel("J(条件化)", fontsize=9.6)
    a1.set_ylim(-0.046, 0.008)
    a1.set_title("标定噪声单调削弱改进；σ=0.2 仍优于 J_ref",
                 fontsize=10.8, color=INK, pad=10)
    # 说明文字移到 fig 底部注，避免与图例 / 红色参考线 / 倾斜刻度重叠
    a1.grid(alpha=0.22, linewidth=0.7)
    a1.legend(frameon=False, fontsize=8.8, loc="lower left",
              bbox_to_anchor=(0.0, 0.12))
    clean_axes(a1, 9.0)

    # 右：坏 corrections
    a2.plot(D["c_bad"], D["J_bad"], "o-", color=C_FLAT, lw=2.2, ms=7,
            label="加 corrections 后 K 轮的 J1")
    a2.axhline(D["baseline"], color=C_CTRL, lw=1.7, ls="--",
               label=f"离线基线（不加数据）={D['baseline']:.4f}")
    for x, v in zip(D["c_bad"], D["J_bad"]):
        a2.annotate(f"{v:.4f}", (x, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.6, color=C_FLAT)
    a2.annotate("纠正偏差过大 → 反而比不加数据更差",
                xy=(4, D["J_bad"][-1]), xytext=(1.4, -0.060),
                fontsize=9.2, color=C_FLAT,
                arrowprops=dict(arrowstyle="->", color=C_FLAT, lw=1.1))
    a2.set_xlabel("纠正动作偏离 θ1 的偏差 c", fontsize=9.6)
    a2.set_ylabel("K 轮后部署回报 J1", fontsize=9.6)
    a2.set_ylim(-0.072, 0.004)
    a2.set_title("坏 corrections 反噬：论文「纠正本身非最优监督」",
                 fontsize=10.6, color=INK, pad=10)
    a2.grid(alpha=0.22, linewidth=0.7)
    a2.legend(frameon=False, fontsize=8.8, loc="upper right")
    clean_axes(a2, 9.2)

    fig.suptitle(f"合成探针 D 组：二值化丢 {D['info_loss']:.4f} 组内方差仍几乎无损，"
                 "但标定误差与坏纠正会把改进还回去",
                 fontsize=11.8, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.text(0.5, 0.055,
             f"仿射平移免疫：δ 取 -0.050~0.050 时 J 恒为 {D['J_affine'][0]:.4f}",
             ha="center", va="bottom", fontsize=8.8, color=MUTE)
    footnote(fig)
    save(fig, "fig6_calibration_cost")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in (("fig1_recap_pipeline", fig_pipeline),
                     ("fig2_conditioning", fig_conditioning),
                     ("fig3_binary_vs_continuous", fig_binary_vs_continuous),
                     ("fig4_iterations", fig_iterations),
                     ("fig5_bestofN", fig_bestofN),
                     ("fig6_calibration_cost", fig_calibration_cost)):
        if only and only not in name:
            continue
        fn()
        print("wrote", name)
