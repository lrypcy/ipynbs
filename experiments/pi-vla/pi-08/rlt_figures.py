# -*- coding: utf-8 -*-
"""08 篇（RLT / RL Token）配图：1 张结构 / 流水线图 + 5 张数据图，全部手工排版成 SVG。

数据图的数字**逐位解析自同目录 results/stdout.txt 的 [MACHINE-BACKFILL-*] 块**
（与「正文回填表」同源），配图不硬编码任何合成数字。结构图按论文 §IV-A/§IV-B
文字重画，标注的是论文写明的常量（H=50、C=10、d=14、50 Hz、chunk=140、
z_rl 维数未给出、β 未给出），不复用任何官方图 / 截图。

字体 PingFang SC；无 Mermaid。画完必须渲染 PNG 用 ViewImage 逐张肉眼过一遍：
bbox_inches="tight" 会把越界元素悄悄撑大画布。

运行：~/Software/miniconda3/bin/python rlt_figures.py
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

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

# 结构图配色（沿用 05–07 篇）
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

C_MAIN = "#185FA5"      # 主曲线 / 完整方法
C_RED = "#A32D2D"       # 对照 / 失效
C_TEAL = "#0F6E56"      # oracle / 上界
C_AMBER = "#854F0B"     # 第三个系列
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


# 标注里出现的数字，全部逐位取自 results/stdout.txt（正文回填表同源）
A_TAU_MAX = grab(r"τ·E\[max_K\] = \+([\d.]+)")
A_TAU_MIN = grab(r"−τ/√π = -([\d.]+)")
A_TD_EXCESS = grab(r"单 Q [\d.]+（\+([\d.]+)）")
A_TD_SHORT = grab(r"clipped [\d.]+（-([\d.]+)）")
A_SPARSE_EP = float(re.findall(r"平均 ([\d.]+) 个 episode", STDOUT_TEXT)[-1])
A_DEC_LABEL = grab(r"决策 (\d+→\d+)", cast=str)
A_DEC_STEP, A_DEC_CHUNK = (int(t) for t in A_DEC_LABEL.split("→"))
A_N_SINGLE = grab(r"(\d+) seed 平均", cast=int)
A_TD_STEPS = grab(r"迭代 TD（(\d+) 步", cast=int)
A_BUFFER = grab(r"好特征 buffer=(\d+)", cast=int)


# ---------------------------------------------------------------- 画图工具
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.0, fs_sub=9.8, lw=0.9, r=0.12,
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


def container(ax, x, y, w, h, title, style, fs=12.2, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(0.25, min(w, h) / 2 - 0.01)}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=1))
    ax.text(x + 0.30, y + h - 0.30, title, ha="left", va="center",
            fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
    return y + h - 0.60


def vstack(ax, x, y_top, w, items, gap=0.20):
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


def clean_axes(ax, fs=10.0):
    ax.tick_params(labelsize=fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBB6")
        ax.spines[s].set_linewidth(0.8)


def val(key, i=0):
    """取回填块里的单个标量，保证逐位来自 stdout.txt。"""
    return float(D[key][i])


def note(ax, text, xy, xytext, color, fs=9.0, ha="left", va="center"):
    """白底标注：文字落在空白区、带描边 bbox，箭头指向数据；避免压住柱/曲线/数字。"""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=color,
                ha=ha, va=va, zorder=6,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor=color, linewidth=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                                shrinkA=4, shrinkB=3))


# ================================================================ 图 1（结构图）
def fig_architecture():
    W, H = 13.6, 9.9
    fig, ax = canvas(W, H)
    label(ax, W / 2, 9.62, "RLT：冻结 VLA 上挂一个小 actor-critic，RL token 是 chunk-level 接口",
          fs=14, color=INK, weight="500")
    label(ax, W / 2, 9.28, "按论文 §IV-A / §IV-B / Alg.1 文字重画 · 标注为论文写明的常量 · 非论文原图",
          fs=9.6, color=HINT)

    # ---- 左：① 冻结 VLA + RL token 抽取（§IV-A）
    lx, lw_, ly, lh = 0.40, 5.85, 3.30, 5.55
    top = container(ax, lx, ly, lw_, lh,
                    "① 冻结 VLA + RL token 抽取（§IV-A，VLA 对 L_ro 冻结）", BLUE)
    vstack(ax, lx + 0.24, top, lw_ - 0.48, [
        (0.72, ["输入：2 腕相机 + 1 基座相机图像、语言 l、本体状态",
                r"$\pi_{0.6}$ = VLM backbone + diffusion action expert"], BLUE_D),
        (0.86, [r"最后层 token 序列  $z_{1:M}\in\mathbb{R}^{M\times d_z}$",
                r"动作块  $\tilde a_{t:t+H-1}\in\mathbb{R}^{H\times d}$（H=50=1 s @50 Hz）"], GRAY_D),
        (0.86, [r"追加可学习 $e_{rl}$，encoder  $g_\varphi([z_{1:M}, e_{rl}])$",
                r"取特殊 token 输出 → RL token  $z_{rl}$（紧凑向量）"], PURPLE_D),
        (0.86, [r"decoder $d_\varphi$ + 线性 $h_\varphi$ 自回归重建原 embedding",
                "L_ro = Σ_i ‖重建_i − z_bar_i‖²，z_bar = sg(z)"], TEAL_D),
    ], gap=0.16)
    # 瓶颈标记
    label(ax, lx + lw_ / 2, 3.62,
          "z_rl 必须留够信息供 decoder 重建 → 信息瓶颈（维数论文未给出）",
          fs=9.4, color=PURPLE["ec"])

    # ---- 右：② 在线 RL：小 actor-critic（§IV-B）
    rx, rw, ry, rh = 7.35, 5.85, 3.30, 5.55
    top2 = container(ax, rx, ry, rw, rh,
                     "② 在线 RL：轻量 actor-critic（§IV-B / TD3）", TEAL)
    vstack(ax, rx + 0.24, top2, rw - 0.48, [
        (0.72, [r"RL state  $x=(z_{rl}, s^{p})$",
                r"$\theta_{vla}$ 与 $\varphi$ 全部冻结，只训 $\pi_\theta / Q_\psi$"], TEAL_D),
        (0.86, [r"critic  $Q_\psi(x, a_{1:C})$ → 标量；离策略 TD",
                r"$\hat Q=\sum_{t'=1}^{C}\gamma^{t'-1}r_{t'}+\gamma^C Q_{\psi'}(x',a')$"], BLUE_D),
        (0.86, ["TD3：双 Q 集成取 min 计算 target + 目标网络",
                "reward 为稀疏二值 r_T=1(成功)/0"], AMBER),
        (0.86, [r"actor  $\pi_\theta(\cdot|x,\tilde a_{1:C})=\mathcal{N}(\mu_\theta,\sigma^2 I)$ — 精炼 VLA 块",
                "VLA 参考块训练时 50% dropout 置零 / 推理始终提供"], PURPLE_D),
    ], gap=0.16)
    label(ax, rx + rw / 2, 3.62,
          r"$a_{1:C}\in\mathbb{R}^{C\times d}=\mathbb{R}^{140}$（C=10<H, d=14）；$L_\pi=-Q+\beta\|a-\tilde a\|^2$",
          fs=9.0, color=TEAL["ec"])

    # 两大块之间的数据流（标签带白底，落在加宽后的间隙内，不压容器/箭头）
    arrow(ax, (lx + lw_, 5.05), (rx, 5.05), color=PURPLE["ec"], lw=2.0)
    ax.text((lx + lw_ + rx) / 2, 5.30, "z_rl（冻结）", ha="center", va="center",
            fontsize=8.6, color=PURPLE["ec"], fontweight="500", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    arrow(ax, (lx + lw_, 4.35), (rx, 4.35), color=TEAL["ec"], lw=1.5, dash=(0, (4, 2)))
    ax.text((lx + lw_ + rx) / 2, 4.60, "参考块 a[1:C]", ha="center", va="center",
            fontsize=8.6, color=TEAL["ec"], zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # ---- 底部：常量带 + 公式带
    box(ax, 0.40, 1.75, 12.80, 1.20, [
        "论文常量：基座 pi_0.6 ｜ 50 Hz ｜ H=50（1 s）｜ d=14 ｜ C=10 ｜ chunk=140 维 ｜ C<H ｜ 稀疏二值 reward ｜ dropout 50%",
        "论文未给出：RL token 瓶颈维数、β 具体数值、γ、σ、warmup 步数、α —— 合成探针里的 b、β=2.5 均为自设",
    ], GRAY, fs=10.8, fs_sub=9.6)
    box(ax, 0.40, 0.30, 12.80, 1.25, [
        "式(1) z_rl = g_phi([z_1:M, e_rl])_(M+1)　·　式(2) L_ro 自回归重建（stop-gradient）　·　"
        "式(3) chunk-level 回报　·　式(4) 高斯 actor 条件于参考块　·　式(5) BC anchor β‖a−a_ref‖²",
        "配方：冻结大 VLA 的感知先验 + 只训小 actor-critic + chunk 级 TD 缩短稀疏奖励下的信用分配 horizon",
    ], GRAY_D, fs=10.2, fs_sub=9.6)
    save(fig, "fig1_rlt_architecture")


# ================================================================ 图 2（A 组 bottleneck）
def fig_bottleneck():
    b = D["B_GRID"].astype(int)
    recon, ctrl, gain = D["RECON"], D["CTRL"], D["RECON_GAIN"]
    knee = int(val("KNEE"))
    floor = val("CTRL_FLOOR")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.1))

    # 左：三段关系（双轴）
    ax2 = a1.twinx()
    l1, = a1.plot(b, recon, "o-", color=C_MAIN, lw=2.1, ms=6, label="L_ro 重建误差")
    l2, = ax2.plot(b, ctrl, "s-", color=C_RED, lw=2.1, ms=6, label="下游控制误差")
    a1.axvline(knee, color=C_TEAL, lw=1.8, ls="--")
    a1.axvspan(0.6, knee - 0.4, color=C_AMBER, alpha=0.06)
    a1.axvspan(knee + 0.4, 12.4, color=C_TEAL, alpha=0.06)
    a1.set_xlabel("RL token 瓶颈维度 b")
    a1.set_ylabel("L_ro 重建误差（越低越好）", color=C_MAIN)
    ax2.set_ylabel("下游控制误差（越低越好）", color=C_RED)
    a1.set_xticks(b)
    a1.set_ylim(0, 14)
    ax2.set_ylim(0, 8)
    a1.set_title("三段关系：维度 → 重建误差 → 控制精度", fontsize=11, color=INK, pad=10)
    a1.legend([l1, l2], ["L_ro 重建误差", "下游控制误差"],
              loc="upper right", frameon=False, fontsize=9.2)
    ax2.annotate(f"拐点 b={knee}=r\n控制已饱和 {floor:.4f}",
                 xy=(knee, floor), xytext=(knee + 2.3, 2.55),
                 fontsize=9.2, color=C_TEAL,
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_TEAL, lw=0.7, alpha=0.92),
                 arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1))
    a1.annotate("b>r：L_ro 仍降，\n控制不再提升", xy=(10, recon[9]),
                xytext=(7.0, 9.3), fontsize=9.0, color=C_AMBER,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_AMBER, lw=0.7, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.1))
    clean_axes(a1, 9.2)
    clean_axes(ax2, 9.2)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    # 右：边际增益
    bars = a2.bar(b, gain, 0.66,
                  color=[C_RED if g > 0.2 else C_TEAL for g in gain],
                  edgecolor="white", linewidth=1.0)
    for bb, g in zip(b, gain):
        a2.text(bb, g + 0.10, f"{g:.2f}", ha="center", fontsize=8.4, color=INK)
    a2.axvline(knee + 0.5, color=C_AMBER, lw=1.6, ls="--")
    a2.set_xlabel("RL token 瓶颈维度 b")
    a2.set_ylabel("每加一维的重建增益")
    a2.set_xticks(b)
    a2.set_ylim(0, 4.7)
    a2.set_title(f"边际增益在 b=r 后塌到噪声级 σ²={gain[8]:.2f}",
                 fontsize=11, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    a2.annotate("信号方向（b≤6）", xy=(1, gain[0]), xytext=(3.1, 4.35),
                fontsize=9.2, color=C_RED,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_RED, lw=0.7, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1))
    a2.annotate(f"纯噪声方向（b>6）\n每维只追回 {gain[8]:.2f}", xy=(9.5, gain[8]),
                xytext=(7.9, 1.9), fontsize=9.2, color=C_TEAL,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_TEAL, lw=0.7, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1))
    clean_axes(a2, 9.2)
    fig.suptitle("合成探针 A 组：RL token 瓶颈（数据逐位来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_bottleneck")


# ================================================================ 图 3（B 组 双 Q）
def fig_doubleq():
    single, decoup, clipped = val("DQ_SINGLE"), val("DQ_DECOUPLED"), val("DQ_CLIPPED")
    vtrue, tds, tdc = val("TD_VTRUE"), val("TD_SINGLE"), val("TD_CLIPPED")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    # 左：单步过估计偏差
    names = ["单 Q\n贪心(max)", "解耦双 Q\n(选/评分离)", "TD3 clipped\n(min Q₁,Q₂)"]
    vals = [single, decoup, clipped]
    cols = [C_RED, C_PURPLE, C_TEAL]
    bars = a1.bar(names, vals, 0.55, color=cols, edgecolor="white", linewidth=1.2)
    for bb, v in zip(bars, vals):
        a1.text(bb.get_x() + bb.get_width() / 2, v + (0.07 if v >= 0 else -0.13),
                f"{v:+.4f}", ha="center", fontsize=9.4, color=INK)
    a1.axhline(0, color=INK, lw=1.0)
    a1.set_ylabel("价值估计偏差（真值 0）")
    a1.set_ylim(-0.95, 2.08)
    a1.set_title(f"单步：双 Q 抑制过估计（{A_N_SINGLE} seed 平均）",
                 fontsize=11, color=INK, pad=10)
    a1.grid(axis="y", alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    note(a1, f"τ·E[max_K]=+{A_TAU_MAX:.4f}", xy=(0, single), xytext=(1.25, 1.80),
         color=C_RED, ha="left")
    note(a1, f"−τ/√π=−{A_TAU_MIN:.4f}\n（可控小幅低估）", xy=(2, clipped),
         xytext=(0.12, -0.74), color=C_TEAL, ha="left")
    clean_axes(a1, 9.2)

    # 右：迭代 TD 复合
    names2 = ["真值 V*", f"单 Q\n({A_TD_STEPS} 步)", f"clipped 双 Q\n({A_TD_STEPS} 步)"]
    vals2 = [vtrue, tds, tdc]
    bars2 = a2.bar(names2, vals2, 0.55,
                   color=[GRAY_D["ec"], C_RED, C_TEAL], edgecolor="white", linewidth=1.2)
    for bb, v in zip(bars2, vals2):
        a2.text(bb.get_x() + bb.get_width() / 2, v + 0.4, f"{v:.2f}",
                ha="center", fontsize=9.4, color=INK)
    a2.axhline(vtrue, color=C_AMBER, lw=1.6, ls="--")
    a2.text(2.35, vtrue + 0.5, "V*=10.00", ha="right", fontsize=9.0, color=C_AMBER)
    a2.set_ylabel("迭代 TD 后的价值估计")
    a2.set_ylim(0, 25)
    a2.set_title("迭代引导：单 Q 高估被复合放大", fontsize=11, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    note(a2, f"+{A_TD_EXCESS:.2f} 高估", xy=(1, tds), xytext=(1.50, 18.9),
         color=C_RED, ha="left")
    note(a2, f"转入 −{A_TD_SHORT:.2f} 低估\n但 |偏差| 小得多", xy=(2, tdc),
         xytext=(1.42, 12.6), color=C_TEAL, ha="left")
    clean_axes(a2, 9.2)
    fig.suptitle("合成探针 B 组：TD3 双 Q 与单 Q 对照（数字来自 results/stdout.txt）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig3_doubleq")


# ================================================================ 图 4（B 组 critic + 方差）
def fig_critic_variance():
    keys = ["Q_GOOD_SMALL", "Q_GOOD_BIG", "Q_GOOD_N50", "Q_BAD_SMALL", "Q_BAD_BIG"]
    qv = [val(k) for k in keys]
    labels = ["好特征\n小critic", "好特征\n多容量", f"好特征\nbuffer={A_BUFFER}",
              "坏特征\n小critic", "坏特征\n多容量"]
    vs, vc = val("VAR_STEP"), val("VAR_CHUNK")
    ratio = val("VAR_RATIO")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.9))

    # 左：冻结特征下小 critic 的 Q-MSE
    cols = [C_MAIN, C_MAIN, C_AMBER, C_RED, C_RED]
    bars = a1.bar(labels, qv, 0.6, color=cols, edgecolor="white", linewidth=1.1)
    for bb, v in zip(bars, qv):
        a1.text(bb.get_x() + bb.get_width() / 2, v + 0.0032, f"{v:.4f}",
                ha="center", fontsize=9.0, color=INK)
    a1.axhline(qv[2], color=C_RED, lw=1.3, ls="--")
    a1.text(0.02, 0.96, f"虚线：好特征 buffer={A_BUFFER} 的 Q-MSE {qv[2]:.4f}",
            transform=a1.transAxes, ha="left", va="top", fontsize=8.5,
            color=C_RED, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.94))
    a1.set_ylabel("Q-MSE（越低越好）")
    a1.set_ylim(0, 0.088)
    a1.set_title("冻结 backbone：信息含量决定 critic 可学习性", fontsize=10.8,
                 color=INK, pad=10)
    a1.grid(axis="y", alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    note(a1, "加容量几乎不掉点\n→ 冻结不是瓶颈", xy=(1, qv[1] * 0.45),
         xytext=(0.15, 0.062), color=C_MAIN, ha="left")
    clean_axes(a1, 8.8)

    # 右：chunk vs step 信用分配方差
    bars2 = a2.bar([f"step-level\n{A_DEC_STEP} 决策", f"chunk-level\n{A_DEC_CHUNK} 决策"],
                   [vs, vc], 0.5,
                   color=[C_RED, C_TEAL], edgecolor="white", linewidth=1.1)
    for bb, v in zip(bars2, [vs, vc]):
        a2.text(bb.get_x() + bb.get_width() / 2, v * 1.25, f"{v:.1f}",
                ha="center", fontsize=9.4, color=INK)
    a2.set_yscale("log")
    a2.set_ylim(10, 12000)
    a2.set_ylabel("REINFORCE 梯度方差（对数轴）")
    a2.set_title("chunk 化把信用分配方差压低一个量级", fontsize=10.8, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7, which="both")
    a2.set_axisbelow(True)
    note(a2, f"比值 {ratio:.1f}×（≈C²）\n决策 {A_DEC_LABEL}", xy=(1, vc),
         xytext=(0.5, 7600), color=C_TEAL, ha="center")
    clean_axes(a2, 9.2)
    fig.suptitle("合成探针 B 组：冻结特征 / 小 critic 与 chunk 级信用分配（数字来自 stdout.txt）",
                 fontsize=11.2, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig4_critic_variance")


# ================================================================ 图 5（C 组 β / dropout）
def fig_beta_dropout():
    betas, cerr, cerr_mc, csucc = D["BETAS"], D["C_ERR"], D["C_ERR_MC"], D["C_SUCC"]
    bstar = val("BETA_STAR")
    ps, mask, corr, norm = D["DROPOUTS"], D["MASK"], D["CORR"], D["NORM"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    x = np.arange(len(betas))

    # 左：β 倒 U（动作误差 + 成功率双轴）
    ax2 = a1.twinx()
    # 让 a1 后绘制，白底标注不被 ax2 的成功率曲线盖住
    a1.set_zorder(ax2.get_zorder() + 1)
    a1.patch.set_visible(False)
    l1, = a1.plot(x, cerr, "-", color=C_MAIN, lw=2.2, label="动作误差(解析)")
    a1.plot(x, cerr_mc, "o", color=C_MAIN, ms=5, alpha=0.65)
    l2, = ax2.plot(x, csucc, "s--", color=C_AMBER, lw=1.8, ms=5, label="成功率(MC)")
    i_star = int(np.argmin(cerr))
    a1.axvline(i_star, color=C_TEAL, lw=1.8, ls="--")
    a1.set_xticks(x, [f"{t:g}" for t in betas], fontsize=8.6)
    a1.set_xlabel("BC anchor 系数 β（探针自设，论文数值未给出）")
    a1.set_ylabel("动作误差（越低越好）", color=C_MAIN)
    ax2.set_ylabel("成功率", color=C_AMBER)
    a1.set_ylim(0, 2.25)
    ax2.set_ylim(0, 0.48)
    a1.set_title(f"β 倒 U 形：解析最优 β*={bstar:g}", fontsize=10.8, color=INK, pad=10)
    a1.legend([l1, l2], ["动作误差(解析/MC)", "成功率(MC)"],
              loc="upper center", frameon=False, fontsize=8.8)
    note(a1, f"β=0 {cerr[0]:.4f}\n(最大单项下降)", xy=(0, cerr[0]),
         xytext=(0.10, 2.13), color=C_RED, ha="left", fs=8.8)
    note(a1, f"β*={bstar:g} → {cerr[i_star]:.4f}", xy=(i_star, cerr[i_star]),
         xytext=(6.1, 0.28), color=C_TEAL, ha="left", fs=9.0)
    note(a1, f"β={betas[-1]:g} {cerr[-1]:.4f}\n(anchor 太强压住改进)",
         xy=(len(betas) - 1, cerr[-1]), xytext=(8.9, 2.03),
         color=C_AMBER, ha="right", fs=8.8)
    clean_axes(a1, 8.8)
    clean_axes(ax2, 8.8)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    # 右：dropout 扫描
    a2.plot(ps, mask, "o-", color=C_RED, lw=2.0, ms=6, label="参考隐藏(masked)")
    a2.plot(ps, corr, "s-", color=C_AMBER, lw=2.0, ms=6, label="参考污染(corrupted)")
    a2.plot(ps, norm, "^-", color=C_MAIN, lw=2.0, ms=6, label="参考正常(normal)")
    a2.axvline(0.5, color=C_TEAL, lw=1.8, ls="--")
    a2.text(0.5, 7.85, "论文 50%", fontsize=9.0, color=C_TEAL, ha="center",
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.94))
    a2.set_xlabel("参考动作 dropout 比例 p")
    a2.set_ylabel("动作误差（越低越好）")
    a2.set_xticks(ps)
    a2.set_ylim(0, 8.2)
    a2.set_title("dropout 太低 → 复制 BC；50% 兑现绝大部分收益", fontsize=10.8,
                 color=INK, pad=10)
    a2.grid(alpha=0.25, linewidth=0.7)
    a2.legend(frameon=False, fontsize=9.0, loc="upper right")
    note(a2, f"p=0 masked {mask[0]:.4f}", xy=(0, mask[0]),
         xytext=(0.06, 4.4), color=C_RED, ha="left", fs=8.8)
    clean_axes(a2, 9.2)
    fig.suptitle("合成探针 C 组：β anchor 与参考动作 dropout（数字来自 results/stdout.txt）",
                 fontsize=11.2, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_beta_dropout")


# ================================================================ 图 6（D 组 边界）
def fig_boundaries():
    ctrl = [val("CTRL_B1"), val("CTRL_B2"), val("CTRL_B3"),
            val("CTRL_B6"), val("CTRL_B12")]
    bs = [1, 2, 3, 6, 12]
    floor = val("CTRL_FLOOR")
    pg = D["P_GOOD"]
    ss, sc = D["SUCC_STEP"], D["SUCC_CHUNK"]
    qonly, qpol = val("Q_VLA_ONLY"), val("Q_WITH_ONPOLICY")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.4, 4.7))

    # (a) token 维度过小
    bars = a1.bar([str(t) for t in bs], ctrl, 0.58,
                  color=[C_RED, C_RED, C_AMBER, C_TEAL, C_TEAL],
                  edgecolor="white", linewidth=1.1)
    for bb, v in zip(bars, ctrl):
        a1.text(bb.get_x() + bb.get_width() / 2, v + 0.12, f"{v:.4f}",
                ha="center", fontsize=8.6, color=INK)
    a1.axhline(floor, color=C_TEAL, lw=1.6, ls="--")
    a1.text(2.45, 1.30, f"饱和地板 {floor:.4f}", ha="left", fontsize=8.6,
            color=C_TEAL, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.94))
    a1.set_xlabel("RL token 瓶颈维度 b")
    a1.set_ylabel("控制误差（越低越好）")
    a1.set_ylim(0, 8.2)
    a1.set_title("① token 过小：信息被掐断", fontsize=10.6, color=INK, pad=10)
    a1.grid(axis="y", alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    clean_axes(a1, 8.8)

    # (b) 回报稀疏
    x = np.arange(len(pg))
    w = 0.36
    a2.bar(x - w / 2, ss, w, color=C_RED, label=f"step ({A_DEC_STEP} 决策)",
           edgecolor="white", linewidth=1.0)
    a2.bar(x + w / 2, sc, w, color=C_TEAL, label=f"chunk ({A_DEC_CHUNK} 决策)",
           edgecolor="white", linewidth=1.0)
    a2.set_yscale("log")
    a2.set_xticks(x, [f"{p:g}" for p in pg])
    a2.set_xlabel(f"单决策成功概率 p_good\n"
                  f"p=0.5 仍需 {A_SPARSE_EP:.1f} episode 才见一次成功(chunk)",
                  fontsize=9.0)
    a2.set_ylabel("P(succ)（对数轴）")
    a2.set_ylim(1e-17, 3)
    a2.set_title("② 回报稀疏：chunk 化不凭空造信息", fontsize=10.6, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7, which="both")
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=8.8, loc="lower left")
    clean_axes(a2, 8.8)

    # (c) 离线数据偏差
    bars3 = a3.bar(["仅 VLA warmup\n离线数据", "+ on-policy\n/ 人类干预"],
                   [qonly, qpol], 0.52, color=[C_RED, C_TEAL],
                   edgecolor="white", linewidth=1.1)
    for bb, v in zip(bars3, [qonly, qpol]):
        a3.text(bb.get_x() + bb.get_width() / 2, v + 0.008, f"{v:.4f}",
                ha="center", fontsize=9.2, color=INK)
    a3.set_ylabel("最优动作区 Q-MSE（越低越好）")
    a3.set_ylim(0, 0.52)
    a3.set_title("③ 离线偏差：warmup 数据覆盖不到 a*", fontsize=10.6,
                 color=INK, pad=10)
    a3.grid(axis="y", alpha=0.25, linewidth=0.7)
    a3.set_axisbelow(True)
    note(a3, "覆盖 a* 邻域后\n误差下降", xy=(1, qpol), xytext=(0.42, 0.34),
         color=C_TEAL, ha="left", fs=9.0)
    clean_axes(a3, 8.8)

    fig.suptitle("合成探针 D 组：三种失效模式（数字来自 results/stdout.txt）",
                 fontsize=11.2, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig6_boundaries")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in (("fig1_rlt_architecture", fig_architecture),
                     ("fig2_bottleneck", fig_bottleneck),
                     ("fig3_doubleq", fig_doubleq),
                     ("fig4_critic_variance", fig_critic_variance),
                     ("fig5_beta_dropout", fig_beta_dropout),
                     ("fig6_boundaries", fig_boundaries)):
        if only and only not in name:
            continue
        fn()
        print("wrote", name)
