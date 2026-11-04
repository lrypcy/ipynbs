# -*- coding: utf-8 -*-
"""05 篇（π0.5）的配图：3 张结构 / 流水线图 + 4 张数据图，全部手工排版成 SVG。

与 01–04 篇一致：字体 PingFang SC，无 Mermaid、无论文截图（π0.5 是
arXiv nonexclusive-distrib 许可，官方图不复用）。画完必须渲染 PNG 肉眼过一遍：
`bbox_inches="tight"` 会把越界元素悄悄撑大画布。

数据图的数字全部来自同目录 `results/stdout.txt`（与 `pi05_lab.py` 的输出逐位对齐）；
结构图按论文 §4 / 附录与 openpi pinned commit 的 tokenizer / pi0_config 重画。

运行：~/Software/miniconda3/bin/python pi05_figures.py    约 40 秒
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
PT = 1.0 / 72.0

C_JOINT = "#185FA5"     # joint / 完整配方
C_FLAT = "#A32D2D"      # flat / 对照
C_IMPLICIT = "#854F0B"  # implicit HL
C_CTRL = "#0F6E56"      # oracle / 对照上界


# ---------------------------------------------------------------- 画图纸
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.5, fs_sub=10.5, lw=0.9, r=0.12,
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


def container(ax, x, y, w, h, title, style, fs=12.5, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(0.25, min(w, h) / 2 - 0.01)}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=1))
    ax.text(x + 0.30, y + h - 0.32, title, ha="left", va="center",
            fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
    return y + h - 0.62


def vstack(ax, x, y_top, w, items, gap=0.22):
    cy = y_top
    for h, lines, st in items:
        cy -= h
        box(ax, x, cy, w, h, lines, st)
        cy -= gap
    return cy + gap


def arrow(ax, p1, p2, color=MUTE, lw=1.5, rad=0.0, dash=None, head="-|>", ms=16):
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


def clean_axes(ax, fs=10.5):
    ax.tick_params(labelsize=fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBB6")
        ax.spines[s].set_linewidth(0.8)


# ================================================================ 图 1
def fig_architecture():
    W, H = 12.6, 10.4
    fig, ax = canvas(W, H)
    label(ax, W / 2, 10.10, "π0.5：同一套权重、两次前向，中间只传一条文本子任务",
          fs=14, color=INK, weight="500")

    # ---- 输入带
    for x, w, lines, st in [
        (0.45, 3.55, ["多视角图像", "$I^1_t,\\ldots,I^n_t$"], BLUE),
        (4.30, 3.55, ["长程任务 / 开放式指令", "$\\ell$：clean the bedroom"], AMBER),
        (8.15, 3.95, ["本体状态（离散化进 prompt）", "$q_t$：256 bins × 18 维"], GRAY_D),
    ]:
        box(ax, x, 8.85, w, 0.90, lines, st, fs=11, fs_sub=10)
    for x in (2.22, 6.08, 10.12):
        arrow(ax, (x, 8.82), (x, 8.41), color=MUTE, lw=1.2)

    # ---- 第 1 次前向
    hx, hw, hy, hh = 0.75, 8.55, 6.35, 2.05
    container(ax, hx, hy, hw, hh,
              "第 1 次前向：高层语义子任务预测（PaliGemma-VLM 2B 主干 / next-token CE）",
              BLUE)
    box(ax, hx + 0.30, 6.60, 3.30, 1.12,
        ["VLM 主干自回归解码", "把 $\\ell$ 与观测翻成下一步该做什么"], BLUE_D,
        fs=11, fs_sub=10)
    box(ax, 4.70, 6.60, 4.00, 1.12,
        ["$\\hat\\ell$：一条原子子任务文本", "例：pick up the pillow"], TEAL,
        fs=11, fs_sub=10)
    arrow(ax, (3.90, 7.16), (4.70, 7.16), color="#185FA5", lw=1.6)

    # ---- ℓ̂：两次前向之间只传这一条文本（一条首尾明确的路径）
    g = "#0F6E56"
    arrow(ax, (8.70, 7.16), (10.35, 7.16), color=g, lw=1.8, head="-")
    arrow(ax, (10.35, 7.16), (10.35, 5.95), color=g, lw=1.8, head="-")
    arrow(ax, (10.35, 5.95), (1.60, 5.95), color=g, lw=1.8, head="-")
    arrow(ax, (1.60, 5.95), (1.60, 5.52), color=g, lw=1.8)
    label(ax, 11.45, 6.68, "两次前向之间", fs=9.5)
    label(ax, 11.45, 6.34, "$\\hat\\ell$ 文本接口", fs=10.5, color=g,
          weight="500")

    # ---- 第 2 次前向
    lx, lw_, ly, lh = 0.75, 8.55, 3.10, 2.40
    container(ax, lx, ly, lw_, lh,
              "第 2 次前向：低层动作生成（flow matching）", PURPLE)
    label(ax, lx + 4.30, 4.94, "$\\hat\\ell$ + 图像 + $q_t$ 重新编码成 prefix",
          fs=9.8, color=MUTE)
    box(ax, lx + 0.30, 3.40, 2.60, 1.20,
        ["$\\hat\\ell$ / 图像 / $q_t$", "prefix 全可见"], PURPLE_D,
        fs=11, fs_sub=10)
    box(ax, lx + 3.15, 3.40, 2.30, 1.20,
        ["共享 VLM 主干", "提供 K / V"], BLUE_D, fs=11, fs_sub=10)
    box(ax, lx + 5.70, 3.40, 2.55, 1.20,
        ["action expert 300M", "10 步去噪 → $A_t$"], TEAL_D, fs=11, fs_sub=10)
    arrow(ax, (lx + 2.90, 4.00), (lx + 3.15, 4.00), color="#534AB7", lw=1.5)
    arrow(ax, (lx + 5.45, 4.00), (lx + 5.70, 4.00), color="#534AB7", lw=1.5)
    label(ax, W / 2, 2.70,
          "信息单向：VLM embedding 从不 attend action expert　·　"
          "输出 $A_t=[a_t,\\ldots,a_{t+H-1}]$，$H=50$ / 50 Hz",
          fs=10, color=MUTE)

    # ---- 底部：公式与对比
    box(ax, 0.75, 0.35, 11.55, 1.55, [
        "$\\pi_\\theta(a_{t:t+H},\\hat\\ell\\mid o_t,\\ell)="
        "\\pi_\\theta(a_{t:t+H}\\mid o_t,\\hat\\ell)\\,"
        "\\pi_\\theta(\\hat\\ell\\mid o_t,\\ell)$",
        "动作只条件于预测出的 $\\hat\\ell$，不直接条件于原始 $\\ell$"
        "　·　Hi Robot 是「两个模型、两套权重」，π0.5 是「一个模型、两次前向」",
    ], GRAY, fs=12.5, fs_sub=10)
    save(fig, "pi05_architecture")


# ================================================================ 图 2
def fig_cotraining():
    W, H = 13.4, 8.9
    fig, ax = canvas(W, H)
    label(ax, W / 2, 8.60, "异构共训练：六类数据先塞进一个模型，再分两阶段改变动作表示",
          fs=14, color=INK, weight="500")

    # ---- 左：数据源
    sx, sw = 0.35, 3.55
    top = container(ax, sx, 1.15, sw, 6.85, "数据来源（不是同一个分布）", GRAY)
    items = [
        (0.78, ["MM 移动操作臂演示", "约 100 个真实家庭 / 约 400 小时"], BLUE),
        (0.78, ["ME 多环境非移动机器人", "更多房间、更多物体摆放"], BLUE),
        (0.78, ["CE 跨本体实验室数据", "含 OXE；仅预训练，后训练去掉"], PURPLE),
        (0.78, ["HL 高层子任务预测", "先定位目标 bbox，再预测子任务"], AMBER),
        (0.78, ["WD 网页多模态数据", "caption / VQA / 目标定位"], TEAL),
        (0.78, ["VI 语言指令", "仅后训练，约占高层移动示例 11%"], RED),
    ]
    vstack(ax, sx + 0.22, top, sw - 0.44, items, gap=0.18)

    # ---- 中：两个阶段
    px, pw = 4.25, 5.15
    top1 = container(ax, px, 4.75, pw, 3.25,
                     "阶段 1　预训练 · 280k steps", BLUE)
    vstack(ax, px + 0.24, top1, pw - 0.48, [
        (0.62, ["动作 = FAST 离散 token"], BLUE_D),
        (0.80, ["目标：next-token 交叉熵", "$\\alpha=0$（还没有 flow matching 损失）"], BLUE),
        (0.70, ["97.6% 的样本不来自目标 MM 家务数据"], TEAL),
    ], gap=0.16)
    top2 = container(ax, px, 1.15, pw, 3.25,
                     "阶段 2　后训练 · 80k steps", PURPLE)
    vstack(ax, px + 0.24, top2, pw - 0.48, [
        (0.62, ["随机初始化 action expert 加入"], PURPLE_D),
        (0.80, ["FAST token CE + 流匹配损失", "$\\alpha=10.0$；10 步去噪"], PURPLE),
        (0.78, ["VI 语言指令加入；CE 实验室数据去掉",
                "VI 约占高层移动示例 11%"], RED),
    ], gap=0.16)
    arrow(ax, (px + pw / 2, 4.75), (px + pw / 2, 4.40),
          color="#534AB7", lw=2.0)

    # ---- 右：推理
    rx, rw = 9.75, 3.30
    top_r = container(ax, rx, 2.55, rw, 4.30, "部署", TEAL)
    vstack(ax, rx + 0.22, top_r, rw - 0.44, [
        (0.72, ["先解码 $\\hat\\ell$ 文本子任务"], TEAL_D),
        (0.72, ["再跑 10 步去噪出动作块"], TEAL_D),
        (0.72, ["50 Hz 控制 / $H=50$"], TEAL_D),
        (0.82, ["动作归一化到 [-1,1]", "按 1% / 99% 分位数"], GRAY_D),
    ], gap=0.20)
    arrow(ax, (px + pw, 3.30), (rx, 4.30), color="#0F6E56", lw=1.8)
    arrow(ax, (px + pw, 6.30), (rx, 5.30), color="#185FA5", lw=1.5)
    label(ax, 7.15, 4.57, "两阶段共享权重", fs=9.5, ha="left")

    label(ax, W / 2, 0.55,
          "第一阶段的赌注：让 VLM 在网页语义 / 别的本体 / 别的家里先把「常识」学够，"
          "再用少量目标平台数据把连续动作接上", fs=10.5, color=MUTE)
    save(fig, "pi05_cotraining")


# ================================================================ 图 3
def fig_attention():
    W, H = 11.6, 8.6
    fig, ax = canvas(W, H)
    label(ax, W / 2, 8.30, "一个序列里的三套注意力规则：两套动作表示共存但不互相看见",
          fs=13.5, color=INK, weight="500")

    groups = [("图像 / 文本 / 状态", 3, BLUE_D),
              ("FAST 动作 token", 3, AMBER),
              ("流匹配动作 embedding", 3, PURPLE_D)]
    n = 9
    # mask：True = 可见
    M = np.zeros((n, n), dtype=bool)
    M[:3, :3] = True                         # prefix 全可见
    M[3:6, :3] = True                       # FAST 看 prefix
    M[3:6, 3:6] = np.tril(np.ones((3, 3), dtype=bool))   # FAST 自回归
    M[6:9, :3] = True                       # flow 看 prefix
    M[6:9, 6:9] = True                      # flow 彼此双向
    # flow 不看 FAST；prefix / FAST 都不看 flow（信息单向）

    x0, y0, cell = 2.55, 1.55, 0.58
    for i in range(n):
        for j in range(n):
            face = "#D6E7F7" if M[i, j] else "#F2F1ED"
            ax.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell),
                                   cell * 0.94, cell * 0.94,
                                   facecolor=face, edgecolor="#FFFFFF",
                                   linewidth=1.1, zorder=2))
    # 3×3 分区：外框 + 两条竖分隔线 + 两条横分隔线（覆盖整个网格，不是只画下半）
    ax.add_patch(Rectangle((x0, y0), n * cell - 0.04, n * cell - 0.04,
                           fill=False, edgecolor="#5F5E5A", linewidth=1.2, zorder=3))
    for k in (1, 2):
        ax.plot([x0 + k * 3 * cell - 0.02] * 2, [y0, y0 + n * cell - 0.04],
                color="#5F5E5A", linewidth=1.3, zorder=3)
        ax.plot([x0, x0 + n * cell - 0.04], [y0 + k * 3 * cell - 0.02] * 2,
                color="#5F5E5A", linewidth=1.3, zorder=3)
    col_names = ["图像 / 文本\n/ 状态", "FAST\n动作 token", "流匹配动作\nembedding"]
    row_names = ["图像 / 文本 / 状态", "FAST 动作 token", "流匹配动作 embedding"]
    for k, (cn, (_, _, st)) in enumerate(zip(col_names, groups)):
        ax.text(x0 + (k * 3 + 1.5) * cell, y0 + n * cell + 0.35, cn,
                ha="center", va="center", fontsize=9.2, color=st["ec"], zorder=4)
        ax.text(x0 - 0.22, y0 + (7.5 - 3 * k) * cell, row_names[k],
                ha="right", va="center", fontsize=9.5, color=st["ec"], zorder=4)
    ax.text(x0 + n * cell / 2, y0 + n * cell + 1.16, "key（被看）",
            ha="center", fontsize=10.5, color=MUTE)
    ax.text(0.30, y0 + n * cell / 2, "query\n（去看）", ha="center", va="center",
            fontsize=10.5, color=MUTE)

    # 右侧说明
    bx = 8.35
    for y, lines, st in [
        (6.95, ["① prefix 全可见", "图像 / 语言 / 离散化状态彼此互看"], BLUE),
        (5.55, ["② FAST token 因果", "看 prefix + 之前的动作 token；", "next-token CE 训练"], AMBER),
        (3.85, ["③ 流匹配 embedding 双向", "看 prefix + 彼此；不看 FAST token"], PURPLE),
        (2.15, ["④ 信息单向", "VLM embedding 从不 attend", "action expert 的 token"], TEAL),
    ]:
        box(ax, bx, y, 2.95, 1.12, lines, st, fs=10.5, fs_sub=9.5)
    label(ax, W / 2, 0.62,
          "τ 经 MLP 投影后用 adaptive RMSNorm 注入 action expert 每层；"
          "这里切断的是注意力可见性，不是梯度（梯度隔离见 03 篇 KI）",
          fs=10, color=MUTE)
    save(fig, "pi05_attention")


# ================================================================ 图 4
def fig_ablation():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    names = ["no WD", "MM only", "no ME", "no CE", "full"]
    web = [0.455, 0.487, 0.667, 0.703, 0.687]
    iderr = [0.256, 0.268, 0.273, 0.259, 0.271]
    env = [0.164, 0.196, 0.199, 0.171, 0.168]
    cols = ["#B9B7B0", "#B9B7B0", "#E3B873", "#B9A6D8", C_JOINT]
    x = np.arange(len(names))
    bars = a1.bar(x, web, 0.62, color=cols, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, web):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9.5, color=INK)
    a1.set_xticks(x, names, fontsize=9.5)
    a1.set_ylim(0, 0.82)
    a1.set_ylabel("技能选择准确率")
    a1.set_title("web-only 类型作目标（分布外）", fontsize=11, color=INK, pad=10)
    a1.grid(axis="y", alpha=0.25, linewidth=0.7)
    a1.set_axisbelow(True)
    a1.annotate("只有 WD 能把未见类型\n的语义接进技能头",
                xy=(4, 0.687), xytext=(0.05, 0.77), ha="left",
                fontsize=9.5, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", color=C_JOINT, lw=1.1))
    clean_axes(a1, 9.5)

    w = 0.36
    a2.bar(x - w / 2, iderr, w, label="ID 端到端误差", color="#9FB6CC",
           edgecolor="white", linewidth=1.0)
    a2.bar(x + w / 2, env, w, label="新环境动作误差（真值技能条件）",
           color="#A8C9BC", edgecolor="white", linewidth=1.0)
    for i in range(len(names)):
        a2.text(x[i] - w / 2, iderr[i] + 0.006, f"{iderr[i]:.3f}",
                ha="center", fontsize=8.8, color=MUTE)
        a2.text(x[i] + w / 2, env[i] + 0.006, f"{env[i]:.3f}",
                ha="center", fontsize=8.8, color=MUTE)
    a2.set_xticks(x, names, fontsize=9.5)
    a2.set_ylim(0, 0.34)
    a2.set_ylabel("相对动作误差")
    a2.set_title("分布内几乎没变，新环境才拉开差距", fontsize=11, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    a2.legend(loc="upper left", frameon=False, fontsize=9.5)
    clean_axes(a2, 9.5)
    fig.suptitle("合成探针 A 组：共训练配方消融（2500 步 × 3 seed 平均）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "pi05_ablation")


# ================================================================ 图 5
def fig_env_scaling():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ns = [3, 6, 12, 25, 50, 100]
    errs = [0.356, 0.239, 0.192, 0.166, 0.167, 0.166]
    ctrl = [0.149, 0.152, 0.152, 0.151, 0.151, 0.152]
    accs = [0.908, 0.911, 0.913, 0.908, 0.929, 0.917]
    a1.plot(ns, errs, "o-", color=C_JOINT, lw=2.1, ms=7, label="只用训练环境")
    a1.plot(ns, ctrl, "s--", color=C_CTRL, lw=1.6, ms=6,
            label="对照：测试环境也进训练池")
    for xx, yy in zip(ns, errs):
        a1.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    a1.set_xscale("log")
    a1.set_xticks(ns, [str(n) for n in ns])
    a1.set_xlabel("训练时见过的环境数 N")
    a1.set_ylabel("新环境动作误差（真值技能条件）")
    a1.set_ylim(0.10, 0.40)
    a1.set_title("「怎么做」随环境数变好", fontsize=11, color=INK, pad=10)
    a1.grid(alpha=0.25, linewidth=0.7)
    a1.legend(frameon=False, fontsize=9.5, loc="upper right")
    clean_axes(a1, 9.5)

    a2.plot(ns, accs, "o-", color=C_IMPLICIT, lw=2.1, ms=7)
    for xx, yy in zip(ns, accs):
        a2.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    a2.set_xscale("log")
    a2.set_xticks(ns, [str(n) for n in ns])
    a2.set_xlabel("训练时见过的环境数 N")
    a2.set_ylabel("技能选择准确率")
    a2.set_ylim(0.85, 0.96)
    a2.set_title("「做什么」几乎不动", fontsize=11, color=INK, pad=10)
    a2.grid(alpha=0.25, linewidth=0.7)
    clean_axes(a2, 9.5)
    fig.suptitle("合成探针 B 组：环境数量 3→100（完整配方，只改环境池大小）",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "pi05_env_scaling")


# ================================================================ 图 6
def fig_joint_vs_flat():
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    metrics = ["ID 端到端", "新环境端到端", "未见复合约束端到端"]
    joint = [0.271, 0.267, 1.159]
    implicit = [0.344, 0.336, 1.005]
    flat = [0.303, 0.290, 0.761]
    x = np.arange(3)
    w = 0.25
    for off, vals, col, name in [
        (-w, joint, C_JOINT, "joint：共训练 + 两阶段推理"),
        (0.0, implicit, C_IMPLICIT, "implicit HL：只共训练，部署不跑高层"),
        (w, flat, C_FLAT, "flat：无子任务监督 / 无技能输入"),
    ]:
        bars = ax.bar(x + off, vals, w, color=col, edgecolor="white",
                      linewidth=1.1, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, metrics, fontsize=10.5)
    ax.set_ylabel("相对动作误差（越低越好）")
    ax.set_ylim(0, 1.32)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("合成探针 C 组：离散子任务接口在分布内省误差、在分布外成单点故障",
                 fontsize=11.5, color=INK, pad=12)
    ax.annotate("分布内 joint 最好", xy=(0 - w, 0.271), xytext=(0.02, 0.52),
                ha="left", fontsize=9.5, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", color=C_JOINT, lw=1.1))
    ax.annotate("分布外 joint 最差", xy=(2 - w, 1.159), xytext=(1.15, 1.22),
                fontsize=9.5, color=C_FLAT,
                arrowprops=dict(arrowstyle="->", color=C_FLAT, lw=1.1))
    clean_axes(ax, 9.5)
    fig.tight_layout()
    save(fig, "pi05_joint_vs_flat")


# ================================================================ 图 7
def fig_state_tokens():
    # 曲线用探针的精确值（stdout 只打印 4 位小数，中间几档会显示成 0.0000）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pi05_lab import state_quant_probe
    q = state_quant_probe()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    bins = [8, 16, 32, 64, 256]
    errs = [q[b] for b in bins]
    floor = q["floor"]
    a1.plot(bins, errs, "o-", color=C_JOINT, lw=2.1, ms=7, label="量化引入的相对误差")
    a1.axhline(floor, color=C_FLAT, lw=1.7, ls="--",
               label=f"噪声下界 {floor:.4f}")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xticks(bins, [str(b) for b in bins])
    a1.set_xlabel("状态量化 bin 数")
    a1.set_ylabel("相对误差（对数轴）")
    a1.set_ylim(1e-06, 0.6)
    a1.set_title("量化精度从来不是瓶颈", fontsize=11, color=INK, pad=10)
    a1.grid(alpha=0.25, linewidth=0.7, which="both")
    a1.legend(frameon=False, fontsize=9.5, loc="lower left")
    a1.annotate("8 bins 也只有 0.0040\n= 下界的 0.03×", xy=(8, q[8]),
                xytext=(16, 0.02), fontsize=9.5, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", color=C_JOINT, lw=1.1))
    a1.annotate(f"256 bins：{q[256]:.2e}", xy=(256, q[256]),
                xytext=(32, 2e-05), fontsize=9.5, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", color=C_JOINT, lw=1.1))
    clean_axes(a1, 9.5)

    # 右：token 槽位预算
    toks = [36, 54, 54, 54, 72]
    a2.bar([str(b) for b in bins], toks, 0.55, color="#9FB6CC",
           edgecolor="white", linewidth=1.1)
    for i, v in enumerate(toks):
        a2.text(i, v + 3, str(v), ha="center", fontsize=9.5, color=INK)
    a2.axhline(48, color=C_FLAT, lw=1.8, ls="--")
    import matplotlib.patheffects as pe
    white_edge = [pe.withStroke(linewidth=3.2, foreground="white")]
    a2.axhline(200, color=C_CTRL, lw=1.8, ls="--")
    a2.text(0.05, 205, "π0.5：200 token 槽", ha="left", fontsize=9.5,
            color=C_CTRL, path_effects=white_edge)
    a2.text(0.05, 56, "π0：48 token 槽", ha="left", fontsize=9.5,
            color=C_FLAT, path_effects=white_edge)
    a2.set_ylim(0, 225)
    a2.set_xlabel("状态量化 bin 数（18 维状态）")
    a2.set_ylabel("prompt 里状态占的 token 数")
    a2.set_title("瓶颈是 token 槽位，不是量化精度", fontsize=11, color=INK, pad=10)
    a2.grid(axis="y", alpha=0.25, linewidth=0.7)
    a2.set_axisbelow(True)
    clean_axes(a2, 9.5)
    fig.suptitle("合成探针 D 组：openpi 的 π0.5 tokenizer 把状态离散化进 prompt",
                 fontsize=11.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "pi05_state_tokens")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in (("pi05_architecture", fig_architecture),
                     ("pi05_cotraining", fig_cotraining),
                     ("pi05_attention", fig_attention),
                     ("pi05_ablation", fig_ablation),
                     ("pi05_env_scaling", fig_env_scaling),
                     ("pi05_joint_vs_flat", fig_joint_vs_flat),
                     ("pi05_state_tokens", fig_state_tokens)):
        if only and only not in name:
            continue
        fn()
        print("wrote", name)
