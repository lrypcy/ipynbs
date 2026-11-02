# -*- coding: utf-8 -*-
"""04 篇（Hi Robot）的配图：2 张架构 / 流水线图 + 4 张数据曲线，全部手工排版成 SVG。

与 `pi-vla/architecture/`、`pi-vla/knowledge-insulation/` 一致：
字体 PingFang SC（Hiragino Sans GB 会静默缺字），坐标以英寸为单位，
`bbox_inches="tight"` 会把越界元素悄悄撑大画布——所以画完**必须渲染 PNG 人眼看一遍**。

图 3 的数字来自官方项目页 https://www.pi.website/research/hirobot （论文 Figure 5 的读数）；
图 4–6 的数字来自同目录 `results/stdout.txt`（与 `hirobot_lab.py` 的输出逐位对齐）。

运行：~/Software/miniconda3/bin/python hirobot_figures.py    约 1 分钟
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

C_FLAT = "#A32D2D"      # 扁平
C_HIER = "#0F6E56"      # 分层
C_ORACLE = "#185FA5"    # 上界 / 参照
C_IDLE = "#888780"


# ---------------------------------------------------------------- 画图纸
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
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(0.25, min(w, h) / 2 - 0.01)}",
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


def label(ax, x, y, text, fs=10.5, color=HINT, ha="center", va="center", weight="400"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, fontweight=weight, zorder=4)


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
def fig_hierarchy():
    W, H = 13.0, 9.5
    fig, ax = canvas(W, H)

    # 左侧三个输入
    box(ax, 0.35, 7.55, 2.55, 1.00,
        ["基座 / 腕部相机图像", "$I^1_t,\\ldots,I^n_t$"], BLUE, fs=11.5, fs_sub=11)
    box(ax, 0.35, 6.25, 2.55, 1.00,
        ["开放式指令 / 插话", "$\\ell_t$（多从句、带约束）"], AMBER, fs=11.5, fs_sub=11)
    box(ax, 0.35, 1.55, 2.55, 1.00,
        ["机器人状态", "$q_t$（关节 + 夹爪）"], GRAY_D, fs=11.5, fs_sub=11)

    # 高层容器
    hx, hw, hy, hh = 3.30, 5.80, 5.00, 3.60
    top_h = container(ax, hx, hy, hw, hh, "高层策略 $p^{\\rm hi}$　System 2（VLM）", BLUE)
    vstack(ax, hx + 0.32, top_h, hw - 0.64, [
        (0.72, ["图像 + $\\ell_t$ → token 序列"], BLUE),
        (0.86, ["PaliGemma-3B VLM 主干 + 语言头",
                "next-token 交叉熵；$\\mathcal{D}_{syn}\\cup\\mathcal{D}_{labeled}$"], BLUE_D),
        (0.90, ["$u_t$：回应 / 澄清（走 TTS，不进低层）",
                "$\\hat\\ell_t$：一条 1–3 秒的原子命令"], TEAL),
    ], gap=0.14)

    # 低层容器
    lx, lw_, ly, lh = 3.30, 5.80, 1.00, 3.60
    top_l = container(ax, lx, ly, lw_, lh, "低层策略 $p^{\\rm lo}$　System 1（$\\pi_0$ VLA）", PURPLE)
    vstack(ax, lx + 0.32, top_l, lw_ - 0.64, [
        (0.72, ["图像 + $\\hat\\ell_t$ + $q_t$ → token 序列"], PURPLE),
        (0.86, ["PaliGemma-3B + flow matching action expert",
                "流匹配损失；$\\mathcal{D}_{labeled}\\cup\\mathcal{D}_{demo}$"], PURPLE_D),
        (0.90, ["$A_t=[a_t,\\ldots,a_{t+H-1}]$，$H=50$",
                "一次调用覆盖约 1 秒，控制频率 50 Hz"], TEAL),
    ], gap=0.14)

    # 数据流箭头
    arrow(ax, (2.90, 8.05), (hx, 8.05), color="#185FA5", lw=1.6)
    arrow(ax, (2.90, 6.75), (hx, 6.75), color="#854F0B", lw=1.6)
    arrow(ax, (2.90, 8.05), (2.90, 4.30), color="#185FA5", lw=1.4)
    arrow(ax, (2.90, 4.30), (lx, 4.30), color="#185FA5", lw=1.4)
    arrow(ax, (2.90, 2.05), (lx, 2.05), color=MUTE, lw=1.6)
    arrow(ax, (hx + hw / 2, hy), (lx + lw_ / 2, ly + lh), color="#0F6E56", lw=2.0)
    label(ax, hx + hw / 2 + 0.72, (hy + ly + lh) / 2, "$\\hat\\ell_t$", fs=12,
          color="#0F6E56", weight="500")
    label(ax, 1.62, 4.55, "图像两条通路共用", fs=9.5)

    # 右列注解
    rx, rw = 9.45, 3.20
    box(ax, rx, 7.35, rw, 1.65,
        ["高层什么时候被调用", "① 距上次推理满 1 秒", "② 收到新的用户语言（事件触发）"],
        GRAY, fs=11.5, fs_sub=10.5)
    box(ax, rx, 5.35, rw, 1.65,
        ["时间尺度", "高层 ≈ 1 Hz（或事件驱动）", "低层 ≈ 10 Hz 决策 / 50 Hz 控制"],
        GRAY, fs=11.5, fs_sub=10.5)
    box(ax, rx, 2.75, rw, 2.30,
        ["单步延迟（论文附录 B.3）", "高层 prefill+decode：",
         "　4090 47 + 13.2 ms；H100 17.3 + 5.7 ms",
         "低层：图像 14 + 观测 32 + 动作 27",
         "　= 73 ms（离板 +WiFi 86 ms）"],
        GRAY, fs=11.5, fs_sub=10)
    box(ax, rx, 1.00, rw, 1.45,
        ["低层语言槽上限（openpi）", "$\\pi_0$ 48 token / $\\pi_{0.5}$ 200 / $\\pi_0$-FAST 250",
         "48 里还要扣掉 BOS 与换行"],
        AMBER, fs=11.5, fs_sub=9.8)

    box(ax, 0.35, 0.22, W - 0.70, 0.62,
        ["两层共用同一个 VLM 主干初始化，但各自微调成两份权重；"
         "中间只有一条文本接口，所以「它在想什么」可以直接读出来，也可以在中途改写"],
        GRAY_D, fs=10.5, fs_sub=10.5)
    save(fig, "hirobot_hierarchy")


# ================================================================ 图 2
def fig_data_pipeline():
    W, H = 13.0, 9.20
    fig, ax = canvas(W, H)

    def col(x, w, y_top, items, gap=0.30):
        return vstack(ax, x, y_top, w, items, gap=gap)

    # 三列数据（底边都要留在两个训练目标容器的上方，否则箭头会穿框）
    TOP = 8.80
    # 列 1：原始演示
    x1, w1 = 0.35, 3.10
    col(x1, w1, TOP, [
        (1.05, ["$\\mathcal{D}_{demo}$：遥操作演示", "整条轨迹只有一个粗粒度目标"], GRAY_D),
        (1.05, ["「make a sandwich」", "（有动作 / 状态，没有逐步标签）"], GRAY),
    ])
    # 列 2：切分
    x2, w2 = 3.85, 3.20
    col(x2, w2, TOP, [
        (1.05, ["切成 1–3 秒的短技能", "再从动作里启发式提运动原语"], BLUE),
        (1.05, ["$\\mathcal{D}_{labeled}$", "$(\\hat\\ell_t,\\ I^1_t,\\ldots)$ 元组"], BLUE_D),
    ])
    # 列 3：合成
    x3, w3 = 7.45, 3.00
    col(x3, w3, TOP, [
        (1.25, ["$p^{\\rm gen}$：一个大 VLM 反着写", "$p^{\\rm gen}(\\ell_t,u_t\\mid I,\\ \\hat\\ell_{0..t},\\ \\mathcal{P})$",
                "三类场景：否定 / 现场纠正 / 具体约束"], AMBER),
        (1.05, ["$\\mathcal{D}_{syn}$：假设的用户提示与插话",
                "「that's not trash」→「I also want a Kitkat」"], AMBER),
    ])
    arrow(ax, (x1 + w1, TOP - 0.55), (x2, TOP - 0.55), color=MUTE, lw=1.6)
    arrow(ax, (x2 + w2, TOP - 0.55), (x3, TOP - 0.55), color=MUTE, lw=1.6)

    # 两个训练目标（内框顶边必须留在 container() 让出的 0.62 英寸以下）
    y_t, h_c = 3.30, 2.60
    ctop = y_t + h_c
    container(ax, 0.35, y_t, 5.60, h_c, "高层训练目标：next-token 交叉熵", BLUE, fs=12)
    box(ax, 0.67, y_t + 0.25, 4.96, 0.85,
        ["训练数据 = $\\mathcal{D}_{syn}\\cup\\mathcal{D}_{labeled}$"], BLUE_D, fs=11.5)
    box(ax, 0.67, y_t + 1.20, 4.96, 0.85,
        ["约 2 小时 / 8×H100；AdamW，lr $10^{-5}$，batch 512"], BLUE, fs=11)

    container(ax, 6.35, y_t, 6.30, h_c, "低层训练目标：flow matching", PURPLE, fs=12)
    box(ax, 6.67, y_t + 0.25, 5.66, 0.85,
        ["训练数据 = $\\mathcal{D}_{labeled}\\cup\\mathcal{D}_{demo}$（不含 $\\mathcal{D}_{syn}$）"],
        PURPLE_D, fs=11)
    box(ax, 6.67, y_t + 1.20, 5.66, 0.85,
        ["就是 $\\pi_0$ 原本的配方，只把语言条件换成原子命令 $\\hat\\ell_t$"], PURPLE, fs=11)

    arrow(ax, (x2 + w2 / 2, TOP - 2.40), (2.60, ctop), color="#185FA5", lw=1.6)
    arrow(ax, (x3 + w3 / 2, TOP - 2.60), (4.20, ctop), color="#854F0B", lw=1.6)
    arrow(ax, (x1 + w1 / 2, TOP - 2.40), (9.20, ctop), color="#534AB7", lw=1.6)
    arrow(ax, (x2 + w2 / 2, TOP - 2.40), (10.90, ctop), color="#534AB7", lw=1.6)

    box(ax, 0.35, 0.22, W - 0.70, 2.55,
        ["为什么合成标注只喂高层",
         "演示里的技能标签 $\\hat\\ell_t$ 是人标的、低层要照着执行的动作，这部分不需要改；",
         "缺的是「什么样的用户语句会引出这一步」——真实演示几乎只覆盖「把桌子清干净」这一种说法，",
         "而真实使用里有否定（只收垃圾，餐具别动）、现场纠正（那个不是垃圾）、具体约束（我对腌黄瓜过敏）。",
         "这些只有高层需要会，所以 $p^{\\rm gen}$ 生成的假设语句只进高层的训练集。",
         "代价：低层见过的措辞是固定的，高层必须把话说到这个集合里（§C 组实测）。"],
        GRAY_D, fs=11.5, fs_sub=10.5)
    save(fig, "hirobot_data_pipeline")


# ================================================================ 图 3
TASKS = ["桌面清理", "做三明治", "购物取货", "三项平均"]
IA = dict(flat=[36, 34, 39, 36], gpt=[35, 13, 41, 30], hi=[74, 83, 72, 76], exp=[100, 100, 100, 100])
TP = dict(flat=[61, 42, 28, 44], gpt=[63, 56, 72, 64], hi=[77, 80, 85, 81], exp=[82, 92, 93, 89])
SERIES = [("flat", "扁平 VLA", C_FLAT), ("gpt", "GPT-4o 作高层", "#854F0B"),
          ("hi", "Hi Robot", C_HIER), ("exp", "专家人工高层（上界）", C_IDLE)]


def _bars(ax, data, title, ymax=105):
    n, m = len(TASKS), len(SERIES)
    idx = np.arange(n)
    width = 0.80 / m
    for j, (k, name, col) in enumerate(SERIES):
        xs = idx - 0.40 + width / 2 + j * width
        ax.bar(xs, data[k], width=width * 0.92, color=col, label=name, zorder=3)
        for x, v in zip(xs, data[k], strict=True):
            ax.text(x, v + 1.6, str(v), ha="center", va="bottom", fontsize=9.2,
                    color=INK, zorder=4)
    ax.set_xticks(idx)
    ax.set_xticklabels(TASKS)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(title, fontsize=11.5, color=INK)
    ax.grid(axis="y", color="#E6E5E1", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    clean_axes(ax)


def fig_results():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for ax, data, title in ((axes[0], IA, "指令跟随准确率 IA（%）"),
                            (axes[1], TP, "任务进度 TP（%）")):
        _bars(ax, data, title)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.02, -0.10), ncol=4,
                   frameon=False, fontsize=10.5)
    axes[1].text(0.02, 0.03, "TP 与 IA 会分离：桌面清理上扁平 VLA 的 TP 61 分不低，\n"
                             "但 IA 只有 36 —— 它执行的是「清空整张桌子」这个默认行为，\n"
                             "恰好就是没人插话时的正确目标。",
                 transform=axes[1].transAxes, fontsize=9.5, color=MUTE, va="bottom")
    fig.suptitle("Hi Robot 与对照方法（每个任务 20 次试验；数据来自官方项目页，论文 Figure 5）",
                 fontsize=12, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(OUT, "hirobot_results.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "hirobot_results.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图 4（A 组）
# 数字来自 results/stdout.txt
STEPS = [250, 500, 1000, 2000, 4000]
ACC_F = [0.440, 0.482, 0.545, 0.618, 0.702]
ACC_H = [0.487, 0.522, 0.599, 0.697, 0.770]
MSE_F = [0.742, 0.700, 0.654, 0.594, 0.522]
MSE_H = [0.996, 0.910, 0.788, 0.630, 0.515]
FLOOR = 0.155
GAPC = {2: (0.794, 0.877), 4: (0.687, 0.762), 8: (0.618, 0.697)}


def fig_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    ax = axes[0]
    ax.plot(STEPS, ACC_F, "o-", color=C_FLAT, lw=2.0, ms=5.5, label="扁平 $(c,x)\\rightarrow a$")
    ax.plot(STEPS, ACC_H, "s-", color=C_HIER, lw=2.0, ms=5.5, label="分层 高层 + 低层")
    ax.annotate("分层 2000 步 0.697\n≈ 扁平 4000 步 0.702",
                xy=(2000, 0.697), xytext=(300, 0.905), fontsize=9.5, color=C_HIER,
                arrowprops=dict(arrowstyle="->", color=C_HIER, lw=1.1))
    ax.set_xlabel("训练步数（batch 64）", fontsize=11)
    ax.set_ylabel("技能选择准确率", fontsize=11.5)
    ax.set_ylim(0.40, 1.00)
    ax.grid(color="#EDEDEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    clean_axes(ax)
    ax.set_title("按「它想执行哪一步」打分（论文给扁平基线的协议）", fontsize=11, color=INK, pad=8)

    ax = axes[1]
    ax.plot(STEPS, MSE_F, "o-", color=C_FLAT, lw=2.0, ms=5.5, label="扁平")
    ax.plot(STEPS, MSE_H, "s-", color=C_HIER, lw=2.0, ms=5.5, label="分层")
    ax.axhline(FLOOR, color=C_ORACLE, ls=(0, (5, 3)), lw=1.4)
    ax.text(4150, FLOOR + 0.02, f"噪声下界 {FLOOR:.3f}", fontsize=9.5, color=C_ORACLE, ha="right")
    ax.annotate("小数据时扁平的 MSE 更低：\n它输出的是几个技能原型的叠加，\nMSE 最优但执行不了",
                xy=(250, 0.742), xytext=(430, 0.40), fontsize=10, color=C_FLAT,
                arrowprops=dict(arrowstyle="->", color=C_FLAT, lw=1.1))
    ax.set_xlabel("训练步数（batch 64）", fontsize=11)
    ax.set_ylabel("动作相对误差", fontsize=11.5)
    ax.set_ylim(0.10, 1.12)
    ax.grid(color="#EDEDEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    clean_axes(ax)
    ax.set_title("按连续动作的均方误差打分", fontsize=11, color=INK, pad=8)

    fig.suptitle("合成探针：8 个训练约束、6 个技能、3 个 seed 平均，未见场景上评测",
                 fontsize=12, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "hirobot_decomposition.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "hirobot_decomposition.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图 5（B 组）
RS = [1, 2, 3, 4, 6, 12]
STEP = {1: 39.0, 2: 44.4, 3: 33.9, 4: 44.5, 6: 55.6, 12: 103.1}
CALL = {1: 39.0, 2: 22.6, 3: 11.3, 4: 11.4, 6: 9.7, 12: 9.3}
LAT_P = {1: 0.12, 2: 0.65, 3: 1.15, 4: 1.65, 6: 2.70, 12: 5.50}
LAT_E = {1: 0.12, 2: 0.20, 3: 0.31, 4: 0.31, 6: 0.38, 12: 0.70}


def fig_replan():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    ax = axes[0]
    xs = np.arange(len(RS))
    ax.bar(xs - 0.20, [STEP[r] for r in RS], width=0.38, color=C_HIER,
           label="完成 10 个子任务所需步数", zorder=3)
    ax.bar(xs + 0.20, [CALL[r] for r in RS], width=0.38, color=C_ORACLE,
           label="高层被调用次数", zorder=3)
    for i, r in enumerate(RS):
        ax.text(i - 0.20, STEP[r] + 1.8, f"{STEP[r]:.0f}", ha="center", fontsize=9, color=INK)
        ax.text(i + 0.20, CALL[r] + 1.8, f"{CALL[r]:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"R={r}" for r in RS])
    ax.set_ylabel("步数 / 调用次数", fontsize=11.5)
    ax.set_ylim(0, 118)
    ax.grid(axis="y", color="#EDEDEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    clean_axes(ax)
    ax.set_title("R = 3（= 一个原子技能的步数）两个指标都最小", fontsize=11, color=INK, pad=8)
    ax.text(0.03, 0.62, "R=1、R=2 都被 R=3 支配：\n步数更多，调用也更多",
            transform=ax.transAxes, fontsize=9.8, color=C_FLAT)
    ax.text(0.55, 0.30, "R>3：拿墙钟时间\n换高层算力", transform=ax.transAxes,
            fontsize=9.8, color=MUTE)

    ax = axes[1]
    ax.plot(RS, [LAT_P[r] for r in RS], "o-", color=C_FLAT, lw=2.0, ms=5.5,
            label="只按周期重算")
    ax.plot(RS, [LAT_E[r] for r in RS], "s-", color=C_HIER, lw=2.0, ms=5.5,
            label="周期 + 事件触发")
    ax.plot(RS, [r / 2 for r in RS], ls=(0, (4, 3)), lw=1.2, color=HINT, label="$R/2$ 参照")
    for r in RS:
        ax.text(r, LAT_P[r] + 0.28, f"{LAT_P[r]:.1f}", ha="center", fontsize=9, color=C_FLAT)
    ax.set_xticks(RS)
    ax.set_xlabel("高层重算周期 $R$（步）", fontsize=11)
    ax.set_ylabel("纠正生效延迟（步）", fontsize=11.5)
    ax.set_ylim(0, 7.2)
    ax.grid(color="#EDEDEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    clean_axes(ax)
    ax.set_title("周期可以放慢，但纠正必须走另一条触发通路", fontsize=11, color=INK, pad=8)

    fig.suptitle("合成探针：子任务长度 L=3 步，高层每次调用有 15% 概率说错，400 个 episode 平均",
                 fontsize=12, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "hirobot_replan.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "hirobot_replan.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图 6（C / D 组）
KS = [0.0, 0.25, 0.5, 1.0, 2.0]
ERR_K = [0.0101, 0.0438, 0.1120, 0.2765, 0.5033]
ERR_AUG = 0.1100
ERR_RAND = 1.710

GROUPS = ["常见单位约束", "稀有单位约束", "复合约束"]
LAB = [0.932, 0.544, 0.412]
SYN = [0.863, 0.620, 0.596]


def fig_interface_data():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    ax = axes[0]
    ax.plot(KS, ERR_K, "o-", color=C_FLAT, lw=2.0, ms=5.5,
            label="低层没做过标签增强")
    ax.plot([0.0, 1.0], [0.0090, ERR_AUG], "s--", color=C_HIER, lw=1.8, ms=5.5,
            label="低层做过标签扰动增强")
    ax.axhline(ERR_RAND, color=HINT, ls=(0, (4, 3)), lw=1.2)
    ax.text(2.0, ERR_RAND + 0.04, f"标签完全随机 {ERR_RAND:.2f}", fontsize=9.5,
            color=HINT, ha="right")
    ax.annotate(f"$\\kappa=1$：{ERR_K[3]:.3f} → {ERR_AUG:.3f}",
                xy=(1.0, ERR_AUG), xytext=(0.05, 0.55), fontsize=10, color=C_HIER,
                arrowprops=dict(arrowstyle="->", color=C_HIER, lw=1.1))
    ax.set_xlabel("高层措辞偏离规范命令的程度 $\\kappa$", fontsize=11)
    ax.set_ylabel("动作相对误差", fontsize=11.5)
    ax.set_ylim(0, 1.85)
    ax.grid(color="#EDEDEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    clean_axes(ax)
    ax.set_title("C 组：高层说出的技能必须落在低层认识的措辞里", fontsize=11, color=INK, pad=8)

    ax = axes[1]
    xs = np.arange(len(GROUPS))
    ax.bar(xs - 0.19, LAB, width=0.36, color=C_IDLE, label="只用人标注 $\\mathcal{D}_{labeled}$", zorder=3)
    ax.bar(xs + 0.19, SYN, width=0.36, color=C_HIER, label="再加合成标注 $\\mathcal{D}_{syn}$", zorder=3)
    for i, (a, b) in enumerate(zip(LAB, SYN, strict=True)):
        ax.text(i - 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(i + 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=9, color=INK)
        d = b - a
        ax.text(i, max(a, b) + 0.055, f"{d:+.3f}", ha="center", fontsize=10.5,
                color=(C_HIER if d > 0 else C_FLAT), fontweight="500")
    ax.set_xticks(xs)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("高层技能选择准确率", fontsize=11.5)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", color="#EDEDEA", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
              fontsize=10.5)
    clean_axes(ax)
    ax.set_title("D 组：合成标注的收益集中在人标注覆盖不到的指令", fontsize=11, color=INK, pad=8)
    ax.text(0.01, 1.09, "头部的常见约束反而降了 0.069：合成数据稀释了它的样本份额",
            ha="left", va="center", fontsize=9.5, color=MUTE)

    fig.suptitle("合成探针：2000 步、3 个 seed 平均（D 组）；C 组用 4000 步训出的低层",
                 fontsize=12, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(OUT, "hirobot_interface_data.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "hirobot_interface_data.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in (("hirobot_hierarchy", fig_hierarchy),
                     ("hirobot_data_pipeline", fig_data_pipeline),
                     ("hirobot_results", fig_results),
                     ("hirobot_decomposition", fig_decomposition),
                     ("hirobot_replan", fig_replan),
                     ("hirobot_interface_data", fig_interface_data)):
        if only and only not in name:
            continue
        fn()
        print("wrote", name)
