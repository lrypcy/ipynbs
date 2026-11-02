"""π 系列文章的架构图（4 张 SVG，手工排版）。

与 `pi-vla/pi0-flow-matching/`、`pi-vla/fast-action-tokenizer/` 里那些**数据曲线**
不同，这里画的是**结构**：模块层次、token 形状、参数量、信息流方向。
Mermaid 只能表达拓扑，画不出这些，所以本目录的图全部手工排版。

输出写在 `figures/`，正文用的是 `assets/img/pi-vla/*.svg`（博客仓库里）。
同时写一份 PNG 方便人眼核对（PNG 只是校对用，不进博客）。

运行：~/Software/miniconda3/bin/python pi_architecture_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
AMBER = dict(fc="#FAEEDA", ec="#854F0B", tc="#412402")
RED = dict(fc="#FCEBEB", ec="#A32D2D", tc="#501313")
GRAY = dict(fc="#F8F7F4", ec="#888780", tc="#2C2C2A")
GRAY_D = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")

INK = "#2C2C2A"
MUTE = "#5F5E5A"
HINT = "#888780"

PT = 1.0 / 72.0  # 1pt 对应多少「英寸」——本文件所有坐标都在英寸尺度上


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def box(ax, x, y, w, h, lines, style, fs=11.5, fs_sub=10.5, lw=0.9, r=0.12, dashed=False):
    """lines: [主标题, 副标题, ...]；第一行用 500 字重。文字块在盒内垂直居中。"""
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
    # 行高按字号算（1.42 倍行距），整体再垂直居中；放不下就整体压缩
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
    """带标题的外框。返回内部可用区域的顶边 y（已让开标题）。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(0.25, min(w, h) / 2 - 0.01)}",
        linewidth=lw, edgecolor=style["ec"], facecolor=style["fc"], zorder=1))
    ax.text(x + 0.30, y + h - 0.32, title, ha="left", va="center",
            fontsize=fs, fontweight="500", color=style["tc"], zorder=3)
    return y + h - 0.62


def vstack(ax, x, y_top, w, items, gap=0.16):
    """items: [(h, lines, style, kwargs), ...]，自上而下排。返回最底边的 y。"""
    cy = y_top
    for h, lines, st, kw in items:
        cy -= h
        box(ax, x, cy, w, h, lines, st, **kw)
        cy -= gap
    return cy + gap


def arrow(ax, p1, p2, color=MUTE, lw=1.4, rad=0.0, dash=None, head="-|>", ms=15):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=head, mutation_scale=ms, linewidth=lw,
        color=color, zorder=2,
        connectionstyle=f"arc3,rad={rad}",
        **({"linestyle": dash} if dash else {})))


def label(ax, x, y, text, fs=10.5, color=HINT, ha="center", va="center", weight="400"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, fontweight=weight, zorder=4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 图 1
def fig_pi0_architecture():
    W, H = 12.6, 7.9
    fig, ax = canvas(W, H)

    # 第一层：两类 token 的构造
    cy, ch = 4.25, 3.15
    top = container(ax, 0.4, cy, 5.2, ch, "prefix ｜ PaliGemma 3B（VLM 预训练权重）", BLUE)
    vstack(ax, 0.75, top, 4.5, [
        (0.70, ["图像 2 到 3 路 → SigLIP", "每相机 256 个 patch token"], BLUE_D, dict(fs=11)),
        (0.70, ["语言指令 token", "PaliGemma 自带 tokenizer"], BLUE_D, dict(fs=11)),
        (0.70, ["拼起来 542 到 798 个 token × 2048"], BLUE_D, dict(fs=10.5)),
    ])
    top = container(ax, 7.0, cy, 5.2, ch, "suffix ｜ action expert 约 300M", PURPLE)
    vstack(ax, 7.35, top, 4.5, [
        (0.70, ["本体状态 1 个 + 加噪动作 50 个"], PURPLE_D, dict(fs=11)),
        (0.70, ["流时间的正弦编码", "每个动作 token 各一份"], PURPLE_D, dict(fs=11)),
        (0.70, ["先过 MLP 升到 1024（π0.5 换 adaRMS）"], PURPLE_D, dict(fs=10.5)),
    ])
    arrow(ax, (3.0, cy), (3.4, 4.03))
    arrow(ax, (9.5, cy), (9.2, 4.03))

    # 第二层：联合注意力（两套权重唯一的交互点）
    box(ax, 2.1, 2.95, 8.4, 1.05,
        ["联合 self-attention × 18 层：两套权重唯一的交互点",
         "三块掩码：prefix 内部双向 ｜ state 只看见自己和 prefix ｜ action 之间双向"],
        GRAY_D, fs=12, fs_sub=10.5)

    # 第三层：各自的 MLP
    box(ax, 0.75, 1.75, 4.5, 0.85, ["主干 MLP", "width 2048，mlp_dim 16384"], BLUE_D, fs=11)
    box(ax, 7.35, 1.75, 4.5, 0.85, ["action expert MLP", "width 1024，mlp_dim 4096"], PURPLE_D, fs=11)
    arrow(ax, (3.0, 2.95), (3.0, 2.63))
    arrow(ax, (9.5, 2.95), (9.5, 2.63))
    label(ax, 6.3, 2.45, "各自独立，所以\nwidth 可以不一样", fs=10, va="top")

    # 第四层：输出
    box(ax, 7.35, 0.72, 4.5, 0.85,
        ["action_out_proj 取 suffix 后 50 个 token", "输出速度场 v（50 × 32）"],
        TEAL, fs=11, fs_sub=10.5)
    arrow(ax, (9.5, 1.75), (9.5, 1.60))

    box(ax, 0.75, 0.72, 4.5, 0.85,
        ["主干这一路不参与输出", "prefix 与动作无关 → 可只算一次、缓存 K/V"],
        GRAY, fs=11, fs_sub=10.5, dashed=True)
    arrow(ax, (3.0, 1.75), (3.0, 1.60), dash="dashed")

    label(ax, 6.3, 0.30, "训练：对 v 做 MSE 逼近 ε − A_t　｜　推理：从 t = 1 欧拉积分 10 步到 t < 0.05",
          fs=10.5, color="#04342C", weight="500")

    save(fig, "pi0_architecture")


# ------------------------------------------------------------------ 图 2
def fig_pi0_inference_loop():
    """推理期的时间线：prefix 只算一次，suffix 的 51 个 token 走十步。"""
    W, H = 12.6, 5.9
    fig, ax = canvas(W, H)

    # ---- 上排：主干流程
    box(ax, 0.4, 4.00, 3.5, 1.10,
        ["prefix 前向一次", "542 到 798 个 token，算完缓存 K/V"], BLUE, fs=12, fs_sub=10.5)
    arrow(ax, (3.92, 4.55), (4.28, 4.55))

    top = container(ax, 4.30, 3.75, 4.70, 1.85, "suffix ｜ 10 步去噪", PURPLE)
    x0, bw, gap, bh = 4.58, 0.38, 0.06, 0.70
    for i in range(10):
        x = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch(
            (x, top - 1.30), bw, bh, boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0.8, edgecolor=PURPLE_D["ec"], facecolor=PURPLE_D["fc"], zorder=2))
        ax.text(x + bw / 2, top - 1.30 + bh / 2, str(i + 1), ha="center", va="center",
                fontsize=9.5, color=PURPLE_D["tc"], zorder=3)
    label(ax, 6.65, top - 1.52, "每一步都是同一组权重", fs=10)

    arrow(ax, (9.02, 4.55), (9.38, 4.55))
    box(ax, 9.40, 4.00, 2.80, 1.10,
        ["输出 1 秒动作块", "50 × 32"], TEAL, fs=12, fs_sub=10.5)

    # ---- 下排：三块注解
    box(ax, 0.4, 1.80, 3.5, 1.35,
        ["token·次（2 相机）", "每步全量重算 5930", "复用 K/V 后 1103，省 5.38×"],
        AMBER, fs=11.5, fs_sub=10.5)
    arrow(ax, (2.15, 4.00), (2.15, 3.18), color=HINT, dash="dashed")

    box(ax, 4.30, 1.80, 4.70, 1.35,
        ["每步只前向 suffix 的 51 个 token", "同一个 action expert 被复用十次"],
        GRAY_D, fs=11.5, fs_sub=10.5)
    arrow(ax, (6.65, 3.75), (6.65, 3.18), color=HINT, dash="dashed")

    box(ax, 9.40, 1.80, 2.80, 1.35,
        ["x ← x − 0.1 · v_t", "t ← t − 0.1，直到 t < 0.05"],
        BLUE_D, fs=11, fs_sub=10)
    arrow(ax, (10.80, 4.00), (10.80, 3.18), color=HINT, dash="dashed")

    label(ax, 6.3, 0.85,
          "去噪步数 10 是 openpi 的默认值；步数越多越准，但每步都要再过一遍 expert",
          fs=10.5, color=MUTE)

    save(fig, "pi0_inference_loop")


# ------------------------------------------------------------------ 图 3
def fig_fast_pipeline():
    W, H = 12.6, 6.3
    fig, ax = canvas(W, H)

    steps = [
        ("动作块 A_t", "50 步 × 14 维", GRAY),
        ("① 归一化", "每维 1%/99% 分位 → −1 到 1", BLUE_D),
        ("② 逐维 DCT-II", "正交、可逆、零训练", BLUE_D),
        ("③ 量化 round(γc)", "γ = 10，唯一的有损环节", RED),
        ("④ 列优先展平", "先频率、后维度", PURPLE_D),
        ("⑤ BPE 无损压缩", "词表 1024，训练几分钟", PURPLE_D),
    ]
    bw, gap, x0, y = 1.72, 0.20, 0.45, 3.62
    for i, (t, s, st) in enumerate(steps):
        x = x0 + i * (bw + gap)
        box(ax, x, y, bw, 1.32, [t, s], st, fs=11, fs_sub=9.8, r=0.10)
        if i:
            arrow(ax, (x - gap + 0.02, y + 0.66), (x - 0.03, y + 0.66), lw=1.3)

    shapes = ["50 × 14", "50 × 14", "14 × 50 系数", "约 16% 非零", "整数序列", "约 44 个 token"]
    for i, s in enumerate(shapes):
        x = x0 + i * (bw + gap)
        label(ax, x + bw / 2, y - 0.30, s, fs=10.5, color=MUTE, weight="500")

    # 「有损」标注
    lx = x0 + 3 * (bw + gap) + bw / 2
    label(ax, lx, y + 1.78, "有损", fs=10.5, color="#A32D2D", weight="500")
    arrow(ax, (lx, y + 2.38), (lx, y + 1.36), color="#A32D2D", lw=1.0, dash="dotted")

    # 解码（下方反向）
    box(ax, 3.0, 1.25, 6.6, 1.00,
        ["⑥ 反解：BPE 解码 → 反展平 → 除以 γ → 逆 DCT", "全部可逆，误差只来自第③步"],
        TEAL, fs=11.5, fs_sub=10.5)
    arrow(ax, (x0 + 6 * (bw + gap) - gap / 2, y - 0.02), (9.55, 2.28), color=TEAL["ec"])
    arrow(ax, (3.0, 1.75), (1.45, 1.75), color=TEAL["ec"])
    label(ax, 1.30, 0.85, "重建的动作块", fs=10.5, color=MUTE, weight="500")

    save(fig, "fast_pipeline")


# ------------------------------------------------------------------ 图 4
def fig_two_heads():
    W, H = 12.6, 6.9
    fig, ax = canvas(W, H)

    box(ax, 4.3, 5.55, 4.0, 0.90,
        ["观测：n 路图像 + 语言指令", "状态离散成 256 档文本（π0.5）"], GRAY, fs=11, fs_sub=10)
    box(ax, 4.3, 4.05, 4.0, 0.95,
        ["PaliGemma 3B 主干", "两个头共用同一套预训练权重"], BLUE, fs=12, fs_sub=10)
    arrow(ax, (6.3, 5.55), (6.3, 5.02))

    # 连续头
    box(ax, 0.5, 2.35, 5.0, 1.15,
        ["连续头：action expert 约 300M", "一次前向出完整个块，10 步去噪"], PURPLE, fs=11.5, fs_sub=10.5)
    box(ax, 0.5, 1.05, 5.0, 1.00,
        ["约 100 ms / 块（4090）", "prefix K/V 可缓存"], PURPLE_D, fs=11.5, fs_sub=10.5)
    arrow(ax, (4.6, 4.05), (3.0, 3.53), rad=0.12)
    arrow(ax, (3.0, 2.35), (3.0, 2.07))

    # 自回归头
    box(ax, 7.1, 2.35, 5.0, 1.15,
        ["自回归头：复用主干自己的输出层", "顺序吐 30 到 60 个 FAST token"], AMBER, fs=11.5, fs_sub=10.5)
    box(ax, 7.1, 1.05, 5.0, 1.00,
        ["约 750 ms / 块（4090）", "每步过一遍 3B 主干，还要反解"], AMBER, fs=11.5, fs_sub=10.5)
    arrow(ax, (8.0, 4.05), (9.6, 3.53), rad=-0.12)
    arrow(ax, (9.6, 2.35), (9.6, 2.07))

    label(ax, 6.3, 3.30, "同一个主干，换一个头", fs=10.5)
    label(ax, 6.3, 0.50, "两个头输出的是同一个形状的动作块（H × d）", fs=10.5, color=MUTE)

    save(fig, "pi0_vs_pi0_fast")


if __name__ == "__main__":
    fig_pi0_architecture()
    fig_pi0_inference_loop()
    fig_fast_pipeline()
    fig_two_heads()
    for f in sorted(os.listdir(OUT)):
        print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) / 1024:.1f} KB")
