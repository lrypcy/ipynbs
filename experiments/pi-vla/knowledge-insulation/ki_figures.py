"""03 篇（KI）的配图：2 张架构图 + 2 张数据曲线，全部手工排版成 SVG。

与 `pi-vla/architecture/` 一致：字体 PingFang SC（Hiragino Sans GB 会有静默缺字），
坐标以英寸为单位，`bbox_inches="tight"` 会把越界元素悄悄撑大画布——
所以画完**必须渲染 PNG 人眼看一遍**，光看 SVG 尺寸是发现不了的。

运行：~/Software/miniconda3/bin/python ki_figures.py    约 3 分钟
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import ListedColormap

import numpy as np

import ki_lib as K

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
GRAY = dict(fc="#F8F7F4", ec="#888780", tc="#2C2C2A")
GRAY_D = dict(fc="#F1EFE8", ec="#5F5E5A", tc="#2C2C2A")

INK = "#2C2C2A"
MUTE = "#5F5E5A"
HINT = "#888780"
PT = 1.0 / 72.0

# 正文里的配色要和数据曲线对得上，这里统一定义
C_JOINT = "#185FA5"      # 不断梯度
C_KI = "#0F6E56"         # 断梯度
C_NAIVE = "#A32D2D"      # 只有流匹配
C_FREEZE = "#888780"


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


# ================================================================ 图 1
def fig_gradient_paths():
    W, H = 12.8, 9.2
    fig, ax = canvas(W, H)

    cx_b, cx_e = 0.30, 8.20          # 两个容器的左边界
    cw = 4.30                        # 容器宽
    cy, ch = 3.60, 5.30              # 容器底边与高
    top = container(ax, cx_b, cy, cw, ch, "backbone（VLM，预训练过）", BLUE)
    container(ax, cx_e, cy, cw, ch, "action expert（随机初始化）", PURPLE)

    inner_w = cw - 0.64
    vstack(ax, cx_b + 0.32, top, inner_w, [
        (1.20, ["观测与指令 token", "图像 + 文本 + 5 个离散动作槽位"], BLUE),
        (1.30, ["backbone 投影", "$W^b_q\\ \\ W^b_k\\ \\ W^b_v$", "PaliGemma 3B"], BLUE_D),
        (1.20, ["输出投影 $W^b_o$", "语言头 / 离散动作头"], BLUE),
    ])
    vstack(ax, cx_e + 0.32, top, inner_w, [
        (1.20, ["带噪动作 token", "$x_t=\\tau A+(1-\\tau)\\omega$，外带 $\\tau$"], PURPLE),
        (1.30, ["expert 投影", "$W^a_q\\ \\ W^a_k\\ \\ W^a_v$", "300M"], PURPLE_D),
        (1.20, ["输出投影 $W^a_o$", "流匹配向量场头"], PURPLE),
    ])

    # K_b / V_b 从 backbone 流出 → 过 sg 闸门 → 进 expert
    ymid = top - 1.20 - 0.22 - 0.65
    gx, gw, gh = 5.35, 1.75, 0.92
    box(ax, gx, ymid - gh / 2, gw, gh, ["sg($\\cdot$)", "梯度到此为止"], RED, fs=13, fs_sub=10)
    for dy, name in ((0.22, "$K_b$"), (-0.22, "$V_b$")):
        y = ymid + dy
        arrow(ax, (cx_b + cw, y), (gx, y), color="#A32D2D", lw=1.8)
        arrow(ax, (gx + gw, y), (cx_e, y), color="#A32D2D", lw=1.8)
        label(ax, (cx_b + cw + gx) / 2, y + 0.20, name, fs=11.5, color="#A32D2D", weight="500")
    label(ax, gx + gw / 2, ymid - gh / 2 - 0.42, "只有 expert 那几行走这一支", fs=10)
    label(ax, gx + gw / 2, ymid - gh / 2 - 0.74, "backbone 行自己的注意力不受影响", fs=10)

    # 三条梯度通路
    bw, bh, by = 3.90, 2.00, 1.35
    for i, (st, lines) in enumerate([
        (TEAL, ["① 流匹配损失", "→ expert 的 5 个矩阵", "照常回传"]),
        (RED, ["② 流匹配损失", "→ $W^b_k$ / $W^b_v$", "被 sg 切断，$\\partial L/\\partial W = 0$"]),
        (TEAL, ["③ 离散动作（FAST）损失", "→ backbone 全部权重", "照常回传"]),
    ]):
        box(ax, 0.30 + i * (bw + 0.25), by, bw, bh, lines, st, fs=12, fs_sub=10.5)

    box(ax, 0.30, 0.30, W - 0.60, 0.80, [
        "顺带一提：$\\partial L_{flow}/\\partial W^b_q$ 与 $\\partial L_{flow}/\\partial W^b_o$ 本来就是 0"
        "（掩码让 backbone 行读不到 action token），所以 sg 只需要加在 $K_b$ / $V_b$ 两个矩阵上"],
        GRAY, fs=11.5, fs_sub=11.5)
    save(fig, "ki_gradient_paths")


# ================================================================ 图 2
def fig_attention_mask():
    M = K.MASK.astype(int)
    n = M.shape[0]
    # 三类底色：允许 / 掩码掉 / 两类动作互不 attend（重点）
    code = np.where(M, 0, 1)
    code[K.N_B:, K.N_O:K.N_B] = 2          # A 行 × R 列
    code[K.N_O:K.N_B, K.N_B:] = 2          # R 行 × A 列
    cmap = ListedColormap(["#B5D4F4", "#F1EFE8", "#F7C9C9"])

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.imshow(code, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    def bracket(x0, x1, text, color=INK):
        """顶部列分块括号。imshow 默认 origin='upper'：y 向下增长，顶部在 y = -0.5。"""
        ax.plot([x0, x0, x1, x1], [-0.5, -0.85, -0.85, -0.5],
                color=color, lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, -1.25, text, ha="center", va="center",
                fontsize=11, color=color, fontweight="500")

    bracket(-0.5, K.N_O - 0.5, "观测 + 指令（6）")
    bracket(K.N_O - 0.5, K.N_B - 0.5, "离散动作槽位（5）")
    bracket(K.N_B - 0.5, n - 0.5, "连续动作 token（5）")

    for y0, text in ((K.N_O / 2 - 0.5, "观测 + 指令（6）"),
                     (K.N_O + K.N_R / 2 - 0.5, "离散动作槽位（5）"),
                     (K.N_B + K.N_A / 2 - 0.5, "连续动作（5）")):
        ax.text(-1.0, y0, text, rotation=90, ha="center", va="center",
                fontsize=11, fontweight="500")

    for b in (K.N_O, K.N_B):
        ax.axhline(b - 0.5, color=INK, lw=1.0)
        ax.axvline(b - 0.5, color=INK, lw=1.0)

    ax.annotate("backbone 行永远看不见 action token\n（右上角整块为空，所以流匹配梯度\n"
                "到不了 $W^b_q$ / $W^b_o$）",
                xy=(K.N_B + 2, 2.0), xytext=(K.N_B + 6.5, 1.2),
                fontsize=10.5, color=MUTE, va="center",
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.2))
    ax.annotate("两类动作表示互不 attend\n（KI 论文明确要求）",
                xy=(K.N_O + 2, K.N_B + 2), xytext=(-6.2, K.N_B + 4.2),
                fontsize=10.5, color="#A32D2D", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#A32D2D", lw=1.2))
    ax.text(n / 2 - 0.5, n + 1.5, "行 = 查询位置，列 = 被 attend 的位置；蓝色 = 允许，"
                                  "浅灰 = 掩码掉，红色 = KI 额外要求的两处",
            ha="center", fontsize=10.5, color=HINT)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.90, bottom=0.08)
    fig.savefig(os.path.join(OUT, "ki_attention_mask.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "ki_attention_mask.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 训练
N_TRAIN, N_EVAL = 2048, 512
sc = K.Scene(seed=4242, n=N_TRAIN + N_EVAL)
EVal = K.sample_batch(sc, np.arange(N_TRAIN, N_TRAIN + N_EVAL),
                      np.random.default_rng(999), fixed_tau=0.5)
ACT_FLOOR = 0.25 * float(np.mean(np.square(EVal["u"])))


def metrics(P):
    o, _ = K.forward(P, EVal, sg=True)
    return dict(act=float(np.mean(np.square(EVal["x_t"] - 0.5 * o["pred_flow"] - EVal["A"]))),
                lang=K.acc(o["logits_lang"], EVal["y_lang"]),
                ar=K.acc(o["logits_ar"], EVal["y_ar"]))


def probe_r2(P, n_fit=1024, lam=1e-3):
    f = lambda i: (sc.x_b(i) @ P["Wv_b"])[:, :K.N_O, :].mean(axis=1)
    Xf = np.concatenate([f(np.arange(n_fit)), np.ones((n_fit, 1))], 1)
    Xe = np.concatenate([f(np.arange(N_TRAIN, N_TRAIN + N_EVAL)), np.ones((N_EVAL, 1))], 1)
    Yf, Ye = sc.A[:n_fit].reshape(n_fit, -1), sc.A[N_TRAIN:].reshape(N_EVAL, -1)
    W = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ Yf)
    return float(1 - np.sum((Xe @ W - Ye) ** 2) / np.sum((Ye - Ye.mean(0)) ** 2))


def run(sg, wa, wl, alpha, steps=3000, lr=5e-3, keys=None, seed=31, pre=None):
    P = K.clone(K.init_params(np.random.default_rng(555)))
    for k in K.BONE:
        P[k] = pre[k].copy()
    opt = K.Adam(P, lr=lr)
    r = np.random.default_rng(seed)
    for _ in range(steps):
        idx = r.integers(0, N_TRAIN, 64)
        b = K.sample_batch(sc, idx, r)
        _, G, _, _ = K.grads(P, b, sg, w_flow=alpha, w_ar=wa, w_lang=wl)
        opt.step(P, G, K.ALL_P if keys is None else keys)
    return P


def pretrain():
    P = K.init_params(np.random.default_rng(1))
    opt = K.Adam(P, lr=3e-3)
    r = np.random.default_rng(7)
    for _ in range(1200):
        idx = r.integers(0, N_TRAIN, 64)
        b = K.sample_batch(sc, idx, r)
        _, G, _, _ = K.grads(P, b, False, w_flow=0.0, w_ar=1.0, w_lang=1.0,
                             ar_tgt=b["y_ar_lang"])
        opt.step(P, G, K.BONE)
    return P


# ================================================================ 图 3
def fig_alpha_sweep(P0):
    ALPHAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    REC = [("joint_vlm", False, 1.0, 1.0, C_JOINT, "联合训练，不断梯度"),
           ("ki_vlm", True, 1.0, 1.0, C_KI, "KI（断梯度）"),
           ("naive", False, 0.0, 0.0, C_NAIVE, "只有流匹配")]
    res = {}
    for name, sg, wa, wl, _, _ in REC:
        rows = []
        for a in ALPHAS:
            P = run(sg, wa, wl, a, pre=P0)
            rows.append((metrics(P), probe_r2(P)))
        res[name] = rows
        print(f"  {name:10s} " + "  ".join(f"a={a:g}:{m['lang']:.3f}/{r:.3f}"
                                           for a, (m, r) in zip(ALPHAS, rows)))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax_i, (ax, key, title, ylim) in enumerate([
            (axes[0], "lang", "语言跟随（越高越好）", (0.0, 1.0)),
            (axes[1], None, "backbone 探针 $R^2$（越高越好）", (0.0, 0.9))]):
        for name, _, _, _, color, desc in REC:
            rows = res[name]
            y = [m[key] for m, _ in rows] if key else [r for _, r in rows]
            ax.plot(ALPHAS, y, marker="o", ms=7, lw=2.0, color=color, label=f"{name}：{desc}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
        ax.set_xlabel("$\\alpha$（流匹配损失权重）")
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, ls=":")
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=10, loc="lower left", framealpha=0.95)
    axes[1].annotate("断梯度后这条线与 α 无关（三种 α 参数逐位相同）\n"
                     "不断梯度时 α 一动，backbone 轨迹就变",
                     xy=(4.0, res["ki_vlm"][2][1]), xytext=(1.02, 0.09),
                     fontsize=10.5, color=MUTE,
                     arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.2))
    fig.suptitle("断梯度之后 $\\alpha$ 就不是旋钮了（各点均为 3000 步，含 VLM 协同训练）",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "ki_alpha_sweep")


# ================================================================ 图 4
def fig_tradeoff(P0):
    RECIPES = [("naive", False, 0.0, 0.0, K.ALL_P, C_NAIVE, "naive\n（π0 式）"),
               ("joint", False, 1.0, 0.0, K.ALL_P, C_JOINT, "joint"),
               ("joint_vlm", False, 1.0, 1.0, K.ALL_P, C_JOINT, "joint\n+ VLM"),
               ("ki", True, 1.0, 0.0, K.ALL_P, C_KI, "ki"),
               ("ki_vlm", True, 1.0, 1.0, K.ALL_P, C_KI, "ki\n+ VLM"),
               ("freeze", True, 0.0, 0.0, K.EXPERT, C_FREEZE, "freeze\n（冻住）")]
    rows = []
    for name, sg, wa, wl, keys, color, xt in RECIPES:
        P = run(sg, wa, wl, 1.0, keys=keys, pre=P0)
        rows.append((xt, color, metrics(P), probe_r2(P)))
        print(f"  {name:10s} lang={rows[-1][2]['lang']:.3f} act={rows[-1][2]['act']:.4f} "
              f"R2={rows[-1][3]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    xs = np.arange(len(rows))
    base_lang = K.acc(K.forward(P0, EVal, sg=True)[0]["logits_lang"], EVal["y_lang"])

    ax = axes[0]
    ax.bar(xs, [r[2]["lang"] for r in rows], color=[r[1] for r in rows], width=0.62)
    ax.axhline(base_lang, color=INK, ls="--", lw=1.4)
    ax.text(len(rows) - 0.5, base_lang + 0.02, f"预训练基线 {base_lang:.3f}",
            ha="right", fontsize=10, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("语言跟随准确率")
    ax.set_title("语言跟随：只有带 VLM 协同训练才保得住", fontsize=12.5, pad=10)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.bar(xs, [r[2]["act"] for r in rows], color=[r[1] for r in rows], width=0.62)
    ax.axhline(ACT_FLOOR, color="#A32D2D", ls="--", lw=1.4)
    # 标注放左边：最右那根 freeze 柱子正好顶在虚线上，放右边会被柱子吃掉
    ax.text(-0.45, ACT_FLOOR + 0.007,
            f"「完全不看 backbone」的下界 {ACT_FLOOR:.3f}", ha="left",
            fontsize=10, color="#A32D2D")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(0.38, 0.56)
    ax.set_ylabel("动作误差（越低越好）")
    ax.set_title("动作误差：冻住 backbone 就卡在下界上", fontsize=12.5, pad=10)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "ki_tradeoff")


if __name__ == "__main__":
    import sys
    print("画架构图 ...")
    fig_gradient_paths()
    fig_attention_mask()
    if "--arch" in sys.argv:
        print("done ->", OUT)
        raise SystemExit(0)
    print("预训练 backbone ...")
    P0 = pretrain()
    print("α 扫描 ...")
    fig_alpha_sweep(P0)
    print("六配方对比 ...")
    fig_tradeoff(P0)
    print("done ->", OUT)
