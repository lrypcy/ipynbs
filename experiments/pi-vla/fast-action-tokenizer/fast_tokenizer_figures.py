"""FAST 动作分词实验的配图（4 张 SVG）。

共用件在 `fast_tokenizer_lib.py`。输出写在 `figures/`，正文用的是
`assets/img/pi-vla/fast_*.svg`（博客仓库里）。

运行：python3 fast_tokenizer_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fast_tokenizer_lib import (
    bin_tokens,
    bpe_train,
    dct_matrix,
    dct_quantize,
    fast_token_count,
    flatten,
    make_dataset,
    normalize,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

C_FAST = "#2176ae"
C_BIN = "#c0392b"
C_DASH = "#8a8a8a"

DIM = 7
VOCAB = 512


# ---------------------------------------------------------------- 图 0
def fig_token_stream():
    """同一个 7 维 50 步动作块，两种分词下的 token 序列。"""
    acts = normalize(make_dataset(32, 7, 50, seed=170))
    coef = dct_quantize(acts)
    merges = bpe_train(flatten(coef), vocab_size=VOCAB)
    tokens = fast_token_count(coef, merges)
    act = acts[:1]
    from fast_tokenizer_lib import bpe_encode
    enc = bpe_encode(flatten(coef)[0], merges)

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 4.6),
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    ax = axes[0]
    ax.imshow(act[0], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"维度 {i + 1}" for i in range(7)])
    ax.set_xlabel("时间步（50 步 = 1 秒 @50 Hz）")
    ax.set_title(f"分箱：每维每步一个 token —— 7 × 50 = {7 * 50} 个", fontsize=12)

    ax = axes[1]
    strip = np.array([enc]) if len(enc) < 200 else np.array([enc[:200]])
    ax.imshow(strip, cmap="viridis", aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("token 序号")
    ax.set_title(f"FAST（DCT + 量化 + BPE）：这 {len(enc)} 个 token 就是同一段动作"
                 f"（本批平均 {tokens:.0f} 个）", fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fast_token_stream.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 1
def fig_token_count():
    hs = [5, 10, 20, 50]
    n_bin, n_fast = [], []
    for h in hs:
        act = normalize(make_dataset(32, DIM, h, seed=100 + h))
        coef = dct_quantize(act)
        merges = bpe_train(flatten(coef), vocab_size=VOCAB)
        n_bin.append(DIM * h)
        n_fast.append(fast_token_count(coef, merges))

    fmaxs = [1.0, 1.5, 2.5, 4.0]
    n_bw = []
    for fmax in fmaxs:
        act = normalize(make_dataset(32, DIM, 50, seed=170, fmax=fmax))
        coef = dct_quantize(act)
        merges = bpe_train(flatten(coef), vocab_size=VOCAB)
        n_bw.append(fast_token_count(coef, merges))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax1.plot(hs, n_bin, "o-", color=C_BIN, lw=2.2, label="分箱 $=d\\times H$")
    ax1.plot(hs, n_fast, "s-", color=C_FAST, lw=2.2, label="FAST（DCT+BPE）")
    for h, b, f in zip(hs, n_bin, n_fast):
        ax1.annotate(f"{b / f:.1f}×", (h, f), textcoords="offset points",
                     xytext=(0, -16), ha="center", color=C_FAST, fontsize=10)
    ax1.set_xlabel("一个动作块的步数 $H$（＝采样率，块长固定 1 秒）")
    ax1.set_ylabel("每个块的 token 数")
    ax1.set_title("采样加密 10 倍：分箱涨 10 倍，FAST 涨 2.03 倍", fontsize=12)
    ax1.legend(frameon=False)
    ax1.set_ylim(0, max(n_bin) * 1.08)

    ax2.plot(fmaxs, n_bw, "s-", color=C_FAST, lw=2.2)
    ax2.set_xlabel("信号带宽上限（Hz，固定 $H=50$）")
    ax2.set_ylabel("FAST token 数")
    ax2.set_title("token 数跟带宽走：1 Hz 到 4 Hz 只涨 60%", fontsize=12)
    for x, y in zip(fmaxs, n_bw):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", color=C_FAST, fontsize=10)
    ax2.set_ylim(0, max(n_bw) * 1.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fast_token_count.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 2
def fig_gamma():
    act = normalize(make_dataset(32, DIM, 50, seed=150))
    d = dct_matrix(50)
    raw = act @ d.T
    gammas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    keep, rmse = [], []
    for g in gammas:
        cq = np.round(g * raw)
        keep.append(float(np.mean(cq != 0)) * 100)
        rmse.append(float(np.sqrt(np.mean(((cq / g) @ d - act) ** 2))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax1.plot(gammas, keep, "o-", color=C_FAST, lw=2.2)
    ax1.set_xscale("log")
    ax1.set_xlabel("量化尺度 $\\gamma$")
    ax1.set_ylabel("保留下来的系数占比（%）")
    ax1.set_title("$\\gamma$ 越大越精细：阈值 $1/(2\\gamma)$ 变小，存活的系数变多", fontsize=12)
    ax1.set_ylim(0, max(keep) * 1.2)

    ax2.plot(gammas, rmse, "o-", color=C_FAST, lw=2.2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("量化尺度 $\\gamma$")
    ax2.set_ylabel("重建 RMSE（动作已归一到 $[-1,1]$）")
    ax2.set_title("保真度：从 0.103 降到 0.0046，边际收益迅速变薄", fontsize=12)
    ax2.axhline(0.02, color=C_DASH, ls="--", lw=1.2)
    ax2.text(gammas[0], 0.0215, "0.02", color=C_DASH, fontsize=10)
    for g in (1.0, 10.0, 50.0):
        i = gammas.index(g)
        ax2.annotate(f"{rmse[i]:.4f}", (g, rmse[i]), textcoords="offset points",
                     xytext=(6, -12), color=C_FAST, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fast_gamma_tradeoff.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 3
def fig_flatten():
    act = normalize(make_dataset(32, 14, 50, seed=250))
    h, dim = 50, 14
    d = dct_matrix(h)
    coef = act @ d.T
    flat_d = flatten(coef, order="dim_first")
    budgets = [n * dim for n in (1, 2, 4, 8, 16)]
    worst_c, worst_r, mean_c, mean_r = [], [], [], []
    for budget in budgets:
        n_freq = budget // dim
        rec_c = (np.where(np.arange(h)[None, None, :] < n_freq, coef, 0.0)) @ d
        trunc = np.zeros_like(flat_d)
        trunc[:, :budget] = flat_d[:, :budget]
        rec_r = trunc.reshape(coef.shape) @ d
        worst_c.append(float(np.max(np.sqrt(np.mean((rec_c - act) ** 2, axis=(0, 2))))))
        worst_r.append(float(np.max(np.sqrt(np.mean((rec_r - act) ** 2, axis=(0, 2))))))
        mean_c.append(float(np.sqrt(np.mean((rec_c - act) ** 2))))
        mean_r.append(float(np.sqrt(np.mean((rec_r - act) ** 2))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax1.plot(budgets, worst_c, "o-", color=C_FAST, lw=2.2, label="列优先（先低频）")
    ax1.plot(budgets, worst_r, "s-", color=C_BIN, lw=2.2, label="行优先（先单维）")
    ax1.set_xscale("log")
    ax1.set_xlabel("系数预算（个）")
    ax1.set_ylabel("最差一维的 RMSE")
    ax1.set_title("行优先要 650 个系数才轮到最后一维", fontsize=12)
    ax1.legend(frameon=False)
    ax1.annotate("最后一维仍是全零", (224, worst_r[-1]), textcoords="offset points",
                 xytext=(-90, -22), color=C_BIN, fontsize=10)

    ax2.plot(budgets, mean_c, "o-", color=C_FAST, lw=2.2, label="列优先")
    ax2.plot(budgets, mean_r, "s-", color=C_BIN, lw=2.2, label="行优先")
    ax2.set_xscale("log")
    ax2.set_xlabel("系数预算（个）")
    ax2.set_ylabel("全体 RMSE")
    ax2.set_title("只看总误差：差距被平均掉了（0.67 vs 0.27）", fontsize=12)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fast_flatten_order.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图 4
def fig_rate_distortion():
    act = normalize(make_dataset(32, 14, 50, seed=300))
    h, dim = 50, 14
    d = dct_matrix(h)
    coef_q = np.round(10 * (act @ d.T)).astype(np.int64)
    merges = bpe_train(flatten(coef_q), vocab_size=VOCAB)

    fast_tok, fast_rmse = [], []
    for n_freq in range(1, h + 1):
        rec = (np.where(np.arange(h)[None, None, :] < n_freq, coef_q / 10.0, 0.0)) @ d
        fast_rmse.append(float(np.sqrt(np.mean((rec - act) ** 2))))
        fast_tok.append(fast_token_count(coef_q, merges, n_freq=n_freq))

    _, act_bin = bin_tokens(act, n_bins=256)
    bin_tok, bin_rmse = [], []
    for s in [1, 2, 4, 8, 16, 50]:
        keep = np.arange(0, h, s)
        rec = np.empty_like(act)
        for ci in range(act.shape[0]):
            for di in range(dim):
                rec[ci, di] = np.interp(np.arange(h), keep, act_bin[ci, di, keep])
        bin_rmse.append(float(np.sqrt(np.mean((rec - act) ** 2))))
        bin_tok.append(dim * len(keep))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(fast_tok, fast_rmse, "-", color=C_FAST, lw=2.4, label="FAST（少发几个高频系数）")
    ax.plot(bin_tok, bin_rmse, "s--", color=C_BIN, lw=2.0, label="分箱（只能降采样换 token）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(0.02, color=C_DASH, ls=":", lw=1.2)
    ax.text(12, 0.0215, "RMSE = 0.02", color=C_DASH, fontsize=10)
    ax.annotate("FAST 71.8 token", (71.8, 0.02), textcoords="offset points",
                xytext=(-10, 18), color=C_FAST, fontsize=10)
    ax.annotate("分箱 350 token", (350, 0.02), textcoords="offset points",
                xytext=(-70, 18), color=C_BIN, fontsize=10)
    ax.set_xlabel("token 预算（每个动作块，对数）")
    ax.set_ylabel("重建 RMSE（对数）")
    ax.set_title("14 维 50 Hz：同一精度下分箱要花 4.9 倍的 token", fontsize=12)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fast_rate_distortion.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_token_stream()
    fig_token_count()
    fig_gamma()
    fig_flatten()
    fig_rate_distortion()
    for f in sorted(os.listdir(OUT)):
        print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) / 1024:.1f} KB")
