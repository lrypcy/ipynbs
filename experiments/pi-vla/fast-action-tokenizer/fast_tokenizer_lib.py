"""FAST 动作分词的共用件：DCT + 量化 + 列优先展平 + BPE。

配套文章 `_posts/2026-11-03-pi-vla-02-fast-action-tokenizer.md`。
`fast_tokenizer_lab.py`（断言）与 `fast_tokenizer_figures.py`（配图）都从这里导入。

只依赖 numpy。
"""

import numpy as np
from collections import Counter

N_ASSERT = 0


def check(cond, msg):
    global N_ASSERT
    N_ASSERT += 1
    if not cond:
        raise AssertionError(msg)


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# %% [markdown]
# ## 共用件：DCT-II 正交矩阵、平滑 1 秒动作块、分箱与 FAST 分词器
#
# 动作块定义为一秒钟内的连续信号：同一段物理轨迹在 H=5（5Hz）和 H=50（50Hz）
# 上是**同一条曲线**的疏密两种采样。这样比较 token 数才是公平的——比的是
# "编码同一段运动需要多少符号"，而不是"比两段不同的运动"。

# %%
def dct_matrix(n):
    """正交归一化 DCT-II 矩阵，D @ x 得到系数。"""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    d[0, :] = np.sqrt(1.0 / n)
    return d


def make_signal_params(rng, dim, n_freq=3, fmax=2.5):
    """每维一条 1 秒轨迹的傅里叶参数，与采样率无关。

    fmax 是带宽（周/秒）。另外给每维一个直流偏置：真实动作是绕着某个位姿
    摆动的，不是零均值振荡——这一点决定了"发不出系数时"误差有多大。
    """
    freq = rng.uniform(0.4, fmax, size=(dim, n_freq))
    amp = rng.uniform(0.3, 1.0, size=(dim, n_freq))
    amp /= amp.sum(axis=1, keepdims=True)
    amp *= 0.35                        # 摆动幅度远小于绝对位姿
    phase = rng.uniform(0, 2 * np.pi, size=(dim, n_freq))
    offset = rng.uniform(-1.0, 1.0, size=(dim, 1))
    return freq, amp, phase, offset


def sample_signal(params, n_steps):
    """把参数采样成 (dim, n_steps) 的动作块，值域约在 [-1, 1]。"""
    freq, amp, phase, offset = params
    t = (np.arange(n_steps) + 0.5) / n_steps                # 0..1 秒
    dim = freq.shape[0]
    x = np.tile(offset, (1, n_steps)).astype(float)
    for d in range(dim):
        for k in range(freq.shape[1]):
            x[d] += amp[d, k] * np.cos(2 * np.pi * freq[d, k] * t + phase[d, k])
    peak = np.abs(x).max(axis=1, keepdims=True)
    return x / np.maximum(peak, 1e-8)


def make_dataset(n_chunks, dim, n_steps, seed=0, fmax=2.5):
    rng = np.random.default_rng(seed)
    return np.stack([sample_signal(make_signal_params(rng, dim, fmax=fmax), n_steps)
                     for _ in range(n_chunks)])


def normalize(actions):
    """按每维 1%/99% 分位缩放到 [-1, 1]（论文做法，对离群点鲁棒）。"""
    lo = np.percentile(actions, 1, axis=(0, 1))
    hi = np.percentile(actions, 99, axis=(0, 1))
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return np.clip((actions - mid) / np.maximum(half, 1e-8), -1.0, 1.0)


def bin_tokens(actions, n_bins=256):
    """分箱分词：每个时间步每维一个 token，token 数恒为 dim*H。"""
    q = np.round((actions + 1) / 2 * (n_bins - 1))
    recon = 2 * q / (n_bins - 1) - 1
    return q.astype(np.int64), recon


def dct_quantize(actions, gamma=10.0):
    """FAST 前三步：逐维 DCT -> 乘 gamma 取整。返回量化系数 (chunk, dim, H)。"""
    h = actions.shape[-1]
    d = dct_matrix(h)
    coef = actions @ d.T                     # 逐维对时间做 DCT
    return np.round(gamma * coef).astype(np.int64)


def flatten(coef, order="freq_first"):
    """展平 (chunk, dim, H)。

    freq_first = 列优先：外层走频率、内层走维度，自回归先吐低频（论文做法）。
    dim_first  = 行优先：外层走维度、内层走频率。
    """
    if order == "freq_first":
        return coef.transpose(0, 2, 1).reshape(coef.shape[0], -1)
    return coef.reshape(coef.shape[0], -1)


def idct_from_prefix(coef, n_freq, order="freq_first"):
    """只用前 n_freq 个频率分量重建动作，验证展平顺序的信息价值。"""
    h = coef.shape[-1]
    d = dct_matrix(h)
    trunc = np.zeros_like(coef, dtype=np.float64)
    if order == "freq_first":
        trunc[:, :, :n_freq] = coef[:, :, :n_freq]
    else:                                    # 行优先下，预算先花在维度 0 的全部频率上
        flat_budget = n_freq * coef.shape[1]
        flat = coef.reshape(coef.shape[0], -1)[:, :flat_budget]
        trunc = flat.reshape(coef.shape[0], coef.shape[1], h)
    return trunc @ d


# %% [markdown]
# ### 一个够用的 BPE 训练器
#
# 论文用的是 HuggingFace `tokenizers` 的 BPE（词表 1024）。这里手写一个最小版，
# 词表给小到 256 是为了在 CPU 上几十秒跑完；结论只看**趋势**，不看绝对值。

# %%
def _apply_merge(seq, a, b, new):
    out, i, n = [], 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return np.asarray(out, dtype=np.int64)


def bpe_train(seqs, vocab_size=256, min_count=2):
    """返回合并表 [(a, b, new_id), ...]，按学习顺序排列。"""
    current = [np.asarray(s, dtype=np.int64) for s in seqs]
    n_sym = int(max(int(s.max()) for s in current) if current else 0) + 1
    merges = []
    while n_sym < vocab_size:
        cnt = Counter()
        for s in current:
            if len(s) < 2:
                continue
            cnt.update(zip(s[:-1].tolist(), s[1:].tolist()))
        if not cnt:
            break
        (a, b), c = cnt.most_common(1)[0]
        if c < min_count:
            break
        merges.append((a, b, n_sym))
        current = [_apply_merge(s, a, b, n_sym) for s in current]
        n_sym += 1
    return merges


def bpe_encode(seq, merges):
    s = np.asarray(seq, dtype=np.int64)
    for a, b, new in merges:
        s = _apply_merge(s, a, b, new)
    return s


def fast_token_count(coef, merges, n_freq=None, order="freq_first"):
    """平均每个 chunk 的 token 数；n_freq 给定时只用前 n_freq 个频率分量。"""
    if n_freq is not None:
        coef = coef[:, :, :n_freq]
    flat = flatten(coef, order=order)
    return float(np.mean([len(bpe_encode(row, merges)) for row in flat]))
