"""Ring attention 的正确性地基：online softmax 的分块递推≡ 一次性全量 softmax。

对应文章：
  https://lrypcy.github.io/2026/10/17/megatron-07-context-parallel-and-long-context/

源码基准：megatron-core 0.20.0 / commit 60e039626。

本模块只做**数值等价性**，不含任何性能测量（纯 CPU、无 GPU）。
四项验证：

  §2  单块（nblocks=1）必须与 `softmax(scores)` 逐位相同
  §3  多块递推与全量 softmax 的最大相对误差；扫块数 1/2/4/8/16/32
  §4  误差随块数增长的规律；以及「块越大越好」的单调性
  §5  ring attention 的真值对照：把序列切成 cp 份、每份只看到 K 的一段，
      按 ring 顺序递推，结果必须与「全 K 一次性 softmax」一致

数学背景：softmax 的分块递推是 online softmax 的核心恒等式。对块 $$b$$ 设
$$m_b = \max_j s_j$$、$$l_b = \sum_j e^{s_j - m_b}$$、$$o_b = \sum_j s_j e^{s_j-m_b}$$
（分母留给最后统一除），则合并两块的递推为

$$m = \max(m_a, m_b),\\quad
l = l_a e^{m_a - m} + l_b e^{m_b - m},\\quad
o = o_a e^{m_a - m} + o_b e^{m_b - m}$$

递推到底再除以 $$l$$，得到的结果与一次算完$$o / l$$ 相同（数学上完全等价，
浮点上因求和次序不同而有舍入差）。
"""

from __future__ import annotations

import numpy as np

SEED = 20261017


# ────────────────────────────────────────────────────────────────────
# 基本件
# ────────────────────────────────────────────────────────────────────
def full_softmax_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """一次性全量 attention：softmax(QKᵀ)V。不分块。归一化在**乘 V 之前**。

    q: [n, d]  k: [m, d]  v: [m, d]
    返回 [n, d]
    """
    scores = (q @ k.T) * (1.0 / np.sqrt(q.shape[-1]))
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def full_softmax_attention_denom_last(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """一次性全量 attention，但归一化放在**乘 V 之后**：$$(e^s V) / \sum e^s$$。

    与 `full_softmax_attention` 数学上等价，浮点上因「先归一化再乘」与
    「先乘再归一化」的操作次序不同而可能差 1 ULP。分块递推天然属于后者
    （分母 $$l$$ 要等所有块都累加完才知道），所以拿它当 §2 的逐位基准。
    """
    scores = (q @ k.T) * (1.0 / np.sqrt(q.shape[-1]))
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    den = e.sum(axis=-1, keepdims=True)
    return (e @ v) / den


def online_softmax_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, n_blocks: int
) -> np.ndarray:
    """online softmax 的分块递推 attention。只在 K/V 的维度上分块。

    q: [n, d]  k: [m, d]  v: [m, d]
    n_blocks: K 被切成几段
    返回 [n, d]
    """
    m_blocks = k.shape[0]
    assert m_blocks % n_blocks == 0, "块数必须整除 K 的长度"
    bs = m_blocks // n_blocks
    scale = 1.0 / np.sqrt(q.shape[-1])

    # running max / running 分母 / running 加权输出
    running_max = np.full(q.shape[0], -np.inf, dtype=np.float64)
    running_den = np.zeros(q.shape[0], dtype=np.float64)
    running_num = np.zeros((q.shape[0], v.shape[1]), dtype=np.float64)

    for bi in range(n_blocks):
        lo, hi = bi * bs, (bi + 1) * bs
        kb, vb = k[lo:hi], v[lo:hi]
        scores_b = (q @ kb.T) * scale  # [n, bs]

        block_max = scores_b.max(axis=-1)  # [n]
        new_max = np.maximum(running_max, block_max)

        # 旧统计量按 (running_max → new_max) 重新缩放
        corr = np.exp(running_max - new_max)
        corr = np.where(np.isfinite(corr), corr, 0.0)

        block_den = np.exp(scores_b - new_max[:, None]).sum(axis=-1)  # [n]
        running_den = running_den * corr + block_den

        block_num = np.exp(scores_b - new_max[:, None]) @ vb  # [n, dv]
        running_num = running_num * corr[:, None] + block_num

        running_max = new_max

    return running_num / running_den[:, None]


# ────────────────────────────────────────────────────────────────────
# §2  单块必须逐位相同
# ────────────────────────────────────────────────────────────────────
def check_e2(n: int, d: int, m: int, dv: int, seed: int = SEED) -> dict:
    """n_blocks=1 时分块递推必须与全量 softmax 逐位相同。"""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((m, d))
    v = rng.standard_normal((m, dv))

    ref = full_softmax_attention_denom_last(q, k, v)
    got = online_softmax_attention(q, k, v, n_blocks=1)
    abs_err = float(np.max(np.abs(got - ref)))
    # 另一个口径：与「先归一化再乘 V」的写法比，差多少
    abs_err_normfirst = float(np.max(np.abs(got - full_softmax_attention(q, k, v))))
    return {
        "n": n, "d": d, "m": m, "dv": dv,
        "abs_err": abs_err,
        "abs_err_normfirst": abs_err_normfirst,
        "bitexact": abs_err == 0.0,
        "ref_absmax": float(np.max(np.abs(ref))),
    }


# ────────────────────────────────────────────────────────────────────
# §3  多块递推 vs 全量：扫块数
# ────────────────────────────────────────────────────────────────────
def check_e3(
    n: int, d: int, m: int, dv: int, nblocks_list, dtype=np.float64, seed: int = SEED
) -> list[dict]:
    """扫块数，返回每档的最大相对/绝对误差。

    `ref` 是 float64 下的一次性全量结果 —— 作为真值。
    `got` 在指定 dtype 下做递推，用于显示低精度下的误差规模。
    """
    rng = np.random.default_rng(seed)
    q64 = rng.standard_normal((n, d))
    k64 = rng.standard_normal((m, d))
    v64 = rng.standard_normal((m, dv))

    ref = full_softmax_attention_denom_last(q64, k64, v64)
    ref_absmax = float(np.max(np.abs(ref)))

    out = []
    for nb in nblocks_list:
        qd = q64.astype(dtype)
        kd = k64.astype(dtype)
        vd = v64.astype(dtype)
        got = online_softmax_attention(qd, kd, vd, n_blocks=nb)
        abs_err = float(np.max(np.abs(got.astype(np.float64) - ref)))
        rel_err = abs_err / ref_absmax
        out.append({
            "nblocks": nb,
            "block_size": m // nb,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "ref_absmax": ref_absmax,
        })
    return out


# ────────────────────────────────────────────────────────────────────
# §4  误差随块数的走向
# ────────────────────────────────────────────────────────────────────
def check_e4(n: int, d: int, m: int, dv: int, nblocks_list, seed: int = SEED) -> dict:
    """误差是否随块数单调增长；相邻块号的相对误差比值。"""
    rows = check_e3(n, d, m, dv, nblocks_list, dtype=np.float64, seed=seed)
    errs = [r["rel_err"] for r in rows]
    ratios = [errs[i + 1] / errs[i] if errs[i] else float("nan") for i in range(len(errs) - 1)]
    monotone = all(errs[i + 1] >= errs[i] for i in range(len(errs) - 1))
    return {
        "rows": rows,
        "errs": errs,
        "ratios": ratios,
        "monotone_nondecreasing": monotone,
        "first_last_ratio": errs[-1] / errs[0] if errs[0] else float("nan"),
    }


# ────────────────────────────────────────────────────────────────────
# §5  ring 顺序递推（真值对照）
# ────────────────────────────────────────────────────────────────────
def ring_attention(
    q_cp: list[np.ndarray], k_cp: list[np.ndarray], v_cp: list[np.ndarray]
) -> list[np.ndarray]:
    """ring attention 的语义：每份 Q 起初只配自己那份 K/V，
    然后按 ring 顺序 (0,1,...,cp-1) 依次吃掉其余 rank 的 K/V 块。

    这与 `online_softmax_attention` 的差别在于「块」不是按连续内存切的，
    而是按 rank 持有的那段切的 —— 数值上两者是同一组分块的不同顺序，
    所以必须给出一致的输出。

    q_cp[i]: [n_i, d]  第 i 个 rank 持有的 Q 段
    返回每 rank 的 [n_i, dv]
    """
    cp = len(q_cp)
    scale = 1.0 / np.sqrt(q_cp[0].shape[-1])
    outs = []
    for i in range(cp):
        q = q_cp[i]
        n = q.shape[0]
        running_max = np.full(n, -np.inf, dtype=np.float64)
        running_den = np.zeros(n, dtype=np.float64)
        running_num = np.zeros((n, v_cp[0].shape[1]), dtype=np.float64)
        for j in range(cp):  # ring 顺序
            kb, vb = k_cp[j], v_cp[j]
            scores_b = (q @ kb.T) * scale
            block_max = scores_b.max(axis=-1)
            new_max = np.maximum(running_max, block_max)
            corr = np.exp(running_max - new_max)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            e = np.exp(scores_b - new_max[:, None])
            running_den = running_den * corr + e.sum(axis=-1)
            running_num = running_num * corr[:, None] + e @ vb
            running_max = new_max
        outs.append(running_num / running_den[:, None])
    return outs


def check_e5(n_cp: int, d: int, m_cp: int, dv: int, seed: int = SEED) -> dict:
    """把序列切成 cp 份做 ring，与「全 K/V 一次性 softmax」逐 rank 对齐。"""
    rng = np.random.default_rng(seed)
    cp = n_cp
    q_cp = [rng.standard_normal((m_cp, d)) for _ in range(cp)]
    k_cp = [rng.standard_normal((m_cp, d)) for _ in range(cp)]
    v_cp = [rng.standard_normal((m_cp, dv)) for _ in range(cp)]

    ring_out = ring_attention(q_cp, k_cp, v_cp)

    # 全量真值：Q 的第 i 段 × 全局全部 K/V
    k_all = np.concatenate(k_cp, axis=0)
    v_all = np.concatenate(v_cp, axis=0)
    ref = full_softmax_attention_denom_last(q_cp[0], k_all, v_all)

    abs_err = float(np.max(np.abs(ring_out[0] - ref)))
    ref_absmax = float(np.max(np.abs(ref)))
    # 每个 rank 各自的误差
    per_rank = []
    for i in range(cp):
        ref_i = full_softmax_attention_denom_last(q_cp[i], k_all, v_all)
        per_rank.append(float(np.max(np.abs(ring_out[i] - ref_i))))

    return {
        "cp": cp,
        "abs_err": abs_err,
        "rel_err": abs_err / ref_absmax,
        "ref_absmax": ref_absmax,
        "per_rank_abs_err": per_rank,
    }