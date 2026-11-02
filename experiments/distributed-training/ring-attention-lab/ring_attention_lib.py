"""序列并行 / Ring Attention 实验的公共函数。

对应文章：分布式训练（05）序列并行——把注意力沿序列切开，Ring Attention 与长上下文
https://lrypcy.github.io/2026/08/31/dist-train-05-sequence-parallel/

四组实验各自独立可复现，纯 numpy、纯 CPU、固定 seed：

  A. online softmax 分块合并 与 全局 softmax 的等价性（含「朴素逐块归一」的反例）
  B. 合并顺序无关性（环上 KV 块的访问次序可以任意）
  C. 因果掩码下的负载分布（contiguous / zigzag / striped 三种切法）
  D. 通信量账与算术强度门槛（ring 的 c >= F/B 条件）

每个函数都返回可断言的结构，断言写在 notebook 的每个分组末尾。
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# ---------------------------------------------------------------- A / B: 注意力


def global_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """单卡参考实现：一次性全局 softmax。Q/K/V 形状 (S, d)。"""
    d = Q.shape[1]
    S = Q @ K.T / math.sqrt(d)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V


def online_merge(
    m_a: np.ndarray,
    l_a: np.ndarray,
    o_a: np.ndarray,
    m_b: np.ndarray,
    l_b: np.ndarray,
    o_b: np.ndarray,
):
    """合并两个部分 softmax 状态 (m, l, O) —— FlashAttention 的 rescale 公式。

    这是 Ring Attention 每轮要做的唯一数学动作：
        m = max(m_a, m_b)
        l = l_a e^{m_a - m} + l_b e^{m_b - m}
        O = o_a e^{m_a - m} + o_b e^{m_b - m}
    """
    m = np.maximum(m_a, m_b)
    e_a = np.exp(m_a - m)
    e_b = np.exp(m_b - m)
    l = l_a * e_a + l_b * e_b
    # O 是 (bs, d)，缩放因子是逐行的，必须显式加行维，否则会按最后一维静默广播错
    o = o_a * e_a[:, None] + o_b * e_b[:, None]
    return m, l, o


def block_state(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """一个 query 块对一个 key/value 块的部分状态。"""
    d = Q.shape[1]
    S = Q @ K.T / math.sqrt(d)
    m = S.max(axis=1)
    P = np.exp(S - m[:, None])
    l = P.sum(axis=1)
    o = P @ V
    return m, l, o


def ring_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    n_blocks: int,
    order: list[int] | None = None,
):
    """模拟 ring：每个 query 块只持有一个 KV 块，逐块 online 合并。

    order 给 KV 块的访问次序（默认 0..n_blocks-1）。返回 (O, 每卡的块访问轨迹)。
    """
    S, d = Q.shape
    bs = S // n_blocks
    assert bs * n_blocks == S, "序列长度必须能被块数整除"
    if order is None:
        order = list(range(n_blocks))

    O = np.zeros((S, d))
    trace = []
    for i in range(n_blocks):
        q = Q[i * bs : (i + 1) * bs]
        m = np.full(bs, -np.inf)
        l = np.zeros(bs)
        o = np.zeros((bs, d))
        seen = []
        for j in order:
            k = K[j * bs : (j + 1) * bs]
            v = V[j * bs : (j + 1) * bs]
            m_b, l_b, o_b = block_state(q, k, v)
            m, l, o = online_merge(m, l, o, m_b, l_b, o_b)
            seen.append(j)
        O[i * bs : (i + 1) * bs] = o / l[:, None]
        trace.append(seen)
    return O, trace


def naive_blockwise_attention(Q, K, V, n_blocks):
    """反例：每块各自 softmax 再按块大小平均（不做全局 max 重标定）。

    这是「朴素地把分块结果拼起来」最容易写出的错误版本，用来量化错得多离谱。
    """
    S, d = Q.shape
    bs = S // n_blocks
    out = np.zeros((S, d))
    for i in range(n_blocks):
        q = Q[i * bs : (i + 1) * bs]
        acc = np.zeros((bs, d))
        for j in range(n_blocks):
            k = K[j * bs : (j + 1) * bs]
            v = V[j * bs : (j + 1) * bs]
            s = q @ k.T / math.sqrt(d)
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            acc += p @ v
        out[i * bs : (i + 1) * bs] = acc / n_blocks
    return out


# ---------------------------------------------------------------- C: 负载分布


def token_owner(n_tokens: int, n_dev: int, scheme: str) -> np.ndarray:
    """把 token 分配给设备，返回长度 n_tokens 的 owner 数组。

    contiguous：连续切片（Ring Attention 原始做法）
    zigzag    ：切成 2P 块，设备 r 拿第 r 块和第 (2P-1-r) 块（Megatron 的做法）
    striped   ：按 token 下标 mod P 分配（Striped Attention 的做法）
    """
    n_tokens = int(n_tokens)
    if scheme == "contiguous":
        step = n_tokens // n_dev
        return np.minimum(np.arange(n_tokens) // step, n_dev - 1)
    if scheme == "zigzag":
        assert n_tokens % (2 * n_dev) == 0
        m = n_tokens // (2 * n_dev)
        blk = np.arange(n_tokens) // m
        owner = np.where(blk < n_dev, blk, 2 * n_dev - 1 - blk)
        return owner
    if scheme == "striped":
        return np.arange(n_tokens) % n_dev
    raise ValueError(scheme)


def ring_rounds(n_tokens: int, n_dev: int, scheme: str):
    """给出每轮、每设备看到的 key token 集合（按环偏移）。

    环上传的是「设备自己的 KV 片」，第 t 轮设备 r 收到设备 (r+t) mod P 的 KV。
    返回 rounds[t][r] = 该轮设备 r 能看到的 token 下标数组。
    """
    owner = token_owner(n_tokens, n_dev, scheme)
    holders = [np.nonzero(owner == d)[0] for d in range(n_dev)]
    rounds = []
    for t in range(n_dev):
        rounds.append([holders[(r + t) % n_dev] for r in range(n_dev)])
    return rounds, owner


def causal_work_matrix(n_tokens: int) -> np.ndarray:
    """1[q >= k] 的因果可见矩阵（含对角）。"""
    i = np.arange(n_tokens)[:, None]
    j = np.arange(n_tokens)[None, :]
    return (i >= j).astype(np.int64)


def simulate_wallclock(n_tokens: int, n_dev: int, scheme: str):
    """按 token 精确计数模拟 ring 的墙钟。

    每轮所有设备并行，墙钟 = 该轮最慢的那台；总墙钟 = 各轮最慢之和（轮与轮之间同步）。
    返回 (wallclock, per_device_total, ideal)，单位是「一个 token 对的注意力量」。
    """
    C = causal_work_matrix(n_tokens)
    rounds, owner = ring_rounds(n_tokens, n_dev, scheme)
    per_dev = np.zeros(n_dev, dtype=np.int64)
    walls = []
    for t, rnd in enumerate(rounds):
        load = np.zeros(n_dev, dtype=np.int64)
        for r in range(n_dev):
            q = np.nonzero(owner == r)[0]
            k = rnd[r]
            load[r] = C[np.ix_(q, k)].sum()
        walls.append(load.max())
        per_dev += load
    total = per_dev.sum()
    ideal = total / n_dev
    return int(np.sum(walls)), per_dev, float(ideal)


# ---------------------------------------------------------------- D: 通信与门槛


def ring_comm_bytes(S: int, d: int, P: int, dtype_bytes: int = 2) -> float:
    """Ring Attention：每卡发送 (P-1) 个 KV 块，每块 2·(S/P)·d 个元素。"""
    return (P - 1) * 2 * (S / P) * d * dtype_bytes


def allgather_comm_bytes(S: int, d: int, P: int, dtype_bytes: int = 2) -> float:
    """all-gather 全部 KV：每卡收到 (P-1)/P 的全量 KV —— 与 ring 同量级。"""
    return (P - 1) / P * 2 * S * d * dtype_bytes


def ulysses_comm_bytes(S: int, h: int, P: int, dtype_bytes: int = 2) -> float:
    """Ulysses：每层两次 all-to-all，聚合量 3Nh（QKV）+ Nh（输出），每链路摊到 1/P。"""
    return 4 * S * h / P * dtype_bytes


def min_block_len(flops_per_sec: float, bytes_per_sec: float) -> float:
    """通信重叠条件 4dc²/F >= 4cd/B 给出的最小块长 c >= F/B。"""
    return flops_per_sec / bytes_per_sec


def min_seq_len(flops_per_sec: float, bytes_per_sec: float, blocks: int = 6) -> float:
    """Ring Attention 论文的 s = 6c（每卡要驻留 6 个块）。"""
    return blocks * min_block_len(flops_per_sec, bytes_per_sec)


HARDWARE = {
    # 名称: (单卡 BF16 dense TFLOPS, 单向互联带宽 GB/s)
    "A100 + NVLink": (312.0, 300.0),
    "A100 + InfiniBand": (312.0, 12.5),
    "TPU v4": (275.0, 268.0),
    "H100 + NVLink": (989.0, 450.0),
    "H100 + IB 400G": (989.0, 25.0),
    "B200 + NVLink5": (2250.0, 900.0),
}
