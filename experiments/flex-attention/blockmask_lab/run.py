#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blockmask_lab：BlockMask 工程那一半的数值验证（纯 numpy / CPU）

01 那篇回答的是「为什么这个编程模型成立」，这一份回答的是「它在工程上到底要付多少成本」：

  1. create_block_mask 的成本结构：它真的会物化一份 Q×KV 的布尔张量吗？多大？
  2. 每步 / 每层重建掩码，划不划算？break-even 稀疏度是多少？
  3. 确定性 dQ：原子累加为什么不确定，预先算好的写入顺序怎么把它修掉
  4. PTRR 负载均衡：BlockMask 的 kv_num_blocks 就是代价模型，重排能压掉多少不均衡
  5. BS × BLOCK_M/N × GQA 的选型账：约束在哪，浪费多少算力

运行：
    python3 run.py              # 跑全部五组
    python3 run.py 2 4          # 只跑第 2、4 组
    python3 run.py --selftest   # 只跑断言（静默）

依赖只有 numpy，固定 seed = 20261018，逐位可复现。
"""

from __future__ import annotations

import math
import sys

import numpy as np

SEED = 20261018
_ASSERTS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _ASSERTS.append(name)
    if not cond:
        raise AssertionError(f"[{name}] {detail}")


# --------------------------------------------------------------------------
# 共用：块网格与 BlockMask 元数据
# --------------------------------------------------------------------------
def block_metadata(mask: np.ndarray, bs: int):
    """从 N×N 布尔掩码构造 (kv_num_blocks, kv_indices) 与 full 版本。

    对应 torch.nn.attention.flex_attention 里的 _convert_mask_to_block_mask
    + _dense_to_ordered（B=H=1）。
    """
    n = mask.shape[0]
    nb = math.ceil(n / bs)
    num_blocks = np.zeros(nb, dtype=np.int64)
    indices = np.zeros((nb, nb), dtype=np.int64)
    full_num = np.zeros(nb, dtype=np.int64)
    full_idx = np.zeros((nb, nb), dtype=np.int64)
    for r in range(nb):
        for c in range(nb):
            blk = mask[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs]
            if blk.any():
                indices[r, num_blocks[r]] = c
                num_blocks[r] += 1
                if blk.all():
                    full_idx[r, full_num[r]] = c
                    full_num[r] += 1
    return num_blocks, indices, full_num, full_idx


def causal_mask(n: int) -> np.ndarray:
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return i >= j


def sliding_mask(n: int, w: int) -> np.ndarray:
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (i - j >= 0) & (i - j < w)


def document_mask(lengths) -> np.ndarray:
    doc = np.concatenate([np.full(L, k) for k, L in enumerate(lengths)])
    return doc[:, None] == doc[None, :]


# --------------------------------------------------------------------------
# 第 1 组：create_block_mask 的成本结构
# --------------------------------------------------------------------------
def run_group1(verbose=True):
    """源码里 create_block_mask → create_mask 会物化一份 (B,H,Q,KV) 的布尔张量。

    两件事要算清楚：这份张量有多大，以及 B=None/H=None 的广播到底省了多少。
    """
    configs = [
        # (标签, B, H, 是否广播, Q=KV)
        ("8B 训练 / 不广播", 8, 16, False, 8192),
        ("8B 训练 / 广播 H", 8, 1, False, 8192),
        ("8B 训练 / 广播 B,H", 1, 1, True, 8192),
        ("长文训练 32k", 1, 1, True, 32768),
        ("长文训练 32k / 不广播", 4, 8, False, 32768),
    ]

    rows = []
    for tag, b, h, bcast, n in configs:
        dense_bytes = b * h * n * n * 1          # bool 按 1 字节
        nb = math.ceil(n / 128)
        meta_bytes = nb * nb * 4 + nb * 4        # kv_indices(int32) + kv_num_blocks
        rows.append((tag, b, h, n, dense_bytes, meta_bytes, dense_bytes / meta_bytes))

    if verbose:
        print("\n【第 1 组】create_block_mask 的成本结构（BLOCK_SIZE=128，int32 索引）")
        print("  配置                    B   H    Q=KV    物化的 bool 掩码    BlockMask 元数据    比值")
        for tag, b, h, n, d, m, r in rows:
            print(
                f"  {tag:22s} {b:2d} {h:3d}  {n:6d}   {d/2**20:10.2f} MiB   {m/2**20:9.4f} MiB   {r:8.0f}×"
            )
        print("  关键：元数据是「压缩后」的成本，但构造过程中那份稠密 bool 张量是真实存在的峰值。")
        print("        B=None/H=None 之所以是官方首推，就是因为它把这份峰值从 B·H·Q·KV 降到 Q·KV。")

    # 广播确实能把峰值降下来；不广播时 8×16×8192² 是 GB 级
    r_bcast = rows[2]
    r_full = rows[0]
    check("G1 广播把峰值降到 GB 以下", r_bcast[4] < 2**30, f"{r_bcast[4]/2**20:.1f} MiB")
    check("G1 不广播时是 GB 级", r_full[4] >= 2**33, f"{r_full[4]/2**30:.1f} GiB")
    check("G1 元数据始终远小于稠密", all(r[4] > r[5] * 1000 for r in rows), "")

    # 元数据随 BLOCK_SIZE 平方下降
    n = 32768
    meta = {}
    for bs in (64, 128, 256, 1024):
        nb = math.ceil(n / bs)
        meta[bs] = nb * nb * 4 + nb * 4
    if verbose:
        print("\n  元数据随 BLOCK_SIZE 的平方下降（N=32768）：")
        for bs, m in meta.items():
            print(f"    BS={bs:5d} → {m/1024:9.2f} KiB")
        print(f"    BS 从 64 提到 1024（16×）→ 元数据降 {meta[64]/meta[1024]:.0f}×（≈256×，差在 nb 的取整）")

    check("G1 BS 翻倍元数据降约 4 倍", 3.5 < meta[128] / meta[256] < 4.5, f"{meta[128]/meta[256]:.2f}")
    return rows


# --------------------------------------------------------------------------
# 第 2 组：每步 / 每层重建，划不划算
# --------------------------------------------------------------------------
def run_group2(verbose=True):
    """把「重建掩码」和「跳过的算力」放在同一把尺子上。

    一次注意力的算力 ≈ 4·Q·KV·D（两次 matmul，每次 2·Q·KV·D 个 MAC）。
    一次 mask_mod 求值 ≈ κ 个 op/元素，κ 取 1 作下界（简单比较）或 4（带算术）。
    重建成本 ≈ κ·Q·KV + (Q/BS)(KV/BS) 的归约，相对于注意力是 κ/(4D)。
    """
    D = 64          # 头维
    BS = 128
    N = 8192
    L = 32          # 层数

    rows = []
    for kappa, tag in ((1.0, "κ=1（纯比较）"), (4.0, "κ=4（带算术）")):
        build_per_attn = kappa / (4.0 * D)
        rows.append((tag, build_per_attn, build_per_attn * 100))

    if verbose:
        print("\n【第 2 组】重建掩码 vs 跳过的算力（D=64，BS=128，N=8192，L=32 层）")
        print("  单次重建相对「一次完整注意力」的成本：")
        for tag, ratio, pct in rows:
            print(f"    {tag:14s}  {ratio:.6f}  =  {pct:.3f}%")
        print()
        print("  摊销方式                单次重建占一层注意力   占整个前向（32 层）")
        for tag, ratio, pct in rows:
            print(f"    每步建一次、32 层复用   {pct/L:.4f}%           {pct/L/L:.5f}%")
            print(f"    每层各建一次            {pct:.3f}%            {pct:.3f}%")
        print()
        print("  → break-even：只要掩码稀疏度让跳过的算力 > 重建成本，就划算。")
        print(f"    每层重建的阈值 ≈ κ/(4D) = {1/(4*D)*100:.3f}% ~ {4/(4*D)*100:.3f}% 的块被跳过。")
        print("    也就是说：**纯 FLOPs 口径下，每层重建几乎总是划算的**（阈值不到 2%）。")
        print("    真实成本不在 FLOPs，而在三处：物化的 bool 峰值显存、独立的 kernel 序列、")
        print("    以及动态 shape 触发的重编译。这正是官方 FAQ 把「每层重建」列为最难场景的原因。")

    # 定量：稀疏度带来的收益 vs 每层重建的代价
    print()
    print("  稀疏度收益 vs 每层重建代价（κ=4）：")
    print("    掩码            块级稀疏度(跳过比例)   省下算力     扣掉重建后净收益")
    results = []
    for name, mask in (
        ("causal", causal_mask(2048)),
        ("sliding w=512", sliding_mask(2048, 512)),
        ("document", document_mask([256, 512, 128, 384, 768])),
    ):
        n = mask.shape[0]
        num_blocks, _, full_num, _ = block_metadata(mask, BS)
        nb = math.ceil(n / BS)
        kept = int(num_blocks.sum())
        total = nb * nb
        skip = 1.0 - kept / total
        build = 4.0 / (4.0 * D)
        net = skip - build
        results.append((name, skip, net))
        print(f"    {name:15s} {skip*100:6.1f}%            {skip*100:6.1f}%      {net*100:+6.1f}%")

    check("G2 每层重建阈值 < 2%", 4.0 / (4 * D) < 0.02, f"{4.0/(4*D)}")
    check("G2 三种掩码净收益都为正", all(r[2] > 0 for r in results), str(results))
    return results


# --------------------------------------------------------------------------
# 第 3 组：确定性 dQ
# --------------------------------------------------------------------------
def run_group3(verbose=True):
    """原子累加是浮点非结合的：同一组部分和，换顺序就换结果。

    PyTorch 2.13 的解法是预先算好「写入顺序」，把累加固定成串行顺序。
    这里复现那个 rank 算法，并验证：按 rank 累加 == 按 n 升序串行累加（逐位）。
    """
    rng = np.random.default_rng(SEED + 3)
    nb_m, nb_n = 8, 16        # query 块数、KV 块数
    bs = 4

    # 一个块稀疏的「贡献者」矩阵：dense[m, n] 表示 (m,n) 这对块是否产生 dQ 贡献
    dense = (rng.random((nb_m, nb_n)) < 0.5).astype(np.int64)
    dense[:, 0] = 1           # 保证每行至少一个贡献者
    dense = dense.astype(np.int64)

    # 每个贡献的真实数值（模拟该块算出的 dQ 分量，float32）
    contrib = rng.standard_normal((nb_m, nb_n, bs)).astype(np.float32)

    # ---- PyTorch 的算法：rank = cumsum(dense) - dense ----
    cumsum = np.cumsum(dense, axis=-1)
    rank_asc = (cumsum - dense).astype(np.int32)
    total_per_m = cumsum[:, -1:]
    rank_spt = (total_per_m - 1 - rank_asc).astype(np.int32)

    # 校验：rank_asc[m,n] 就是「n 之前（不含 n）有多少个贡献者」
    for m in range(nb_m):
        for n in range(nb_n):
            if dense[m, n]:
                expected = int(dense[m, :n].sum())
                assert rank_asc[m, n] == expected, (m, n, rank_asc[m, n], expected)
    check("G3 rank_asc 等于「n 之前的贡献者数」", True)

    # ---- 非确定性：随机顺序累加 ----
    def accumulate(m, order):
        acc = np.zeros(bs, dtype=np.float32)
        for n in order:
            if dense[m, n]:
                acc = (acc + contrib[m, n]).astype(np.float32)
        return acc

    serial_asc = np.stack([accumulate(m, range(nb_n)) for m in range(nb_m)])
    serial_desc = np.stack([accumulate(m, reversed(range(nb_n))) for m in range(nb_m)])
    d_desc = float(np.max(np.abs(serial_asc - serial_desc)))

    # 多个随机顺序的离散度
    spread = []
    for _ in range(64):
        perm = rng.permutation(nb_n)
        spread.append(float(np.max(np.abs(serial_asc - np.stack(
            [accumulate(m, perm) for m in range(nb_m)])))))
    spread_max = max(spread)
    spread_mean = float(np.mean(spread))

    # ---- 确定性路径：按 rank 排序后再累加 == 串行升序 ----
    by_rank_asc = []
    for m in range(nb_m):
        pairs = [(int(rank_asc[m, n]), n) for n in range(nb_n) if dense[m, n]]
        pairs.sort()
        by_rank_asc.append(accumulate(m, [n for _, n in pairs]))
    by_rank_asc = np.stack(by_rank_asc)
    d_rank = float(np.max(np.abs(serial_asc - by_rank_asc)))

    by_rank_spt = []
    for m in range(nb_m):
        pairs = [(int(rank_spt[m, n]), n) for n in range(nb_n) if dense[m, n]]
        pairs.sort()
        by_rank_spt.append(accumulate(m, [n for _, n in pairs]))
    by_rank_spt = np.stack(by_rank_spt)
    d_spt = float(np.max(np.abs(serial_desc - by_rank_spt)))

    if verbose:
        print("\n【第 3 组】确定性 dQ：同一组部分和，换顺序就换结果")
        print(f"  query 块 {nb_m} × KV 块 {nb_n}，块大小 {bs}，float32")
        print(f"  |升序串行 − 降序串行|               = {d_desc:.3e}")
        print(f"  64 个随机顺序相对升序的最大偏差     = {spread_max:.3e}（均值 {spread_mean:.3e}）")
        print(f"  |升序串行 − 按 rank_asc 累加|       = {d_rank:.3e}   ← 逐位一致")
        print(f"  |降序串行 − 按 rank_spt 累加|       = {d_spt:.3e}   ← 逐位一致")
        print("  结论：rank 表把「谁先写」提前定死，累加顺序不再取决于 SM 的到达次序；")
        print("        dq_kv_order=True 只是把顺序反过来（SPT），两种都是确定的，只是基准不同。")

    check("G3 顺序会影响结果", d_desc > 0.0 or spread_max > 0.0, f"{d_desc} {spread_max}")
    check("G3 rank_asc 累加逐位等于串行升序", d_rank == 0.0, f"{d_rank:.3e}")
    check("G3 rank_spt 累加逐位等于串行降序", d_spt == 0.0, f"{d_spt:.3e}")
    return d_rank, d_spt, spread_max


# --------------------------------------------------------------------------
# 第 4 组：PTRR 负载均衡
# --------------------------------------------------------------------------
def ptrr_scheduling(process_time: np.ndarray, group_size: int) -> np.ndarray:
    """复现 torch.distributed.tensor.experimental._context_parallel._load_balancer
    ._PTRRLoadBalancer.ptrr_scheduling 的算法。

    1. 按 process_time 降序稳定排序；
    2. 每 group_size 个一组，交替把组内顺序翻转；
    3. 转置成 (group_size, n // group_size)：第 i 个任务进第 (i mod group_size) 组；
    4. 组内排序（仅为了好看）。
    """
    n = process_time.shape[0]
    if n % group_size != 0:
        raise NotImplementedError(f"num_tasks {n} must be divisible by group_size {group_size}")
    order = np.argsort(-process_time, kind="stable")
    # 每 group_size 个一组，交替翻转（源码：i // group_size % 2 == 0 用原序，否则翻转）
    flipped = order.reshape(-1, group_size).copy()
    for g in range(flipped.shape[0]):
        if g % 2 == 1:
            flipped[g] = flipped[g][::-1]
    flipped = flipped.reshape(-1)
    groups = flipped.reshape(-1, group_size).T          # (group_size, n // group_size)
    return np.sort(groups, axis=1)


def headtail_pairs(seq_blocks: int, world_size: int) -> np.ndarray:
    """经典 head-tail 配对：块 i 与块 (n-1-i) 配成一对，再把 n/2 对轮转分给各 rank。"""
    n = seq_blocks
    assert n % (world_size * 2) == 0
    npairs = n // 2
    chunk = npairs // world_size
    groups = []
    for r in range(world_size):
        g = []
        for k in range(chunk):
            i = r * chunk + k          # 第 (r*chunk + k) 对
            g.extend([i, n - 1 - i])
        groups.append(sorted(g))
    return np.array(groups, dtype=np.int64)


def run_group4(verbose=True):
    """BlockMask 的 kv_num_blocks 就是代价模型：一个 query 块要算几个 KV 块。

    上下文并行把 query 块分给各 rank，朴素连续切分会因为 causal 而严重不均衡。
    """
    N, BS = 4096, 128
    CP = 8
    nb = N // BS

    cases = []
    for name, mask in (
        ("causal", causal_mask(N)),
        ("sliding w=1024", sliding_mask(N, 1024)),
        # 4 篇文档，长度都是 BS 的整数倍，总和恰好 N
        ("document", document_mask([1024, 1536, 512, 1024])),
    ):
        num_blocks, _, full_num, _ = block_metadata(mask, BS)
        cost = (num_blocks + full_num).astype(np.float64)   # 每块的 KV 块数 = 处理时间
        assert cost.shape[0] == nb, (name, cost.shape, nb)
        cases.append((name, cost))

    if verbose:
        print(f"\n【第 4 组】PTRR 负载均衡（N={N}，BS={BS}，CP={CP}，query 块数 {nb}）")
        print("  代价模型 = 该 query 块要算的 KV 块数（kv_num_blocks + full_kv_num_blocks）")
        print()
        print("  掩码            切分方式   各 rank 代价(max/min)   理想均值   max/理想   峰谷差")
        for name, cost in cases:
            rank_mean = cost.sum() / CP
            naive = cost.reshape(CP, -1).sum(axis=1)
            try:
                ht = np.array([cost[g].sum() for g in headtail_pairs(nb, CP)])
            except AssertionError:
                ht = None
            ptrr = ptrr_scheduling(cost, CP)
            p = np.array([cost[g].sum() for g in ptrr])
            for tag, s in (("朴素连续", naive), ("head-tail", ht), ("PTRR", p)):
                if s is None:
                    print(f"  {name:15s} {tag:10s}  —（块数不满足 world_size*2 整除）")
                    continue
                print(
                    f"  {name:15s} {tag:10s}  {s.max():7.0f} / {s.min():7.0f}    {rank_mean:7.1f}   "
                    f"{s.max()/rank_mean:7.3f}   {(s.max()-s.min()):7.0f}"
                )
            print()

    # 断言：causal 下朴素切分严重不均衡，PTRR 明显改善
    name, cost = cases[0]
    rank_mean = cost.sum() / CP
    naive = cost.reshape(CP, -1).sum(axis=1)
    ptrr = ptrr_scheduling(cost, CP)
    p = np.array([cost[g].sum() for g in ptrr])
    naive_ratio = naive.max() / rank_mean
    ptrr_ratio = p.max() / rank_mean

    if verbose:
        print(f"  causal：朴素连续 max/理想 = {naive_ratio:.3f} → PTRR {ptrr_ratio:.3f}（1.0 为完美均衡）")

    check("G4 朴素切分确实不均衡", naive_ratio > 1.3, f"{naive_ratio:.3f}")
    check("G4 PTRR 显著改善", ptrr_ratio < 1.15, f"{ptrr_ratio:.3f}")
    check("G4 PTRR 每个块恰好分一次", sorted(ptrr.reshape(-1).tolist()) == list(range(nb)), "")

    # PTRR 的结构不变量：组大小相等、覆盖全部
    for name, cost in cases:
        g = ptrr_scheduling(cost, CP)
        assert g.shape == (CP, nb // CP)
        assert sorted(g.reshape(-1).tolist()) == list(range(nb))
    check("G4 PTRR 三种掩码都满足结构不变量", True)
    return naive_ratio, ptrr_ratio


# --------------------------------------------------------------------------
# 第 5 组：BS × BLOCK_M/N × GQA 的选型账
# --------------------------------------------------------------------------
def run_group5(verbose=True):
    """块粒度是有损的：块越大，为了少数几个必须算的元素而陪算的元素越多。"""
    N, D = 8192, 64

    masks = {
        "causal": causal_mask(N),
        "sliding w=1024": sliding_mask(N, 1024),
        "document": document_mask(
            [1024, 2048, 512, 1536, 512, 1024, 1024, 512]
        ),
    }

    rows = []
    for name, mask in masks.items():
        need = int(mask.sum())
        for bs in (64, 128, 256):
            num_blocks, _, full_num, _ = block_metadata(mask, bs)
            nb = math.ceil(N / bs)
            computed = int(num_blocks.sum()) * bs * bs
            waste = (computed - need) / need
            rows.append((name, bs, nb, int(num_blocks.sum()), need, computed, waste))

    if verbose:
        print(f"\n【第 5 组】块粒度的浪费率（N={N}）")
        print("  掩码            BS   块数/行   保留块   必需元素      实算元素     浪费率")
        for name, bs, nb, kept, need, computed, waste in rows:
            print(
                f"  {name:14s} {bs:4d}   {nb:5d}   {kept:5d}   {need:10d}   {computed:11d}   {waste*100:7.2f}%"
            )
        print("  浪费率 =（整块陪算的元素 − 真正需要的元素）/ 真正需要的元素。")

    # 约束检查：SPARSE_KV_BLOCK_SIZE 必须被 BLOCK_N 整除，否则 Inductor 直接报错
    print()
    print("  约束：SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0 且 SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0")
    print("    (BS, BLOCK_N)   合法？")
    for bs in (64, 128, 256):
        for bn in (32, 64, 128):
            ok = (bs % bn == 0) and (bs >= bn)
            print(f"    ({bs:4d}, {bn:4d})      {'✓' if ok else '✗  → raise_flex_kernel_options_error'}")
            if bs == 128 and bn == 64:
                check("G5 128 % 64 == 0 合法", ok, "")

    # GQA：FlexDecoding 要求 Hq/Hkv 是 2 的幂；BS 必须能被 BLOCK_M/N 整除
    print()
    print("  FlexDecoding 的准入条件（flex_decoding.py:66-132）：")
    print("    GQA 比值   是 2 的幂？   FlexDecoding 可用？")
    for ratio in (1, 2, 3, 4, 6, 8):
        pw = (ratio > 0) and (ratio & (ratio - 1)) == 0
        print(f"      {ratio:3d}         {'✓' if pw else '✗'}            {'✓' if pw else '✗'}")
    check("G5 3 不是 2 的幂", (3 & 2) != 0, "")
    check("G5 8 是 2 的幂", (8 & 7) == 0, "")

    # 块级密度 vs 元素密度（浪费的另一种说法）
    print()
    print("  同一件事的两种说法：")
    for name, bs, nb, kept, need, computed, waste in rows:
        if bs == 128:
            elem_density = need / (N * N)
            block_density = kept / (nb * nb)
            print(
                f"    {name:14s} 元素密度 {elem_density*100:5.2f}%  →  块级密度 {block_density*100:5.2f}%"
                f"   （差 {(block_density-elem_density)*100:5.2f} 个百分点）"
            )

    check("G5 浪费率非负", all(r[6] >= 0 for r in rows), "")
    check("G5 BS 越大浪费越多", all(
        [r for r in rows if r[0] == n and r[1] == 256][0][6]
        >= [r for r in rows if r[0] == n and r[1] == 64][0][6]
        for n in masks
    ), "")
    return rows


GROUPS = {1: run_group1, 2: run_group2, 3: run_group3, 4: run_group4, 5: run_group5}


def main(argv):
    if "--selftest" in argv:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for i in sorted(GROUPS):
                GROUPS[i](verbose=False)
        print(f"selftest OK: {len(_ASSERTS)} 条断言全部通过")
        return 0

    sel = [int(a) for a in argv if a.isdigit()]
    keys = sel if sel else sorted(GROUPS)
    quiet = "--quiet" in argv

    print("=" * 78)
    print("blockmask_lab：BlockMask 工程那一半（numpy / CPU / seed=%d）" % SEED)
    print("=" * 78)
    for i in keys:
        GROUPS[i](verbose=not quiet)

    print("\n" + "=" * 78)
    print(f"总计 {len(_ASSERTS)}/{len(_ASSERTS)} 通过")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
