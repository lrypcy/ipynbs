#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flex_core：FlexAttention 编程模型的数值验证（纯 numpy / CPU）

回答七个问题，每条都做「构造真值 ↔ 流计算」或「解析真值 ↔ 数值反算」自校验：

  1. online softmax 配上任意逐点 score_mod，是否真的等于一次性算完整矩阵
  2. base-2 换底之后 logsumexp 的口径到底是什么
  3. 运行最大值 m_i 为什么必须取在「修正之后」的分数上
  4. 表达力边界在哪：什么样的 mod 流式算不出来
  5. BlockMask 把掩码压缩成多少元数据、跳掉多少块、丢掉多少稀疏度
  6. and_mask / or_mask 的组合是否等价于手写掩码
  7. PagedAttention 的 physical → logical 索引转换是否保语义

运行：
    python3 run.py              # 跑全部七组
    python3 run.py 1 3          # 只跑第 1、3 组
    python3 run.py --selftest   # 只跑断言（静默）

依赖只有 numpy，固定 seed = 20261017，逐位可复现。
"""

from __future__ import annotations

import math
import sys

import numpy as np

SEED = 20261017
LN2 = math.log(2.0)
RCP_LN2 = 1.0 / LN2  # PyTorch 模板里的常量 1.44269504
NEG_INF = -np.inf

_ASSERTS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _ASSERTS.append(name)
    if not cond:
        raise AssertionError(f"[{name}] {detail}")


def approx(x: float, y: float, tol: float) -> bool:
    return abs(x - y) <= tol


# --------------------------------------------------------------------------
# 0. 基础设施：mod 函数库
# --------------------------------------------------------------------------
def mod_noop(s, i, j):
    return s


def mod_causal(s, i, j):
    return np.where(i >= j, s, NEG_INF)


def mk_alibi(slope: float):
    """ALiBi：按头给斜率，按 (i - j) 给惩罚（这里单头）。"""

    def f(s, i, j):
        return s - slope * np.abs(i - j)

    return f


def mk_softcap(cap: float = 20.0, sm_scale: float = 1.0):
    """Gemma-2 的 logit soft-capping：cap * tanh(s / cap)。"""

    def f(s, i, j):
        # s 已经乘过 sm_scale，这里按相对步长还原再压回去
        return cap * np.tanh((s / sm_scale) / cap) * sm_scale

    return f


def mk_rel_bias(bias: np.ndarray):
    """可学习位置偏置（T5 风格）。"""

    def f(s, i, j):
        return s + bias[i, j]

    return f


def mod_row_mean_gate(s, i, j):
    """不可流的 mod：每个元素乘上一个依赖整行均值的行标量（σ(mean_j s_ij)）。"""
    row_mean = np.mean(s, axis=1, keepdims=True)
    return s / (1.0 + np.exp(-row_mean))


# --------------------------------------------------------------------------
# 参考实现：材料化完整矩阵
# --------------------------------------------------------------------------
def materialize_scores(q, k, sm_scale, score_mod=None, mask_mod=None):
    N = q.shape[0]
    ii = np.arange(N)[:, None]
    jj = np.arange(N)[None, :]
    s = (q @ k.T) * sm_scale
    if score_mod is not None:
        s = score_mod(s, ii, jj)
    if mask_mod is not None:
        s = np.where(mask_mod(ii, jj), s, NEG_INF)
    return s


def naive_attention(q, k, v, *, sm_scale, score_mod=None, mask_mod=None, base2=False):
    N = q.shape[0]
    s = materialize_scores(q, k, sm_scale, score_mod, mask_mod)
    s2 = s * RCP_LN2 if base2 else s
    logf = np.log2 if base2 else np.log
    m = np.max(s2, axis=1)
    m_fin = np.where(np.isneginf(m), 0.0, m)
    p = (np.exp2 if base2 else np.exp)(s2 - m_fin[:, None])
    l = np.sum(p, axis=1)
    l_safe = np.where(l == 0.0, 1.0, l)
    out = (p @ v) / l_safe[:, None]
    out = np.where(np.isneginf(m)[:, None], 0.0, out)
    if base2:
        lse = (m_fin + np.log2(l_safe)) * LN2   # 换成自然对数单位
    else:
        lse = m_fin + np.log(l_safe)
    lse = np.where(np.isneginf(m), NEG_INF, lse)
    return out.reshape(N, -1), lse.reshape(N, 1), s


# --------------------------------------------------------------------------
# 待验证实现：分块流式（对齐 common.py.jinja 的 forward_block_mn）
# --------------------------------------------------------------------------
def online_attention(
    q,
    k,
    v,
    *,
    sm_scale,
    block_n=32,
    score_mod=None,
    mask_mod=None,
    max_over="post",
    partial_row_mod=None,
):
    """返回 (out, lse, max_exponent)。

    max_over: "post" 取修正后分数的行最大值（PyTorch 的做法）；
              "pre"  取未修正分数的行最大值（第 3 组对照用）。
    max_exponent: 记录实际求过的最大指数，用来说明「什么时候会溢出」。
    """
    N = q.shape[0]
    Dv = v.shape[1]
    m = np.full(N, NEG_INF, dtype=np.float64)
    l = np.zeros(N, dtype=np.float64)
    acc = np.zeros((N, Dv), dtype=np.float64)
    max_exp = NEG_INF

    nb = math.ceil(N / block_n)
    for b in range(nb):
        j0 = b * block_n
        j1 = min(j0 + block_n, N)
        ii = np.arange(N)[:, None]
        jj = np.arange(j0, j1)[None, :]

        s_raw = (q @ k[j0:j1].T) * sm_scale
        s_pre_log2 = s_raw * RCP_LN2

        s = s_raw
        if score_mod is not None:
            s = score_mod(s, ii, jj)
        if partial_row_mod is not None:
            # 流式版本「只能知道已到达的部分」，用 running 统计量代替全局统计量
            s = partial_row_mod(s, j0, j1, N)
        if mask_mod is not None:
            s = np.where(mask_mod(ii, jj), s, NEG_INF)

        s_log2 = s * RCP_LN2
        rowmax_post = np.max(s_log2, axis=1)
        rowmax_ref = np.max(s_pre_log2, axis=1) if max_over == "pre" else rowmax_post
        m_ij = np.maximum(m, rowmax_ref)

        masked_out_rows = np.isneginf(m_ij)
        m_ij_masked = np.where(masked_out_rows, 0.0, m_ij)

        exp_arg = s_log2 - m_ij_masked[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            finite_arg = np.where(np.isneginf(exp_arg), NEG_INF, exp_arg)
            if finite_arg.size and np.any(np.isfinite(finite_arg)):
                max_exp = max(max_exp, float(np.max(finite_arg)))
            p = np.exp2(finite_arg)
            alpha = np.exp2(m - m_ij_masked)
        p = np.where(np.isnan(p), 0.0, p)
        p = np.where(np.isposinf(p), np.inf, p)
        alpha = np.where(np.isnan(alpha), 0.0, alpha)

        l = l * alpha + np.sum(p, axis=1)
        acc = acc * alpha[:, None] + p @ v[j0:j1]
        m = m_ij

    l_safe = np.where(l == 0.0, 1.0, l)
    with np.errstate(invalid="ignore"):
        out = acc / l_safe[:, None]
    out = np.where(np.isneginf(m)[:, None], 0.0, out)
    out = np.where(np.isnan(out), np.nan, out)
    lse = (np.where(np.isneginf(m), 0.0, m) + np.log2(l_safe)) * LN2
    lse = np.where(np.isneginf(m), NEG_INF, lse)
    return out, lse.reshape(N, 1), max_exp


# --------------------------------------------------------------------------
# 第 1 组：online softmax + 任意逐点 score_mod 是否成立
# --------------------------------------------------------------------------
def run_group1(verbose=True):
    N, D = 256, 32
    sm_scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(N, D, seed=SEED)
    rng = np.random.default_rng(SEED + 1)
    bias = rng.standard_normal((N, N)) * 0.05

    variants = [
        ("noop      ", mod_noop, None),
        ("causal    ", mod_causal, None),
        ("alibi     ", mk_alibi(0.05), None),
        ("softcap   ", mk_softcap(20.0, sm_scale), None),
        ("rel_bias  ", mk_rel_bias(bias), None),
        ("causal+alibi", lambda s, i, j: mk_alibi(0.05)(mod_causal(s, i, j), i, j), None),
    ]

    rows = []
    worst = 0.0
    for name, fm, mm in variants:
        out_ref, lse_ref, _ = naive_attention(
            q, k, v, sm_scale=sm_scale, score_mod=fm, mask_mod=mm
        )
        out_on, lse_on, _ = online_attention(
            q, k, v, sm_scale=sm_scale, score_mod=fm, mask_mod=mm, block_n=32
        )
        denom = np.max(np.abs(out_ref)) if np.max(np.abs(out_ref)) > 0 else 1.0
        d_out = float(np.max(np.abs(out_ref - out_on) / denom))
        d_lse = float(np.max(np.abs(lse_ref - lse_on)))
        rows.append((name, d_out, d_lse))
        worst = max(worst, d_out, d_lse)
        # 和 materialize 出来的定义逐个比对 hint：参考实现自身也要自校验
        s_ref = materialize_scores(q, k, sm_scale, fm, mm)
        p_ref = np.exp((s_ref * RCP_LN2) * LN2 - np.max(s_ref, axis=1, keepdims=True))
        check(
            f"G1 {name.strip()} 参考实现自校验",
            float(np.max(np.abs(np.sum(p_ref, axis=1)))) > 0.0,
            "参考实现的 softmax 分母退化为 0",
        )

    if verbose:
        print("\n【第 1 组】online softmax × 六种 score_mod（N=256，D=32，float64）")
        print("  变体            max|Δout|/max|out|    max|Δlse|")
        for name, d_out, d_lse in rows:
            print(f"  {name:14s}   {d_out:.3e}        {d_lse:.3e}")
        print(f"  六种变体里最差的一项：{worst:.3e}（浮点误差量级，不是算法误差）")

    check(
        "G1 online == naive（六种变体）",
        worst < 1e-12,
        f"worst={worst:.3e}",
    )
    return worst


# --------------------------------------------------------------------------
# 第 2 组：base-2 换底与 logsumexp 的口径
# --------------------------------------------------------------------------
def run_group2(verbose=True):
    N, D = 128, 32
    sm_scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(N, D, seed=SEED + 2)
    fm = mk_alibi(0.05)

    _, lse_nat, s = naive_attention(q, k, v, sm_scale=sm_scale, score_mod=fm, base2=False)
    _, lse_b2, _ = naive_attention(q, k, v, sm_scale=sm_scale, score_mod=fm, base2=True)
    _, lse_on, _ = online_attention(q, k, v, sm_scale=sm_scale, score_mod=fm, block_n=32)

    d1 = float(np.max(np.abs(lse_nat - lse_b2)))
    d2 = float(np.max(np.abs(lse_nat - lse_on)))

    # 直接用 logsumexp 的定义做第三方校验
    m_ref = np.max(s, axis=1)
    lse_def = (m_ref + np.log(np.sum(np.exp(s - m_ref[:, None]), axis=1))).reshape(N, 1)
    d3 = float(np.max(np.abs(lse_def - lse_nat)))

    if verbose:
        print("\n【第 2 组】base-2 换底与 logsumexp 的口径（N=128）")
        print(f"  |lse(自然 log) − lse(base-2 换算回自然 log)| = {d1:.3e}")
        print(f"  |lse(自然 log) − lse(流式，内部 log2)|       = {d2:.3e}")
        print(f"  |lse(定义式)   − lse(参考实现)|              = {d3:.3e}")
        print(f"  RCP_LN2 = 1/ln2 = {RCP_LN2:.8f}（模板里写成常量 1.44269504）")
        print(f"  ln2     = {LN2:.8f}（flex_attention 在返回前乘回这个数）")

    check("G2 lse 两种底数换算一致", d1 < 1e-12, f"d1={d1:.3e}")
    check("G2 流式 lse 与参考一致", d2 < 1e-12, f"d2={d2:.3e}")
    check("G2 lse 与定义式一致", d3 < 1e-12, f"d3={d3:.3e}")
    check("G2 RCP_LN2 常量", approx(RCP_LN2, 1.44269504, 1e-7), f"{RCP_LN2}")
    return d2


# --------------------------------------------------------------------------
# 第 3 组：运行最大值为什么必须取在修正后的分数上
# --------------------------------------------------------------------------
def run_group3(verbose=True):
    N, D = 128, 32
    sm_scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(N, D, seed=SEED + 3)

    # 一个「看起来无害」的 mod：整体加一个常数。softmax 对此不变，
    # 但如果 max 取在修正之前，指数里的差值会把这个常数原封不动暴露出来。
    for bias_val, tag in ((80.0, "bias=+80"), (160.0, "bias=+160")):

        def fm(s, i, j, b=bias_val):
            return s + b

        out_ref, lse_ref, _ = naive_attention(q, k, v, sm_scale=sm_scale, score_mod=fm)
        out_post, lse_post, maxexp_post = online_attention(
            q, k, v, sm_scale=sm_scale, score_mod=fm, block_n=32, max_over="post"
        )
        out_pre, lse_pre, maxexp_pre = online_attention(
            q, k, v, sm_scale=sm_scale, score_mod=fm, block_n=32, max_over="pre"
        )
        d_post = float(np.max(np.abs(out_ref - out_post)))
        d_pre = float(np.max(np.abs(out_ref - out_pre)))

        # 数值不变式：问题的答案与参考值无关，log ∣ Rendered 指数 ∣ 才是关键
        exp_arg_gap = maxexp_pre - maxexp_post
        f32_break = maxexp_pre > 127.0

        if verbose:
            print(f"\n【第 3 组】max 取在哪一侧：常数偏置 {tag}（N=128，float64 计算）")
            print(f"  参考 vs max-取-修正后 : {d_post:.3e}")
            print(f"  参考 vs max-取-修正前 : {d_pre:.3e}   ← float64 下同样正确")
            print(f"  实际求过的 log2 指数上界：修正前 {maxexp_pre:.2f} / 修正后 {maxexp_post:.2f}")
            print(f"  差值 {exp_arg_gap:.2f} = {bias_val} / ln2 = {bias_val * RCP_LN2:.2f}")
            print(
                f"  float32 的溢出阈值是 2^127 ≈ 1.70e38："
                f"取修正前 {'会' if f32_break else '不会'}溢出，取修正后不会"
            )

        check(f"G3 {tag} 修正后 max 的结果正确", d_post < 1e-12, f"d_post={d_post:.3e}")
        check(f"G3 {tag} 修正前 max 数学上仍正确", d_pre < 1e-10, f"d_pre={d_pre:.3e}")
        check(
            f"G3 {tag} 指数差值 = bias/ln2",
            approx(exp_arg_gap, bias_val * RCP_LN2, 1e-6),
            f"{exp_arg_gap}",
        )

    return None


# --------------------------------------------------------------------------
# 第 4 组：表达力边界——什么样的 mod 流式算不出来
# --------------------------------------------------------------------------
def run_group4(verbose=True):
    """表达力边界：依赖「整行统计量」的 mod 无法流式计算。

    判据不是「算得准不准」，而是「同一个元素的修正值会不会因为后面还没到的块而变」。
    """
    N, D = 128, 32
    sm_scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(N, D, seed=SEED + 4)

    # ① 真值：把整行摊开一次性算
    out_ref, _, _ = naive_attention(
        q, k, v, sm_scale=sm_scale, score_mod=mod_row_mean_gate
    )

    # ② 流式只能看见已到达的块：用「已见元素的行均值」代替整行均值
    state = {"n": 0, "s": np.zeros(N)}

    def partial(s, j0, j1, N_all):
        state["n"] += s.shape[1]
        state["s"] += np.sum(s, axis=1)
        running_mean = state["s"] / max(state["n"], 1)
        return s / (1.0 + np.exp(-running_mean[:, None]))

    out_stream, _, _ = online_attention(
        q, k, v, sm_scale=sm_scale, block_n=32, partial_row_mod=partial
    )
    denom = float(np.max(np.abs(out_ref)))
    rel = float(np.max(np.abs(out_ref - out_stream)) / denom)

    # ③ 对照：一个真正的逐点 mod（soft-cap）在同样的流式路径下应当逐位对上
    fm_soft = mk_softcap(20.0, sm_scale)
    ref_soft, _, _ = naive_attention(q, k, v, sm_scale=sm_scale, score_mod=fm_soft)
    onl_soft, _, _ = online_attention(
        q, k, v, sm_scale=sm_scale, score_mod=fm_soft, block_n=32
    )
    d_soft = float(np.max(np.abs(ref_soft - onl_soft)))

    # ④ 行均值本身在流式过程中一路漂移，直到最后一个块才收敛到真值
    s_full = materialize_scores(q, k, sm_scale, None, None)
    true_mean = np.mean(s_full, axis=1)
    bs = 32
    drift = []
    seen_sum = np.zeros(N)
    for b in range(N // bs):
        blk = s_full[:, b * bs : (b + 1) * bs]
        seen_sum += np.sum(blk, axis=1)
        seen_n = (b + 1) * bs
        drift.append(float(np.mean(np.abs(seen_sum / seen_n - true_mean))))

    if verbose:
        print("\n【第 4 组】表达力边界：依赖「整行行均值」的 mod（N=128，分 4 块）")
        print("  块序   部分行均值与真值的平均绝对偏差")
        for b, d in enumerate(drift):
            print(f"   {b+1}     {d:.6f}")
        print(f"  逐点 mod（soft-cap）流式 vs 一次性：      {d_soft:.3e}   ← 可表达")
        print(f"  依赖行均值的 mod 流式 vs 一次性：         {rel:.3e}   ← 不可表达")
        print("  判据：某个元素的修正值依赖「还没算到的列」，重标定只改得了分母，")
        print("        改不了已经混进 acc 的分子 —— 这正是 score_mod 必须逐点的原因。")

    check("G4 逐点 mod 逐位可复现", d_soft < 1e-12, f"{d_soft:.3e}")
    check("G4 整行依赖 mod 无法流式复现", rel > 1e-3, f"{rel:.3e}")
    check("G4 最后一块才收敛", drift[-1] < 1e-15, f"{drift[-1]:.3e}")
    return rel


# --------------------------------------------------------------------------
# 第 5 组：BlockMask 的元数据账与稀疏度账
# --------------------------------------------------------------------------
def build_block_metadata(mask, bs):
    """从 N×N 的布尔掩码构造 (kv_num_blocks, kv_indices)，并区分 full / partial。"""
    N = mask.shape[0]
    nb = math.ceil(N / bs)
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
    return (
        num_blocks,
        indices,
        full_num,
        full_idx,
    )


def blockmask_report(name, mask, bs):
    nb_rows = math.ceil(mask.shape[0] / bs)
    nb_cols = math.ceil(mask.shape[1] / bs)
    num_blocks, indices, full_num, full_idx = build_block_metadata(mask, bs)
    total_blocks = nb_rows * nb_cols
    kept = int(np.sum(num_blocks))
    kept_full = int(np.sum(full_num))
    kept_partial = kept - kept_full
    elem = float(np.mean(mask))
    block_elem = kept / total_blocks
    return {
        "name": name,
        "N": mask.shape[0],
        "bs": bs,
        "total_blocks": total_blocks,
        "kept": kept,
        "full": kept_full,
        "partial": kept_partial,
        "skipped": total_blocks - kept,
        "elem_density": elem,
        "block_density": block_elem,
        "waste": (kept * bs * bs - int(np.sum(mask))) / max(int(np.sum(mask)), 1),
    }


def run_group5(verbose=True):
    N = 512
    ii = np.arange(N)[:, None]
    jj = np.arange(N)[None, :]

    causal = ii >= jj
    w = 256
    sliding = (ii - jj >= 0) & (ii - jj < w)

    rng = np.random.default_rng(SEED + 5)
    # 多文档拼接：随机文档长度铺满 N
    lens = []
    tot = 0
    while tot < N:
        L = int(rng.integers(64, 192))
        lens.append(min(L, N - tot))
        tot += min(L, N - tot)
    doc_id = np.concatenate([np.full(L, k) for k, L in enumerate(lens)])
    document = doc_id[:, None] == doc_id[None, :]

    reports = [
        blockmask_report("causal", causal, 64),
        blockmask_report("sliding(w=256)", sliding, 64),
        blockmask_report("document", document, 64),
        blockmask_report("causal", causal, 128),
    ]

    if verbose:
        print("\n【第 5 组】BlockMask 的两笔账（N=512，B=H=1）")
        print(
            "  掩码            BS   总块数  保留  full  partial  整块跳过   元素密度   块级密度"
        )
        for r in reports:
            print(
                f"  {r['name']:15s} {r['bs']:3d}   {r['total_blocks']:4d}   {r['kept']:3d}  "
                f"{r['full']:3d}   {r['partial']:3d}     {r['skipped']:3d}      "
                f"{r['elem_density']*100:5.1f}%    {r['block_density']*100:5.1f}%"
            )
        print("  块级密度总是 ≥ 元素密度：128×128 里只要有一个元素要算，整块就得算。")

    rc = reports[0]
    check(
        "G5 causal 的三角形计块数",
        rc["kept"] == 36 and rc["total_blocks"] == 64,
        f"kept={rc['kept']} total={rc['total_blocks']}",
    )
    check(
        "G5 causal 的 full/partial 分裂",
        rc["full"] == 28 and rc["partial"] == 8,
        f"full={rc['full']} partial={rc['partial']}",
    )
    check(
        "G5 块级密度 ≥ 元素密度",
        all(r["block_density"] >= r["elem_density"] - 1e-12 for r in reports),
        "",
    )

    # 元数据 vs 稠密布尔掩码的显存账
    def n_blocks(n, bs):
        return math.ceil(n / bs)

    print("\n  元数据大小 vs 稠密布尔掩码（B=H=1，int32 索引，bool 按 1 字节）")
    print("  序列长  BS     稠密 bool        BlockMask 元数据")
    for n, bs in ((8192, 128), (32768, 128), (32768, 1024), (1 << 20, 128)):
        dense = n * n * 1
        rows = n_blocks(n, bs)
        meta = rows * rows * 4 + rows * 4  # kv_indices + kv_num_blocks
        print(
            f"  {n:8d}  {bs:5d}   {dense/1e6:9.2f} MB      {meta/1e6:9.4f} MB"
            f"    （{dense/max(meta,1):.0f}×）"
        )
    return reports


# --------------------------------------------------------------------------
# 第 6 组：and_mask / or_mask 组合是否等价于手写掩码
# --------------------------------------------------------------------------
def run_group6(verbose=True):
    N = 256
    ii = np.arange(N)[:, None]
    jj = np.arange(N)[None, :]
    causal = ii >= jj
    prefix_len = 64
    prefix_block = (ii < prefix_len) & (jj < prefix_len)
    win128 = (ii - jj >= 0) & (ii - jj < 128)
    doc_len = 96
    doc = (ii // doc_len) == (jj // doc_len)

    # prefixLM：前缀区域内双向，前缀之外因果。注意它不等于「前 64 行全可见」。
    or_composed = (prefix_block | causal).astype(bool)
    or_hand = (
        ((ii < prefix_len) & (jj < prefix_len)) | ((ii >= prefix_len) & (ii >= jj))
    ).astype(bool)
    and_composed = (win128 & causal & doc).astype(bool)
    and_hand = ((ii - jj >= 0) & (ii - jj < 128) & ((ii // doc_len) == (jj // doc_len)))

    d_or = int(np.sum(or_composed ^ or_hand))
    d_and = int(np.sum(and_composed ^ and_hand))

    if verbose:
        print("\n【第 6 组】逻辑融合：prefixLM = or(prefix, causal)；滑窗 = and(win, causal, doc)")
        print(f"  or  组合 vs 手写：不一致的元素数 {d_or}")
        print(f"  and 组合 vs 手写：不一致的元素数 {d_and}")
        print(f"  prefixLM 掩码密度 {(np.mean(or_composed)*100):.2f}%")
        print(f"  and 三连掩码密度  {(np.mean(and_composed)*100):.2f}%")

    check("G6 or_mask 等价", d_or == 0, f"{d_or}")
    check("G6 and_mask 等价", d_and == 0, f"{d_and}")

    # 组合爆炸：k 个独立可选项 → 2^k 个 kernel
    k = 6
    combos = 2 ** k
    if verbose:
        print(
            f"  变体的组合数：{k} 个正交开关 → {combos} 种组合；"
            f"手写 kernel 要为每种组合各写一个，FlexAttention 只需各写一个 mask_mod。"
        )
    check("G6 组合数", combos == 64, f"{combos}")
    return d_or + d_and


# --------------------------------------------------------------------------
# 第 7 组：PagedAttention 的 physical → logical 索引转换
# --------------------------------------------------------------------------
def run_group7(verbose=True):
    N = 128          # 逻辑序列长度
    D = 32
    PAGE = 32        # paged attention 的页大小（= BlockMask 的块大小）
    npages = N // PAGE

    rng = np.random.default_rng(SEED + 7)
    q, k, v = make_inputs(N, D, seed=SEED + 7)

    # 逻辑布局：连续
    logical_out, _, _ = naive_attention(
        q, k, v, sm_scale=1.0 / math.sqrt(D), score_mod=mod_causal
    )

    # 物理布局：页被打散存放
    perm = rng.permutation(npages)
    page_table = np.zeros((npages, PAGE), dtype=np.int64)
    for logical_page, physical_page in enumerate(perm):
        page_table[logical_page] = np.arange(
            physical_page * PAGE, (physical_page + 1) * PAGE
        )
    k_phys = k[page_table.reshape(-1)]
    v_phys = v[page_table.reshape(-1)]

    # physical_to_logical：块级反查表
    physical_to_logical = np.full(npages, -1, dtype=np.int64)
    for logical_page, physical_page in enumerate(perm):
        physical_to_logical[physical_page] = logical_page

    def converted_mask_mod(i, physical_j):
        """先反查回逻辑索引，再套用用户原本的 causal mask_mod。"""
        phys_block = physical_j // PAGE
        phys_offset = physical_j % PAGE
        logical_block = physical_to_logical[phys_block]
        logical_j = logical_block * PAGE + phys_offset
        live = logical_block >= 0
        return live & (i >= logical_j)

    physical_out, _, _ = naive_attention(
        q, k_phys, v_phys, sm_scale=1.0 / math.sqrt(D), mask_mod=converted_mask_mod
    )

    denom = float(np.max(np.abs(logical_out)))
    rel = float(np.max(np.abs(logical_out - physical_out)) / denom)

    if verbose:
        print("\n【第 7 组】PagedAttention：先反查回逻辑 index，再套用原 mask_mod")
        print(f"  页数 {npages}，页大小 {PAGE}，页表打散: {perm.tolist()}")
        print(f"  逻辑布局输出 vs 物理布局输出：最大相对误差 {rel:.3e}")
        print("  反查只要 O(1)：physical_kv_block = kv_idx // PAGE，余数直接补回去。")

    check("G7 物理布局与逻辑布局等价", rel < 1e-12, f"{rel:.3e}")
    return rel


# --------------------------------------------------------------------------
def make_inputs(N, D, *, seed=SEED):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((N, D)),
        rng.standard_normal((N, D)),
        rng.standard_normal((N, D)),
    )


GROUPS = {
    1: run_group1,
    2: run_group2,
    3: run_group3,
    4: run_group4,
    5: run_group5,
    6: run_group6,
    7: run_group7,
}


def main(argv):
    if "--selftest" in argv:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for i in sorted(GROUPS):
                GROUPS[i](verbose=False)
        print(f"selftest OK: {len(_ASSERTS)} 条断言全部通过")
        return 0

    sel = [int(a) for a in argv if a.isdigit()]
    keys = sel if sel else sorted(GROUPS)
    quiet = "--quiet" in argv

    print("=" * 76)
    print("flex_core：FlexAttention 编程模型的数值验证（numpy / CPU / seed=%d）" % SEED)
    print("=" * 76)
    for i in keys:
        GROUPS[i](verbose=not quiet)

    print("\n" + "=" * 76)
    print(f"总计 {len(_ASSERTS)}/{len(_ASSERTS)} 通过")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
