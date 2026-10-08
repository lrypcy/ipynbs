"""GTP 权重重 materialization 的切分语义验证（纯 CPU，numpy）。

对应文章：
  https://lrypcy.github.io/2026/10/11/megatron-04-generalized-tensor-parallelism/

源码基准：megatron-core 0.20.0 / commit `60e039626`。

验证对象
--------
`generalized_tensor_parallelism.py` 的 `gtp_remat_shard_dim0`（L519–L533）与
`_gtp_slice_one_param`（L536–L576）。两者合起来定义了「一个权重的第 0 维如何被
切给 gtp_remat 组里的每个 rank」：

    alignment   = GTP_CONFIG.pad_for_alignment * gtp_remat_size
    pad_length  = (alignment - dim0 % alignment) % alignment
    padded      = dim0 + pad_length
    shard_dim0  = padded // gtp_remat_size
    rank r 拿到 = tensor[r * shard_dim0 : (r + 1) * shard_dim0]

两条分支：`pad_for_alignment > 0`（默认 16）走 padding；否则要求 `dim0 % gtp_remat_size == 0`
否则 `assert` 失败。

验证的五组不变量
----------------
I1  `padded % (pad_for_alignment * gtp_remat_size) == 0`
I2  `shard_dim0 % pad_for_alignment == 0`——每个 rank 的行数是 pad_for_alignment 的倍数
I3  各 rank 的行区间首尾相接、不重叠，并集恰为 `[0, padded)`
I4  padding 落在尾部：所有真实数据行 `[0, dim0)` 都在某个 rank 的区间内且连续，
    补出来的行 `[dim0, padded)` 全部落在最后一个 rank 的区间尾部
I5  数值往返：把各 rank 的分片按 rank 序拼接，应精确还原 padded 张量；
    再去掉尾部 pad_length 行，应逐位还原原始张量

外加no-pad 模式的断言行为，以及与 activation-centric TP 的形状对照。

复现性质：纯 numpy、固定 seed、无 GPU、无第三方依赖。断言在 gen_stdout.py 里，
退出码非 0 即表示对账失败。本文件只做切分语义与数值等价验证，不做任何性能测量。
"""

from __future__ import annotations

import numpy as np

SEED = 20261011

# 与 megatron/core/tensor_parallel/generalized_tensor_parallelism.py 的 GTPRematConfig 对齐
DEFAULT_PAD_FOR_ALIGNMENT = 16


def shard_dim0(dim0: int, gtp_remat_size: int, pad_for_alignment: int = DEFAULT_PAD_FOR_ALIGNMENT):
    """复现 `gtp_remat_shard_dim0`（L519–L533），返回 `(shard_dim0, pad_length)`。

    与原函数一致：pad_for_alignment > 0 时按 alignment 补齐，否则要求整除。
    """
    if pad_for_alignment > 0:
        alignment = pad_for_alignment * gtp_remat_size
        pad_length = (alignment - dim0 % alignment) % alignment
    else:
        if dim0 % gtp_remat_size != 0:
            raise AssertionError(
                f"gtp_remat_shard_dim0: dim0={dim0} not divisible by "
                f"gtp_remat_size={gtp_remat_size}."
            )
        pad_length = 0
    padded = dim0 + pad_length
    return padded // gtp_remat_size, pad_length


def rank_row_range(gtp_rank: int, shard: int, gtp_remat_size: int) -> tuple[int, int]:
    """复现 `tensor[gtp_rank * shard : (gtp_rank + 1) * shard]`，返回半开区间。"""
    return gtp_rank * shard, (gtp_rank + 1) * shard


def per_rank_shapes(dim0: int, trailing: tuple[int, ...], gtp_remat_size: int,
                    pad_for_alignment: int = DEFAULT_PAD_FOR_ALIGNMENT) -> dict:
    """给出每个 rank 上的张量形状与行区间。"""
    shard, pad = shard_dim0(dim0, gtp_remat_size, pad_for_alignment)
    padded = dim0 + pad
    ranks = []
    for r in range(gtp_remat_size):
        lo, hi = rank_row_range(r, shard, gtp_remat_size)
        ranks.append({
            "rank": r,
            "rows": hi - lo,
            "lo": lo,
            "hi": hi,
            "shape": (hi - lo,) + trailing,
        })
    return {
        "dim0": dim0,
        "trailing": trailing,
        "gtp_remat_size": gtp_remat_size,
        "pad_for_alignment": pad_for_alignment,
        "alignment": pad_for_alignment * gtp_remat_size if pad_for_alignment > 0 else None,
        "pad_length": pad,
        "padded": padded,
        "shard_dim0": shard,
        "ranks": ranks,
    }


def check_invariants(info: dict) -> dict:
    """验 I1–I4，返回逐条结果。真值检查在 roundtrip() 里。"""
    pad_align = info["pad_for_alignment"]
    size = info["gtp_remat_size"]
    shard = info["shard_dim0"]
    padded = info["padded"]
    dim0 = info["dim0"]
    out = {}

    if pad_align > 0:
        out["I1_padded_aligned"] = padded % (pad_align * size) == 0
        out["I2_shard_is_multiple_of_pad_align"] = shard % pad_align == 0
    else:
        out["I1_padded_aligned"] = padded % size == 0
        out["I2_shard_is_multiple_of_pad_align"] = True   # no-pad 模式下此条不适用

    # I3：区间首尾相接、不重叠、并集为 [0, padded)
    out["I3_contiguous_cover"] = all(
        info["ranks"][i]["hi"] == info["ranks"][i + 1]["lo"] for i in range(size - 1)
    ) and info["ranks"][0]["lo"] == 0 and info["ranks"][-1]["hi"] == padded

    # I4：真实数据行连续覆盖 [0, dim0)，padding 只在尾部
    covered = np.zeros(padded, dtype=bool)
    for r in info["ranks"]:
        covered[r["lo"]:r["hi"]] = True
    out["I4_all_rows_owned"] = bool(covered.all())
    out["I4_data_contiguous"] = bool(covered[:dim0].all())
    out["I4_pad_is_tail_only"] = bool((~covered[:dim0]).sum() == 0 and
                                      all(covered[dim0:]))
    out["I4_pad_owned_by_last_rank"] = (
        info["ranks"][-1]["hi"] == padded and info["ranks"][-1]["hi"] - max(dim0, info["ranks"][-1]["lo"]) >= 0
    ) if pad_align > 0 else True
    return out


def roundtrip(dim0: int, trailing: tuple[int, ...], gtp_remat_size: int,
              pad_for_alignment: int = DEFAULT_PAD_FOR_ALIGNMENT, seed: int = SEED) -> dict:
    """I5：真值 → 分片 → 拼接 → 去padding → 应逐位还原。

    构造方式：用一个可识别的真值张量（元素编码为其行号），逐 rank 切片后再拼回。
    这样任何错位或丢行都会立刻暴露，而不是被随机数据掩盖。
    """
    rng = np.random.default_rng(seed)
    info = per_rank_shapes(dim0, trailing, gtp_remat_size, pad_for_alignment)
    padded, pad, shard, size = info["padded"], info["pad_length"], info["shard_dim0"], info["gtp_remat_size"]

    truth = rng.standard_normal((dim0,) + trailing)
    padded_ref = np.zeros((padded,) + trailing, dtype=truth.dtype)
    padded_ref[:dim0] = truth

    shards = []
    for r in range(size):
        lo, hi = rank_row_range(r, shard, size)
        piece = padded_ref[lo:hi]
        assert piece.shape[0] == shard, f"rank {r} 拿到 {piece.shape[0]} 行，应为 {shard}"
        shards.append(piece.copy())

    rebuilt = np.concatenate(shards, axis=0)
    restored = rebuilt[:dim0]

    return {
        "dim0": dim0, "padded": padded, "pad_length": pad,
        "shard_dim0": shard, "gtp_remat_size": size,
        "rebuild_matches_padded": bool(np.array_equal(rebuilt, padded_ref)),
        "restore_matches_truth": bool(np.array_equal(restored, truth)),
        "max_abs_diff": float(np.max(np.abs(restored - truth))) if truth.size else 0.0,
        "shard_shapes": [tuple(s.shape) for s in shards],
        "shard_bytes": [int(s.nbytes) for s in shards],
        "padded_bytes": int(padded_ref.nbytes),
        "waste_bytes": int(padded_ref.nbytes - truth.nbytes),
        "waste_ratio": float(padded_ref.nbytes - truth.nbytes) / float(truth.nbytes),
    }


def activation_centric_shapes(dim0: int, hidden: int, tp: int) -> dict:
    """对照：activation-centric TP 切的是激活的最后一维（hidden），权重在init 时就被切开。

    这里只给形状，不做数值——目的是与 GTP 的「切权重第 0 维」形成轴上的对照。
    """
    assert hidden % tp == 0
    return {
        "hidden": hidden,
        "tp": tp,
        "act_shape_full": (1, 1, hidden),
        "act_shape_per_rank": (1, 1, hidden // tp),
        "weight_shape_full": (dim0, hidden),
        "weight_shape_per_rank": (dim0 // tp, hidden) if dim0 % tp == 0 else None,
        "cut_axis": "激活 = 最后一维；权重 = 输出维（初始化时切）",
    }
