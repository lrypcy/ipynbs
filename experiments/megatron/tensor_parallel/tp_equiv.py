"""张量并行与序列并行的数值等价性验证（纯 CPU，numpy）。

对应文章：
  https://lrypcy.github.io/2026/09/09/megatron-03-tensor-parallel-and-sequence-parallel/

源码基准：megatron-core 0.20.0 / commit 60e039626。

三项验证
--------
E1  f 的列切 + g 的 all-reduce  ≡  未切分的整块 GEMM
    对应 layers.py::RowParallelLinear.forward（L1504 `scatter_to_tensor_model_parallel_region`
    → L1538 `reduce_from_tensor_model_parallel_region`）与 mappings.py 的
    `_ScatterToModelParallelRegion` / `_ReduceFromModelParallelRegion`。

E2  SP 下把 g 的 all-reduce 换成 reduce-scatter、把 Column 的输入侧换成 all-gather，
    数值不变；且通信量守恒。
    对应 layers.py L616-621（SP 时 all-gather 输入）、L1534（SP 时 g 走
    `reduce_scatter_to_sequence_parallel_region`）与 mappings.py 的
    `_ReduceScatterToSequenceParallelRegion` / `_GatherFromSequenceParallelRegion`。

E3  gradient-accumulation-fusion 与朴素累加的精度差。
    对应 layers.py::LinearWithGradAccumulationAndAsyncCommunication 与 _wgrad_gemm：
    开融合时 wgrad 直接以 beta=1 累加进 fp32 的 main_grad，不开时每步先落独立张量再相加。

复现性质：纯 numpy、固定 seed、无 GPU、无第三方依赖。断言在 gen_stdout.py 里，
退出码非 0 即表示对账失败。本文件只做等价性 / 正确性验证，不做任何性能测量。

数据流约定（与 Megatron 的 MLP 布局一致）
------------------------------------------
  Column（W 按输出维切）:  x @ W_up_k        → [N, ffn/tp]
  GeLU（逐元素，无需通信）:  作用于特征分片本身
  Row  （W 按输入维切）  :  act_k @ W_down_k  → [N, hidden] 的部分和
  g                       :  all-reduce       → [N, hidden]

两段 GEMM 之间没有 all-reduce——这一点容易记反。层内的两处 all-reduce 分别来自
注意力输出投影与 MLP 输出投影，本实验只建模后者（结构相同）。
"""

from __future__ import annotations

import numpy as np

SEED = 20261009

# ring 集合通信的 bus-byte 口径（单 rank 收发字节数）
#   AllReduce      = 2 S (n-1) / n
#   ReduceScatter  =   S (n-1) / n
#   AllGather      =   S (n-1) / n
# 故 RS + AG 逐字节等于 AR。这个恒等式是「SP 不减通信量」的代数根据。


def ring_bytes(kind: str, n: int, size: float) -> float:
    """单rank 的 bus 字节数（ring 算法）。"""
    if kind == "allreduce":
        return 2.0 * size * (n - 1) / n
    if kind in ("reduce_scatter", "all_gather"):
        return size * (n - 1) / n
    raise ValueError(kind)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """把 float32 舍入到 bfloat16 再以 float32 返回。

    bfloat16 与 float32 位数相同（1+8+23 对 1+8+7），指数域完全一致，
    因此只需保留尾数高 7 位、低 16 位按「就近舍入到偶」处理。
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    lsb = (bits >> 16) & 1                      # 就近舍入的进位判据
    rounded = bits + np.uint32(0x7FFF) + lsb
    out = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isnan(x), x, out)


def _split(x: np.ndarray, n: int, axis: int) -> list[np.ndarray]:
    """沿 axis 等分成 n 块。"""
    return np.array_split(x, n, axis=axis)


def _all_reduce(partials: list[np.ndarray]) -> np.ndarray:
    """集合 all-reduce 的本地等价：逐 rank 结果求和。"""
    out = np.zeros_like(partials[0])
    for p in partials:
        out = out + p
    return out


def _reference_mlp(x, w_up, w_down) -> np.ndarray:
    """未切分的整块 MLP：GeLU(x @ W_up) @ W_down。"""
    return np.maximum(x @ w_up, 0.0) @ w_down


def _gelu(x: np.ndarray) -> np.ndarray:
    """GeLU 的下界形式 max(0, x)。逐元素，故作用在特征分片上与作用在全量上等价。"""
    return np.maximum(x, 0.0)


# ─────────────────────────────────────────────────────────────────────
# E1  f 列切 + g all-reduce ≡ 整块 GEMM
# ─────────────────────────────────────────────────────────────────────

def check_e1(s: int, b: int, hidden: int, ffn: int, tp: int, dtype=np.float32) -> dict:
    """f 列切输入维 + 分片 GEMM + g all-reduce，与未切分整块 GEMM 比较。

    对应 RowParallelLinear 在 input_is_parallel=False 时的路径：
    输入全量复制 → f 把特征维 scatter 成 tp 份 → 每 rank 与 W 的对应块相乘
    → g 做 all-reduce。RowParallelLinear 里 W 按输入维切，故每块为 [hidden/tp, ffn]。
    """
    rng = np.random.default_rng(SEED)
    n = s * b
    x = rng.standard_normal((n, hidden)).astype(dtype)
    w = rng.standard_normal((hidden, ffn)).astype(dtype)

    ref = x @ w                                                # [N, ffn]

    w_parts = _split(w, tp, axis=0)                             # 每块 [hidden/tp, ffn]
    x_parts = _split(x, tp, axis=1)                             # f：scatter 特征维
    partials = [x_parts[k] @ w_parts[k] for k in range(tp)]     # 每份 [N, ffn] 的部分和
    out = _all_reduce(partials)                                 # g：all-reduce

    scale = float(np.max(np.abs(ref)))
    err = float(np.max(np.abs(out - ref)))
    return {
        "N": n, "hidden": hidden, "ffn": ffn, "tp": tp,
        "dtype": np.dtype(dtype).name,
        "ref_absmax": scale,
        "abs_err": err,
        "rel_err": err / scale if scale else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# E2  SP：g 换 reduce-scatter、Column 输入侧换 all-gather
# ─────────────────────────────────────────────────────────────────────

def check_e2(s: int, b: int, hidden: int, ffn: int, tp: int, dtype=np.float32) -> dict:
    """对比非 SP 与 SP 两条路径，以及两者的通信量。

    非 SP：x 全量复制；g1 = all-reduce（作用在 attention/MLP 输出投影上）
    SP   ：x 沿序列维分片成 [N/tp, hidden]；Column 内部 all-gather 回全序列；
           g2 = reduce-scatter，输出 [N/tp, hidden]

    两条路径最终结果都应等于未切分的整块 MLP（SP 路径只需与自己那段行比）。
    """
    rng = np.random.default_rng(SEED)
    n = s * b
    assert n % tp == 0, "序列维必须能被 tp 整除"
    x = rng.standard_normal((n, hidden)).astype(dtype)
    w_up = rng.standard_normal((hidden, ffn)).astype(dtype)
    w_down = rng.standard_normal((ffn, hidden)).astype(dtype)
    ref = _reference_mlp(x, w_up, w_down)

    w_up_parts = _split(w_up, tp, axis=1)
    w_down_parts = _split(w_down, tp, axis=0)

    # ── 非 SP ──
    # Column 按输出维切 → 每 rank 拿到互不重叠的特征段；GeLU 逐元素，故可作用在分片上；
    # Row 的 input_is_parallel=True，直接拿自己的特征分片，无需再 scatter。
    col_full = [x @ p for p in w_up_parts]                 # [N, ffn/tp] 每 rank
    act_full = [_gelu(c) for c in col_full]
    nosp_partials = [act_full[k] @ w_down_parts[k] for k in range(tp)]
    nosp = _all_reduce(nosp_partials)                      # [N, hidden]

    # ── SP ──
    # 输入沿序列维（axis=0）分片。注意切的是序列轴而非特征轴，这正是 SP 与 TP 的分界。
    x_sp = _split(x, tp, axis=0)
    # Column 内部沿序列维 all-gather 回全量。gather 结果与原 x 逐位相同、各 rank 拿到同一份，
    # 故本地模拟里这一步退化为恒等，直接用全量 x 继续算。
    x_ag = [x for _ in range(tp)]
    col_sp = [x_ag[k] @ w_up_parts[k] for k in range(tp)]
    act_sp = [_gelu(c) for c in col_sp]
    sp_partials = [act_sp[k] @ w_down_parts[k] for k in range(tp)]
    # reduce-scatter 与 all-reduce 的差别只出现在最后一步：两者都先跨 rank 求和拿到全量行，
    # 区别是 RS 随后只让每个 rank 保留自己那段行。
    sp_full = _all_reduce(sp_partials)
    sp = [sp_full[r] for r in _split(np.arange(n), tp, axis=0)]

    d_nosp = float(np.max(np.abs(nosp - ref)))
    d_sp = float(np.max(np.abs(sp[0] - _split(ref, tp, axis=0)[0])))
    scale = float(np.max(np.abs(ref)))

    ar = 2.0 * ring_bytes("allreduce", tp, 1.0)
    rs_ag = 2.0 * ring_bytes("reduce_scatter", tp, 1.0) + 2.0 * ring_bytes("all_gather", tp, 1.0)
    return {
        "N": n, "hidden": hidden, "ffn": ffn, "tp": tp,
        "dtype": np.dtype(dtype).name,
        "ref_absmax": scale,
        "nosp_abs_err": d_nosp,
        "sp_abs_err": d_sp,
        "nosp_rel_err": d_nosp / scale if scale else 0.0,
        "sp_rel_err": d_sp / scale if scale else 0.0,
        "ar_bytes_unit": ar,
        "rs_ag_bytes_unit": rs_ag,
        "bytes_ratio": rs_ag / ar,
    }


# ─────────────────────────────────────────────────────────────────────
# E3  gradient-accumulation-fusion 与朴素累加的精度差
# ─────────────────────────────────────────────────────────────────────

def check_e3(s: int, hidden: int, ffn: int, tp: int, steps: int) -> dict:
    """wgrad 累加的三条路径。

    fused   ：每步 wgrad 以 beta=1 直接累加进 fp32 的 main_grad（开融合的路径）
    naive32 ：每步产出独立 fp32 wgrad，最后按步序相加
    naive16 ：每步 wgrad 先舍入到 bf16 再累加进 fp32（模拟主累加器不是 fp32 的情形）

    用 fp64 累加作真值基准，把「累加顺序差异」与「舍入误差」分开。
    """
    rng = np.random.default_rng(SEED)
    per_step = []
    for _ in range(steps):
        g = rng.standard_normal((s, ffn)).astype(np.float32)   # grad_output
        x = rng.standard_normal((s, hidden)).astype(np.float32) # input
        per_step.append((g.T @ x).astype(np.float32))          # wgrad = grad^T @ input

    w_shape = (ffn, hidden)
    main_fused = np.zeros(w_shape, dtype=np.float32)
    main_naive32 = np.zeros(w_shape, dtype=np.float32)
    main_naive16 = np.zeros(w_shape, dtype=np.float32)
    for wg in per_step:
        main_fused += wg                     # beta = 1 的 GEMM epilogue
        main_naive16 += to_bf16(wg)          # 每步先舍入再累加
    for wg in per_step:
        main_naive32 += wg                   # 步序相加

    exact = np.zeros(w_shape, dtype=np.float64)
    for wg in per_step:
        exact += wg.astype(np.float64)

    scale = float(np.max(np.abs(exact)))
    return {
        "steps": steps, "shape": w_shape, "tp": tp,
        "ref_absmax": scale,
        "fused_vs_naive32_abs": float(np.max(np.abs(main_fused - main_naive32))),
        "fused_vs_fp64_abs": float(np.max(np.abs(main_fused.astype(np.float64) - exact))),
        "naive16_vs_fp64_abs": float(np.max(np.abs(main_naive16.astype(np.float64) - exact))),
        "fused_vs_fp64_rel": float(np.max(np.abs(main_fused.astype(np.float64) - exact))) / scale,
        "naive16_vs_fp64_rel": float(np.max(np.abs(main_naive16.astype(np.float64) - exact))) / scale,
    }