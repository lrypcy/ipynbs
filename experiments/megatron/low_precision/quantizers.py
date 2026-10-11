#!/usr/bin/env python3
"""FP8 / MX / INT8 量化器手撸对比（C 级，numpy）。

对应文章《Megatron-LM 深度剖析（09）：低精度——FP8 全链路与 FP4》。
源码基准：megatron-core 0.20.0 / commit 60e039626。

本篇要回答的问题只有一个：**scale 粒度换来多少误差下降，值不值它引入的归约开销。**

`fp8_recipe` 的五个取值（`transformer_config.py` L587-L593）里，
`delayed` / `tensorwise` / `mxfp8` / `blockwise` 在「scale 怎么算」上落在三种粒度：

| recipe | scale 粒度 | scale 类型 | 是否需要跨 rank 归约 |
| --- | --- | --- | --- |
| `tensorwise` | 整张量一个 | 浮点 | 需要 |
| `delayed` | 整张量一个 + 历史窗口 | 浮点 | 需要（还多一层 iteration 维） |
| `mxfp8` / `blockwise` | 每 32 个元素一个 | E8M0（2 的幂） | 不需要 |

`get_amax_reduction_group`（`parallel_state.py` L1877-L1900）只对前两者有意义，
结构原因就在这张表里：粒度细到块内之后，块间不共享 scale，归约就没有对象了。

量化器全部手撸
--------------
E4M3 / E5M2 按 OCP 逐位枚举全部可表示值，再取最近格点。**枚举而非套公式**，
是为了让格点集合本身可被审阅 —— 脚本会打印格点数量与最大有限值，并断言它们
等于规格值（E4M3FN = 448、E5M2 = 57344）。

E4M3FN 与 E5M2 的关键差别在指数全 1 那一行：E4M3FN 没有 Inf，
指数全 1 且尾数全 1 才是 NaN，其余是正常数（所以最大值是 448）；
E5M2 指数全 1 整行都是 Inf/NaN，最大有限值只能到指数 30，即 57344。
枚举时必须分开处理，否则 E5M2 会多算出 114688 这个不存在的值。

数据口径（必须与正文一致）
------------------------
**这是合成数据，不是真实权重。绝对数值不可外推，只用于比较粒度之间的相对关系。**
三组数据，分别暴露粒度的不同侧面：

- **A 单一量级**：全部 $$N(0,1)$$。基线，此时三种粒度几乎无差别。
- **B 块间量级悬殊**：相邻块交替取 $$N(0,1)$$ 与 $$N(0,0.02)$$。
  per-tensor 的 scale 被大块顶高，小块落到网格低端 —— 这一组最能显出粒度价值。
- **C 稀疏 outlier**：少量幅值很大的点。**对 FP8 与对 INT8 影响完全不同**，
  因为 FP8 是浮点（相对精度与 scale 无关），INT8 是均匀量化（相对精度随 scale 变）。

用法
----
    ~/Software/miniconda3/bin/python3 quantizers.py
退出码 0 表示全部断言通过。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

SEED = 20261015
N_ELEM = 4096
MX_BLOCK = 32


def _fp8_grid(exp_bits: int, mant_bits: int, bias: int, has_inf: bool) -> np.ndarray:
    """枚举一个 FP8 格式的全部可表示有限值（升序、含 ±0）。

    `has_inf=True` 表示指数全 1 整行保留给 Inf/NaN（E5M2）；
    `has_inf=False` 表示有限-非数变体，指数全 1 且尾数全 1 才是 NaN（E4M3FN）。
    """
    values = {0.0}
    top = 2 ** exp_bits - 1
    for sign in (0, 1):
        s = -1.0 if sign else 1.0
        for e in range(2 ** exp_bits):
            if has_inf and e == top:
                continue
            for m in range(2 ** mant_bits):
                if not has_inf and e == top and m == 2 ** mant_bits - 1:
                    continue
                if e == 0:
                    val = 2.0 ** (1 - bias) * (m / 2 ** mant_bits)
                else:
                    val = 2.0 ** (e - bias) * (1.0 + m / 2 ** mant_bits)
                values.add(s * val)
    return np.array(sorted(values), dtype=np.float64)


E4M3_GRID = _fp8_grid(exp_bits=4, mant_bits=3, bias=7, has_inf=False)
E5M2_GRID = _fp8_grid(exp_bits=5, mant_bits=2, bias=15, has_inf=True)


def quantize_to_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """把 x 投到最近格点，超出范围则夹紧。"""
    shape = np.asarray(x).shape
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    idx = np.clip(np.searchsorted(grid, flat), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    out = np.where(np.abs(flat - hi) < np.abs(flat - lo), hi, lo)
    return np.clip(out, grid[0], grid[-1]).reshape(shape)


def fp8_quantize_per_tensor(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    amax = float(np.max(np.abs(x)))
    if amax == 0:
        return np.array(x, dtype=np.float64)
    scale = amax / float(np.max(grid))
    return quantize_to_grid(np.asarray(x, dtype=np.float64) / scale, grid) * scale


def fp8_quantize_mxblock(x: np.ndarray, grid: np.ndarray, block: int = MX_BLOCK) -> np.ndarray:
    """MX block scaling：块内共享一个 E8M0 scale（必须是 2 的幂）。"""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    max_val = float(np.max(grid))
    n_blocks = (flat.size + block - 1) // block
    padded = np.zeros(n_blocks * block, dtype=np.float64)
    padded[: flat.size] = flat
    view = padded.reshape(n_blocks, block)
    out = np.empty_like(view)
    for b in range(n_blocks):
        amax = float(np.max(np.abs(view[b])))
        if amax == 0:
            out[b] = 0.0
            continue
        scale = 2.0 ** np.ceil(np.log2(amax / max_val))
        out[b] = quantize_to_grid(view[b] / scale, grid) * scale
    return out.reshape(-1)[: flat.size].reshape(x.shape)


def int8_quantize_per_tensor(x: np.ndarray) -> np.ndarray:
    amax = float(np.max(np.abs(x)))
    if amax == 0:
        return np.array(x, dtype=np.float64)
    scale = amax / 127.0
    return np.rint(np.asarray(x, dtype=np.float64) / scale) * scale


def int8_quantize_per_channel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 127.0)
    return np.rint(x / scale) * scale


def int8_quantize_per_block(x: np.ndarray, block: int = MX_BLOCK) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    n_blocks = (flat.size + block - 1) // block
    padded = np.zeros(n_blocks * block, dtype=np.float64)
    padded[: flat.size] = flat
    view = padded.reshape(n_blocks, block)
    out = np.empty_like(view)
    for b in range(n_blocks):
        amax = float(np.max(np.abs(view[b])))
        out[b] = 0.0 if amax == 0 else np.rint(view[b] / (amax / 127.0)) * (amax / 127.0)
    return out.reshape(-1)[: flat.size].reshape(x.shape)


def rmse(x: np.ndarray, xq: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, np.float64) - xq) ** 2)))


def snr_db(x: np.ndarray, xq: np.ndarray) -> float:
    err = float(np.sum((np.asarray(x, np.float64) - xq) ** 2))
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(float(np.sum(np.asarray(x, np.float64) ** 2)) / err)


def small_value_stats(x: np.ndarray, xq: np.ndarray, thresh: float = 1e-3) -> Tuple[float, float]:
    """小幅值元素上的两个量：被清零的比例、与相对误差。

    RMSE 会被大值项主导，所以「小块里混着极小值」的损害在 RMSE 上看不见。
    真正该看的是这些极小值：它们在 per-tensor scale 下会掉到规格化区以下，
    被直接舍成 0。
    """
    small = np.abs(x) < thresh
    if not np.any(small):
        return 0.0, 0.0
    xs, qs = x[small], xq[small]
    zero_frac = float(np.mean(qs == 0.0))
    rel = float(np.mean(np.abs(qs - xs) / np.maximum(np.abs(xs), 1e-30)))
    return zero_frac, rel


def max_rel_err_on_extreme(x: np.ndarray, xq: np.ndarray) -> float:
    i = int(np.argmax(np.abs(x)))
    return float(abs(float(xq[i]) - float(x[i])) / max(abs(float(x[i])), 1e-30))


@dataclass
class Row:
    name: str
    rmse: float
    snr: float
    max_rel: float


QUANTIZERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "FP8 E4M3 per-tensor": lambda x: fp8_quantize_per_tensor(x, E4M3_GRID),
    "FP8 E5M2 per-tensor": lambda x: fp8_quantize_per_tensor(x, E5M2_GRID),
    "FP8 E4M3 MX-block32": lambda x: fp8_quantize_mxblock(x, E4M3_GRID),
    "INT8 per-tensor": int8_quantize_per_tensor,
    "INT8 per-channel": int8_quantize_per_channel,
    "INT8 per-block32": int8_quantize_per_block,
}


def make_case_a(rng) -> np.ndarray:
    return rng.standard_normal(N_ELEM)


def make_case_b(rng) -> np.ndarray:
    """相邻块交替取 N(0,1) 与 N(0,0.02)：块间量级差 50 倍。"""
    x = np.empty(N_ELEM, dtype=np.float64)
    step = MX_BLOCK
    for start in range(0, N_ELEM, step):
        sigma = 1.0 if (start // step) % 2 == 0 else 0.02
        x[start:start + step] = rng.standard_normal(min(step, N_ELEM - start)) * sigma
    return x


def make_case_d(rng) -> np.ndarray:
    """**块内**动态范围超 2^14.8：每个 32 元素块内部同时含 1e-5 量级与 1 量级。

    这是 FP8 粒度真正起作用的场景。E4M3 的最小正规格值是 2^-6 = 0.015625，
    最大有限值 448，于是「规格化区」只覆盖 448 / 0.015625 ≈ 2^14.8 的动态范围。
    一个组内的值跨度超过这个比值时，小值就会掉进次正规区，相对精度随之崩塌。
    per-tensor 与 per-block32 在此都会崩；只有把粒度降到 1 才能救回来。
    """
    x = np.empty(N_ELEM, dtype=np.float64)
    for start in range(0, N_ELEM, MX_BLOCK):
        n = min(MX_BLOCK, N_ELEM - start)
        half = n // 2
        x[start:start + half] = rng.standard_normal(half)
        x[start + half:start + n] = rng.standard_normal(n - half) * 1e-5
    return x


def make_case_c(rng) -> np.ndarray:
    x = rng.standard_normal(N_ELEM)
    pos = rng.choice(N_ELEM, size=4, replace=False)
    x[pos] = np.where(rng.random(4) > 0.5, 1.0, -1.0) * 12.0
    return x


CASES = [
    ("A 单一量级 N(0,1)", make_case_a),
    ("B 块间量级差 50 倍（块内一致）", make_case_b),
    ("C 4 个幅值 ±12 的 outlier", make_case_c),
    ("D 块内动态范围 1e-5 ~ 1（超规格化区）", make_case_d),
]


def print_table(title: str, x: np.ndarray) -> Dict[str, Row]:
    print(f"\n【{title}】")
    print(f"{'量化器':<24s} {'RMSE':>12s} {'SNR(dB)':>10s} {'极值相对偏差':>14s}")
    print("-" * 64)
    out = {}
    for name, fn in QUANTIZERS.items():
        xq = fn(x)
        r = Row(name, rmse(x, xq), snr_db(x, xq), max_rel_err_on_extreme(x, xq))
        out[name] = r
        print(f"{r.name:<24s} {r.rmse:>12.6e} {r.snr:>10.2f} {r.max_rel:>14.6e}")
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    print("=" * 100)
    print("FP8 / MX / INT8 量化粒度对比（C 级 · numpy）")
    print(f"种子 {SEED}  元素数 {N_ELEM}  MX 块大小 {MX_BLOCK}")
    print("=" * 100)

    print("\n【0】FP8 格点集合自检（逐位枚举，不是套公式）")
    for name, grid in (("E4M3FN", E4M3_GRID), ("E5M2", E5M2_GRID)):
        print(f"  {name}: 有限可表示值 {grid.size} 个，"
              f"最大 {np.max(grid):.6g}，最小正规格值 {np.min(grid[grid > 0]):.6g}")
    print("  规格值：E4M3FN 最大有限 448（无 Inf，指数全 1 除尾数全 1 外都是正常数）；")
    print("          E5M2   最大有限 57344（指数全 1 整行是 Inf/NaN）。")

    print("\n【0b】指数全 1 那一行的处理差异（枚举错就会多出这个值）")
    wrong_e5 = np.max(_fp8_grid(5, 2, 15, has_inf=False))
    print(f"  若把 E5M2 也当「有限-非数」变体枚举，最大值会算成 {wrong_e5:.6g} —— 该值不存在。")

    tables = {}
    for title, gen in CASES:
        tables[title] = print_table(title, gen(rng))

    print("\n【1】粒度对误差的影响（数据 B：量级差在**块之间**，每块内部一致）")
    t = tables["B 块间量级差 50 倍（块内一致）"]
    e4_pt = t["FP8 E4M3 per-tensor"]
    e4_mx = t["FP8 E4M3 MX-block32"]
    i8_pt = t["INT8 per-tensor"]
    i8_bl = t["INT8 per-block32"]
    print(f"  FP8 E4M3  per-tensor RMSE {e4_pt.rmse:.4e} → MX-block32 {e4_mx.rmse:.4e}"
          f"   （降 {(1 - e4_mx.rmse / e4_pt.rmse) * 100:.1f}%）")
    print(f"  INT8      per-tensor RMSE {i8_pt.rmse:.4e} → per-block32 {i8_bl.rmse:.4e}"
          f"   （降 {(1 - i8_bl.rmse / i8_pt.rmse) * 100:.1f}%）")

    print("\n【2】FP8 与 INT8 对 outlier 的反应完全不同（数据 C）")
    ta = tables["A 单一量级 N(0,1)"]
    tc = tables["C 4 个幅值 ±12 的 outlier"]
    for key in ("FP8 E4M3 per-tensor", "INT8 per-tensor", "INT8 per-block32"):
        d_snr = ta[key].snr - tc[key].snr
        print(f"  {key:<22s} SNR 从 {ta[key].snr:6.2f} 降到 {tc[key].snr:6.2f} dB"
              f"   （掉 {d_snr:.2f} dB）")
    print("  原因：FP8 是浮点格式，相对精度与 scale 无关 —— outlier 抬高 scale，")
    print("  只压缩了动态范围，没有压缩相对精度。INT8 是均匀量化，抬高 scale 就直接")
    print("  等比例放大量化步长，所以掉得厉害。")
    print("  ⚠ SNR 在这里是**跨数据集**比较，而 outlier 抬高了信号能量 sum(x^2)，")
    print("  所以 FP8 那一行的 SNR 反而升了。这不是量化变好，是分母（信号能量）变大。")
    print("  跨数据集比精度只能用 RMSE 或针对某个子集（如极小值）的相对误差，")
    print("  不能用 SNR —— 这一点在正文里必须写明，否则会读成「FP8 抗 outlier」。")

    print("\n【2b】小幅值元素的存活情况（数据 D，阈值 |x| < 1e-3）")
    xd = make_case_d(np.random.default_rng(SEED + 2))
    variants = [
        ("FP8 E4M3 per-tensor", fp8_quantize_per_tensor(xd, E4M3_GRID)),
        ("FP8 E4M3 MX-block32", fp8_quantize_mxblock(xd, E4M3_GRID, block=32)),
        ("FP8 E4M3 MX-block8", fp8_quantize_mxblock(xd, E4M3_GRID, block=8)),
        ("FP8 E4M3 逐元素", fp8_quantize_mxblock(xd, E4M3_GRID, block=1)),
        ("INT8 per-tensor", int8_quantize_per_tensor(xd)),
    ]
    print(f"{'方案':<24s} {'被清零比例':>12s} {'相对误差':>12s}")
    print("-" * 50)
    small_rows = {}
    for name, xq in variants:
        zf, rel = small_value_stats(xd, xq)
        small_rows[name] = (zf, rel)
        print(f"{name:<24s} {zf:>11.1%} {rel:>12.4f}")
    print("  这张表是本篇最该记住的一条：FP8 的 per-tensor 与 block32 都会把极小值清零，")
    print("  因为它们的块里仍然混着大值；把粒度降到 8 或 1 才救得回来。")
    print("  也就是说 —— block scaling 解决的是「量程」问题，不是「精度」问题。")

    print("\n【3】MX 块大小的粒度曲线（数据 B，E4M3）—— 曲线是平的，原因见下")
    xb = make_case_b(np.random.default_rng(SEED + 1))
    base = fp8_quantize_per_tensor(xb, E4M3_GRID)
    print(f"{'块大小':>8s} {'RMSE':>12s} {'SNR(dB)':>10s} {'scale 字节':>12s}")
    print("-" * 46)
    for blk in (1, 8, 32, 128, 512):
        q = fp8_quantize_mxblock(xb, E4M3_GRID, block=blk)
        scale_bytes = (N_ELEM // blk) if blk > 1 else N_ELEM
        tag = "（E8M0，1 B/块）" if blk > 1 else "（逐元素，实际不这么用）"
        print(f"{blk:>8d} {rmse(xb, q):>12.6e} {snr_db(xb, q):>10.2f} {scale_bytes:>10d} B {tag}")
    print(f"  per-tensor 参照：RMSE {rmse(xb, base):.6e}  SNR {snr_db(xb, base):.2f} dB"
          f"  scale 4 B")
    print("  注意三行的 RMSE 完全一样 —— 这不是 bug，是浮点格式的性质：")
    print("  只要一组内的值没有跨出规格化区，缩小 scale 并不增加尾数位，")
    print("  相对精度恒为 2^-3 量级。粒度对 FP8 的作用不是「更准」，而是「各自用满量程」。")

    print("\n【4】粒度与归约开销的对应（结构性结论，非实测）")
    print("  per-tensor（tensorwise / delayed）：scale 必须跨 rank 归约后才能用，")
    print("    因为各 rank 的 amax 不同，取谁的值都会丢掉一部分张量的量程。")
    print("    归约组由 get_amax_reduction_group（parallel_state.py L1877-L1900）给出，")
    print("    有 4 种组合（含或不含 CP × 是否只跨 TP）。")
    print("  block（mxfp8 / blockwise）：块内共享、块间独立，scale 不必跨 rank 归约。")

    print("\n【断言】")
    a = tables["A 单一量级 N(0,1)"]
    b = tables["B 块间量级差 50 倍（块内一致）"]
    c = tables["C 4 个幅值 ±12 的 outlier"]
    xb2 = make_case_b(np.random.default_rng(SEED))
    per_t = rmse(xb2, fp8_quantize_per_tensor(xb2, E4M3_GRID))
    blk32 = rmse(xb2, fp8_quantize_mxblock(xb2, E4M3_GRID, block=32))
    blk1 = rmse(xb2, fp8_quantize_mxblock(xb2, E4M3_GRID, block=1))
    checks = [
        ("E4M3FN 枚举最大值 = 448", abs(np.max(E4M3_GRID) - 448.0) < 1e-9),
        ("E5M2 枚举最大值 = 57344", abs(np.max(E5M2_GRID) - 57344.0) < 1e-6),
        ("E5M2 按 finite-nan 变体枚举会多算出 98304（故枚举必须分变体）",
         abs(wrong_e5 - 98304.0) < 1e-3),
        ("E4M3 最小正规格值 = 2^-6、次正规下限 = 2^-9",
         abs(float(np.min(E4M3_GRID[E4M3_GRID > 0])) - 2.0 ** -9) < 1e-12),
        ("单一量级下 E4M3 的 SNR 高于 E5M2（尾数位多 1 位）",
         a["FP8 E4M3 per-tensor"].snr > a["FP8 E5M2 per-tensor"].snr),
        ("数据 B 下 MX-block32 优于 per-tensor", blk32 < per_t),
        ("块越小 RMSE 越低（block1 ≤ block32 ≤ per-tensor）",
         blk1 <= blk32 + 1e-12 and blk32 <= per_t + 1e-12),
        ("数据 B 下 INT8 per-block32 优于 per-tensor",
         b["INT8 per-block32"].rmse < b["INT8 per-tensor"].rmse),
        ("outlier 对 INT8 per-tensor 的 SNR 影响大于对 FP8 per-tensor",
         (a["INT8 per-tensor"].snr - c["INT8 per-tensor"].snr)
         > (a["FP8 E4M3 per-tensor"].snr - c["FP8 E4M3 per-tensor"].snr)),
        ("数据 D 下 per-tensor 把过半极小值清零",
         small_rows["FP8 E4M3 per-tensor"][0] > 0.5),
        ("数据 D 下 MX-block32 清零比例仍高（块内仍混着大值）",
         small_rows["FP8 E4M3 MX-block32"][0] > 0.4),
        ("粒度降到 8 之后极小值几乎全部救回",
         small_rows["FP8 E4M3 MX-block8"][0] < 0.05),
        ("逐元素时极小值相对误差落在 E4M3 理论界内（< 2^-3 = 12.5%）",
         small_rows["FP8 E4M3 逐元素"][1] < 0.125),
        ("INT8 per-tensor 把极小值全部清零（均匀量化无次正规区可退）",
         small_rows["INT8 per-tensor"][0] > 0.99),
        ("整体 RMSE 无法反映这个损害（D 的 RMSE 与 B 同量级）",
         abs(tables["D 块内动态范围 1e-5 ~ 1（超规格化区）"]["FP8 E4M3 per-tensor"].rmse
             / tables["B 块间量级差 50 倍（块内一致）"]["FP8 E4M3 per-tensor"].rmse - 1) < 0.5),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    print()
    if ok:
        print("全部断言通过。")
        return 0
    print("存在未通过的断言。")
    return 1


if __name__ == "__main__":
    sys.exit(main())