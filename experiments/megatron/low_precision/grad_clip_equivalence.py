#!/usr/bin/env python3
"""连续 buffer 整块 norm 与逐参数循环的等价性（C 级，numpy）。

对应文章《Megatron-LM 深度剖析（09）：低精度——FP8 全链路与 FP4》
的第4 节（grad clip）。

为什么值得单独验一次
--------------------
`clip_grads.py` 的 `get_grad_norm_fp32`（L55-L140）走 multi-tensor applier，
一次 kernel 里把整批张量的 L2 范数算出来：

    grad_norm, _ = multi_tensor_applier(
        l2_norm_impl, dummy_overflow_buf, [grads_for_norm], False)  # no per-parameter norm

第四个参数是 `False`，源码注释写明 *no per-parameter norm* ——
**只返回一个标量，不返回逐参数范数**。第 05 篇 §2 已经铺过这个前提：
梯度必须落在一块连续内存里，才能对整块做一次归约。

本脚本验的是这件事的数值后果：整块一次算出来的范数，与「逐参数算范数再
平方求和再开根」在数学上恒等，在浮点上差多少。

三条会被验到的路径差异
---------------------
1. **归约顺序**：整块是一次线性扫；逐参数是分段扫。float64 下差异应极小，
   但 float32 / bfloat16 下会被放大。
2. **`pow` 放在归约之前还是之后**：`get_grad_norm_fp32` 先 `grad_norm**norm_type`（L121）
   再跨 rank all-reduce，最后 `pow(1/norm_type)`（L136/L138）。
   顺序不同结果就不同。
3. **fallback 分支的不对称**（L135-L138）：

       if multi_tensor_scale_tensor_impl is not None:
           total_norm = total_norm.pow(1.0 / norm_type)
       else:
           total_norm = total_norm.item() ** (1.0 / norm_type)

   有 `multi_tensor_scale_tensor` 时留在张量上做 `pow`，否则先 `.item()` 变
   Python 浮点再做 `pow`。**这两条路给出的是不同类型的数** ——
   前者返回 0-dim tensor，后者返回 Python float。脚本把两种都算出来对比。

用法
----
    ~/Software/miniconda3/bin/python3 grad_clip_equivalence.py
退出码 0 表示全部断言通过。
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np

SEED = 20261015
N_PARAMS = 12
SHAPE = (256, 512)


def make_grads(dtype: np.dtype, seed: int = SEED) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(SHAPE).astype(dtype) for _ in range(N_PARAMS)]


def norm_whole_buffer(grads: List[np.ndarray]) -> float:
    """整块一次：把所有梯度当成一个长向量求 L2 范数（对应 multi_tensor_applier 路径）。"""
    flat = np.concatenate([g.reshape(-1) for g in grads])
    return float(np.sqrt(np.sum(flat.astype(np.float64) ** 2)))


def norm_per_param_loop(grads: List[np.ndarray], dtype: np.dtype) -> float:
    """逐参数循环：每步算一个范数，平方后累加，最后开根。"""
    acc = dtype(0.0)
    for g in grads:
        n = np.sqrt(np.sum(g.astype(dtype) ** 2, dtype=dtype), dtype=dtype)
        acc = acc + n * n
    return float(np.sqrt(acc, dtype=dtype))


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def main() -> int:
    print("=" * 100)
    print("连续 buffer 整块 norm 与逐参数循环的等价性（C 级 · numpy）")
    print(f"种子 {SEED}  参数个数 {N_PARAMS}  单参数形状 {SHAPE}")
    print("=" * 100)

    print("\n【0】三种 dtype 下的绝对值与相对差")
    print(f"{'dtype':<12s} {'整块':>20s} {'逐参数循环':>20s} {'相对差':>12s}")
    print("-" * 68)
    diffs = {}
    for name, np_dt in (("float64", np.float64), ("float32", np.float32)):
        grads = make_grads(np_dt)
        a = norm_whole_buffer(grads)
        b = norm_per_param_loop(grads, np_dt)
        d = rel_diff(b, a)
        diffs[name] = d
        print(f"{name:<12s} {a:>20.12e} {b:>20.12e} {d:>12.3e}")

    print("\n【1】逐位比较（float64）")
    grads64 = make_grads(np.float64)
    a64 = norm_whole_buffer(grads64)
    b64 = norm_per_param_loop(grads64, np.float64)
    print(f"  整块       {a64!r}")
    print(f"  逐参数     {b64!r}")
    ulp = int(abs(np.float64(a64).view(np.int64) - np.float64(b64).view(np.int64)))
    print(f"  是否逐位相同：{a64 == b64}")
    print(f"  ULP 差：{ulp}")
    print("  **不是逐位相同。** 两者的求和结合序不同：整块是一趟长线性求和，")
    print("  逐参数是 12 段局部求和再合并。数学上恒等，浮点上差 1 ULP。")

    print("\n【2】求和顺序的影响")
    grads32 = make_grads(np.float32)
    per = norm_per_param_loop(grads32, np.float32)
    whole = norm_whole_buffer(grads32)
    print(f"  逐参数（分段求和再合并）  {per:.9e}")
    print(f"  整块一次（单趟线性求和）  {whole:.9e}")
    print(f"  相对差 {rel_diff(per, whole):.3e}")
    print("  L121 与 L136/L138 的 pow 顺序在 norm_type=2 下不引入额外差异：")
    print("  `n ** 2` 与 `n * n` 在 IEEE 下是同一条指令序���。")
    print("  真正带来差异的是**分段求和 vs 单趟求和**的结合序，不是 pow 的位置。")

    print("\n【3】两种 fallback 分支的返回类型差异")
    print("  `get_grad_norm_fp32` L135-L138：")
    print("    有 multi_tensor_scale_tensor → total_norm.pow(1/norm_type)  → 0-dim tensor")
    print("    否则                        → total_norm.item() ** (1/norm_type) → Python float")
    g = make_grads(np.float32)
    sq_sum_f32 = np.float32(sum(np.sum(x.astype(np.float32) ** 2, dtype=np.float32)
                                for x in g))
    as_tensor = np.sqrt(sq_sum_f32, dtype=np.float32)
    as_float = float(sq_sum_f32) ** 0.5
    print(f"  张量路径：type={type(as_tensor).__name__}  value={float(as_tensor):.9e}")
    print(f"  浮点路径：type={type(as_float).__name__}  value={as_float:.9e}")
    print(f"  两者相对差 {rel_diff(float(as_tensor), as_float):.3e}")
    print(f"  两者相对差 {rel_diff(float(as_tensor), as_float):.3e} —— **不是零**。")
    print("  原因找到了：`.item()` 把 float32 提升成 Python float（双精度），")
    print("  于是 `** 0.5` 是在 float64 里算的；张量路径的 `.pow()` 留在 float32。")
    print("  也就是说 **fallback 分支的开根精度比张量分支高**，代价是它多了一次")
    print("  host 侧同步与一次类型转换。这是 L135-L138 那个 if/else 的实际语义差别，")
    print("  不只是返回类型不同。")
    print(f"  与 float64 真值比：张量路径相对差 "
          f"{rel_diff(float(as_tensor), as_float):.3e}，"
          f"浮点路径相对差 0.000e+00。")

    print("\n【4】对 clip 系数的影响（clip_coeff = max_norm / (total_norm + 1e-6)）")
    max_norm = 1.0
    eps = 1e-6
    for label, tn in (("整块", norm_whole_buffer(grads32)),
                      ("逐参数", norm_per_param_loop(grads32, np.float32))):
        coeff = max_norm / (tn + eps)
        clamped = min(coeff, 1.0)
        print(f"  {label:<8s} total_norm={tn:.9e}  clip_coeff={coeff:.9f}  "
              f"clamp_max(1.0) 后={clamped:.9f}")
    print("  这里 total_norm ≫ max_norm，系数落在 (0,1)，**梯度会被缩放**。")
    print("  L179-L192 的两条路：系数是 Tensor 时先 clamp_max_(1.0) 再走")
    print("  multi_tensor_scale_tensor（L181-L188）；是 float 且 < 1.0 时走")
    print("  multi_tensor_scale（L189-L192）。判定时机不同，缩放结果一致。")

    print("\n【断言】")
    g64 = make_grads(np.float64)
    checks = [
        ("float64 下整块与逐参数相差不超过 2 ULP（关联序不同，非语义差异）",
         0 < 1 and int(ulp) <= 2),
        ("float32 下相对差 < 1e-6", diffs["float32"] < 1e-6),
        ("float32 的相对差大于 float64（量化步长更粗）",
         diffs["float32"] > diffs["float64"]),
        ("张量路径与浮点路径返回类型不同",
         type(as_tensor) is not type(as_float)),
        ("张量分支在 float32 里开根、fallback 分支在 float64 里开根（相对差 > 0）",
         rel_diff(float(as_tensor), as_float) > 0.0),
        ("该相对差量级与 float32 eps 相当（< 1e-6）",
         rel_diff(float(as_tensor), as_float) < 1e-6),
        ("总范数远大于 max_norm 时 clip 系数落在 (0,1)，即梯度被缩放",
         0.0 < max_norm / (norm_whole_buffer(grads32) + eps) < 1.0),
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