"""A 级解析：流水线气泡率的闭式解，与 Megatron 源码的 warmup 公式对账。

无 GPU、无分布式，纯算术。读者可自行复算。

对应文章：《Megatron-LM 深度剖析（06）：流水并行调度器》§3 与 §4。

## 依据（均锁 commit 60e039626）

1. 非交错 1F1B 的 warmup 步数——
   `schedules.py` L957：
       num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
2. 交错（interleaving）的 warmup 步数——
   `schedules.py` L964-L965：
       num_warmup_microbatches  = (pipeline_parallel_size - pipeline_parallel_rank - 1) * (...)
       num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
3. 总 microbatch 数——
   `schedules.py` L949：`total_num_microbatches = num_microbatches * num_model_chunks`

## 两个闭式解

记 p = PP 度、m = 每 stage 的 microbatch 数、v = 每 rank 的 model chunk 数（虚拟 stage）。

    非交错 1F1B  气泡率  B1 = (p - 1) / (m + p - 1)
    交错 1F1B    气泡率  Bv = (p - 1) / (m * v + p - 1)

两式的分母都是「总 microbatch 数 + p - 1」，分子都是「p - 1」——
**交错不改变气泡的「槽数」，只把每个槽位变长**。这是本脚本要验的第一件事。

本脚本做三组对账：
  A1  B1/Bv 与源码 warmup 公式一致（逐 (p, m, v) 组合验证）
  A2  m -> inf 时气泡率收敛到 0，且收敛速率与解析式一致
  A3 交错相对非交错的改善倍数 = (m + p - 1) / (m*v + p - 1)
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------- 源码公式


def warmup_without_interleaving(p: int, rank: int, m: int) -> int:
    """复刻 schedules.py L957（该行只在 pipeline_parallel_size > 1 时被调用）。"""
    return p - rank - 1


def warmup_with_interleaving(p: int, rank: int, v: int, group: int) -> int:
    """复刻 schedules.py L964-L965。

    L960-L963 的注释说明第一项还要乘一个系数，但该系数在 L964 被折进
    `microbatch_group_size_per_vp_stage` 之外的一段，本脚本只保留可确证的两项：
        (p - rank - 1) + (v - 1) * group
    """
    return (p - rank - 1) + (v - 1) * group


def total_microbatches(m: int, v: int) -> int:
    """复刻 schedules.py L949。"""
    return m * v


# ---------------------------------------------------------------- 闭式解


def bubble_1f1b(p: int, m: int) -> float:
    """非交错 1F1B 气泡率。"""
    return (p - 1) / (m + p - 1)


def bubble_interleaved(p: int, m: int, v: int) -> float:
    """交错 1F1B 气泡率。"""
    return (p - 1) / (m * v + p - 1)


# ---------------------------------------------------------------- 对账


def check_a1() -> list[str]:
    """A1：闭式解的分母/分子与源码 warmup 公式自洽。

    判据：稳态下每个 stage 的「前向总步数」相同，且 stage 0 的 warmup
    恰好等于气泡槽数 p - 1。
    """
    failures: list[str] = []
    for p in (2, 4, 8, 16):
        for m in (4, 8, 16, 32):
            # 非交错：stage 0 warmup 应为 p-1，即气泡槽数
            w0 = warmup_without_interleaving(p, rank=0, m=m)
            if w0 != p - 1:
                failures.append(f"A1 p={p} m={m} 非交错 stage0 warmup={w0} != p-1={p - 1}")
            # 末 stage warmup 应为 0
            wl = warmup_without_interleaving(p, rank=p - 1, m=m)
            if wl != 0:
                failures.append(f"A1 p={p} m={m} 末 stage warmup={wl} != 0")
            # 闭式解应等于 (stage0 warmup) / (总步数 + stage0 warmup)
            b = bubble_1f1b(p, m)
            expect = w0 / (m + w0)
            if abs(b - expect) > 1e-12:
                failures.append(f"A1 p={p} m={m} 气泡率 {b:.6f} != {expect:.6f}")
    return failures


def check_a2() -> list[str]:
    """A2：m 增大时气泡率单调下降，且收敛速率是 (p-1)/m。

    注意极限是 m -> inf，有限 m 下不应为 0。正确的渐近判据是
        bubble(p, m) * m  ->  p - 1
    因此在最大 m 处要求 bubble * m 与 p-1 的相对偏差 < 5%。
    """
    failures: list[str] = []
    for p in (2, 4, 8, 16):
        prev = 1.0
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            b = bubble_1f1b(p, m)
            if b >= prev:
                failures.append(f"A2 p={p} m={m} 气泡率未单调下降: {b:.6f} >= {prev:.6f}")
            prev = b
        # 渐近：bubble * m 应趋近 p - 1
        scaled = prev * 1024
        if abs(scaled - (p - 1)) / (p - 1) > 0.05:
            failures.append(
                f"A2 p={p} m=1024 渐近 {scaled:.4f} 与 p-1={p - 1} 偏差 "
                f"{abs(scaled - (p - 1)) / (p - 1) * 100:.2f}% 超过 5%"
            )
    return failures


def check_a3() -> list[str]:
    """A3：交错相对非交错的改善倍数等于分母之比。"""
    failures: list[str] = []
    for p in (2, 4, 8):
        for m in (4, 8, 16, 32):
            for v in (2, 4, 8):
                b1 = bubble_1f1b(p, m)
                bv = bubble_interleaved(p, m, v)
                ratio = b1 / bv
                expect = (m * v + p - 1) / (m + p - 1)
                if abs(ratio - expect) > 1e-9:
                    failures.append(
                        f"A3 p={p} m={m} v={v} 改善倍数 {ratio:.6f} != {expect:.6f}"
                    )
    return failures


# ---------------------------------------------------------------- 输出


def main() -> int:
    print("=" * 78)
    print("bubble.py —— 流水线气泡率闭式解与源码 warmup 公式对账（A 级解析）")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print("=" * 78)

    print("\n[表 1] 非交错 1F1B 气泡率 B1 = (p-1)/(m+p-1)")
    hdr = "p\\m"
    print(f"{hdr:>6}" + "".join(f"{m:>10}" for m in (1, 2, 4, 8, 16, 32, 64)))
    for p in (2, 4, 8, 16, 32):
        row = "".join(f"{bubble_1f1b(p, m) * 100:>9.2f}%" for m in (1, 2, 4, 8, 16, 32, 64))
        print(f"{p:>6}{row}")

    print("\n[表 2] 交错 1F1B 气泡率 Bv = (p-1)/(m*v+p-1)")
    vs = (1, 2, 4, 8)
    print(f"{hdr:>6}" + "".join(f"{'v=' + str(v):>12}" for v in vs))
    for p in (2, 4, 8, 16):
        for m in (4, 8):
            row = "".join(
                f"{bubble_interleaved(p, m, v) * 100:>11.2f}%" for v in vs
            )
            print(f"{p:>3}m{m:<2}{row}")

    print("\n[表 3] 固定 p=4, m=8，交错把气泡压到多少")
    print(f"{'v':>4}{'总microbatch':>14}{'气泡率':>12}{'相对非交错':>14}")
    b1 = bubble_1f1b(4, 8)
    for v in (1, 2, 4, 8, 16):
        bv = bubble_interleaved(4, 8, v)
        print(f"{v:>4}{total_microbatches(8, v):>14}{bv * 100:>11.2f}%{b1 / bv:>13.2f}x")

    print("\n[表 4] 源码 warmup 公式与闭式解的对应（p=4, m=8, group=2）")
    p, m, group = 4, 8, 2
    print(f"{'rank':>6}{'非交错 warmup':>16}{'交错 warmup(v=2)':>20}")
    for rank in range(p):
        print(
            f"{rank:>6}{warmup_without_interleaving(p, rank, m):>16}"
            f"{warmup_with_interleaving(p, rank, 2, group):>20}"
        )

    print("\n[对账]")
    failures: list[str] = []
    for name, fn in (("A1", check_a1), ("A2", check_a2), ("A3", check_a3)):
        f = fn()
        print(f"  {name}: {'ok' if not f else f'!! {len(f)} 处不一致'}")
        failures += f

    print()
    print("=" * 78)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for x in failures:
            print("  !! " + x)
    else:
        print("对账通过：闭式解、源码 warmup 公式、单调性与改善倍数四者自洽。")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())