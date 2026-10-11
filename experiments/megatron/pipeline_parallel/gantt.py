"""C 级复现：1F1B 执行时序模拟，产出甘特图区间并独立测出气泡率。

用离散事件模拟跑完整的 1F1B 调度，再：
  C1  复刻 schedules.py L989 的 get_schedule_table，逐位对账官方 docstring 的两张表
  C2  模拟出的 makespan 与 2m + 2(p-1) 的一致
  C3  模拟测得的气泡率与 A 级闭式 (p-1)/(m+p-1) 一致
  C4  甘特图区间：每个 stage 的忙/闲段，供正文画图

无 GPU、无 torch，纯 Python。与 bubble.py 的输出交叉对账。

对应文章：《Megatron-LM 深度剖析（06）：流水并行调度器》§2、§3、§4。
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- C1 表格复刻


def get_schedule_table(
    num_microbatches: int, num_model_chunks: int, microbatch_group_size_per_vp_stage: int
) -> List[Tuple[int, int]]:
    """逐行复刻 schedules.py L989-L1016 的 get_schedule_table。"""
    schedule_table: List[Tuple[int, int]] = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table


# 官方 docstring 里的两张表（逐字抄录，用于对账）
DOC_CONFIG = {
    # model_parallel_config.py L441-L444：PP=2 VP=2 group=2 num_microbatches=4
    "config_group2_m4": (
        [0, 1, 0, 1, 2, 3, 2, 3],
        [0, 0, 1, 1, 0, 0, 1, 1],
    ),
}
# schedules.py L1227-L1229：PP2 N3M5 VP2 group=3 num_microbatches=5
DOC_SCHEDULES = {
    "sched_group3_m5": (
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    ),
}


def check_c1() -> List[str]:
    failures: List[str] = []
    for name, (exp_mb, exp_chunk) in DOC_CONFIG.items():
        m = 4 if "m4" in name else 5
        group = 2 if "group2" in name else 3
        table = get_schedule_table(m, 2, group)
        got_mb = [mb for mb, _ in table]
        got_chunk = [ch for _, ch in table]
        if got_mb != exp_mb:
            failures.append(f"C1 {name} microbatch_id: {got_mb} != 文档 {exp_mb}")
        if got_chunk != exp_chunk:
            failures.append(f"C1 {name} model_chunk_id: {got_chunk} != 文档 {exp_chunk}")
    return failures


# ---------------------------------------------------------------- 1F1B 模拟


def build_1f1b_order(p: int, rank: int, m: int) -> List[Tuple[str, int]]:
    """按 schedules.py 的三阶段（L2333 warmup / L2378 steady / L2450 cooldown）
    生成某个 stage 的操作序列，元素为 ('F'|'B', microbatch_id)。

    warmup = p - rank - 1（schedules.py L2277）
    再 min(warmup, num_microbatches)（schedules.py L2278）——这个封顶是
    m < p 仍然合法的唯一原因：否则 stage 0 的 warmup 会超过 microbatch 总数。
    """
    num_warmup = min(p - rank - 1, m)
    order: List[Tuple[str, int]] = []
    # warmup：只前向
    for j in range(num_warmup):
        order.append(("F", j))
    # steady：剩下 num_microbatches_remaining 个，每个先 F 再 B
    remaining = m - num_warmup
    for k in range(remaining):
        j = num_warmup + k
        order.append(("F", j))
        order.append(("B", k))
    # cooldown：剩余未做的 B
    for k in range(remaining, m):
        order.append(("B", k))
    return order


def simulate(p: int, m: int) -> Tuple[Dict[int, List[Tuple[int, int, str]]], int]:
    """事件驱动模拟。返回 (甘特区间, makespan)。

    资源约束：每个 stage 一次只做一个操作，且必须按 1F1B 顺序出队。
    数据依赖：F(s, j) 需 F(s-1, j) 完成；B(s, j) 需 B(s+1, j) 完成。
    每个操作耗时 1 个时间单位。

    注意 done 的键必须带 stage——同一个 microbatch j 在每个 stage 上都有
    自己的 F 与 B，不带 stage 会互相覆盖。
    """
    orders = {s: build_1f1b_order(p, s, m) for s in range(p)}
    ptr = {s: 0 for s in range(p)}
    done: Dict[Tuple[int, str, int], int] = {}
    busy: Dict[int, List[Tuple[int, int, str]]] = {s: [] for s in range(p)}
    last_end = {s: 0 for s in range(p)}

    total = sum(len(o) for o in orders.values())
    executed = 0
    guard = 0
    limit = total * (p + 2) + 16

    while executed < total:
        guard += 1
        if guard > limit:
            raise RuntimeError(f"调度未收敛: 已执行 {executed}/{total}")
        progressed = False
        for s in range(p):
            if ptr[s] >= len(orders[s]):
                continue
            op, j = orders[s][ptr[s]]
            dep: int | None = 0
            if op == "F" and s > 0:
                dep = done.get((s - 1, "F", j))
                if dep is None:
                    continue
            elif op == "B" and s < p - 1:
                dep = done.get((s + 1, "B", j))
                if dep is None:
                    continue
            start = last_end[s] if dep is None else max(last_end[s], dep)
            finish = start + 1
            if start > last_end[s]:
                busy[s].append((last_end[s], start, "idle"))
            busy[s].append((start, finish, f"{op}{j}"))
            last_end[s] = finish
            done[(s, op, j)] = finish
            ptr[s] += 1
            executed += 1
            progressed = True
        if not progressed:
            raise RuntimeError(f"调度死锁: 已执行 {executed}/{total}，无 stage 可推进")

    makespan = max(last_end.values())
    for s in range(p):
        if last_end[s] < makespan:
            busy[s].append((last_end[s], makespan, "idle"))
    return busy, makespan


# ---------------------------------------------------------------- 对账


def bubble_formula(p: int, m: int) -> float:
    return (p - 1) / (m + p - 1)


def check_c2_c3() -> List[str]:
    failures: List[str] = []
    for p in (2, 3, 4, 6, 8):
        for m in (4, 8, 12, 16):
            _, makespan = simulate(p, m)
            ideal = 2 * m
            if makespan != ideal + 2 * (p - 1):
                failures.append(f"C2 p={p} m={m} makespan={makespan} != 2m+2(p-1)={ideal + 2 * (p - 1)}")
            measured = (makespan - ideal) / makespan
            expect = bubble_formula(p, m)
            if abs(measured - expect) > 1e-9:
                failures.append(f"C3 p={p} m={m} 模拟气泡率 {measured:.6f} != 闭式 {expect:.6f}")
    return failures


# ---------------------------------------------------------------- 输出


def print_gantt(p: int, m: int) -> None:
    busy, makespan = simulate(p, m)
    print(f"\n[甘特图] p={p} m={m} makespan={makespan} （每格 1 单位时间，F=前向 B=反向 i=空闲）")
    for s in range(p):
        cells = []
        for t in range(makespan):
            ch = "."
            for a, b, label in busy[s]:
                if a <= t < b:
                    ch = label[0]
                    break
            cells.append(ch)
        print(f"  stage {s}: |{'|'.join(cells)}|")


def main() -> int:
    print("=" * 78)
    print("gantt.py —— 1F1B 时序模拟与气泡率交叉对账（C 级复现）")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print("=" * 78)

    print("\n[C1] get_schedule_table 与官方 docstring 两张表逐位对账")
    for name, (exp_mb, exp_chunk) in list(DOC_CONFIG.items()) + list(DOC_SCHEDULES.items()):
        m = 4 if "m4" in name else 5
        group = 2 if "group2" in name else 3
        table = get_schedule_table(m, 2, group)
        got_mb = [mb for mb, _ in table]
        got_chunk = [ch for _, ch in table]
        ok_mb = got_mb == exp_mb
        ok_ch = got_chunk == exp_chunk
        print(f"  {name}: microbatch_id {'✓' if ok_mb else '✗'} | model_chunk_id {'✓' if ok_ch else '✗'}")
        print(f"      实算 mb  = {got_mb}")
        print(f"      文档 mb  = {exp_mb}")

    print("\n[C2/C3] 模拟 makespan 与气泡率对账")
    print(f"{'p':>4}{'m':>5}{'makespan':>11}{'理论 2m+2(p-1)':>17}{'模拟气泡率':>13}{'闭式':>10}")
    for p in (2, 4, 8):
        for m in (8, 16):
            _, makespan = simulate(p, m)
            ideal = 2 * m + 2 * (p - 1)
            print(
                f"{p:>4}{m:>5}{makespan:>11}{ideal:>17}"
                f"{(makespan - 2 * m) / makespan * 100:>12.2f}%{bubble_formula(p, m) * 100:>9.2f}%"
            )

    print_gantt(4, 4)
    print_gantt(3, 6)

    print("\n[对账]")
    failures: List[str] = []
    for name, fn in (("C1", check_c1), ("C2/C3", check_c2_c3)):
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
        print("对账通过：schedule table 复刻逐位一致；模拟气泡率与 A 级闭式完全一致。")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())