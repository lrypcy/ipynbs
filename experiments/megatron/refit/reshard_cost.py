"""A 级 · 排期与通信量解析模型——用真实 planner 的 op 列表算字节与 op 数。

不给任何实测数字：全部由 (布局, 参数形状) 解析算出，再借真实
`build_plan_from_rosters` 产出的 op 列表把「解析量」落实成「多少个 op、
多少字节、其中多少是本地拷贝」。

关键区分（对应四种 copy service 里 local/remote 的分派）：
  本地字节  src_rank == dst_rank  → 各 backend 走 copy_()，不过网络
  远端字节  src_rank != dst_rank  → 走 NCCL / Gloo / NIXL / NVSHMEM

这解释了「同构 vs 异构」的差别不是「快一点」而是「字节数差一个数量级」。

对应文章：《Megatron-LM 深度剖析（11）：Refit / Resharding》§5。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from planner_determinism import (  # noqa: E402
    COMMIT, LAYERS, INP, OUT, build_rosters, load_real_planner, snapshot,
)


def _numel(sl, local_shape):
    """由 tuple-of-slice 还原元素数。

    slice 是**本 rank 本地分片坐标**下的：send 落在源分片 (OUT/src_tp, INP)，
    recv 落在目的分片 (OUT/dst_tp, INP)。开区间切片按该维本地长度补全。
    """
    n = 1
    for dim, s in enumerate(sl):
        if isinstance(s, slice):
            if s.stop is None:
                full = local_shape[dim] if dim < len(local_shape) else 1
                start, stop, step = (s.start or 0), full, (s.step or 1)
            else:
                start, stop, step = (s.start or 0), s.stop, (s.step or 1)
            n *= len(range(start, stop, step))
        else:
            n *= 1
    return n


def measure(planner, utils, src_tp: int, dst_tp: int, collocated: bool):
    """跑真实 planner，算出一组 (src_tp, dst_tp, 是否共址) 的解析代价。"""
    # 共址时目的 rank 从 0 起编——必须**在建 roster 时就编对**：
    # 事后只改 roster 的 key 而不同步改 ParameterMetadata.tensor_parallel_group_ranks
    # 的话，_plan_tp 会因为 my_global_rank 不在该组里而静默返回空排期。
    base = 0 if collocated else 100
    dst_by_rank, src_by_name, all_ranks = build_rosters(utils, src_tp, dst_tp, dst_rank_base=base)
    if collocated:
        all_ranks = sorted(set(all_ranks))

    sends, recvs = snapshot(planner, dst_by_rank, src_by_name, all_ranks)

    elem = 2  # bf16
    src_local = (OUT // src_tp, INP)
    dst_local = (OUT // dst_tp, INP)
    local_b = remote_b = 0
    per_rank = {}
    for ops in sends.values():
        for rank, _pname, my_slice, _peer_slice, peer in ops:
            nb = _numel(my_slice, src_local) * elem
            per_rank[rank] = per_rank.get(rank, 0) + nb
            if peer == rank:
                local_b += nb
            else:
                remote_b += nb

    return {
        "src_tp": src_tp,
        "dst_tp": dst_tp,
        "collocated": collocated,
        "task_ids": len(sends),
        "send_ops": sum(len(v) for v in sends.values()),
        "recv_ops": sum(len(v) for v in recvs.values()),
        "local_bytes": local_b,
        "remote_bytes": remote_b,
        "total_bytes": local_b + remote_b,
        "per_rank": per_rank,
    }


# ---------------------------------------------------------------- 模型自检


def check_model(planner, utils):
    """两组不变量：
    M1 总字节 == 模型参数总量（bf16），异构时每个字节恰好搬一次
    M2 task_id 数 == 参数数 × src_tp（TP 整除时每源分片各发一次）
    """
    bad = []
    for src_tp, dst_tp, colo in ((8, 4, False), (8, 8, False), (8, 8, True), (8, 2, False)):
        m = measure(planner, utils, src_tp, dst_tp, colo)
        # 模型参数量
        per_param = OUT * INP
        total_params = LAYERS * 4 * per_param
        expect_bytes = total_params * 2  # bf16
        if m["total_bytes"] != expect_bytes:
            bad.append(
                f"M1 src_tp={src_tp} dst_tp={dst_tp} colocated={colo}: "
                f"总字节 {m['total_bytes']} != 参数量×2 {expect_bytes}")
        n_params = LAYERS * 4
        if m["task_ids"] != n_params * src_tp:
            bad.append(
                f"M2 src_tp={src_tp} dst_tp={dst_tp} colocated={colo}: "
                f"task_id {m['task_ids']} != 参数数×src_tp {n_params * src_tp}")
    return bad


# ---------------------------------------------------------------- main


def main() -> int:
    print("=" * 78)
    print("reshard_cost.py —— 同构 vs 异构 resharding 的通信量解析模型（A 级）")
    print(f"源码基准 Megatron-LM {COMMIT}；op 列表来自真实 build_plan_from_rosters")
    print(f"模型：{LAYERS} 层 × 4 类权重，OUT={OUT} INP={INP} bf16")
    print("=" * 78)

    planner, utils, tmp = load_real_planner()

    print("\n[表 1] 异构：训练侧 TP=8 固定，推理侧 TP 变（non-collocated）")
    print(f"{'dst_tp':>7}{'task_id':>9}{'send':>7}{'recv':>7}{'远端字节':>14}{'本地字节':>12}{'总字节':>14}{'远端占比':>10}")
    for dst_tp in (8, 6, 4, 2, 1):
        m = measure(planner, utils, 8, dst_tp, False)
        pct = m["remote_bytes"] / m["total_bytes"] * 100
        print(f"{dst_tp:>7}{m['task_ids']:>9}{m['send_ops']:>7}{m['recv_ops']:>7}"
              f"{m['remote_bytes']:>14,}{m['local_bytes']:>12,}{m['total_bytes']:>14,}{pct:>9.1f}%")

    print("\n[表 2] 同构：TP=8 → TP=8，但目的 rank 与源 rank 共址（热切换）")
    m = measure(planner, utils, 8, 8, True)
    print(f"  task_id {m['task_ids']} | send {m['send_ops']} | recv {m['recv_ops']}")
    print(f"  远端字节 {m['remote_bytes']:,} | 本地字节 {m['local_bytes']:,} | 总 {m['total_bytes']:,}")
    print(f"  → 远端占比 {m['remote_bytes'] / m['total_bytes'] * 100:.1f}%"
          "（同构且共址时全部是本地 copy_()，不过网络）")

    print("\n[表 3] 同构但不共址：TP=8 → TP=8，rank 空间错开")
    m = measure(planner, utils, 8, 8, False)
    print(f"  远端字节 {m['remote_bytes']:,} | 本地字节 {m['local_bytes']:,} | 总 {m['total_bytes']:,}")
    print(f"  → 远端占比 {m['remote_bytes'] / m['total_bytes'] * 100:.1f}%"
          "（布局相同但 rank 不同，仍要整份搬）")

    print("\n[表 4] 每 rank 负载（TP=8→4，non-collocated）")
    m = measure(planner, utils, 8, 4, False)
    for r in sorted(m["per_rank"]):
        tag = "源(训练)" if r < 8 else "目的(推理)"
        print(f"  rank {r:>3} {tag:<10} {m['per_rank'][r]:>12,} B")

    print("\n[表 5] op 数与 src_tp 的关系（dst_tp=4 固定）")
    print(f"{'src_tp':>8}{'task_id':>9}{'send':>7}{'每参数 op 数':>14}")
    for src_tp in (1, 2, 4, 8, 16):
        m = measure(planner, utils, src_tp, 4, False)
        n_params = LAYERS * 4
        print(f"{src_tp:>8}{m['task_ids']:>9}{m['send_ops']:>7}"
              f"{m['task_ids'] / n_params:>14.2f}")

    print("\n[对账]")
    failures = []
    for name, fn in (("M1/M2 模型不变量", lambda: check_model(planner, utils)),):
        try:
            f = fn()
        except Exception as e:  # noqa: BLE001
            f = [f"{name} 抛异常: {type(e).__name__}: {e}"]
        print(f"  {name}: {'ok' if not f else f'!! {len(f)} 处不一致'}")
        failures += f

    print()
    print("=" * 78)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for x in failures:
            print("  !! " + x)
    else:
        print("对账通过：字节与 op 数均由 (布局, 形状) 解析得出，与真实 planner 排期一致。")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())