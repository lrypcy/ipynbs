"""C 级 · planner 确定性验证——直接跑**真实**的 build_plan_from_rosters。

本脚本不重实现 planner。它在运行时用 `git show 60e039626:` 把
`megatron/core/resharding/{planner,utils}.py` 抽到一个临时包再 import，
因此测的就是仓库里那份代码（只读，不改动 Megatron-LM）。

planner.py 的 docstring 明确说 build_plan_from_rosters 是
"Pure and collective-free, so it can be tested or reused with preassembled
rosters without touching the process group" —— 这正是本实验成立的前提。

五组断言：
  C1 确定性      同输入跑两次，task_id -> op 的映射逐位相同
  C2 互补性      所有 rank 的 send_ops 合起来，每个 task_id 恰好出现一次
  C3 收发自洽    同一 task_id 上 sender 的 my_slice == receiver 的 peer_slice
  C4 插入序无关  roster 的 dict 插入顺序被打乱后，task_id 分配不变
  C5 稀疏 rank 集 roster 的 key 换成非连续值（0,3,7）后结果不变

对应文章：《Megatron-LM 深度剖析（11）：Refit / Resharding》§2。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

MEGATRON = Path.home() / "Desktop/Projects/github/Megatron-LM"
COMMIT = "60e039626"
PKG_FILES = ["planner.py", "utils.py"]


# ---------------------------------------------------------------- 抽取真实源码


def load_real_planner():
    """把 planner.py + utils.py 抽成临时包并 import 进来。"""
    if not MEGATRON.is_dir():
        raise RuntimeError(f"找不到 Megatron-LM 仓库: {MEGATRON}")
    head = subprocess.run(
        ["git", "-C", str(MEGATRON), "log", "-1", "--format=%H"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not head.startswith(COMMIT):
        raise RuntimeError(f"HEAD 是 {head}，期望 {COMMIT}…；不许 pull/checkout，请人工确认")

    tmp = Path(tempfile.mkdtemp(prefix="refit_planner_"))
    pkg = tmp / "resharding_snapshot"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for name in PKG_FILES:
        src = subprocess.run(
            ["git", "-C", str(MEGATRON), "show", f"{COMMIT}:megatron/core/resharding/{name}"],
            capture_output=True, text=True, check=True,
        ).stdout
        (pkg / name).write_text(src, encoding="utf-8")

    sys.path.insert(0, str(tmp))
    import importlib
    mod = importlib.import_module("resharding_snapshot.planner")
    utils = importlib.import_module("resharding_snapshot.utils")
    return mod, utils, str(tmp)


# ---------------------------------------------------------------- 造 roster

# OUT 取 1152 = 2^7 * 3^2，使其可被 1/2/3/4/6/8/12/16 整除。
# 真实 planner 对 TP 维度做**精确**的尺寸守恒校验（_build_descriptors_for_param），
# OUT=2048 时 dst_tp=6 会被直接拒掉：8*256=2048 != 6*341。
OUT, INP, LAYERS = 1152, 1024, 6


def _mk(utils, name, resolved_name, tp_ranks, rank, dtype_size=2):
    """造一个 TP 分片后的 ParameterMetadata。

    切分沿 dim0，每个 rank 拿 OUT/len(tp_ranks) 行。
    """
    ParameterMetadata = utils.ParameterMetadata
    tp = len(tp_ranks)
    return ParameterMetadata(
        name=name,
        shape=(OUT // tp, INP),
        dtype="bfloat16",
        element_size=dtype_size,
        is_tp=tp > 1,
        partition_dim=0 if tp > 1 else None,
        partition_stride=1 if tp > 1 else None,
        partition_sizes=None,
        is_ep=False,
        num_experts=None,
        owner_rank=rank,
        tensor_parallel_group_ranks=list(tp_ranks),
        expert_parallel_group_ranks=None,
        data_parallel_group_ranks=None,
        pipeline_parallel_group_ranks=None,
        resolved_name=resolved_name,
        global_expert_index=None,
    )


def build_rosters(utils, src_tp: int, dst_tp: int, dst_rank_base: int = 100):
    """训练侧 src_tp 路 TP、推理侧 dst_tp 路 TP，各一个 TP 组。

    源 rank 编号 0..src_tp-1。目的 rank 默认从 100 起编，刻意让两套 rank
    空间不相交，模拟 non-collocated refit；传 dst_rank_base=0 即共址热切换。
    """
    src_ranks = list(range(src_tp))
    dst_ranks = [dst_rank_base + i for i in range(dst_tp)]

    src_by_name, dst_by_rank = {}, {r: {} for r in dst_ranks}

    for layer in range(LAYERS):
        for kind in ("self_attention.linear_qkv.weight",
                     "self_attention.linear_proj.weight",
                     "mlp.linear_fc1.weight",
                     "mlp.linear_fc2.weight"):
            resolved = f"decoder.layers.{layer}.{kind}"
            for r in src_ranks:
                m = _mk(utils, resolved, resolved, src_ranks, r)
                src_by_name.setdefault(resolved, []).append(m)
            for r in dst_ranks:
                m = _mk(utils, resolved, resolved, dst_ranks, r)
                dst_by_rank[r][resolved] = m

    return dst_by_rank, src_by_name, src_ranks + dst_ranks


# ---------------------------------------------------------------- 快照


def snapshot(planner, dst_by_rank, src_by_name, all_ranks):
    """跑一遍所有 rank，把 (task_id -> (kind, rank, param, my_slice, peer_slice)) 收齐。"""
    sends, recvs = {}, {}
    for r in all_ranks:
        plan = planner.build_plan_from_rosters(dst_by_rank, src_by_name, r)
        for op in plan.send_ops:
            sends.setdefault(op.task_id, []).append(
                (r, op.param_name, tuple(op.my_slice), tuple(op.peer_slice), op.peer_rank))
        for op in plan.recv_ops:
            recvs.setdefault(op.task_id, []).append(
                (r, op.param_name, tuple(op.my_slice), tuple(op.peer_slice), op.peer_rank))
    return sends, recvs


# ---------------------------------------------------------------- 五组断言


def check_c1_determinism(planner, dst_by_rank, src_by_name, all_ranks):
    a, _ = snapshot(planner, dst_by_rank, src_by_name, all_ranks)
    b, _ = snapshot(planner, dst_by_rank, src_by_name, all_ranks)
    if a != b:
        only_a = sorted(set(a) - set(b))[:5]
        only_b = sorted(set(b) - set(a))[:5]
        return [f"C1 两次 task_id 集合不同: 仅A={only_a} 仅B={only_b}"]
    diff = [t for t in a if a[t] != b[t]]
    if diff:
        return [f"C1 task_id {diff[:5]} 的 op 内容两次不同"]
    return []


def check_c2_complementarity(planner, dst_by_rank, src_by_name, all_ranks):
    sends, recvs = snapshot(planner, dst_by_rank, src_by_name, all_ranks)
    bad = []
    dup = [t for t, v in sends.items() if len(v) > 1]
    if dup:
        bad.append(f"C2 有 {len(dup)} 个 task_id 被多个 rank 同时作为 sender: {sorted(dup)[:5]}")
    missing = sorted(set(recvs) - set(sends))
    if missing:
        bad.append(f"C2 有 {len(missing)} 个 recv 没有对应 send: {missing[:5]}")
    orphan = sorted(set(sends) - set(recvs))
    if orphan:
        bad.append(f"C2 有 {len(orphan)} 个 send 没有对应 recv: {orphan[:5]}")
    return bad


def check_c3_pairing(planner, dst_by_rank, src_by_name, all_ranks):
    """同一 task_id：sender.my_slice 必须等于 receiver.peer_slice，反之亦然。"""
    sends, recvs = snapshot(planner, dst_by_rank, src_by_name, all_ranks)
    bad = []
    for t in sorted(set(sends) & set(recvs)):
        s = sends[t][0]
        r = recvs[t][0]
        # sends 里存的是 (rank, param, my_slice, peer_slice, peer_rank)
        s_my, s_peer = s[2], s[3]
        r_my, r_peer = r[2], r[3]
        if s_my != r_peer or s_peer != r_my:
            bad.append(f"C3 task_id={t} 切片不互为镜像: send my={s_my} peer={s_peer} / recv my={r_my} peer={r_peer}")
        if s[4] != r[0]:
            bad.append(f"C3 task_id={t} sender 认为对端是 {s[4]}，而对端确实是 rank {r[0]}")
    return bad


def check_c4_insertion_order(planner, utils, src_tp, dst_tp):
    """打乱 roster 的插入顺序，task_id 分配应当不变。

    docstring 说迭代顺序 "depends only on the rosters"，且源码用
    sorted(dst_param_metadata_by_rank) 遍历 —— 所以插入序不该有影响。
    这条正是「确定性」承诺里最容易实现错的一环。
    """
    dst_by_rank, src_by_name, all_ranks = build_rosters(utils, src_tp, dst_tp)

    def shuffled(d):
        items = list(d.items())
        items.reverse()
        return dict(items)

    base, _ = snapshot(planner, dst_by_rank, src_by_name, all_ranks)

    # 只打乱 dst 的 rank 维插入序
    rev_dst = {r: dict(reversed(list(d.items()))) for r, d in
               [(k, shuffled(v)) for k, v in shuffled(dst_by_rank).items()]}
    try:
        s1, _ = snapshot(planner, rev_dst, src_by_name, all_ranks)
    except Exception as e:  # noqa: BLE001
        return [f"C4 打乱 dst 插入序后 planner 抛异常: {type(e).__name__}: {e}"]

    bad = []
    if set(s1) != set(base):
        bad.append(f"C4 打乱 dst rank 插入序后 task_id 集合变了: 仅基准={sorted(set(base)-set(s1))[:5]}")
    else:
        diff = [t for t in base if base[t] != s1[t]]
        if diff:
            bad.append(f"C4 有 {len(diff)} 个 task_id 的 sender op 变了，例: t={diff[0]} {base[diff[0]]} vs {s1[diff[0]]}")
    return bad


def check_c5_sparse_ranks(planner, utils):
    """roster 的 key 换成非连续值，planner 不该依赖 range(world_size)。"""
    dst_by_rank, src_by_name, all_ranks = build_rosters(utils, 4, 2)
    base_sends, _ = snapshot(planner, dst_by_rank, src_by_name, all_ranks)

    # 只保留源侧，构造一个稀疏的 dst key 集合
    sparse_dst = {r: dst_by_rank[r] for r in sorted(dst_by_rank)[:1]}
    src_ranks = sorted(src_by_name and {m.owner_rank for v in src_by_name.values() for m in v})
    try:
        sp_sends, _ = snapshot(planner, sparse_dst, src_by_name, src_ranks + sorted(sparse_dst))
    except Exception as e:  # noqa: BLE001
        return [f"C5 稀疏 roster 抛异常: {type(e).__name__}: {e}"]

    bad = []
    if not base_sends:
        bad.append("C5 基准没有产生任何 send，无法比较")
    # 稀疏集合下产生的 task_id 必须是基准的子集，且每个 task_id 内容一致
    for t in sorted(sp_sends):
        if t not in base_sends:
            bad.append(f"C5 稀疏 roster 产生了基准里不存在的 task_id={t}")
        elif sp_sends[t] != base_sends[t]:
            bad.append(f"C5 task_id={t} 在稀疏 roster 下内容变了")
    if not bad:
        n = len(sp_sends)
        print(f"    （稀疏 roster：源 {len(src_ranks)} rank + 目的 {len(sparse_dst)} rank，"
              f"产出 {n} 个 task_id，全部是基准的子集）")
    return bad


# ---------------------------------------------------------------- main


def main() -> int:
    print("=" * 78)
    print("planner_determinism.py —— 真实 build_plan_from_rosters 的确定性验证（C 级）")
    print(f"源码基准 Megatron-LM {COMMIT}（运行时 git show 抽取，不改动仓库）")
    print("=" * 78)

    planner, utils, tmp = load_real_planner()
    print(f"\n已加载真实 planner：{planner.__file__}")

    print(f"\n[拓扑] 训练侧 TP=8（源 rank 0..7），推理侧 TP=4（目的 rank 100..103）")
    print(f"[模型] {LAYERS} 层 × 4 类权重，OUT={OUT} INP={INP} bf16")
    dst_by_rank, src_by_name, all_ranks = build_rosters(utils, 8, 4)

    sends, recvs = snapshot(planner, dst_by_rank, src_by_name, all_ranks)
    total_ops = sum(len(v) for v in sends.values()) + sum(len(v) for v in recvs.values())
    print(f"[规模] task_id 共 {len(sends)} 个；send op {sum(len(v) for v in sends.values())} 个，"
          f"recv op {sum(len(v) for v in recvs.values())} 个，合计 {total_ops} 个")
    busy = sorted({r for v in sends.values() for (r, *_ ) in v} |
                  {r for v in recvs.values() for (r, *_) in v})
    print(f"[参与] {len(busy)} 个 rank 都有 op（源 8 + 目的 4 = 12）")

    print("\n[对账]")
    failures = []
    for name, fn in (
        ("C1 确定性", lambda: check_c1_determinism(planner, dst_by_rank, src_by_name, all_ranks)),
        ("C2 互补性", lambda: check_c2_complementarity(planner, dst_by_rank, src_by_name, all_ranks)),
        ("C3 收发自洽", lambda: check_c3_pairing(planner, dst_by_rank, src_by_name, all_ranks)),
        ("C4 插入序无关", lambda: check_c4_insertion_order(planner, utils, 8, 4)),
        ("C5 稀疏 rank 集", lambda: check_c5_sparse_ranks(planner, utils)),
    ):
        try:
            f = fn()
        except Exception as e:  # noqa: BLE001
            f = [f"{name} 抛异常: {type(e).__name__}: {e}"]
        print(f"  {name}: {'ok' if not f else f'!! {len(f)} 处不一致'}")
        failures += f

    shutil.rmtree(Path(tmp).parent, ignore_errors=True)

    print()
    print("=" * 78)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for x in failures:
            print("  !! " + x)
    else:
        print("对账通过：真实 planner 在五组断言下均确定性且自洽。")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())