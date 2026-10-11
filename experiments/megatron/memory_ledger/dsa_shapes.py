#!/usr/bin/env python3
"""distribute_saved_activations 的逐 rank 形状推导（C 级，numpy）。

对应文章《Megatron-LM 深度剖析（08）：显存账本——重计算、卸载与分布保存》。
源码基准：megatron-core 0.20.0 / commit 60e039626。

要推的东西只有一条，但它决定了这一节所有结论
------------------------------------------------
`distribute_saved_activations`（下称 DSA）到底把哪个张量摊成了什么形状。

源码路径：
- 前向 `tensor_parallel/random.py` L621-L625：
      ctx.input_0_shape = args[0].data.shape
      safely_set_viewless_tensor_data(
          args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))
- 反向 L647-L650：
      safely_set_viewless_tensor_data(
          inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))
- 切分 `tensor_parallel/utils.py` L62-L76：
      partition_size = numel // tp_size
      start = partition_size * rank ; end = start + partition_size
      new_buffer=True 时 **新分配一块并拷贝**（L66-L73），不是 view
- 聚合 `tensor_parallel/utils.py` L88-L93：
      numel_gathered = numel * tp_size ; dist_all_gather_func(...)

本脚本逐条推出四组结论
----------------------
1. 保存下来的是**一维**张量，长度 `numel // tp_size`，形状不再是 `(T, h)`。
2. 只摊 `args[0]` 一个张量 —— 源码 L622 / L648 写死下标 0，
   所以 DSA 的收益上限是「一个张量的 (1 - 1/tp)」，与层内张量总数无关。
3. 与 SP 冲突的机制：DSA 切的是**扁平化后**的连续块，在 `(T, h)` 且行主序
   的情况下等价于沿序列维切；SP 已经沿序列维切过一遍，两者切的是同一个轴。
   这解释了 `transformer_config.py` L1912-L1916 为什么直接禁止 DSA + SP。
4. `numel` 不能被 `tp_size` 整除时，`//` 向下取整会**丢掉余数**，
   而聚合端按 `numel * tp_size` 还原，长度对不上，`view` 必然失败。

用法
----
    python3 dsa_shapes.py
退出码 0 表示全部断言通过。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Shard:
    rank: int
    start: int
    end: int
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        return self.end - self.start


def split_shape(shape: Tuple[int, ...], tp_size: int) -> List[Shard]:
    """复刻 `split_tensor_into_1d_equal_chunks` 的下标算术。"""
    numel = int(np.prod(shape))
    partition = numel // tp_size
    out = []
    for rank in range(tp_size):
        start = partition * rank
        end = start + partition
        out.append(Shard(rank=rank, start=start, end=end, shape=(partition,)))
    return out


def main() -> int:
    print("=" * 96)
    print("distribute_saved_activations 的逐 rank 形状推导（C 级 · numpy）")
    print("源码：tensor_parallel/utils.py L48-L93 / tensor_parallel/random.py L621-L650")
    print("=" * 96)

    T, H = 4096, 4096
    TP = 8
    shape = (T, H)
    numel = T * H
    print(f"\n设层输入 hidden_states 形状 = {shape}，numel = {numel}，tp_size = {TP}")
    print(f"bf16 字节数 = {numel * 2}")

    shards = split_shape(shape, TP)
    print("\n【1】切分后各 rank 手上的形状")
    print(f"{'rank':>4s} {'start':>10s} {'end':>10s} {'numel':>10s} {'保存形状':>12s} {'字节':>12s}")
    for s in shards:
        print(f"{s.rank:>4d} {s.start:>10d} {s.end:>10d} {s.numel:>10d} "
              f"{str(s.shape):>12s} {s.numel * 2:>12d}")
    group_total = sum(s.numel for s in shards) * 2
    per_rank = shards[0].numel * 2
    print(f"每 rank 保存 = {per_rank} 字节 = {per_rank / 1024 ** 2:.1f} MiB")
    print(f"原始 = {numel * 2} 字节 = {numel * 2 / 1024 ** 2:.1f} MiB")
    print(f"全组合计 = {group_total} 字节（与原始相等：数据没少，只是摊到各 rank）")
    print(f"单 rank 节省比例 = 1 - 1/tp = 1 - 1/{TP} = {1 - per_rank / (numel * 2):.6f}")

    print("\n【2】只摊 args[0] 一个张量")
    print("源码 random.py L622 写 ctx.input_0_shape = args[0].data.shape，L648 写")
    print("gather_split_1d_tensor(inputs[0]) —— 下标 0 是硬编码的。")
    print("所以 DSA 的收益 = 一个 [T,h] 张量的 (1 - 1/tp)，与该层保存了多少张量无关。")
    one = numel * 2 * (1 - 1 / TP)
    print(f"本例收益上限 = {numel * 2} × (1 - 1/{TP}) = {one:.0f} 字节 = "
          f"{one / 1024 ** 2:.1f} MiB")

    print("\n【3】往返一致性（split → all-gather → view）")
    rng = np.random.default_rng(20261012)
    x = rng.standard_normal(shape)
    local = [x.reshape(-1)[s.start:s.end].copy() for s in shards]
    gathered = np.concatenate(local)
    restored = gathered.reshape(shape)
    exact = np.array_equal(x, restored)
    print(f"  切分再聚合后与原张量逐位相同：{exact}")
    print(f"  聚合后长度 = {gathered.size}，原长度 = {numel}，"
          f"{'一致' if gathered.size == numel else '不一致'}")

    print("\n【4】SP 已经切过序列维时，DSA 的净收益")
    sp_T = T // TP
    sp_shape = (sp_T, H)
    sp_numel = sp_T * H
    print(f"  SP 下本 rank 的 hidden_states 形状 = {sp_shape}（序列维已切），"
          f"numel = {sp_numel}")
    sp_part = sp_numel // TP
    print(f"  DSA 再切一次：partition_size = {sp_numel} // {TP} = {sp_part}")
    print(f"  若允许 DSA+SP 同时开，单张量再省 {1 - 1 / TP:.6f}，但这条张量")
    print("  已经被 SP 摊到 1/tp；两次摊的是同一个轴（扁平化后的连续块在")
    print("  (T,h) 行主序下就是沿序列维切）。所以两项不是可叠加的省显存手段。")
    print("  transformer_config.py L1912-L1916 因此直接 raise ValueError 禁止组合。")
    print(f"  验证：SP 形状被 DSA 切两次后长度 = {sp_part}，"
          f"占原始 numel 的 {sp_part / numel:.8f}（= 1/tp² = {1 / TP ** 2:.8f}）")

    print("\n【5】numel 不能被 tp_size 整除的边界")
    print(f"{'numel':>10s} {'tp':>4s} {'numel//tp':>10s} {'tp×(numel//tp)':>15s} {'余数':>6s} {'能还原?':>8s}")
    for numel_e, tp in [(4096 * 4096, 8), (4095, 8), (100, 8), (4096, 5), (1023, 7), (10, 4)]:
        part = numel_e // tp
        recovered = part * tp
        rem = numel_e - recovered
        ok = (rem == 0)
        print(f"{numel_e:>10d} {tp:>4d} {part:>10d} {recovered:>15d} {rem:>6d} "
              f"{'是' if ok else '否':>8s}")
    print("  `//` 向下取整，余数被静默丢弃；聚合端按 numel × tp_size 还原，")
    print("  长度不足时 random.py L649 的 .view(ctx.input_0_shape) 会抛形状不匹配。")
    print("  实际配置里 (T, h) 的 numel 一般能被 tp 整除（h 通常是 tp 的倍数），")
    print("  所以这条边界平时碰不到 —— 但它说明 DSA 没有余数处理逻辑。")

    print("\n【断言】")
    checks = [
        ("切分后形状是一维 (numel//tp,)",
         all(len(s.shape) == 1 for s in shards)),
        ("各 rank 区间首尾相接且覆盖全表",
         all(shards[i].end == shards[i + 1].start for i in range(len(shards) - 1))
         and shards[0].start == 0 and shards[-1].end == shards[-1].numel * TP),
        ("往返逐位一致", exact),
        ("单 rank 持有量恰为 1/tp",
         abs(per_rank / (numel * 2) - 1 / TP) < 1e-12),
        ("全组合计与原始等长（摊分不丢数据）", group_total == numel * 2),
        ("非整除的样例确实有余数", any((n // t) * t != n for n, t in
                                   [(4095, 8), (100, 8), (4096, 5), (1023, 7), (10, 4)])),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print()
    if ok:
        print("全部断言通过。")
        return 0
    print("存在未通过的断言。")
    return 1


if __name__ == "__main__":
    sys.exit(main())