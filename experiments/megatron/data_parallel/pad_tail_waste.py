"""bucket padding 的 tail 浪费：逐字复刻 Megatron 的两套装箱逻辑。

对应文章：
  《Megatron-LM 深度剖析（05）：数据并行与优化器——三套实现的分野》
  https://lrypcy.github.io/2026/10/08/megatron-05-data-parallel-and-optimizers/

源码基准 megatron-core 0.20.0 / commit 60e039626。

────────────────────────────────────────────────────────────────────────
两个容易被忽略的前提，都来自源码，不是本文的推测
────────────────────────────────────────────────────────────────────────

1) **默认那条路根本不 padding。**
   `param_and_grad_buffer.py` L954-L968 的 docstring 明写（原文三引号内容，
   这里改成缩进引用以免嵌套截断）：

       def _compute_default_per_buffer_param_layout(params, bucket_size):
           Compute parameter layout for the non-distributed-optimizer case.

           No padding is applied. Parameters are iterated in reverse order
           (backprop order) and grouped into buckets of approximately
           bucket_size elements.

   所以「tail 浪费」只在 `use_distributed_optimizer=True` 时才存在。

2) **桶容量来自 `resolve_ddp_bucket_size`**（`training/training.py` L2172-L2187）：

       if ddp_config.num_buckets is not None:
           bucket_size = num_parameters // ddp_config.num_buckets
       elif ddp_config.bucket_size is None:
           bucket_size = max(40000000, 1000000 * get_pg_size(dp_cp_group))
       else:
           bucket_size = ddp_config.bucket_size
       if not overlap_grad_reduce:
           bucket_size = None          # 不分桶

   注意最后一行：**关掉 overlap_grad_reduce 会直接让 bucket_size 变成 None**，
   也就是「不分桶」，这与「padding 开销」是同一个开关的两面。

复刻对象：

   `optimizer/param_layout.py` L19-L42
       pad_to_divisor / pad_param_start(64) / bucket_end_divisor / pad_bucket_end

   `distributed/param_and_grad_buffer.py` L954-L1002   （默认路，no padding）
   `optimizer/distrib_optimizer.py`          L506-L580 （DistOpt 路，有 padding）

三个常数 64 / 128 / 2**16 的**设计动机源码未写**，本文只陈述现象，不做推测。
"""

from __future__ import annotations

import argparse
import math
import sys

# ══ 逐字复刻 param_layout.py L19-L42（不要改，改了就对不上正文） ══════════
def pad_to_divisor(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)

def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)

def bucket_end_divisor(data_parallel_world_size: int, pad_for_high_nccl_busbw: bool) -> int:
    """Divisor used to pad bucket ends for DP-divisibility (and optional NCCL busbw)."""
    if pad_for_high_nccl_busbw:
        return math.lcm(data_parallel_world_size, 128, 2 ** 16)
    return math.lcm(data_parallel_world_size, 128)

def pad_bucket_end(bucket_end_index: int, data_parallel_world_size: int,
                   pad_for_high_nccl_busbw: bool) -> int:
    return pad_to_divisor(bucket_end_index,
                          bucket_end_divisor(data_parallel_world_size, pad_for_high_nccl_busbw))


# ══ 复刻 resolve_ddp_bucket_size（training.py L2172-L2187） ═══════════════
def resolve_ddp_bucket_size(dp_cp_size: int, overlap_grad_reduce: bool,
                            num_buckets=None, bucket_size=None) -> int | None:
    if num_buckets is not None:
        return None  # 需要全量 numel，本实验不覆盖此分支
    if bucket_size is None:
        bucket_size = max(40_000_000, 1_000_000 * dp_cp_size)
    if not overlap_grad_reduce:
        return None
    return bucket_size


# ══ 复刻默认路 param_and_grad_buffer.py L975-L996（no padding） ═════════
def pack_default(params_numel: list[int], bucket_size: int | None) -> list[tuple[int, int]]:
    """逆序遍历；装满就切桶。**不做任何 padding。**"""
    bucket_indices: list[tuple[int, int]] = []
    param_start_index = 0
    bucket_start_index = 0
    n_in_bucket = 0
    for this_numel in params_numel[::-1]:
        param_end_index = param_start_index + this_numel
        n_in_bucket += 1
        if bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size:
            bucket_indices.append((bucket_start_index, param_end_index))
            bucket_start_index = param_end_index
            n_in_bucket = 0
        param_start_index = param_end_index
    if n_in_bucket > 0:
        bucket_indices.append((bucket_start_index, param_end_index))
    return bucket_indices


# ══ 复刻 DistOpt 路 distrib_optimizer.py L534-L580（有 padding） ════════
def pack_distopt(params_numel: list[int], bucket_size: int | None,
                 dp_world_size: int, pad_high: bool,
                 shared_embedding_idx: set[int] | None = None) -> list[tuple[int, int]]:
    """逆序遍历；每个 param 起点 pad_param_start；每个桶尾 pad_bucket_end；
    shared_embedding 的 param 单独成桶。返回 padded 后的 bucket_indices。"""
    shared_embedding_idx = shared_embedding_idx or set()
    bucket_indices: list[tuple[int, int]] = []
    param_start_index = 0
    bucket_start_index = 0
    n_in_bucket = 0

    def finalize(bucket_end_index: int, bucket_start: int) -> tuple[int, int]:
        """_finalize_bucket L542-L550"""
        end = pad_bucket_end(bucket_end_index, dp_world_size, pad_high)
        bucket_indices.append((bucket_start, end))
        return end, 0

    # 逆序遍历时 index 要翻转，shared_embedding 判定用原始下标
    for orig_i, this_numel in enumerate(params_numel[::-1]):
        real_i = len(params_numel) - 1 - orig_i
        is_shared = real_i in shared_embedding_idx

        param_start_index = pad_param_start(param_start_index)
        param_end_index = param_start_index + this_numel

        # shared embedding 单独成桶（L556-L561）
        if is_shared and n_in_bucket > 0:
            bucket_start_index, _ = finalize(param_start_index, bucket_start_index)
            n_in_bucket = 0

        n_in_bucket += 1
        # 装满或遇到 shared embedding 就切桶（L568-L572）
        if (bucket_size is not None
                and (param_end_index - bucket_start_index) >= bucket_size) or is_shared:
            bucket_start_index, n_in_bucket = finalize(param_end_index, bucket_start_index)
        param_start_index = param_end_index

    if n_in_bucket > 0:
        finalize(param_end_index, bucket_start_index)
    return bucket_indices


# ══ 模型参数规模（按 TP 切分后的每 rank numel） ═════════════════════════
def transformer_numels(n_layers: int, d_model: int, ffn: int, vocab: int,
                       tp: int, pp: int) -> tuple[list[int], set[int]]:
    """返回 (每 rank 的 param numel 列表, shared_embedding 的下标集合)。

    TP 沿 d_model / ffn / vocab 切，PP 沿层切。顺序按 forward（backpack 逆序遍历）。
    """
    dloc = d_model // tp
    ffloc = ffn // tp
    vloc = vocab // tp
    n_layer_loc = n_layers // pp
    nums: list[int] = []
    shared: set[int] = set()
    for _ in range(n_layer_loc):
        nums.append(3 * d_model * dloc)   # qkv   （row 切）
        nums.append(dloc * d_model)       # out   （col 切）
        nums.append(3 * dloc * ffloc)     # SwiGLU gate/up/down
    shared.add(len(nums))
    nums.append(vloc * d_model)           # embedding，共享输入输出
    nums.append(d_model)                  # final norm
    return nums, shared


MODELS = {
    "llama-7b":  dict(n_layers=32, d_model=4096,  ffn=11008, vocab=32000),
    "qwen-14b":  dict(n_layers=48, d_model=5120,  ffn=13824, vocab=151936),
    "gpt3-175b": dict(n_layers=96, d_model=12288, ffn=49152, vocab=50257),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama-7b,qwen-14b")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--dp", default="2,8,32,128")
    args = ap.parse_args(argv)

    print("=" * 92)
    print("bucket padding 的 tail 浪费（复刻 60e039626 的两套装箱逻辑）")
    print("=" * 92)

    # ── 前提 1：桶容量 ────────────────────────────────────────────────
    print("\n【前提1】resolve_ddp_bucket_size 默认给多少（training.py L2182）")
    print(f"  {'dp_cp':>6}  {'bucket_size':>14}   {'overlap_grad_reduce=False':>26}")
    for dp in [int(x) for x in args.dp.split(",")]:
        bs = resolve_ddp_bucket_size(dp, True)
        bs_no = resolve_ddp_bucket_size(dp, False)
        print(f"  {dp:>6}  {bs:>14,}   {str(bs_no) + '  即不分桶':>26}")

    # ── 前提 2：默认路不 padding ─────────────────────────────────────
    print("\n【前提2】默认路（use_distributed_optimizer=False）padding 为 0")
    for name in args.models.split(","):
        nums, _ = transformer_numels(tp=args.tp, pp=args.pp, **MODELS[name])
        total = sum(nums)
        bi = pack_default(nums, 40_000_000)
        padded_total = bi[-1][1]
        print(f"  {name:>10}  真实 {total:>14,}  桶尾 {padded_total:>14,}  "
              f"桶数 {len(bi):>3}  差值 {padded_total - total:>3}")
    print("  ↑ 与 param_and_grad_buffer.py L959 的 'No padding is applied' 一致")

    # ── 前提 3：三个常数怎么跳变 ─────────────────────────────────────
    print("\n【前提3】bucket_end_divisor 的因子（param_layout.py L29-L34）")
    print(f"  {'dp':>5}  {'lcm(dp,128)':>13}  {'lcm(dp,128,2^16)':>18}")
    for dp in [int(x) for x in args.dp.split(",")]:
        print(f"  {dp:>5}  {bucket_end_divisor(dp, False):>13}  "
              f"{bucket_end_divisor(dp, True):>18}")

    # ── 正文主结果：DistOpt 路的逐桶浪费 ─────────────────────────────
    print(f"\n【主结果】DistOpt 路 padding 浪费（TP={args.tp} PP={args.pp}，桶容量取默认）")
    print(f"  {'模型':>10}  {'dp':>4}  {'pad_high':>9}  {'桶数':>5}  "
          f"{'真实元素':>14}  {'padded 元素':>14}  {'浪费元素':>10}  {'浪费率':>9}")
    for name in args.models.split(","):
        nums, shared = transformer_numels(tp=args.tp, pp=args.pp, **MODELS[name])
        total = sum(nums)
        for dp in [int(x) for x in args.dp.split(",")]:
            bs = resolve_ddp_bucket_size(dp, True)
            for pad_high in (False, True):
                bi = pack_distopt(nums, bs, dp, pad_high, shared)
                padded_total = bi[-1][1]
                waste = padded_total - total
                print(f"  {name:>10}  {dp:>4}  {str(pad_high):>9}  {len(bi):>5}  "
                      f"{total:>14,}  {padded_total:>14,}  {waste:>10,}  "
                      f"{waste / padded_total:>8.4%}")

    # ── 前提 4：什么情况下 padding 才真的咬人 ─────────────────────────
    print("\n【前提4】只有「param numel 不是 128 的倍数」时 padding 才产生浪费")
    print("  稠密模型的 d_model/ffn 通常是 512/1024 的倍数，除以 TP 后仍是")
    print("  128 的倍数，于是 pad_param_start(64) 与 pad_bucket_end(128) 都落空。")
    print("  真正制造浪费的是不对齐的 param——MoE 的单专家权重、不对齐的 bias。")
    print(f"  {'场景':>34}  {'numel':>10}  {'start 浪费':>11}  {'end 浪费':>10}")
    scenarios = [
        ("稠密层 qkv（dloc=512）", 3 * 512 * 4096),
        ("稠密层 mlp（dloc=512,ffl=1376）", 3 * 512 * 1376),
        ("MoE 单专家 up（dloc=512,ffl=1408）", 2 * 512 * 1408),
        ("MoE 单专家 down（dloc=512,ffl=1408）", 512 * 1408),
        ("非对齐 bias（专家数量 7）", 7 * 512),
        ("非对齐 router bias（专家 11）", 11 * 512),
        ("非对齐 layernorm 分片（3 路）", 3 * 401),
    ]
    dp = 8
    for label, numel in scenarios:
        end_pad = pad_bucket_end(numel, dp, False) - numel
        start_pad = pad_param_start(numel) - numel
        print(f"  {label:>34}  {numel:>10}  {start_pad:>11}  {end_pad:>10}")

    print("\n【前提5】连续排布多个非对齐 param：每个 param 都补一次 64")
    mixed = [2 * 512 * 1408, 512 * 1408, 7 * 512, 11 * 512, 3 * 401, 512 * 1408]
    bs = 40_000_000
    for pad_high in (False, True):
        bi = pack_distopt(mixed, bs, dp, pad_high, set())
        real = sum(mixed)
        padded = bi[-1][1]
        print(f"  pad_high={str(pad_high):>5}  真实 {real:>10,}  padded {padded:>10,}  "
              f"浪费 {padded - real:>8,}  {(padded - real) / padded:>7.3%}  桶数 {len(bi)}")

    # ── 结论 ─────────────────────────────────────────────────────────
    print("\n【结论】")
    print("  1. 浪费是**逐桶 + 逐 param** 累加的：pad_bucket_end 每个桶补一次，")
    print("     pad_param_start 每个 param 补一次。桶数与 param 数都是放大因子。")
    print("  2. pad_high=False 时 lcm(dp,128) 在 dp<=128 恒等于 128。稠密模型的")
    print("     param numel 基本都是 128 的倍数，这一项**补不出任何东西**。")
    print("  3. pad_high=True 时 lcm 跃升到 65536，对齐良好的模型每个桶尾平均补")
    print("     32768 个元素，是 pad_high=False 的数百倍。代价可忽略但非零。")
    print("  4. 真正的大头是 pad_param_start(64)：只要 param numel 不是 64 的倍数，")
    print("     每个 param 都要补，最多 63 个元素。MoE 的单专家权重最容易踩。")
    print("  5. overlap_grad_reduce=False 让 bucket_size 变 None（不分桶），")
    print("     桶数降到 1，桶尾 padding 随之消失——这是同一开关的两面。")
    print("  6. 三个常数 64 / 128 / 2**16 的设计动机**源码未写**，正文标「待验证」。")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())