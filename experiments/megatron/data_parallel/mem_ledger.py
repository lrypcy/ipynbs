"""三套数据并行路径的显存账本。

对应文章：
  《Megatron-LM 深度剖析（05）：数据并行与优化器——三套实现的分野》
  https://lrypcy.github.io/2026/10/08/megatron-05-data-parallel-and-optimizers/

源码基准 megatron-core 0.20.0 / commit 60e039626。

本脚本只做**算术账本**：给定参数量、优化器状态数、桶配置，按源码里
写死的比例算出每 rank 的各项字节数。它不模拟分配器、不预测峰值，
也不假装精确——真实峰值由 CUDA caching allocator 的碎片决定。

六项口径（与正文 §10 的取舍表一一对应）：

  1. param      每 rank 保留的权重。TP/PP 切分后 = 总量 / (TP*PP)
  2. grad       梯度缓冲。三套路径都是「每 rank 一份完整梯度」——
               DDP 复制、DistOpt 复制后 reduce、FSDP 分片。
  3. optim      优化器状态。AdamW 一阶+二阶 = 2 份 param。
  4. activation 激活。与 DP 路径无关，只随 microbatch 变化，此处不估。
  5. temp_bucket 重叠期临时桶。overlap_param_gather 开启时，
               all-gather 目标 buffer 与主权重 buffer 同时存在。
  6. comm_prec 通信精度带来的副本。fp32 累加 reduce-scatter
               （`reduce_scatter_with_fp32_accumulation.py`）需要额外
               一份 fp32 暂存。

分派来源 `training/training.py` L2476-L2483：

    if args.use_torch_fsdp2:   DP = torch_FSDP
    elif args.use_megatron_fsdp: DP = FullyShardedDataParallel
    else:                       DP = DDP
"""

from __future__ import annotations

import argparse
import sys

BYTES = {  # 每元素字节数
    "bf16": 2, "fp16": 2, "fp32": 4,
}


def ledger(total_params: int, tp: int, pp: int, dp: int,
           path: str, microbatch: int, seq_len: int,
           overlap_param_gather: bool, overlap_grad_reduce: bool,
           rs_fp32_accum: bool, param_dtype: str = "bf16",
           optim_dtype: str = "fp32") -> dict[str, int]:
    """返回该配置下每 rank 的各项字节数。

    path 取 'ddp' | 'distopt' | 'torch_fsdp2' | 'megatron_fsdp'。
    """
    psz = BYTES[param_dtype]
    osz = BYTES[optim_dtype]

    # TP*PP 切分后每 rank 的参数量
    shard = total_params // (tp * pp)
    out: dict[str, int] = {}

    # 1. param：三条路径都是每 rank 保留自己的 TP*PP 切片
    out["param"] = shard * psz

    # 2. grad：Megatron 的 grad buffer 始终是**每 rank 一份完整梯度**，
    #    即使 DistOpt 做 reduce-scatter，落地的仍然是一份完整大小的缓冲，
    #    只是每个 rank 只写自己负责的那 1/dp 区间。
    if path in ("ddp", "distopt"):
        out["grad"] = shard * psz
    else:
        # FSDP2 / MFSDP 用 flat param + 分片梯度，梯度也是分片的
        out["grad"] = (shard // dp) * psz

    # 3. optim 状态：一阶 m + 二阶 v，各一份 optim_dtype
    if path == "ddp":
        out["optim"] = shard * osz * 2
    elif path == "distopt":
        # DistOpt 的关键收益：状态按 DP 切片，只存 1/dp
        out["optim"] = (shard // dp) * osz * 2
    else:
        # FSDP2 / MFSDP 同样分片优化器状态
        out["optim"] = (shard // dp) * osz * 2

    # 4. activation：此处只按「与 DP 无关」记账，不做估算
    out["activation"] = 0

    # 5. temp_bucket：overlap_param_gather 时 all-gather 目标与源并存
    out["temp_bucket"] = shard * psz if overlap_param_gather else 0

    # 6. comm_prec：fp32 累加 reduce-scatter 需要一份 fp32 暂存
    if rs_fp32_accum and path == "distopt":
        out["comm_prec"] = (shard // dp) * 4
    else:
        out["comm_prec"] = 0

    out["_total"] = sum(v for k, v in out.items() if not k.startswith("_"))
    return out


PATHS = ["ddp", "distopt", "torch_fsdp2", "megatron_fsdp"]
NAMES = {
    "ddp": "DDP（全量 all-reduce）",
    "distopt": "DistributedOptimizer（reduce-scatter）",
    "torch_fsdp2": "torch FSDP2",
    "megatron_fsdp": "Megatron FSDP",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-params", type=int, default=7_000_000_000)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=8)
    args = ap.parse_args(argv)

    gib = 1024 ** 3
    print("=" * 94)
    print(f"三套 DP 路径的显存账本  总量 {args.total_params/1e9:.1f}B 参数  "
          f"TP={args.tp} PP={args.pp} DP={args.dp}  bf16 权重 + fp32 优化器状态")
    print("=" * 94)

    rows = []
    for path in PATHS:
        for opg, ogr, rs32 in [(False, False, False), (True, True, True)]:
            r = ledger(args.total_params, args.tp, args.pp, args.dp, path,
                       microbatch=1, seq_len=4096,
                       overlap_param_gather=opg, overlap_grad_reduce=ogr,
                       rs_fp32_accum=rs32)
            rows.append((path, opg, r))

    print(f"\n{'路径':<36} {'overlap':>8} {'param':>9} {'grad':>9} {'optim':>9} "
          f"{'临时桶':>9} {'fp32暂存':>9} {'合计 GiB':>10}")
    print(f"  {'':<34} {'':>8} " + " ".join(f"{'GiB':>9}" for _ in range(5)) + f" {'GiB':>10}")
    for path, opg, r in rows:
        print(f"  {NAMES[path]:<34} {str(opg):>8} "
              f"{r['param']/gib:>9.2f} {r['grad']/gib:>9.2f} {r['optim']/gib:>9.2f} "
              f"{r['temp_bucket']/gib:>9.2f} {r['comm_prec']/gib:>9.2f} "
              f"{r['_total']/gib:>10.2f}")

    print("\n【读数要点】")
    d = rows[0][2]        # DDP, overlap=False
    dist = rows[2][2]     # DistOpt, overlap=False
    dist_ov = rows[3][2]  # DistOpt, overlap=True
    print(f"  1. 优化器状态是 DistOpt 省显存的主要来源：")
    print(f"     DDP {d['optim']/gib:.2f} GiB  →  DistOpt {dist['optim']/gib:.2f} GiB，"
          f"省下 {100*(1-dist['optim']/d['optim']):.0f}%。")
    print(f"     原因是 grad buffer 仍是完整一份（{d['grad']/gib:.2f} GiB 不变），"
          f"省的是 m/v 两份状态。")
    print(f"  2. 开 overlap_param_gather 的代价是一整份 param 的临时桶："
          f"{dist_ov['temp_bucket']/gib:.2f} GiB。")
    print(f"     这是拿显存换延迟，账本上必须留出这笔预算。")
    print(f"  3. fp32 累加 reduce-scatter 再加 {dist_ov['comm_prec']/gib:.2f} GiB 暂存，"
          f"换取梯度累加不掉精度。")
    print(f"     两项叠加后 DistOpt 从 {dist['_total']/gib:.2f} 涨到 "
          f"{dist_ov['_total']/gib:.2f} GiB，仍低于 DDP 的 {d['_total']/gib:.2f}。")
    print(f"  4. 激活项本脚本记 0：它与 DP 路径无关，由 microbatch 数与重计算策略决定，")
    print(f"     属于第 06、07 篇的口径。**因此上表的合计不能当作训练峰值显存。**")
    print("=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(main())