# experiments/megatron/data_parallel

《Megatron-LM 深度剖析（05）：数据并行与优化器——三套实现的分野》的配套实验。

- **文章**：https://lrypcy.github.io/2026/10/08/megatron-05-data-parallel-and-optimizers/
- **源码基准**：`megatron-core` 0.20.0 / commit `60e039626`
- **复现**：`python3 gen_stdout.py`

## 三个实验各自证明什么

### 1. `pad_tail_waste.py` — bucket padding 到底浪费多少

逐字复刻两套装箱逻辑，跑它们的差：

| 路径 | 函数 | padding |
|---|---|---|
| 默认（DDP） | `param_and_grad_buffer.py` L954-L1002 | **无**，docstring 明写 "No padding is applied" |
| DistOpt | `distrib_optimizer.py` L506-L580 | 有，调用 `param_layout.py` L19-L42 |

两个容易被忽略的结论：

1. padding **只在 `use_distributed_optimizer=True` 时存在**。默认那条路一个元素都不补。
2. `resolve_ddp_bucket_size`（`training/training.py` L2172-L2187）里
   `overlap_grad_reduce=False` 会让 `bucket_size` 直接变成 `None`，
   即**不分桶**——桶数降到 1，桶尾 padding 随之消失。这是同一开关的两面。

实测：稠密模型的 param numel 基本都是 128 的倍数，
`pad_bucket_end`（`lcm(dp,128)` 在 `dp<=128` 时恒等于 128）**补不出任何东西**；
真正制造浪费的是 `pad_param_start(64)` 遇到非对齐 param（MoE 单专家权重最常见），
每个最多补 63 个元素。开 `pad_buckets_for_high_nccl_busbw` 后
`lcm` 跃升到 65536，非对齐场景的浪费率从 0.003% 涨到 1.869%。

三个常数 64 / 128 / 2\*\*16 的设计动机**源码没有写**，本文只陈述现象。

### 2. `rs_shard_equivalence.py` — 分片更新与不分片 AdamW 严格等价

`DistributedOptimizer` 把参数按 DP 切片、梯度用 reduce-scatter 落到本 rank。
因为 AdamW 是**逐元素**运算，切片不改变任何元素的更新式，
所以分片路径应当与非分片路径**逐位相同**（不是近似）。

float64 实测：

| dp | 结果 | 最大 ULP 差 |
|---|---|---|
| 2 / 8 / 32 / 64（2 的幂） | 逐位相同 | 0 |
| 3 / 6 / 12 / 5（非 2 的幂） | 不逐位相同 | 4 – 24 |

2 的幂那一组测不出差异，是因为 `1/dp` 可被二进制精确表示（只降阶数），
「先求和后缩放」与「先缩放后求和」必然相同。**这正是实践中最常见的情形。**

结论：分片不改变收敛语义，只改变显存布局与通信量。
换 DP 组大小做 bit-wise 复现时，若 dp 不是 2 的幂需留意上述 ULP 差。

### 3. `mem_ledger.py` — 三套路径的显存账本

六项口径：param / grad / optim / activation / 临时桶 / fp32 暂存。
7B 参数、TP=8、DP=8、bf16 权重 + fp32 优化器状态：

| 路径 | param | grad | optim | 临时桶 | fp32 暂存 | 合计 |
|---|---|---|---|---|---|---|
| DDP | 1.63 | 1.63 | 6.52 | 0.00 | 0.00 | 9.78 GiB |
| DistOpt | 1.63 | 1.63 | 0.81 | 0.00 | 0.00 | 4.07 GiB |
| DistOpt + overlap | 1.63 | 1.63 | 0.81 | 1.63 | 0.41 | 6.11 GiB |
| FSDP2 / MFSDP | 1.63 | 0.20 | 0.81 | 0.00 | 0.00 | 2.65 GiB |

要点：

- DistOpt 省显存的主要来源是**优化器状态**（6.52 → 0.81 GiB，省 88%）。
  grad buffer 仍是完整一份——reduce-scatter 只是让每个 rank 只写自己那 1/dp 区间。
- 开 `overlap_param_gather` 的代价是一整份 param 的临时桶。
- **activation 项记 0**，它与 DP 路径无关。因此合计不能当作训练峰值显存。

## 文件

```
pad_tail_waste.py        A 级：padding 浪费
rs_shard_equivalence.py  C 级：等价性逐位验证
mem_ledger.py            A 级：显存账本
gen_stdout.py            依次运行并固化输出
results/stdout.txt       固化输出（167 行）
```

`gen_stdout.py` 在任一实验非零退出时**拒绝写入** `results/stdout.txt`，
避免把失败的输出固化成基线。