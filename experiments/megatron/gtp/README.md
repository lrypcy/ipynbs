# GTP 切分语义与通信量验证 · 对应《深度剖析（04）》

## 对应文章

- 《Megatron-LM 深度剖析（04）：GTP——weight-centric 张量并行与权重重 materialization》
  https://lrypcy.github.io/2026/10/11/megatron-04-generalized-tensor-parallelism/

## 这是什么

两件互不依赖的事，都不需要 GPU：

| 部分 | 等级 | 验证对象 |
|---|---|---|
| 切分语义 | C（CPU 微型复现） | `gtp_remat_shard_dim0` / `_gtp_slice_one_param` 的 padding 与切片 |
| 通信量预算 | A（解析脚本） | 每元素线上字节数，与官方 §1.3 表逐格对账 |

切分语义部分的验证对象是 `generalized_tensor_parallelism.py` 的两个函数：

| 源码 | 行号 | 作用 |
|---|---|---|
| `gtp_remat_shard_dim0` | L519–L533 | 给定 `dim0` 与 GTP 组，算出每 rank 的行数与 padding 长度 |
| `_gtp_slice_one_param` | L536–L576 | 把补齐后的张量按 rank 切片，包成 `GTPShardedParam` |

复现的公式（`pad_for_alignment` 默认 16，见 `GTPRematConfig` L418）：

```text
alignment  = pad_for_alignment * gtp_remat_size
pad_length = (alignment - dim0 % alignment) % alignment
padded     = dim0 + pad_length
shard_dim0 = padded // gtp_remat_size
rank r 拿到 = tensor[r * shard_dim0 : (r + 1) * shard_dim0]
```

源码基准：`megatron-core` 0.20.0 / commit `60e039626`。固定 `seed=20261011`，
纯 numpy，无 GPU、无第三方依赖。

## 如何运行

```bash
cd ~/Desktop/Projects/sandbox/ipynbs/experiments/megatron/gtp

~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏
~/Software/miniconda3/bin/python3 comm_volume.py           # 写 results/comm_volume.txt
```

两个脚本退出码非 0 即表示有对账失败项，可直接当 CI 用。

## 文件

| 文件 | 等级 | 作用 |
|---|---|---|
| `gtp_shard.py` | C | 切分语义的实现：`shard_dim0` / `rank_row_range` / `per_rank_shapes` / `check_invariants` / `roundtrip` / `activation_centric_shapes` |
| `gen_stdout.py` | C | 切分语义逐格对账，产出 `results/stdout.txt` |
| `results/stdout.txt` | C | 固化的对账输出（58 行），读者不必自己跑 |
| `comm_volume.py` | A | 通信量预算解析：按块大小算每元素线上字节，产出 `results/comm_volume.txt` |
| `results/comm_volume.txt` | A | 固化的通信量对账输出，读者不必自己跑 |

## 结论

### §3 padding 在标准维度上是 no-op

`pad_for_alignment=16`、`gtp_remat_size=8` 时 `alignment = 128`，而常见维度都已是 128 的倍数：

| dim0 | size | align | pad | padded | shard | shard % 16 | 每 rank 形状 |
|---|---|---|---|---|---|---|---|
| 4096 | 4 | 64 | 0 | 4096 | 1024 | 0 | (1024, 14336) |
| 4096 | 8 | 128 | 0 | 4096 | 512 | 0 | (512, 14336) |
| 5120 | 8 | 128 | 0 | 5120 | 640 | 0 | (640, 14336) |
| 11008 | 8 | 128 | 0 | 11008 | 1376 | 0 | (1376, 14336) |
| 13824 | 8 | 128 | 0 | 13824 | 1728 | 0 | (1728, 14336) |

**padding 只在两种情况下才真正产生开销**：维度不是标准值，或 `gtp_remat_size` 整除不上标准维度。

| dim0 | size | align | pad | padded | shard | shard % 16 | 浪费 |
|---|---|---|---|---|---|---|---|
| 4100 | 4 | 64 | 60 | 4160 | 1040 | 0 | 1.463% |
| 777 | 4 | 64 | 55 | 832 | 208 | 0 | 7.079% |
| 11008 | 3 | 48 | 32 | 11040 | 3680 | 0 | 0.291% |
| 11008 | 6 | 96 | 32 | 11040 | 1840 | 0 | 0.291% |

注意 `11008` 那一行：`size=8` 时 `pad=0`，`size=3` 与 `size=6` 时 `pad=32`。**同一维度换个 GTP 组大小就会触发 padding**。

### §4 五组不变量在 8 组配置上全部成立

| 编号 | 断言 |
|---|---|
| I1 | `padded % (pad_for_alignment * gtp_remat_size) == 0` |
| I2 | `shard_dim0 % pad_for_alignment == 0`——每个 rank 的行数是对齐粒度的倍数 |
| I3 | 各 rank 行区间首尾相接、不重叠，并集恰为 `[0, padded)` |
| I4 | 真实数据行 `[0, dim0)` 连续覆盖且全在某个 rank 的区间内；padding 只落在尾部 |
| I5 | 数值往返：各 rank 分片按 rank 序拼接精确还原 padded 张量，去尾部 pad 后逐位还原原始张量 |

覆盖 `(4096,4) (4096,8) (5120,8) (4100,4) (777,4) (11008,3) (11008,8) (28672,2)`，
六项检查全 `True`，往返全部「一致」。

I2 是这套机制存在的理由：`alignment` 取 `pad_for_alignment * gtp_remat_size`
正是为了让 `shard_dim0 = padded / gtp_remat_size` 恒为 `pad_for_alignment` 的倍数——
每个 rank 拿到的行数本身就是对齐的。

### §5 no-pad 模式的断言行为

`pad_for_alignment=0` 时源码要求 `dim0 % gtp_remat_size == 0`，否则 `assert`：

| dim0 | dim0 % 4 | 行为 |
|---|---|---|
| 4096 | 0 | `shard=1024, pad=0` |
| 4100 | 0 | `shard=1025, pad=0` |
| 4101 | 1 | ✓ 抛 `AssertionError` |
| 777 | 1 | ✓ 抛 `AssertionError` |

no-pad 模式的代价写在源码注释里：*No-pad mode: dim-0 must divide gtp_remat_size or
AG output loses tail rows*——不补齐就会在 all-gather 后丢尾部行。

### §6 与 activation-centric TP 的形状对照

| | 切什么 | 每 rank 形状（dim0=hidden=4096, tp=4） |
|---|---|---|
| activation-centric | 激活的最后一维 | 激活 `(1, 1, 1024)` |
| activation-centric | 权重（初始化时按输出维切） | 权重 `(1024, 4096)` |
| GTP | **权重的第 0 维** | 权重 `(1024, 4096)` |

两者**每 rank 的形状可能相同，但切分的轴与时机不同**：activation-centric 在初始化时
就把权重切掉、之后不再动；GTP 把权重沿第 0 维切给 GTP 组，并在使用时 all-gather 回全量
（这就是 remat）。

### §11 通信量预算（A 级解析：低精度到底省了多少）

`comm_volume.py` 回答一个纯算术问题：**每microbatch、每权重，走GTP_remat 的
通信量是多少字节，BF16 vs MXFP8 vs NVFP4 分别省多少。** 口径是每元素字节，
假设 wgrad 按 bf16 做 reduce-scatter。

口径定义（与官方 GTP 设计文档 §1.3 的 *Communication volume breakdown* 一致）：

```text
Per-elem = 数据 B/elem + scale_inv B/elem    # 一份量化权重缓冲的线上字节
Fwd AG   = Per-elem                          # 前向聚合一次
Bwd AG   = Per-elem                          # 反向再聚合同一分片（columnwise 视角）
Wgrad RS = 2.0                               # 梯度恒按 bf16 归约，与权重精度无关
Total    = Fwd AG + Bwd AG + Wgrad RS
```

两种原生格式都**没有** per-microbatch 的 amax all-reduce：MXFP8 是 microscale，
scale 随数据一起走；NVFP4 的 block scale 在被聚合的缓冲里，per-tensor scale 在
优化器步里定。所以 `Total` 里不含 amax 一项。

| 格式 | 块 | 数据 B/elem | scale_inv B/elem | Per-elem | Fwd AG | Bwd AG | Wgrad RS | 合计 | vs BF16 |
|---|---|---|---|---|---|---|---|---|---|
| BF16 | n/a | 2.0000 | — | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 6.0000 | 1.00× |
| MXFP8 | 32 | 1.0000 | 1/32 = 0.0313 | 1.0313 | 1.0313 | 1.0313 | 2.0000 | 4.0626 | 0.68× |
| NVFP4 | 16 | 0.5000 | 1/16 = 0.0625 | 0.5625 | 0.5625 | 0.5625 | 2.0000 | 3.1250 | 0.52× |

**聚合侧省了，归约侧没省。** 这是这个表最要紧的一层含义：

| 格式 | AG 两项占总预算 | wgrad RS 占|
|---|---|---|
| BF16 | 67% | 33% |
| MXFP8 | 51% | 49% |
| NVFP4 | 36% | **64%** |

BF16 下聚合是大头；到 NVFP4，**归约取代聚合成了主路径**。这解释了为什么
NVFP4 路线必须同时配 `--fp4-param-gather`：只压聚合而不动归约，收益会很快见底。

脚本的 `scale_inv` 由块大小推出（一个 scale 字节摊到 `block` 个元素上），
并逐格与官方表对账，20 格全部一致（`1/32` 与 `1/16` 的四位小数取整差异在
`5e-4` 容差内）。

**本节不声称任何吞吐 / MFU 结论**——那是B 级（真机实测）的范围。

## 它验证了本文哪个数字

`results/stdout.txt` 里 8 条断言全部 `ok`、0 个 `MISMATCH`，覆盖正文：

- §3-a 表的五行 `pad` / `padded` / `shard` / 每 rank 形状
- §3-b 表的四行 `pad` / `shard` / 浪费率，其中 `777` 的 `7.078507%`（相对容差 `1e-5`）
- §5 表的四行断言行为 + `no-pad shard=1024`
- §6 的两个形状断言

另有不出现在正文列表、但作为门禁生效的断言：§3-a 每行 `pad` 必须为 0、
§3-b 与 §4 每组配置的 `shard % 16` 必须为 0、§4 六项不变量全为 `True`、
§4 的拼接必须精确还原 padded 张量。

`comm_volume.py` 另外对账正文 §11 的通信量表：三种格式各 4 列
（`Per-elem` / `Fwd AG` / `Bwd AG` / `Total B/elem`）加一个 `vs BF16` 相对值，
共 20 格，全部一致（容差 `5e-4`）；并推出 AG 与 wgrad RS 各自占预算的比例
（NVFP4 下 RS 占 64%）。

## 一个实现上的观察

`gtp_remat_shard_dim0` 只返回 `(shard_dim0, pad_length)`，真正的 padding 与切片在
`_gtp_slice_one_param`（L536–L576）里，且那里是**先补后切**：

```python
    if GTP_CONFIG.pad_for_alignment > 0:
        # Pad before slicing so shards stay alignment-divisible and padding
        # ends up contiguous at the tail of the gathered result.
        alignment = GTP_CONFIG.pad_for_alignment * gtp_remat_size
        dim0 = tensor.shape[0]
        pad_length = (alignment - dim0 % alignment) % alignment
        if pad_length > 0:
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_length))
```

补在**尾部**（`F.pad` 的 `(0,0,0,pad_length)` 只动第 0 维后端），所以 padding
在all-gather 结果里也是连续的尾部——这正是 §4 的 I4 能验「padding 只落尾部」的原因。
若改成头部补齐，各 rank 的分片里就会夹着真实数据与padding 的边界，I4 不再成立。