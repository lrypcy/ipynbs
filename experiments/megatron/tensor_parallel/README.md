# 张量并行 / 序列并行等价性验证 · 对应《深度剖析（03）》

## 对应文章

- 《Megatron-LM 深度剖析（03）：张量并行与序列并行——`f` / `g` 算子与 RS + AG 的真实实现》
  https://lrypcy.github.io/2026/09/09/megatron-03-tensor-parallel-and-sequence-parallel/

## 这是什么

纯 CPU 的**等价性 / 正确性**验证，不含任何性能测量。三项断言：

| 编号 | 验证内容 | 对应源码 |
|---|---|---|
| §3 | `f` 的列切 + `g` 的 all-reduce ≡ 未切分的整块 GEMM | `RowParallelLinear.forward` L1504 / L1538 |
| §6 | SP 路径（`g` 换 reduce-scatter、`Column` 输入侧换 all-gather）数值不变；且通信量守恒 | L616–L621、L1534、`mappings.py` L355–L381 |
| §9 | `gradient-accumulation-fusion` 与朴素累加的精度差 | `LinearWithGradAccumulationAndAsyncCommunication`、`_wgrad_gemm` |

源码基准：`megatron-core` 0.20.0 / commit `60e039626`。固定 `seed=20261009`，纯 numpy，
无 GPU、无第三方依赖。

## 如何运行

```bash
cd ~/Desktop/Projects/sandbox/ipynbs/experiments/megatron/tensor_parallel

~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏
```

`gen_stdout.py` 退出码非 0 即表示有对账失败项，可直接当 CI 用。

## 文件

| 文件 | 作用 |
|---|---|
| `tp_equiv.py` | 三项验证的实现：`check_e1` / `check_e2` / `check_e3`，外加 `to_bf16` 与 `ring_bytes` |
| `gen_stdout.py` | 逐格对账，产出 `results/stdout.txt` |
| `results/stdout.txt` | 固化的对账输出（41 行），读者不必自己跑 |

## 结论

### §3 `f` 列切 + `g` all-reduce ≡ 整块 GEMM

规格 `s=8 b=4 hidden=16 ffn=48`，参考量级 `absmax = 15.8227`。

| tp | 相对误差 | 绝对误差 |
|---|---|---|
| 2 | 1.205e-07 | 1.907e-06 |
| 4 | 1.205e-07 | 1.907e-06 |
| 8 | 1.808e-07 | 2.861e-06 |

误差停在 float32 的机器精度量级（`1.2e-7`），**不随 tp 增大而累积**——这是「分片 + 求和」与整块 GEMM 在数值上等价的直接证据。

### §6 SP 数值等价，且通信量逐字节守恒

| tp | 非 SP 相对误差 | SP 相对误差 |
|---|---|---|
| 2 | 1.706e-07 | 1.706e-07 |
| 4 | 1.706e-07 | 1.138e-07 |
| 8 | 2.275e-07 | 1.138e-07 |

通信量（ring bus-byte 口径，单位化到 $$S = 1$$）：

| tp | AllReduce | RS + AG | 比值 |
|---|---|---|---|
| 2 | 2.000000 | 2.000000 | 1.000000000000000 |
| 4 | 3.000000 | 3.000000 | 1.000000000000000 |
| 8 | 3.500000 | 3.500000 | 1.000000000000000 |

**比值逐位为 1**，因为 $$\text{AllReduce} = \text{ReduceScatter} \circ \text{AllGather}$$ 在
ring 算法下是恒等式：$$\text{AR} = \frac{2S(n-1)}{n}$$，$$\text{RS} + \text{AG} = \frac{S(n-1)}{n} \times 2 = \frac{2S(n-1)}{n}$$。
所以**SP 不减少通信量**，它的收益在别处（激活显存）。

### §9 融合不改变数值；改变数值的是主累加器的 dtype

| steps | fused vs naive32 | fused vs fp64 | naive16 vs fp64 |
|---|---|---|---|
| 4 | **0.000e+00** | 1.113e-07 | 2.070e-03 |
| 16 | **0.000e+00** | 9.761e-08 | 1.421e-03 |
| 64 | **0.000e+00** | 2.829e-07 | 1.901e-03 |

两条结论：

1. **`fused_vs_naive32` 恒为 0**——只要主累加器是 fp32，开不开 `gradient-accumulation-fusion`
   的结果**逐位相同**。这个开关是纯性能优化，不是数值语义变更。
2. **劣化全部来自 dtype，不来自融合方式**。steps=16 时 fp32 主累加器相对误差
   `9.761e-08`，若每步先把 wgrad 舍入到 bf16 再累加则为 `1.421e-03`，
   **劣化 14558.9 倍**（`1.4559e4`）。

所以「要不要开融合」的答案与数值无关，只与显存和速度有关；而「主累加器用 fp32」才是
数值稳定性的关键。

## 它验证了本文哪个数字

`results/stdout.txt` 里 4 条断言全部 `ok`、0 个 `MISMATCH`，覆盖正文：

- §3 表的三行相对误差与绝对误差（tp = 2/4/8）
- §6 表的两列相对误差（非 SP、SP）与三行通信量比值（全部 `1.000000000000000`）
- §9 表的三行 `fused vs naive32`（全为 0）与两列相对误差，以及 steps=16 的劣化倍数
  `14558.94262295`（相对容差 `1e-9`）

另有 4 条不在正文列表、但作为门禁生效的断言：`tp=2/4/8` 的等价性误差不得越过
`TOL_REL = 1e-5`、通信量比值不得越过 `1e-12`、`fused vs naive32` 必须**严格为 0**、
bf16 路径的劣化倍数必须大于 `100x`。

## 一个实现上的坑（写这个实验时踩到）

Megatron 的 MLP 里，**两段 GEMM 之间没有 all-reduce**。Column 的 `W` 按输出维切开后，
每个 rank 拿到的是**互不重叠的特征段**；GeLU 逐元素，所以可以直接作用在特征分片上；
Row 的 `input_is_parallel=True`，直接拿自己那份特征分片去乘 `W` 的对应块。

第一次写这个实验时，我在Column 之后又按特征维切了一次给 Row，结果维度对不上。
`RowParallelLinear.forward` L1503–L1504 的分叉就是这个不变量：

- `input_is_parallel=True`：输入已按特征维分片，直接用
- `input_is_parallel=False`：先 `scatter_to_tensor_model_parallel_region` 再用（这才是
  §3 验的那条 `f` 列切路径）

`f` 的列切属于后者，`input_is_parallel=True` 的 MLP 路径不需要它。