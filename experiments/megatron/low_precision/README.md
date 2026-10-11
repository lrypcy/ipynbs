# experiments/megatron/low_precision

《Megatron-LM 深度剖析（09）：低精度——FP8 全链路与 FP4》的配套实验。

- **文章**：https://lrypcy.github.io/2026/10/15/megatron-09-low-precision-fp8-fp4/
- **源码基准**：`megatron-core` 0.20.0 / commit `60e039626`
- **复现**：`~/Software/miniconda3/bin/python3 gen_stdout.py`

两个脚本共 22 条断言，全部通过，输出固化在 `results/stdout.txt`（196 行）。

## 1. `quantizers.py` — 粒度换来的是量程，不是精度

手撸四类量化器，在四组合成数据上对比。**数据是合成的正态 + 人为注入的极值，不是真实权重；绝对数值不可外推，只用于比较方案之间的相对关系。**

### 为什么用「逐位枚举」而不是套公式

E4M3 / E5M2 的格点集合是**枚举**出来的，不是按公式算出来的 —— 这样格点本身可被审阅。脚本会打印格点数量与最大有限值，并断言它们等于规格值：

| 量 | 枚举结果 | 规格 |
| --- | --- | --- |
| E4M3FN 有限值个数 | 253 | —— |
| E4M3FN 最大有限 | 448 | 448 |
| E5M2 有限值个数 | 247 | —— |
| E5M2 最大有限 | 57344 | 57344 |

**枚举时必须分开处理两个格式的「指数全 1」那一行**，这是本脚本最容易写错的地方：

- E4M3FN **没有 Inf**，指数全 1 且尾数全 1 才是 NaN，其余是正常数 → 最大 448
- E5M2 指数全 1 **整行**是 Inf/NaN → 最大只能到 57344

把 E5M2 当成 finite-nan 变体枚举，最大值会算成 **98304** —— 这个值不存在。脚本第 0b 段专门演示了这个错误。

枚举出的规格值与两处官方来源交叉核对通过：TE 的 [FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)（448 / 57344）与 [OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)（E4M3 无 Inf、最小规格化 2^-6、最小次正规 2^-9）。

### 四组数据各自暴露什么

| 组 | 构造 | 暴露的现象 |
| --- | --- | --- |
| A | 全部 N(0,1) | 基线，此时粒度几乎无差别 |
| B | 相邻 32 元素块交替 N(0,1) 与 N(0,0.02) | 量级差在**块之间**、块内一致 |
| C | 4 个幅值 ±12 的 outlier | FP8 与 INT8 的分歧 |
| D | 每个 32 块**内部**同时含 1 与 1e-5 | 量级差在**块之内** |

### 三条主要结论

**结论一 · 粒度曲线是平的。** MX 块大小取 1 / 8 / 32 / 128 / 512，RMSE **完全相同**（1.945893e-02），而 per-tensor 是 1.889401e-02。原因是 FP8 是浮点格式：只要一组内的值没跨出规格化区，缩小 scale 不增加尾数位，相对精度恒在 2^-3 量级。E4M3 的规格化区覆盖 448 / 2^-6 ≈ 2^14.8 的动态范围。

**结论二 · 粒度真正起作用的地方是极小值存活率。** 数据 D 上：

| 方案 | 极小值被清零比例 | 相对误差 |
| --- | --- | --- |
| FP8 E4M3 per-tensor | 60.9% | 0.7751 |
| FP8 E4M3 MX-block32 | 47.6% | 0.6416 |
| FP8 E4M3 MX-block8 | 0.0% | 0.0225 |
| FP8 E4M3 逐元素 | 0.0% | 0.0225 |
| INT8 per-tensor | 100.0% | 1.0000 |

per-tensor 的 scale 被同一块里的大值顶高，极小值除以 scale 后落到规格化区以下被舍成 0。MX-block32 只好一些，因为 32 元素的块里仍混着大值。**所以 block scaling 解决的是量程问题，不是精度问题。**

**结论三 · 同一笔「粒度换误差」的账，INT8 划算 FP8 不划算。** 数据 B 上 INT8 per-tensor → per-block32 误差降 **58.4%**，而 FP8 E4M3 per-tensor → MX-block32 反而**升 5.8%**（E8M0 scale 必须是 2 的幂，比浮点 scale 更受限）。

### 一条度量上的坑

脚本在数据 C 上得到 FP8 的 SNR **不降反升**（31.64 → 32.19 dB）。这不是量化变好，而是 outlier 抬高了信号能量 sum(x²)，SNR 的分子跟着涨。**跨数据集比精度只能用 RMSE，或针对某个子集算相对误差，不能用 SNR。** 同理，「极值相对偏差」这一列只在极小值那个子集上有信息量 —— 因为 scale 按 amax 算，最大值必然被精确还原。

## 2. `grad_clip_equivalence.py` — 整块归约与逐参数循环

验 `clip_grads.py` 里那个 multi-tensor applier 路径的数值后果。float64、12 个 `(256, 512)` 梯度：

| 读数 | 值 |
| --- | --- |
| 整块一次 | 1254.6424425718765 |
| 逐参数循环 | 1254.6424425718762 |
| 是否逐位相同 | **否** |
| ULP 差 | **1** |
| float32 相对差 | 1.069e-08 |

数学上恒等（√(Σ‖gᵢ‖²) = ‖g‖），浮点上差 1 ULP，来源是**求和结合序不同**：整块是一趟长线性求和，逐参数是 12 段局部求和再合并。

顺带验掉了一个常见误解：`norm_type=2` 时 `n ** 2` 与 `n * n` 在 IEEE 下是同一条指令序列，所以 `clip_grads.py` L121 那个 `pow` 的位置**不引入额外差异**，真正带来差异的只有分段 vs 单趟求和。

### 一条值得记的 fallback 差异

`clip_grads.py` L135-L138 的两条分支看起来只是「张量版 vs Python 版」：

```python
if multi_tensor_scale_tensor_impl is not None:
    total_norm = total_norm.pow(1.0 / norm_type)
else:
    total_norm = total_norm.item() ** (1.0 / norm_type)
```

**但 `.item()` 会把 float32 提升成 Python float（双精度），所以后一条分支的开根是在 float64 里做的。** 实测两条分支相对差 `2.145e-08`，与 float32 的 eps 同阶 —— **没有 `multi_tensor_scale_tensor` 的那条分支精度反而更高**，代价是一次 host 侧同步。源码没提这件事。

## 依赖

| 脚本 | 解释器 | 需要 |
| --- | --- | --- |
| `quantizers.py` | `~/Software/miniconda3/bin/python3` | numpy |
| `grad_clip_equivalence.py` | `~/Software/miniconda3/bin/python3` | numpy |

`gen_stdout.py` 为每个脚本显式指定解释器。