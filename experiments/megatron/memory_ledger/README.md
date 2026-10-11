# experiments/megatron/memory_ledger

《Megatron-LM 深度剖析（08）：显存账本——重计算、卸载与分布保存》的配套实验。

- **文章**：https://lrypcy.github.io/2026/10/12/megatron-08-memory-ledger/
- **源码基准**：`megatron-core` 0.20.0 / commit `60e039626`
- **复现**：`~/Software/miniconda3/envs/torch/bin/python gen_stdout.py`

## 三个实验各自证明什么

### 1. `pareto.py`（A 级 · 解析）—— 05 篇留下的那个 0 怎么填上

上一篇的 `data_parallel/mem_ledger.py` 按六项口径算账时，**激活项显式记 0**，
并写明理由：「它与 DP 路径无关，由 microbatch 数与重计算策略决定。
因此上表不能当作训练峰值显存」。本脚本接着那句话往下做：把激活这一项按
`recompute_granularity` / `recompute_method` / `recompute_num_layers` /
`recompute_modules` / `distribute_saved_activations` / offload 三档展开，
扫出前沿。

代价轴用**额外重算 FLOPs**，不用吞吐。本机没有 GPU，吞吐是时延量、
依赖卡型与 kernel 实现；额外重算 FLOPs 是纯结构量，前沿形状可被他人复算。

四条读数（llama-7b / seq 4096 / tp 8 / mbs 1 / SP on）：

| 读数 | 结论 |
| --- | --- |
| 基线激活 | 9.19 GiB |
| `full` + `uniform`，k 取 1/2/4/8 | 恒为 **0.125 GiB**，k 不改变显存 |
| `full` + `block`，k=1/2/4/8 | 8.90 / 8.62 / 8.05 / 6.92 GiB，代价 3.1% / 6.2% / 12.5% / 25.0% |
| `selective core_attn` | 9.06 GiB，代价 9.7%（extra/F） |

`uniform` 的 k 不改变显存这件事在源码里有直接出处：`recompute.py`
L157-L166 的 `while` 循环把全部层划进 chunk 且**每段都 checkpoint**，
k 只决定 chunk 边界数量，不决定哪些层被 checkpoint。`block` 才是
「只 checkpoint 前 k 层」（L167-L181）。

`distribute_saved_activations` 那一栏是本脚本最值得看的结果：

| SP | DSA 前后激活 | 省 |
| --- | --- | --- |
| on | 0.12 → 0.12 GiB | **0.000 GiB** |
| off | 1.00 → 0.12 GiB | 0.875 GiB |

SP 开着时 DSA 的净收益**恰好是 0** —— 它摊的正是层输入那条张量，
而 SP 已经把这条沿序列维摊过了。源码因此在 `transformer_config.py`
L1912-L1916 直接禁止 DSA 与 SP 同时开：二者不是可叠加的两项。

### 2. `selective_equivalence.py`（C 级 · CPU torch float64）—— 重算是不是数值恒等

要验四件事，全部以断言形式给出：

| 断言 | 结果 |
| --- | --- |
| normal 与 output-discarding 的**前向**逐位相同 | 0 ULP |
| normal 的全部梯度逐位相同 | 0 ULP |
| output-discarding **段内**参数（up/down）梯度逐位相同 | 0 ULP |
| output-discarding **段外**梯度相对误差 < 1e-14 | 2.20e-16（≈ 1 个 float64 eps） |
| 丢弃动作确实把字节归零，且形状/句柄仍可用 | 8192 → 0 字节 |
| 去掉 RNG 恢复后梯度确实变了 | 512 ULP |

最有意思的一条是 ULP 差的**位置**：段内 0 ULP，段外 1 个 ULP 量级，
边界张量的输入梯度 512 ULP。原因不是语义差异，而是 `detach_variable`
在重算图与外层图之间多插了一次梯度累加，浮点加法不结合。
normal 路径整块 checkpoint，段外没有第二条路径，故全 0 ULP。

### 3. `dsa_shapes.py`（C 级 · numpy）—— DSA 到底把张量摊成了什么形状

逐条复刻 `split_tensor_into_1d_equal_chunks`（`tensor_parallel/utils.py`
L48-L76）与 `gather_split_1d_tensor`（L79-L93）的下标算术，推出四点：

1. 保存下来的是**一维**张量，长度 `numel // tp_size`，形状不再是 `(T, h)`。
   `(4096, 4096)` + tp=8 → 每 rank 拿到 `(2097152,)`。
2. 只摊 `args[0]` 一个张量 —— `random.py` L622 / L648 的下标 0 是硬编码的。
   收益上限是「一个张量的 (1 - 1/tp)」，与该层保存了多少张量无关。
3. 与 SP 冲突的机制：切的是**扁平化后**的连续块，在 `(T, h)` 行主序下
   等价于沿序列维切 —— 与 SP 切的是同一个轴。SP 下再切一次会到 1/tp²。
4. `numel` 不能被 `tp_size` 整除时，`//` 向下取整**静默丢掉余数**，
   而聚合端按 `numel × tp_size` 还原，长度对不上，`.view()` 必然失败。
   实际配置里 `(T, h)` 的 numel 一般能被 tp 整除，所以这条边界平时碰不到，
   但它说明 DSA 没有余数处理逻辑。

## 与 04 篇 `gtp_shard.py` 的关系

第 3 个实验的方法论（给定形状与并行度，逐 rank 推出摊分后的形状）与
`experiments/megatron/gtp/gtp_shard.py` 同源：都是**只推形状、不做通信**。
代码是各自独立写的，没有复制。

## 三个脚本的依赖

| 脚本 | 解释器 | 需要 |
| --- | --- | --- |
| `pareto.py` | `~/Software/miniconda3/bin/python3` | 无（纯标准库） |
| `selective_equivalence.py` | `~/Software/miniconda3/envs/torch/bin/python` | torch |
| `dsa_shapes.py` | `~/Software/miniconda3/bin/python3` | numpy |

`gen_stdout.py` 为每个脚本显式指定解释器，所以用哪个跑它都行。