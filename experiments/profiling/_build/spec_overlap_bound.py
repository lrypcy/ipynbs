"""notebook spec: 重叠收益的上界（α-β 模型）。

对应博客：《性能剖析（01）：训练栈 profiling》
https://lrypcy.github.io/2026/09/29/profiling-01-training/
"""

CONFIGS_NAMES = ["ModelSpec", "MODELS", "HardwareSpec", "HARDWARE", "BenchConfig"]
COMM_NAMES = ["collective_bytes", "collective_time_s", "CommEvent",
              "tensor_parallel_events"]

SHIM = '''
# ---- 模块名垫片：让原脚本里的 `megatron_bench.comm.X` / `configs.X` 原样可用 ----
import types

configs = types.SimpleNamespace(BenchConfig=BenchConfig, MODELS=MODELS,
                                HARDWARE=HARDWARE)
comm = types.SimpleNamespace(collective_bytes=collective_bytes,
                             collective_time_s=collective_time_s,
                             tensor_parallel_events=tensor_parallel_events)
'''

# 复用脚本自己的 argparse 默认值，不在 notebook 里另写一套参数
ARGS = '''
# ---- 脚本 CLI 的默认参数（与 overlap_bound.py 的 argparse 一致）----
ARGS = argparse.Namespace(
    alpha_us=2.0, model="llama3-8b", hardware="h100-80g",
    tp=8, cp=1, pp=1, dp=8, micro_batch=4, grad_accum=8,
    seq_len=None, mfu=0.40, dump=None)
'''

SPEC = {
    "slug": "profiling-01-overlap-bound",
    "title": "重叠收益的上界：α-β 模型下一个能藏多少、什么情况下根本藏不动",
    "timeout": 300,
    "cells": [
        ("md", r"""
# 重叠收益的上界

> 对应博客篇目：《[性能剖析（01）：训练栈 profiling](https://lrypcy.github.io/2026/09/29/profiling-01-training/)》
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/overlap_bound.py` 与 `tools/megatron_bench/comm.py`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。纯 CPU，无第三方依赖。

## 这个 notebook 回答三个问题

1. **一次集合通信里，有多少是「可以藏」的？**
   $\alpha$-$\beta$ 模型把时间分成两截：`steps·α`（延迟项，依赖链，与计算量无关）与 `bytes/BW`（带宽项，有独立计算可跑就能藏住）。两项相等发生在 $D^* = n \cdot \alpha \cdot BW$。**这条比「通信占比」有用得多**——消息小于 $D^*$ 时是延迟主导，做重叠收益很小。

2. **一次迭代里通信的理论下界是什么？**
   完全不重叠 $T = C + M$；完美重叠 $T \ge \max(C, M)$；现实 $T \approx C + M - \eta\min(C, M)$。

3. **nccl-tests 的 algbw 和 busbw 差多少、差在哪。**
   $\text{busbw} = \text{algbw} \times 2(N-1)/N$（AllReduce）。**但 busbw 在 NVLS / CollNet / 分层算法下不再对应单一物理链路**，拿它反推线速会把「算法利用率」读成「链路带宽」。
"""),

        ("md", r"""
## 第 0 步 · 环境与共享记账

**这步在做什么**：把两个依赖摊平——

- `configs`：模型与硬件标称规格（Llama-3-8B / A100 / H100 / B200）
- `comm`：集合通信的 $\alpha$-$\beta$ 代价模型。约定**带宽用单向链路带宽**，时间公式采用 nccl-tests 的 bus-bandwidth 口径

集合通信的单卡搬运量：

$$\text{bytes} = \begin{cases} \frac{2(n-1)}{n} D & \text{AllReduce} \\ \frac{n-1}{n} D & \text{ReduceScatter / AllGather / All-to-All}\end{cases} \qquad t = \text{steps} \cdot \alpha + \frac{\text{bytes}}{BW}$$
"""),

        ("imports_py", (["tools/megatron_bench/configs.py",
                         "tools/megatron_bench/comm.py",
                         "tools/profiling_bench/overlap_bound.py"],
                        ("megatron_bench", "profiling_bench"))),
        ("raw_py", "tools/megatron_bench/configs.py", CONFIGS_NAMES, ""),
        ("raw_py", "tools/megatron_bench/comm.py", COMM_NAMES, ""),
        ("code", SHIM),
        ("code", ARGS),

        ("md", r"""
## 第 1 节 · $\alpha$ 项与 $\beta$ 项的交叉点 $D^* = n \cdot \alpha \cdot BW$

**这步在做什么**：以 AllReduce ring 为例，

$$t = 2(n-1)\alpha + \frac{2(n-1)}{n}\frac{D}{BW} \qquad \text{两项相等} \Rightarrow \alpha = \frac{D}{n \cdot BW} \Rightarrow D^* = n \cdot \alpha \cdot BW$$

AllGather / ReduceScatter 的 steps 是 $(n-1)$，比值同为 $n$，结果一样。

读法：

- $D^*$ **以下 = 延迟主导**。此时改重叠、加 buffer 都没多少用，真正有效的是「合并小消息」或换算法（NVLS）。
- $D^*$ **以上 = 带宽主导**。这才是重叠能发挥作用的区间。
- $D^*$ 与 $n$ 成正比：卡越多，依赖链越长，小消息越吃亏。
"""),

        ("raw_py", "tools/profiling_bench/overlap_bound.py",
         ["crossover_bytes", "section_crossover"],
         "section_crossover(ARGS.alpha_us)"),

        ("md", r"""
> **两个最容易记错的结论**：
>
> 1. **IB 的 $D^*$ 只有几百 KB**（25 GB/s 档约 0.2 MB），而 NVLink 的 $D^*$ 高出两个数量级——所以「跨机通信藏不住」的第一个原因是**它本来就落在延迟主导区**，第二个原因才是带宽窄。
> 2. **DP 梯度 reduce 的消息远在 $D^*$ 之上** → 带宽主导，可重叠。而 **TP 每层每次 AR 的消息量小得多，正好落在 $D^*$ 附近** → 这就是 `tp_comm_overlap` 收益不稳定、且强依赖 shape 的原因。
"""),

        ("md", r"""
## 第 2 节 · 一次迭代的通信量与重叠收益上界

**这步在做什么**：给定一套并行配置（默认 Llama-3-8B on H100，tp=8 / dp=8 / 8 卡一个节点，micro_batch=4 / grad_accum=8），算 TP 与 DP 的通信量、计算时长 $C$，然后给出四种口径下的迭代时间。

**先看一个容易被误当成结论的数字**：TP only 与 TP+SP 的通信总时长之比。这一节会算出来是 **1.00×**。

这不是模型不够精细，而是结论本身：**序列并行（SP）不改变 TP 的通信总量。** 恒等式 $\text{AR}(d) = \text{RS}(d) + \text{AG}(d)$ 决定总量守恒——baseline TP 每层 4 次 AR（每次搬 $2(n-1)/n \cdot d$），SP 每层 4 次 AG + 4 次 RS（每次搬 $(n-1)/n \cdot d$），两边都等于 $8(n-1)/n \cdot d$。

SP 分的是**激活显存**（LN / dropout / residual 从每卡存一份 $(s,b,h)$ 变成存 $(s/t,b,h)$，缩小 $t$ 倍）；真正的提速来自「显存省下来了，可以把 activation recompute 关掉」。**通信量口径上把 SP 当降本手段是常见的误解。**
"""),

        ("raw_py", "tools/profiling_bench/overlap_bound.py",
         ["fmt_b", "section_bound"], "section_bound(ARGS)"),

        ("md", r"""
> **怎么读这张上界表**：
>
> - `max(C, M)` 是任何调度器都突破不了的下界——**它是上界收益，不是目标值**。
> - $\eta$ 从 0 到 1 的落差 = 「重叠这件事的全部价值」。若只有几个百分点，所有 overlap 调参都不值得做，瓶颈在别处。
> - 本例 $\min(C, M) = M$，说明通信总量小于计算总量：**重叠在原理上够用**，那么「通信没藏住」就一定是调度/依赖问题，而不是通信量太大。
> - 反过来若 $M > C$，即使 $\eta = 1$ 也只能降到 $M$：此时唯一有效的手段是**减少通信量本身**（换并行策略、增大 micro batch 摊薄、改通信算法/拓扑）。注意**开序列并行不在此列**。
> - 实测 $\eta$ 的算法：从 trace 里量 `overlap_time / min(C, M)`。
"""),

        ("md", r"""
## 第 3 节 · nccl-tests：algbw 与 busbw 差多少、能反推什么

**这步在做什么**：$\text{busbw} = \text{algbw} \times \text{factor}$，factor 只取决于集合类型与卡数。给出换算表，并演示**algbw 的排名与 busbw 的排名可以相反**。
"""),

        ("raw_py", "tools/profiling_bench/overlap_bound.py",
         ["busbw_factor", "section_busbw"], "section_busbw()"),

        ("md", r"""
> **口径陷阱**：
>
> - **N=2 时 AllReduce 的 factor 只有 1.0**——两卡 AllReduce 就是 SendRecv，没有 busbw 口径优势。拿两卡测出来的 busbw 去推 8 卡性能会严重高估。
> - **busbw 是双向口径，链路标称带宽通常是单向口径**。所以 ring AllReduce 在大 N 下 busbw $\approx 2 \times$ 单向链路是**正常的**（N=8 时 1.75×），看到它超过链路标称值不要急着报故障。
> - 真正会让口径失效的是 **NVLS / CollNet / 分层算法**：此时上面这个 factor 的推导前提（数据在 N 卡链路上都跑了一遍）不成立，busbw 会超出 2× 这个经验上界。**判据：busbw / 单向链路 > 2 时，不要再拿它反推线速。**
> - **跨集合类型比带宽数字，是这张表要防的唯一一件事。**

## 参考文献

- [nccl-tests 源码](https://github.com/NVIDIA/nccl-tests)
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) —— 序列并行
- 通信条数的独立复述见 [arXiv:2311.02382](https://arxiv.org/abs/2311.02382) §3
"""),
    ],
    "readme": r"""
## 对应文章

- 《性能剖析（01）：训练栈 profiling》
  https://lrypcy.github.io/2026/09/29/profiling-01-training/

## 如何运行

```bash
# 纯标准库，无第三方依赖，秒级完成
jupyter nbconvert --to notebook --execute profiling-01-overlap-bound.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io`：

- `tools/profiling_bench/overlap_bound.py`
- `tools/megatron_bench/comm.py`（α-β 集合通信代价模型）
- `tools/megatron_bench/configs.py`（模型与硬件标称规格）

由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **第 1 节 · 交叉点 $D^*$**：$D^* = n \alpha BW$ 的全硬件 × 全链路表，
  区分「延迟主导」与「带宽主导」两个区间
- **第 2 节 · 重叠收益上界**：$C$ / $M$ 的解析计算，四种口径（不重叠 / 各级 $\eta$ / 完美重叠）
- **第 3 节 · algbw vs busbw**：换算因子表 + 排名反转的演示

## 关键结论

- **序列并行不改变 TP 通信总量**（比值恒 1.00×）——恒等式 $\text{AR} = \text{RS} \circ \text{AG}$ 决定总量守恒。
  SP 的收益在**激活显存**，提速来自可以关掉 activation recompute。
- **跨机通信藏不住**，第一个原因是它落在延迟主导区（IB 的 $D^*$ 只有几百 KB），第二个才是带宽窄。
- **DP 梯度 reduce 在带宽主导区**（可重叠），**TP 每层 AR 恰好落在 $D^*$ 附近**（这就是 `tp_comm_overlap` 收益不稳定的原因）。
- **N=2 的 AllReduce 没有 busbw 口径优势**；**busbw / 单向链路 > 2 时不能反推线速**（NVLS / CollNet / 分层算法）。
""",
}
