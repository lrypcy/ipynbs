"""notebook spec: 内置 Timer 的同步式计时语义。

对应博客：《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》
https://lrypcy.github.io/2026/09/29/profiling-00-metrics/

注意：实验一用真线程 + sleep 模拟设备侧耗时，绝对值有线程调度抖动
（同机重跑约 5%），所以正文里只主张趋势与比值。
"""

SPEC = {
    "slug": "profiling-00-timer-semantics",
    "title": "内置 Timer 的同步式计时语义：它测的是「排空耗时」，不是「阶段耗时」",
    "timeout": 900,
    "cells": [
        ("md", r"""
# 内置 Timer 的同步式计时语义

> 对应博客篇目：《[性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准](https://lrypcy.github.io/2026/09/29/profiling-00-metrics/)》§4
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/timer_semantics.py`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。纯 CPU，不需要 numpy。

## 这个 notebook 回答什么

Megatron 内置的 `Timer` 用的不是 CUDA Event，而是 `torch.cuda.synchronize()` + `time.time()`（`megatron/core/timers.py:143` / `:156`）。同步点会等**所有流排空**，于是**边界处还在飞的工作被算进了被测量的区间**。

而「在飞的工作」正是你辛苦做重叠的那部分。所以：

> **重叠做得越好，内置 timer 的读数膨胀得越厉害。优化的方向和指标的方向反了。**

本 notebook 分四步：

| 步骤 | 问题 |
|:---|:---|
| 1 | 同步点数量 × 通信占比 → 计时膨胀多少 |
| 2 | 多 rank 的 min/max 能不能相加 |
| 3 | `filter(x > 0.0)` 静默丢了谁 |
| 4 | `reset()` 不清 `_active_time` 会让占空比变成什么 |
"""),

        ("md", r"""
## 第 0 步 · 环境

只需要标准库（`threading` / `time` / `statistics`）。本 notebook 的实验一用**真线程 + sleep** 模拟设备侧耗时，语义与 CUDA 的异步发射一致，但线程调度会带来额外开销（每 chunk 多少由脚本自己打印，见实验一输出）。

脚本会把每 chunk 的调度开销显式打出来，所以看趋势与比值即可，不要纠结绝对值。
"""),

        ("imports_py", ["tools/profiling_bench/timer_semantics.py"]),

        ("md", r"""
## 第 1 步 · 同步式计时对重叠的破坏

**这步在做什么**：构造两条流——compute 流逐 chunk 计算，comm 流的第 $i$ 个 chunk 依赖 compute 流第 $i$ 个 chunk 完成（1F1B 反向重叠的语义）。chunk 时长 10 ms，共 8 个 chunk。

然后对比两种计时口径：

- **时间线口径**：一次发射到底，不在中间插同步点 → 这是「真实时间线」
- **同步式计时**：每 `phase_len` 个 chunk 调一次 `stop()`（排空所有流）再 `start()` → 这是内置 timer 的语义

膨胀量的解析式就是边界处被迫串行掉的通信：读数 = $$L(T_c + T_m)$$，真实 = $$L \cdot T_c + T_m$$，差 $$(L-1)T_m$$。

$T_m$ 越大，重叠越值得做，读数膨胀也越厉害。
"""),

        ("raw_py", "tools/profiling_bench/timer_semantics.py",
         ["_ComputeWorker", "_CommWorker", "_run_async", "_run_synced",
          "experiment_sync_inflation"],
         "experiment_sync_inflation()"),

        ("md", r"""
> **这是一条与直觉相反的规律**：通信占计算比例越高（`Tm` 越大、越值得藏），「每 chunk 一个阶段」那一档的膨胀率越夸张。优化的方向和指标的方向反了。
>
> **正确做法**：要「阶段耗时」就用时间线口径——CUDA Event 打点，或从 nsys / torch.profiler 的 trace 里读该区间的实际跨度。要「端到端步时」就只在外层打一对点，中间不要插同步点。

### 关于这一格数字的抖动

输出里「每 chunk 的线程调度开销」那一行是**实测值**，它随机器负载变化很大（本 notebook
这次记录到的值见上面输出）。它会同时抬高分子和分母，所以：

- **膨胀率的单调性是稳的**——`Tm` 越大膨胀越大，这个趋势每次运行都成立；
- **膨胀率的具体数值不恒定**——调度开销越大，`L·(T_c+T_m)` 里被 `L` 放大的那一项占比越高，膨胀率读数会偏低。

博客正文引用的是另一次运行的输出，与这里的**同趋势、绝对值不同**。想复现正文的绝对值，
请在机器空闲时重跑，并以脚本自己打印的调度开销那一行判断可信度。

实验二、三、四是**确定性计算**，没有这个抖动，数字与博客正文逐位一致。
"""),

        ("md", r"""
## 第 2 步 · 多 rank 的 min/max 不是同一张卡

**这步在做什么**：`timers.py:318` 的 `_get_global_min_max_time()` 对每个阶段分别取各 rank 的 min 与 max。**min 与 max 可能来自不同的卡。**

构造两种形态做对照（确定性合成，不用随机数）：

- **形态 A（掉队卡固定）**：rank 3 在每个阶段都慢 1.6× → 各阶段 max 落在同一张卡上
- **形态 B（短板轮转）**：每个阶段换一张卡慢 → 各阶段 max 落在不同的卡上

两种形态对「逐阶段相加」的影响完全不同。
"""),

        ("raw_py", "tools/profiling_bench/timer_semantics.py",
         ["_synth_ranks", "experiment_minmax_additivity"],
         "experiment_minmax_additivity()"),

        ("md", r"""
> **形态 A（掉队卡固定）**：各阶段 max 恰好都落在同一张卡上，相加碰巧等于最慢卡真实值。
> **形态 B（短板轮转）**：max 来自不同的卡，相加得到一个**没有任何卡跑出过的数**。
>
> 所以「把各项 max 加起来」既可能碰巧对，也可能系统性错——取决于短板是否固定。这不是可以靠约定消除的，只能靠改观测方案：要么采逐 rank 原始值，要么采 top-1 最慢 rank 的整步时长。
"""),

        ("md", r"""
## 第 3 步 · `filter(x > 0.0)` 的静默丢弃

**这步在做什么**：没进过该 timer 的 rank 会被 `filter` 掉，于是 min 变成「**活跃子集里的最快**」，与「全体的最快」不是同一个量。

过滤本身是合理的（没进过 timer 不该报 0），**但它不告诉你少了谁**。
"""),

        ("raw_py", "tools/profiling_bench/timer_semantics.py",
         ["experiment_zero_filter"], "experiment_zero_filter()"),

        ("md", r"""
## 第 4 步 · `reset()` 不清 `_active_time`

**这步在做什么**：`timers.py:171` 的 `reset()` 只清 `_elapsed`，而 `_active_time`（`timers.py:212`）从 timer 构造起只增不减。于是两者的比值是一个**占空比**，但只在稳态才成立。

同一个数在早期迭代里完全没有意义——**要拿占空比，必须用稳态窗口内的差值，不能拿累计量直接除。**
"""),

        ("raw_py", "tools/profiling_bench/timer_semantics.py",
         ["experiment_active_time"], "experiment_active_time()"),

        ("md", r"""
## 小结

内置 timer 报的是「**排空后的时间**」，不是「阶段占用的时间」。三条推论：

1. **overlap 越好 → 边界处在飞的工作越多 → 读数膨胀越大。** 拿它评价重叠优化会得到反向结论。
2. **多 rank 的 min/max 相加没有物理含义**，取决于短板是否固定。
3. **`filter(x > 0.0)` 会静默缩小统计范围**，min 变成「活跃子集里的最快」。

替代方案：CUDA Event 打点（异步、不改变执行），或从 trace 里读区间跨度。

## 源码依据

基准 commit `60e039626`（Megatron-LM v0.20.0，2026-08-21 main）：

- `megatron/core/timers.py:109` `Timer`
- `megatron/core/timers.py:143` / `:156` `start()` / `stop()` 里的 `torch.cuda.synchronize()`
- `megatron/core/timers.py:171` `reset()`（注释就写着 `Don't reset _active_time`）
- `megatron/core/timers.py:212` `active_time`
- `megatron/core/timers.py:318` `_get_global_min_max_time()` 里的 `filter(lambda x: x > 0.0, ...)`
"""),
    ],
    "readme": r"""
## 对应文章

- 《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》§4
  https://lrypcy.github.io/2026/09/29/profiling-00-metrics/

## 如何运行

```bash
# 纯标准库，无第三方依赖
jupyter nbconvert --to notebook --execute profiling-00-timer-semantics.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io` 的 `tools/profiling_bench/timer_semantics.py`，
由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **实验一 · 同步点 × 通信占比**：真线程复刻 `Timer` 的 `synchronize()` 语义，
  对比时间线口径与同步式计时的读数，量化膨胀率
- **实验二 · min/max 不可加**：掉队卡固定 vs 短板轮转，两种形态下
  「各项 max 相加」与最慢卡真实值的偏差
- **实验三 · 静默丢弃**：`filter(x > 0.0)` 如何把 min 变成「活跃子集里的最快」
- **实验四 · 占空比失真**：`reset()` 不清 `_active_time` 导致早期迭代的比值虚高

## 说明：为什么绝对值有抖动

实验一用真线程 + `sleep` 模拟设备侧耗时，每个 chunk 的线程调度开销与机器负载直接相关。
脚本会把这一行显式打出来（见 notebook 输出），**以它为准判断本次运行的可信度**。

**看趋势与比值，不要看绝对值。** 膨胀率的单调性（`Tm` 越大膨胀越大）每次运行都成立，
但膨胀率的具体数值不恒定——调度开销越大，膨胀率读数越偏低。因此本 notebook 记录到的
绝对值与博客正文引用的那次不同，这是预期内的。

实验二、三、四是**确定性计算**，无抖动，与博客正文逐位一致（如短板轮转时
「各阶段 max 相加」比最慢卡真实值高 29.6%）。
""",
}
