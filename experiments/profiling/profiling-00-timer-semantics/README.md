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
