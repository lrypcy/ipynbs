## 对应文章

- 《性能剖析（01）：训练栈 profiling》
  https://lrypcy.github.io/2026/09/29/profiling-01-training/

## 如何运行

```bash
# 纯标准库（argparse / gzip / json / statistics / bisect），无第三方依赖
jupyter nbconvert --to notebook --execute profiling-01-trace-agg.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io` 的 `tools/profiling_bench/trace_agg.py`，
由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **区间代数**：并集 / 交集 / 补集，overlap 的定义基础
- **事件分类**：comm / compute / host-work / host-launch / marker
- **合成 trace**：按预置真值构造 8 rank × 3 迭代的 chrome trace
- **真值反算自校验**：六个核心量并排对比（最大相对误差应 < 2%）
- **idle 三分**：host-wait / kernel-wait / unknown，判据与修法
- **overlap 双分母**：`ov/comm`（效率）vs `ov/iter`（上界），随重叠率扫描分道扬镳
- **短板形态对比**：固定短板 vs 轮转短板，聚合统计量相同但排查方向完全不同

## 关键数字

- 真值反算六个主量最大相对误差 **0.00%**（确定性计算，每次运行逐位一致）
- idle 三分绝对偏差最大 < 1 us（依赖 `min_fill` 判据，容忍度 5 us）
- 短板轮转时「各阶段 max 相加」比最慢卡真实值高 **29.6%**（见
  [profiling-00-timer-semantics](../profiling-00-timer-semantics/)）

## 读真实 trace

同一份代码可以直接吃 Perfetto / kineto 的 chrome trace：

```python
res = analyze(load_trace("torch_profile/rank-0.json.gz"), verbose=True)
report_real(res)
```
