## 对应文章

- 《性能剖析（02）：推理服务 profiling》
  https://lrypcy.github.io/2026/09/29/profiling-02-inference/

## 如何运行

```bash
# 纯 CPU，无第三方依赖
jupyter nbconvert --to notebook --execute profiling-02-serving-schema.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io` 的 `tools/profiling_bench/serving_schema.py`
（第 3 节用到的队列模型符号取自同目录的 `serving_model.py`），
由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **第 1 节 · 字段映射表**：A / B / C 三组，逐字段标可信度
  （已核验 / 条件出现 / 版本相关 / 推断 / 自造）
- **第 2 节 · ITL vs TPOT 的分母**：固定 TTFT = 100 ms、E2EL = 180 ms，
  只改每个流式输出带几个 token，看两个数差几倍
- **第 3 节 · 逐字段回读校验**：跑一遍 → 落盘 → 断言 A 组字段一个不缺

## 关键结论

- **`p90_*` 是条件出现的字段**：`--metric-percentiles` 默认值是 99，所以默认跑出来的
  结果文件里**没有** `p90_ttft_ms` 这个键。要它得显式加进参数。
- **goodput 结构里的 `ttft_ms` / `tpot_ms` 是 SLO 阈值**，不是测量值。
- **ITL / TPOT = (n_out − 1) / n_streamed_gaps**：每个输出恰好一个 token 时比值为 1，
  这正是「ITL 和 TPOT 差不多」这句经验话成立的**唯一**条件。
- **`request_goodput` 的字段名是推断的**，不要拿它做自动化断言。

## 相关

队列模型本体（TTFT/TPOT 分位、goodput、拐点判读）见
[profiling-02-serving-model](../profiling-02-serving-model/)。
