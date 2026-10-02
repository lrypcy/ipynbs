"""notebook spec: 指标口径对齐（vLLM --save-result 字段映射 + ITL/TPOT 分母）。

对应博客：《性能剖析（02）：推理服务 profiling》
https://lrypcy.github.io/2026/09/29/profiling-02-inference/

依赖 serving_model 的几个符号，所以这个 notebook 会把它们一起摊开。
"""

SIM_DEPS = [
    "DEFAULT_COST", "ServerCfg", "_Req", "arrival_times", "_gamma_sample",
    "pct", "meets_slo", "summarize", "cfg_in_len", "_pct_gain", "_ratio",
    "_finite", "steady_state", "steady_window", "simulate",
]

SPEC = {
    "slug": "profiling-02-serving-schema",
    "title": "指标口径对齐：队列模型输出 ↔ vLLM --save-result 字段，模型给不出的量",
    "timeout": 600,
    "cells": [
        ("md", r"""
# 指标口径对齐

> 对应博客篇目：《[性能剖析（02）：推理服务 profiling](https://lrypcy.github.io/2026/09/29/profiling-02-inference/)》
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/serving_schema.py`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。

## 为什么需要这一步

「TTFT」「吞吐」「goodput」这些词在工具之间**同名不同义**。不逐字段钉一遍，就会拿 A 工具的字段去比 B 工具的字段，然后得出一个自己都不知道在比什么的结论。

本 notebook 做三件事：

1. **字段映射表**：每个能对齐的字段标明来源与可信度（已核验 / 条件出现 / 版本相关 / 推断 / 自造）。
2. **能力边界表**：明确列出**队列模型给不出的量**。报得出来的和报不出来的都要写清楚，否则读者会以为模型能替代压测。
3. **ITL 与 TPOT 的分母演示**：这两个量在流式输出下不是一回事，差多少有一个闭式表达，**不依赖任何仿真**。

## 字段名的可信度分级（不是所有名字都同样硬）

| 标签 | 含义 |
|:---|:---|
| 已核验 | vLLM 文档或 `benchmark_serving.py` 源码里直接出现的名字 |
| 条件出现 | 只有 `--metric-percentiles` 里包含该分位才有；**默认值是 99**，所以 `p90_*` 默认跑出来的结果文件里没有这个键 |
| 版本相关 | 来自结果文件的实际用法，不同版本可能增删；跨版本对比前先 `--save-result` 跑一次看实际键名 |
| 推断 | 由文档描述推出，未经一手确认。**不要拿推断项做自动化断言** |
| 自造 | 本系列自己的合成量，工具里没有对应字段 |
"""),

        ("md", r"""
## 第 0 步 · 环境与依赖

这个 notebook 的第 3 节要真跑一次队列模型，所以先把 `serving_model` 里用到的符号摊开（与 [profiling-02-serving-model](../profiling-02-serving-model/) 同一份代码）。
"""),

        ("imports_py", (["tools/profiling_bench/serving_schema.py",
                         "tools/profiling_bench/serving_model.py"],
                        ("profiling_bench", "megatron_bench"))),
        ("raw_py", "tools/profiling_bench/serving_model.py", SIM_DEPS, ""),

        ("md", r"""
## 第 1 节 · 字段映射：队列模型输出 ↔ vLLM `--save-result`

三组：

- **A 组** —— 队列模型能给，可与压测输出逐字段对照
- **B 组** —— 需要真实流式时序，队列模型**给不出**
- **C 组** —— 服务端才有，客户端压测看不到
"""),

        ("raw_py", "tools/profiling_bench/serving_schema.py",
         ["FIELD_MAP", "GROUP_TITLE", "CONF_TAG", "print_field_map"],
         "print_field_map()"),

        ("md", r"""
> **不要把 B/C 组的量当成「也是模型算的」。** 队列模型回答「拐点在哪、由什么决定」；它**不回答「ITL 的尾部形态」**。

## 第 2 节 · ITL 与 TPOT 的分母：同一个流，两个数可以差几倍

**这是本文最容易读错的一处。** vLLM 官方文档给的定义：

- **ITL** 记录**相邻两次流式输出**之间的间隔。一个流式输出里如果含多个 token（投机解码下一个 engine step 接受多个 draft token 时就会这样），**这些 token 不额外产生 ITL 样本**。
- **TPOT** 是**每条请求算一次**、排除首 token 后摊到每个输出 token 上，再跨请求聚合。

于是两者相差一个分母：

$$\frac{\text{mean\_ITL}}{\text{TPOT}} = \frac{n_{\text{out}} - 1}{n_{\text{streamed\_gaps}}}$$
"""),

        ("raw_py", "tools/profiling_bench/serving_schema.py",
         ["itl_vs_tpot", "print_itl_demo"], "print_itl_demo()"),

        ("md", r"""
> **实操含义**：开投机解码后每个流式输出接受的 token 数上升，$n_{\text{streamed\_gaps}}$ 下降，**mean ITL 按 $(n_{\text{out}}-1)/n_{\text{gaps}}$ 同比例放大**，而 TPOT（每条请求摊到每个 token）不受这个口径影响。
>
> **只盯 ITL 会把这个口径放大读成「服务变慢了」。**

## 第 3 节 · 落盘与逐字段回读

**这步在做什么**：跑一遍队列模型 → 按 `--save-result` 的形状组织成 JSON → 回读，断言 A 组要求的字段一个不缺。
"""),

        ("raw_py", "tools/profiling_bench/serving_schema.py",
         ["build_result", "check"], "ok = check()\nprint('\\nA 组字段校验: %s' % ('通过' if ok else '失败'))"),

        ("md", r"""
> **一个容易看错的地方**：goodput 结构里的 `ttft_ms` / `tpot_ms` 是 **SLO 阈值**，不是测量值。别把它和逐请求真值数组 `ttfts` 看混。

## 小结

- **同名不同义是常态**，跨工具比数之前必须逐字段对齐，并对齐到「可信度」这一级。
- **模型给不出的量要写出来**（B 组：ITL 族；C 组：服务端指标）。不写就会有人以为队列模型能替代压测。
- **ITL 与 TPOT 差一个分母**，比值 = $(n_{\text{out}}-1)/n_{\text{streamed\_gaps}}$。投机解码会让 mean ITL 上升而每请求生成速度不变。
- **推断项不要用于自动化断言**——`request_goodput` 的字段名就是推断的。

## 参考文献

- [vLLM 文档](https://docs.vllm.ai/) —— `--save-result` / `--save-detailed` / `--metric-percentiles` / `--goodput` 的字段定义
- vLLM `benchmarks/benchmark_serving.py`
"""),
    ],
    "readme": r"""
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
""",
}
