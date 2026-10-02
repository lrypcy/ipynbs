"""notebook spec: goodput 与 MFU 的分工。

对应博客：《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》
https://lrypcy.github.io/2026/09/29/profiling-00-metrics/
"""

SPEC = {
    "slug": "profiling-00-goodput",
    "title": "goodput 与 MFU 的分工：算得快不快 vs 时间用在正事上没有",
    "timeout": 300,
    "cells": [
        ("md", r"""
# goodput 与 MFU 的分工

> 对应博客篇目：《[性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准](https://lrypcy.github.io/2026/09/29/profiling-00-metrics/)》§5
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/goodput_calc.py`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。纯 CPU，无第三方依赖。

## 这个 notebook 回答什么

$$\text{MFU} = \frac{\text{模型 FLOPs}}{\text{时间} \times \text{峰值算力}} \qquad\qquad \text{goodput} = \frac{\text{计入 goodput 的时长}}{\text{挂钟时长}}$$

- **MFU** 回答「算得快不快」
- **goodput** 回答「时间用在正事上没有」

两者可以背离，且**背离的方向直接告诉你该优化哪一边**：

| 形态 | 含义 | 该做什么 |
|:---|:---|:---|
| MFU 高、goodput 低 | 计算很猛，但大量时间花在数据加载 / 通信漏出 / 日志上 | 查 `data_loading` 与 `communication` span，不要换 kernel |
| MFU 低、goodput 高 | 时间都在 forward_backward 里，但每步内部访存受限 | 查算子与并行切分，不要缩减非计算开销 |

## 口径来源

Megatron v0.20 的原生 OTel 观测栈（commit `60e039626`）：

- `megatron/core/telemetry/span_groups.py` —— `MegatronSpanGroup` 的 span 分组
- `megatron/core/telemetry/training_metrics.py` —— 8 个 OTel 指标
- 各埋点处用 `is_goodput_span=True` 标记「算正事」的 span：
  `training/checkpointing.py:810` / `:896` / `:2515`、`training/training.py:4482`
"""),

        ("md", r"""
## 第 1 步 · 把一个训练步拆成 span，算 goodput

**这步在做什么**：把一次稳定迭代拆成叶子 span，按 `is_goodput_span` 标记累加。

一个容易犯的错：`megatron.step` 是**父 span**（时长 = 整个迭代），把父子的时长相加就重复计了。下面只列叶子 span，它们的和恰好等于父 span。

span 时长是**量级示意**（依据 Megatron 的 span 语义与 step 时间构成），重点是**口径与算法**，不是绝对数。
"""),

        ("raw_py", "tools/profiling_bench/goodput_calc.py",
         ["_fmt_span", "steady_state_iteration", "checkpoint_cost", "report"],
         "report()"),

        ("md", r"""
## 第 2 步 · 两种背离形态

上面已经把两种形态列出来了。这里再量化其中一种：把 `data_loading` 与 `communication` 的漏出压缩掉，goodput 能回到多少。

关键在于——**分子没动**（这两项本来就不计 goodput），只是分母变小，所以 goodput 自然回升。

> **goodput 的变化只反映「正事占比」，不能推出 FLOPs 利用率也同步改善。** 这是两个指标必须一起看的根本原因。

## 小结

- **只报 MFU** 会漏掉「算得快但整天在等数据」——那种情况下 MFU 看起来很漂亮。
- **只报 goodput** 会把「时间都在算、但算得很慢」当成健康——访存受限的 kernel 照样能拿到 100% goodput。

两个一起看，且**看它们背离的方向**：MFU 高 / goodput 低 → 去查非计算开销；MFU 低 / goodput 高 → 去查算子效率。

## 参考文献

- Megatron-LM `megatron/core/telemetry/`（基准 commit `60e039626`，v0.20.0）
- [OpenTelemetry 规范](https://opentelemetry.io/docs/specs/otel/)
"""),
    ],
    "readme": r"""
## 对应文章

- 《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》§5
  https://lrypcy.github.io/2026/09/29/profiling-00-metrics/

## 如何运行

```bash
# 纯标准库，无第三方依赖，秒级完成
jupyter nbconvert --to notebook --execute profiling-00-goodput.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io` 的 `tools/profiling_bench/goodput_calc.py`，
由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **span 拆分与 goodput**：把一个稳定迭代拆成叶子 span，按 `is_goodput_span` 累加
- **checkpoint 摊薄**：ckpt 间隔从「不存」到 500 步，有效吞吐掉多少、goodput 动多少
- **两种背离形态**：MFU 高 / goodput 低 与 MFU 低 / goodput 高 各指向什么优化动作
- **量化形态一**：把 data_loading 与 communication 漏出压缩到 1/4，goodput 回到多少

## 关键结论

- 存 checkpoint 越勤，**有效吞吐掉得越多，而 goodput 几乎不动**（甚至微升，因为 ckpt 算正事）
- 压缩非计算开销时 **goodput 的分子没动**，只是分母变小 —— 所以 goodput 变好不能推出 MFU 也变好
""",
}
