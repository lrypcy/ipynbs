"""notebook spec: 推理服务的队列模型。

对应博客：《性能剖析（02）：推理服务 profiling》
https://lrypcy.github.io/2026/09/29/profiling-02-inference/

注意：代价常数是示意值，notebook 只主张结构。
"""

SIM_NAMES_COMMON = [
    "DEFAULT_COST", "ServerCfg", "_Req", "arrival_times", "_gamma_sample",
    "simulate", "pct", "meets_slo", "summarize", "cfg_in_len",
    "_pct_gain", "_ratio", "_finite", "_check", "selftest", "hygiene_ok",
    "steady_state", "steady_window",
]
SIM_NAMES_SWEEP = [
    "SWEEP_HEADER", "sweep_concurrency", "sweep_rate", "print_sweep",
    "throughput_knee", "capacity_boundary",
]
SIM_NAMES_SECTIONS = [
    "section_preempt_ab", "section_loop_compare", "section_pattern_compare",
    "prefix_cache_demo",
]

SPEC = {
    "slug": "profiling-02-serving-model",
    "title": "推理服务的队列模型：TTFT / TPOT 分位、goodput、拐点由什么决定",
    "timeout": 1800,
    "cells": [
        ("md", r"""
# 推理服务的队列模型

> 对应博客篇目：《[性能剖析（02）：推理服务 profiling](https://lrypcy.github.io/2026/09/29/profiling-02-inference/)》
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/serving_model.py`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。纯 CPU，秒级到十几秒跑完。

## 这个 notebook 回答什么

服务侧的一堆读数（TTFT、TPOT、ITL、E2EL、吞吐、goodput）在压测报告里是一组平行数字，**看不出它们由什么决定**。把它们写成一个可解析的排队系统之后，三件事就能算出来而不是猜出来：

1. **拐点位置由什么决定。** 吞吐什么时候走平、TTFT 什么时候起飞，不是调参调出来的，是「步长 → 批量」这条曲线的膝盖位置决定的。
2. **两种上限谁先 bind。** `max_num_seqs`（调度槽位）与 KV 容量（块池）是两个独立约束，长请求会让后者先到。**压着槽位加并发不加吞吐，就是 KV 在 bind。**
3. **goodput 与吞吐差多少。** 达标率随到达率单调下降，「最大不达标率仍满足 $\alpha$ 的到达率」才是容量边界——它和峰值吞吐经常差一倍以上。

> **代价常数是示意值**（一组自洽的默认参数），本 notebook 只主张**结构**：哪些量互为因果、拐点由什么位置决定。**绝对数值不代表任何具体硬件或模型。**
"""),

        ("md", r"""
## 第 0 步 · 代价模型：先平后斜的膝盖
**这步在做什么**：单步时间按 roofline 写，取两种资源的较大者：

$$\text{prefill\_step\_ms}(n_{\text{tok}}) = \max(\text{PREFILL\_FLOOR\_MS},\ \text{PREFILL\_MS\_PER\_TOK} \cdot n_{\text{tok}})$$
$$\text{decode\_step\_ms}(n_{\text{seq}}) = \max(\text{DECODE\_FLOOR\_MS},\ \text{DECODE\_MS\_PER\_SEQ} \cdot n_{\text{seq}})$$

两者都先由一个与批量无关的**下限**决定（权重搬运、kernel launch、采样），批量大到超过膝盖 $n^* = \text{FLOOR}/\text{PER\_UNIT}$ 之后才转成随批量线性增长。

**膝盖两侧的优化手段完全不同**：

| 位置 | 单步时间 | 加并发的效果 |
|:---|:---|:---|
| 膝盖左侧 | 与批量无关 | 加并发几乎免费，吞吐线性上升，TPOT 不变 |
| 膝盖右侧 | $\propto$ 批量 | 吞吐饱和在上限 $1000/\text{PER\_UNIT}$，TPOT 线性上升 |

chunked prefill 下 prefill 与 decode 在同一个 step 内**串行**执行，所以单步时间是两段之和。
"""),

        ("imports_py", (["tools/profiling_bench/serving_model.py"],
                        ("profiling_bench", "megatron_bench"))),
        ("raw_py", "tools/profiling_bench/serving_model.py",
         ["DEFAULT_COST", "ServerCfg", "_Req", "arrival_times", "_gamma_sample"],
         ""),

        ("md", r"""
## 第 1 步 · 到达模型：constant / Poisson / gamma

**这步在做什么**：三种到达模式**均值相同、分布不同**——这正是「三者不可互推」的来源。

- `constant`：等间隔，CV = 0。平滑到不真实，但它是复现性最好的基线。
- `poisson`：指数间隔，CV = 1。
- `gamma`：Gamma(shape=$k$, rate=$k\lambda$)，均值 = $1/\lambda$，CV = $1/\sqrt{k}$。$k=1$ 退化成 Poisson；$k>1$ 更规则；$k<1$ 更突发。

> **一处容易踩的坑**：工具的旋钮名字和分布的平滑程度不一定同向。不同实现里「burstiness」可能映射到 shape 本身，也可能映射到它的倒数。**所以这里不按旋钮名推断，而是把实测 CV 打出来**——换工具时先看 CV 对不对得上，再比数。
"""),

        ("code", r"""
print("%-28s %10s %10s" % ("到达模式", "均值 (s)", "实测 CV"))
print("-" * 52)
for label, pattern, shape in (("constant（等间隔）", "constant", 1.0),
                              ("poisson（k=1）", "poisson", 1.0),
                              ("gamma, k=4（规则）", "gamma", 4.0),
                              ("gamma, k=0.35（突发）", "gamma", 0.35)):
    t, cv = arrival_times(pattern, 5.0, 2000, seed=0, shape=shape)
    mean_gap = t[-1] / (len(t) - 1)
    print("%-28s %10.5f %10.3f" % (label, mean_gap, cv))
print("\n四种模式的平均间隔都收敛到 1/5.0 = 0.2 s 附近，但 CV 从 0 到 1.9 差了一个量级。")
print("均值把这个信息完全抹掉了 —— 这就是「同一 λ 下队列行为可以差很多」的来源。")
"""),

        ("md", r"""
## 第 2 步 · 仿真内核：槽位与 KV 两个约束分开建模

**这步在做什么**：跑一遍连续批处理调度。每个 step 的顺序是固定的，**顺序本身就是语义**：

1. 空闲则把时间跳到下一个到达时刻（不能凭空推进）
2. **准入**：从等待队列头部取请求，同时受槽位与 KV 池限制
3. 填 prefill token 预算（从已在 prefill 的请求队头开始）
4. 推进 decode（每个在跑的请求 +1 token）
5. 结算本步代价、推进时间、收尾

第 2 步与第 3 步之间不做抢占，抢占只发生在「准入失败」时。
"""),

        ("raw_py", "tools/profiling_bench/serving_model.py",
         ["pct", "meets_slo", "summarize", "cfg_in_len", "_pct_gain",
          "_ratio", "_finite", "steady_state", "steady_window", "simulate"],
         ""),

        ("md", r"""
## 第 3 步 · 自校验：把纸面推导和仿真结果对账

**这步在做什么**：这是整个模型的**可信度来源**。四个可以从纸面推出来的结论：

1. **无争用单请求**：`TTFT == prefill_step_ms(in_len)`、`TPOT == decode_step_ms(1)`、`E2EL == TTFT + (out_len-1)·TPOT`，三者都是**精确值**，不是近似。
2. **同批到达 N 个请求**（不超预算、不被槽位挡住）：N 个请求的 TTFT **完全相同**，且等于 `prefill_step_ms(N·in_len)`——prefill 是批处理，不是逐个串行。
3. **吞吐饱和**：decode 段 `decode_ms_per_seq > 0` 时，输出吞吐严格小于上限 $1000/\text{decode\_ms\_per\_seq}$，且随并发单调不减。
4. **卫生条件**：所有时延 $\ge 0$、利用率/达标率 $\in [0,1]$、P99/P50 $\ge 1$。

> 第 4 条不是形式主义：`tools/rl_framework_budget.py` 第一版曾跑出**负空闲率**。比率类量的取值范围必须显式断言。
"""),

        ("raw_py", "tools/profiling_bench/serving_model.py",
         ["_check", "hygiene_ok", "selftest"], "selftest()"),

        ("md", r"""
## 第 3.5 步 · 三个对照实验的实现

**这步在做什么**：把后面第 6～9 步要用的对照实验函数一次性摊开——它们的实现都在同一个文件里，放在这里是为了让后面每个实验单元只留「调用 + 读数」。
"""),

        ("raw_py", "tools/profiling_bench/serving_model.py",
         SIM_NAMES_SECTIONS, ""),

        ("md", r"""
## 第 4 步 · 并发扫描（closed loop）：拐点判读

**这步在做什么**：固定并发数扫一遍，看吞吐什么时候走平、TTFT 什么时候起飞。

两个口径并列给出：

- **「全」= 整个压测段**
- **「稳态」= 丢掉第一波在途请求（爬坡段）**

**两列差多少，就是「冷启动有多慢」被算进分位数里的程度。**

> **闭环比开环好看，这是工具的问题不是服务的问题。** closed loop 下等待队列永远是空的（在途数被客户端卡住），所以 TTFT 里被排队那一块是**测不到**的——它只反映「prefill 本身花多久」。真实系统的 TTFT 主要输在排队上，**那部分只有 open loop 能看见**。
"""),

        ("raw_py", "tools/profiling_bench/serving_model.py",
         SIM_NAMES_SWEEP,
         "CFG = dict(max_num_batched_tokens=2048, kv_token_capacity=262144,\n"
         "           block_size=16, preempt=True)\n"
         "rows_c = sweep_concurrency([1, 2, 4, 8, 16, 32, 64, 128, 256, 512],\n"
         "                          in_len=1024, out_len=128,\n"
         "                          slo_ttft_ms=300.0, slo_tpot_ms=30.0,\n"
         "                          cfg_kwargs=CFG)\n"
         "print_sweep(rows_c, '并发扫描（closed loop，固定并发数）', 300.0, 30.0)"),

        ("md", r"""
> **怎么读拐点判读那一段**：
>
> - **吞吐饱和** = 边际收益 < 5% 的那一档。若扫描范围内没出现，说明还可以继续加并发。
> - **容量边界 goodput** = 达标率首次跌破 90% 之前的那一档。**它和峰值吞吐经常差一倍以上**——峰值吞吐那个数在容量边界之外，是不可用的。
> - **P99/P50** 是唯一能把「正在变饱和」和「已经饱和」区分开的读数。工具不报这个比值，但它是本系列自造的合成量。
"""),

        ("md", r"""
## 第 5 步 · 到达率扫描（open loop）：不达标率是显式的

**这步在做什么**：固定到达率扫一遍。closed loop 看不到的排队延迟，这一档能看到。
"""),

        ("code", r"""
rows_r = sweep_rate([1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32],
                    in_len=1024, out_len=128, n_prompts=600,
                    slo_ttft_ms=300.0, slo_tpot_ms=30.0,
                    cfg_kwargs=CFG, pattern="poisson", seed=0)
print_sweep(rows_r, "到达率扫描（open loop，poisson）", 300.0, 30.0)
"""),

        ("md", r"""
## 第 6 步 · 对照实验：抢占（recompute）值多少吞吐

**这步在做什么**：抢占开 / 关各跑一遍。

设计上遵循一条教训——**先确认基线归零**：并发数还没超过 KV 池装得下的路数时，抢占开关应该**完全不影响结果**（因为一次抢占都不会发生）。**如果基线档位上两列不相等，说明脚本本身有问题，结论一律不采信。**

`max_num_seqs` 与 KV 容量是两个独立约束：压着槽位加并发不加吞吐，就是 **KV 在 bind**。
"""),

        ("code", r"""
section_preempt_ab(in_len=1024, out_len=128,
                  kv_token_capacity=262144,
                  max_num_batched_tokens=8192)
"""),

        ("md", r"""
> **本模型只实现 recompute 型抢占。** vLLM 另有 swap 型（把块换出去再换回来），代价低得多，所以这里的跌幅是 recompute 形态的量级，**不能直接外推**。
"""),

        ("md", r"""
## 第 7 步 · closed loop vs open loop：两条曲线不可互推

**这步在做什么**：同一组代价参数下，两种负载生成方式各测一遍峰值。
"""),

        ("code", r"""
section_loop_compare(in_len=1024, out_len=128, n_prompts=800,
                     slo_ttft_ms=300.0, slo_tpot_ms=30.0,
                     cfg_kwargs=CFG)
"""),

        ("md", r"""
> **两条结论**：
>
> 1. **closed loop 的稳态尾部是结构性地平的。** 在途数被客户端卡住，等待队列永远为空，所以 TTFT 里没有「排队」这一项，稳态 p99 与 p50 逐位相等（比值 1.00）。
> 2. **closed loop 的峰值吞吐更高。** 它总能把批次填满；open loop 在低到达率下会空转。**拿 closed loop 的峰值做容量规划，会把容量估高。**
>
> **报峰值吞吐可以用 closed loop，做容量规划必须用 open loop。**

## 第 8 步 · 同一 λ，三种到达分布：goodput 差七成，吞吐只差十几个点

**这步在做什么**：把「分布」这一个变量单独拎出来。
"""),

        ("code", r"""
section_pattern_compare(rate=5.0, in_len=1024, out_len=128, n_prompts=500,
                       slo_ttft_ms=300.0, slo_tpot_ms=30.0,
                       cfg_kwargs=CFG, seed=0)
"""),

        ("md", r"""
> **这是「三者不可互推」的量化版本**：
>
> - **平均吞吐几乎看不出差别**（最大最小只差百分之十几）。原因：突发之间的空档里服务器空转（少干），突发期间超出容量的部分排队（干不完），两个方向的偏差互相抵消。**只看吞吐会得出结论「分布无所谓」。**
> - **goodput 差别很大**：同一组 λ 下能差七成以上。因为达标率对瞬时过载非常敏感，而均值和峰值分位把它平均掉了。
>
> **到达分布这件事，在吞吐上看不见，在 goodput 上一览无余。** 等间隔（CV = 0）是所有分布中最乐观的一个，**拿它做基线会把容量估高**。

## 第 9 步 · 前缀缓存为什么会把 TTFT 读数改掉

**这步在做什么**：解析式，不依赖仿真。命中比例 $f$ 把真正要算的 prefill token 数从 $S$ 降到 $S(1-f)$，于是

$$\frac{\text{TTFT}(f)}{\text{TTFT}(0)} \ge 1 - f \quad (\text{带宽主导时取等}) \qquad\qquad \frac{\text{TTFT}(f)}{\text{TTFT}(0)} \to 1 \quad (\text{命中到 floor 主导时饱和})$$

**它是分母上的乘法，不是模型变快了。**
"""),

        ("code", r"""
prefix_cache_demo(in_len=1024, ratios=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99),
                  cfg_kwargs=CFG)
"""),

        ("md", r"""
> **所以「第二轮压测 TTFT 掉了一半」这个读数里，有一部分是缓存、有一部分才是预热——两者必须分开。** 复测前必须在「换种子 / 每轮之间重启服务 / 用 `vllm bench sweep serve`」里选一条（后者会显式调 `/reset_prefix_cache`）。
>
> 另：warmup（首请求的 kernel 编译与显存分配）和缓存命中是**两个**效应，`--num-warmups` 只处理后者之外的那一个，**不要指望它替你清缓存**。

## 小结

- **拐点不是调参调出来的**，是「步长 → 批量」曲线的膝盖位置决定的，而这个膝盖位置 $\text{FLOOR}/\text{PER\_UNIT}$ 是可以从配置读出来的。
- **两种上限要分开看**：槽位（`max_num_seqs`）与 KV 容量。压着槽位加并发不加吞吐，就是 KV 在 bind。
- **容量边界 ≠ 峰值吞吐**。「达标率首次跌破 $\alpha$ 之前的那一档」才是能用的数。
- **closed loop 与 open loop 不可互推**：前者峰值更高、尾部结构性偏平。报峰值可用 closed loop，做容量规划必须 open loop。
- **到达分布只在 goodput 上可见**，在吞吐上被平均掉了。
- **前缀缓存改的是测量口径**，不是服务速度。

## 参考文献

- [vLLM 文档](https://docs.vllm.ai/) —— `--save-result` 字段、`--goodput` SLO 口径、`benchmark_serving.py`
- Megatron-LM `training/checkpointing.py` 的 `is_goodput_span` 标记（goodput 概念来源）
"""),
    ],
    "readme": r"""
## 对应文章

- 《性能剖析（02）：推理服务 profiling》
  https://lrypcy.github.io/2026/09/29/profiling-02-inference/

## 如何运行

```bash
# 纯 CPU。numpy 只用于 gamma 采样与分位，缺了也能跑（stdlib 回退）
jupyter nbconvert --to notebook --execute profiling-02-serving-model.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io` 的 `tools/profiling_bench/serving_model.py`，
由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **代价模型**：roofline 两段（prefill / decode），先平后斜的膝盖
- **到达模型**：constant / Poisson / Gamma，打印实测 CV
- **自校验**：四个纸面结论与仿真对账（精确解，非近似）
- **并发扫描**（closed loop）：吞吐膝盖、容量边界、P99/P50
- **到达率扫描**（open loop）：不达标率随 λ 的变化
- **抢占 A/B**：先验基线归零，再看 recompute 型抢占的代价
- **closed vs open loop**：两条曲线不可互推
- **到达分布对比**：吞吐只差十几个点，goodput 差七成
- **前缀缓存**：解析式说明它改的是测量口径

## ⚠️ 代价常数是示意值

`DEFAULT_COST` 是一组自洽的默认参数，**绝对数值不代表任何具体硬件或模型**。
本 notebook 只主张**结构**：哪些量互为因果、拐点由什么位置决定。
要换硬件就整组换常数，不要单独调某一个。

## 可信度来源

`selftest()` 把四个可以从纸面推出来的结论和仿真结果对账，全部是**精确解**：

1. 无争用单请求：`TTFT` / `TPOT` / `E2EL` 三个量都有闭式解
2. 同批 N 条：TTFT 全等且等于批 prefill 代价
3. 吞吐严格小于上限 $1000/\text{decode\_ms\_per\_seq}$，且随并发单调不减
4. 卫生条件：时延 $\ge 0$、比率 $\in [0,1]$、P99/P50 $\ge 1$

## 相关

字段口径对齐（`--save-result` 逐字段映射、模型给不出的量、ITL vs TPOT 的分母）见
[profiling-02-serving-schema](../profiling-02-serving-schema/)。
""",
}
