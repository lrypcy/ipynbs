"""notebook spec: 多 rank trace 的聚合分析（真值反算自校验）。

对应博客：《性能剖析（01）：训练栈 profiling》
https://lrypcy.github.io/2026/09/29/profiling-01-training/
"""

SPEC = {
    "slug": "profiling-01-trace-agg",
    "title": "多 rank trace 的聚合分析：时间分解、idle 三分、overlap 双分母、掉队卡",
    "timeout": 900,
    "cells": [
        ("md", r"""
# 多 rank trace 的聚合分析

> 对应博客篇目：《[性能剖析（01）：训练栈 profiling](https://lrypcy.github.io/2026/09/29/profiling-01-training/)》
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/trace_agg.py`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。纯标准库，无第三方依赖。

## 这个 notebook 回答什么

一次训练迭代慢在哪？三个核心量——**overlap、idle、straggler**。

**为什么必须自校验**：这三个量都有多种同样「说得通」的算法，不同算法在边缘 case 下差出几十个百分点很常见。没有真值反算，一份看起来很漂亮的报告可能整体是错的。

所以本 notebook 的做法是：**按预置真值构造一份多 rank chrome trace，再分析它，把「量出来的值」和「造进去的真值」并排打印。** 分析实现有 bug 时，两者对不上。

## 口径说明（这些选择都会影响结论）

- **comm**：名字含 nccl / AllReduce / AllGather / ReduceScatter / SendRecv 的 kernel，加上 `cat` 为 `gpu_memcpy` / `communication` 的事件
- **compute**：设备侧其余 kernel
- **idle**：迭代墙钟里既无 compute 也无 comm 的时间
- **overlap**：comm 区间与 compute 区间的**交集长度**（区间代数，不是「各取一半」的估算）

### overlap 有两个分母，给出的是两个完全不同的结论

| 口径 | 含义 |
|:---|:---|
| `overlap / comm_total` | 「通信有多少被藏住了」（效率视角） |
| `overlap / iter_time` | 「通信还能再藏多少」（收益上界视角） |

报告里两个都给，并明确标注是哪一种。这是 HTA `get_comm_comp_overlap()` 的语义。

### idle 三分（沿用 HTA 的官方措辞）

| 成因 | 判据 | 修法 |
|:---|:---|:---|
| `host-wait` | 空隙期间 host 在跑非 launch 的活（dataloader / python_function） | 查数据管线 |
| `kernel-wait` | 空隙前 host 已 post 了下一个 kernel，设备没开工 | 查流内依赖 / launch 开销 |
| `unknown` | 窗口内看不到任何 host 侧解释 | 通常是等对端，或 host 侧采样缺失 |

三者的修法完全不同，混在一起统计会把人引向错误的优化方向。

> **注意本实现的判据是自己的**（按 host 侧事件的时间重叠关系推断），与 HTA 基于 Kineto CPU 侧算子链的推断规则**不完全一致**。跨工具比数之前必须先对齐判据，否则同一个 trace 会给出两组不同的三分占比。
"""),

        ("imports_py", (["tools/profiling_bench/trace_agg.py"],
                        ("profiling_bench", "megatron_bench"))),

        ("md", r"""
## 第 1 步 · 区间代数：并集、交集、补集

**这步在做什么**：三个基础量，后面所有分析都建立在它们之上。

- `union`：合并重叠区间，得到按起点排序的不相交区间
- `intersection_len`：两个区间集合的**交集总长度**——先各自并集化，再双指针求交
- `sparse_union_len`：sweep line 求并集长度，比 pairwise 快，用于大 trace

`intersection_len` 是 overlap 的定义所在：**不是「各取一半」的估算，是真正的区间交集**。
"""),

        ("raw_py", "tools/profiling_bench/trace_agg.py",
         ["US", "union", "total_len", "intersection_len", "sparse_union_len"],
         ""),

        ("md", r"""
## 第 2 步 · 事件分类：什么算 comm、什么算 compute

**这步在做什么**：把 chrome trace 的事件按 `cat` 与 `name` 分成六类。

有一个容易漏的点：`user_annotation` 里名字含 `iteration` 的事件是**迭代标记**，由 `analyze()` 单独处理，**不参与 compute/comm 统计**。真实训练 trace 都有这层标注（NVTX / `record_function`）。
"""),

        ("raw_py", "tools/profiling_bench/trace_agg.py",
         ["COMM_HINTS", "classify", "_ev"],
         ""),

        ("md", r"""
## 第 3 步 · 合成 trace：把真值造进去

**这步在做什么**：按预置真值构造一份 8 rank × 3 迭代的 chrome trace。每个 rank、每次迭代的时间线是：

- compute 切成 4 个 kernel 铺满 $[0, c]$
- comm 里 `overlap_frac · m` 落在 compute 区间内部（被藏住），其余 $(1 - \text{overlap\_frac}) \cdot m$ 暴露在外
- 之后再加 idle 空隙，按 50/30/20 分成三种成因

idle 被切成 10 个空隙，让 50/30/20 不被量化吃掉；空隙之间用小 kernel 隔开，因为真实 trace 里的空隙不会互相贴着。
"""),

        ("raw_py", "tools/profiling_bench/trace_agg.py",
         ["build_synthetic"],
         "trace, truth = build_synthetic(n_ranks=8, iters=3, straggler=3, overlap_frac=0.5)\n"
         "print('合成 trace：%d 个事件' % len(trace['traceEvents']))"),

        ("md", r"""
## 第 4 步 · 分析：迭代窗口、idle 三分、掉队卡

**这步在做什么**：把上面造好的 trace 算成结构化结果。

**分段策略很关键**：

- trace 里有名字含 `iteration` 的 `user_annotation` 区间 → 以它为迭代窗口，逐个迭代分析后取**平均**。真实训练 trace 都有这层标注。
- 没有标注 → 退化成「整段跨度」口径，此时**跨迭代的间隙会被算成 idle**，结果系统性偏大。这种情况会打印警告，**不静默给数**。
"""),

        ("raw_py", "tools/profiling_bench/trace_agg.py",
         ["_complement", "_covered", "_triage_split", "load_trace", "analyze"],
         "res = analyze(trace, verbose=True)"),

        ("md", r"""
## 第 5 步 · 真值反算：量出来的和造进去的差多少

**这步在做什么**：把六个核心量并排打印。这是整个 notebook 的可信度来源。
"""),

        ("raw_py", "tools/profiling_bench/trace_agg.py",
         ["pct", "report_selftest"],
         "ok = report_selftest(res, truth)\nprint('\\n自校验结果: %s' % ('通过' if ok else '未通过'))"),

        ("md", r"""
> **怎么读这一格**：
>
> - 六个主量的最大相对误差应该在 2% 以内（本实现是**确定性计算**，每次运行逐位一致）。
> - **idle 三分**的偏差容忍度更松（5 us），因为它依赖 `_triage_split` 的 `min_fill` 判据，量化到具体空隙上必然有偏差。
> - **逐迭代最慢卡**那一栏是本节最有实践价值的输出：每轮都是同一张卡 → 固定短板，去查那张卡本身（NIC / 拓扑 / 掉频）；每轮换卡 → 多半是资源争抢或负载不均。**这个区分靠「平均值」永远看不出来。**
"""),

        ("md", r"""
## 第 6 步 · overlap 率扫描：两个分母怎么分道扬镳

**这步在做什么**：把重叠率从 0% 扫到 100%，同时看两个分母的反应。
"""),

        ("code", r"""
print("=" * 74)
print("重叠率扫描：两个分母给出的结论差距")
print("=" * 74)
print("%8s %14s %14s %12s" % ("overlap", "ov/comm (%)", "ov/iter (%)", "iter(us)"))
for of in (0.0, 0.25, 0.5, 0.75, 1.0):
    tr, _ = build_synthetic(n_ranks=8, straggler=None, overlap_frac=of)
    rs = analyze(tr)
    d = rs["per_rank"][0]
    print("%7.0f%% %14.1f %14.1f %12.1f"
          % (of * 100, pct(d["overlap_us"], d["comm_us"]),
             pct(d["overlap_us"], d["iter_us"]), d["iter_us"]))
"""),

        ("md", r"""
> **读法**：随着重叠变好，`ov/comm` 趋近 100%（通信基本藏完），而 `ov/iter` 反而「看起来很低」——因为分母是迭代时长，它本来就不该接近 100%。
>
> 把这两个分母混用，就会出现「我们 overlap 只有 20%，还能大改」这种结论：**那个 20% 是 `ov/iter`，真正该看的是 `ov/comm`。**
"""),

        ("md", r"""
## 第 7 步 · 短板形态对比：固定 vs 轮转

**这步在做什么**：同样的平均拖慢量，一种让短板固定在一张卡上，一种每轮换卡。看两种形态下逐迭代的形态差多少。
"""),

        ("code", r"""
import statistics

print("=" * 74)
print("短板形态对比：固定 vs 轮转（同样的平均拖慢量）")
print("=" * 74)
for label, s in (("固定 rank 3", 3), ("每轮换卡", -1)):
    tr, _ = build_synthetic(n_ranks=8, straggler=s)
    rs = analyze(tr)
    print("\n[%s]" % label)
    n_iter = max(len(rs["per_rank"][r]["iter_durs"]) for r in rs["ranks"])
    for i in range(n_iter):
        d = {r: rs["per_rank"][r]["iter_durs"][i] for r in rs["ranks"]
             if len(rs["per_rank"][r]["iter_durs"]) > i}
        if not d:
            continue
        sr = max(d, key=d.get)
        mv = statistics.median(d.values())
        print("   迭代 %d: 最慢 rank %d  %.1f us  (+%.2f%% vs 中位数)"
              % (i, sr, d[sr], (d[sr] / mv - 1.0) * 100.0))
    print("   均值 %.1f / 中位数 %.1f / 最大 %.1f us"
          % (rs["iter_mean_us"], rs["iter_median_us"], rs["iter_max_us"]))
"""),

        ("md", r"""
> **两种形态的均值/中位数/最大值几乎一样，但它们指向完全不同的排查方向。** 这就是为什么只报聚合统计量不够——必须逐迭代看「最慢的是哪张卡」。
>
> 分布式里整步时长由**最慢卡**决定，所以「均值」这个数永远是乐观的：它把掉队卡的超时摊给了其它卡。

## 怎么用它读真实 trace

同一份分析代码也能直接吃真实 trace——它吃的是 Perfetto / kineto 的 chrome trace 格式（`traceEvents` + `ph == "X"`），也就是：

- PyTorch `torch.profiler` 导出的 `torch_profile/rank-N.json.gz`
- Megatron `--profile` 的产物（注意配置字段名其实是 `use_nsys_profiler`，`dest` 才叫 `profile`）
- nsys 的 `.json` 导出

```python
# res = analyze(load_trace("torch_profile/rank-0.json.gz"), verbose=True)
# report_real(res)
```

## 小结

- **必须自校验**：overlap / idle / straggler 三个量都有多种说得通的算法，没有真值反算的报告可能整体是错的。
- **overlap 两个分母不能混用**：`ov/comm` 是效率，`ov/iter` 是上界。
- **idle 三分要分开看**：host-wait / kernel-wait / unknown 的修法完全不同。
- **短板形态要逐迭代看**：固定短板查那张卡，轮转短板查资源争抢。
"""),
    ],
    "readme": r"""
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
""",
}
