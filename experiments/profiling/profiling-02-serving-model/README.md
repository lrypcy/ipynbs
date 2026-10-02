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
