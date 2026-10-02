"""notebook spec: MFU / HFU 的口径实验。

对应博客：《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》
https://lrypcy.github.io/2026/09/29/profiling-00-metrics/

代码全部从 lrypcy.github.io 的 tools/ 按顶层符号抽取，逐字一致。
"""

CONFIGS_NAMES = ["ModelSpec", "MODELS", "HardwareSpec", "HARDWARE", "BenchConfig"]
FLOPS_NAMES = ["FlopsBreakdown", "forward_flops_per_token",
               "training_flops_per_token", "approx_6nd"]
SHIM = '''
# ---- 模块名垫片：让原脚本里 `configs.X` / `flops.X` 的写法原样可用 ----
import types

configs = types.SimpleNamespace(BenchConfig=BenchConfig, MODELS=MODELS,
                                HARDWARE=HARDWARE)
flops = types.SimpleNamespace(forward_flops_per_token=forward_flops_per_token,
                              training_flops_per_token=training_flops_per_token,
                              approx_6nd=approx_6nd)
'''

SPEC = {
    "slug": "profiling-00-mfu-accounting",
    "title": "MFU / HFU 的口径实验：分子换一下、分母换一下，结果差多少",
    "timeout": 300,
    "cells": [
        ("md", r"""
# MFU / HFU 的口径实验

> 对应博客篇目：《[性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准](https://lrypcy.github.io/2026/09/29/profiling-00-metrics/)》
>
> 代码逐字摘自博客仓库 `tools/profiling_bench/mfu_accounting.py` 与 `tools/megatron_bench/`（[lrypcy/ipynbs](https://github.com/lrypcy/ipynbs)）。
> 全部纯 CPU，只需要 numpy 栈里的标准库部分，秒级跑完。

## 这个 notebook 回答什么

同一行训练日志，可以合法地报成 20.2% 或 128.2% 的 MFU。差异全部来自分母：用的是稠密还是带 2:4 稀疏的规格、是 BF16 还是 FP8、是不是同一代卡。

**MFU 不是一个数，是「分子口径 + 分母口径 + 硬件」的三元组。** 报 MFU 不写分母，等于没报。

本 notebook 分六步把这件事拆开：

| 步骤 | 问题 |
|:---|:---|
| 1 | 分子固定，只换分母，MFU 差多少 |
| 2 | 分子换 recompute 策略，MFU 与 HFU 分道扬镳多少 |
| 3 | 6ND 近似随序列变长失真多少 |
| 4 | 换一代卡，机器平衡点抬高多少 |
| 5 | 一次 forward 的 FLOPs 构成里有多少不是 GEMM |
| 6 | 把官方 log 的 TFLOP/s/GPU 换回 MFU |
"""),

        ("md", r"""
## 第 0 步 · 环境与共享记账

**这步在做什么**：把三个依赖摊平到 notebook 里，之后每个实验单元都是纯粹的「口径对比」。

- `configs`：模型与硬件的标称规格（Llama-3-8B / A100 / H100 / B200），每个数字都带出处。
- `flops`：FLOPs 记账，区分**逐算子精确**与**6ND 近似**两种口径。

`forward_flops_per_token()` 返回的是**单层**的量；乘上 `n_layers` 才是完整模型。lm_head 每次 forward 只算一次，所以它不乘层数——这一点在第 5 步会专门验证。
"""),

        ("imports_py", ["tools/megatron_bench/configs.py", "tools/megatron_bench/flops.py"]),

        ("raw_py", "tools/megatron_bench/configs.py", CONFIGS_NAMES, ""),
        ("raw_py", "tools/megatron_bench/flops.py", FLOPS_NAMES, ""),
        ("code", SHIM),

        ("md", r"""
## 第 1 步 · 分母口径：同一行 log，换个分母就换一个 MFU

**这步在做什么**：分子完全固定（同一份每 token FLOPs），只把分母在六种规格之间换来换去。

入口取的是「官方 log 那一行」的 TFLOP/s/GPU，而不是自己编吞吐——因为这正是读者会遇到的那个数。400 TFLOP/s/GPU 对 8B 模型在 H100 上是常见量级（≈40% MFU）。

峰值来源均为官方产品页（核验日期 2026-09-29）：

- [A100 产品页](https://www.nvidia.com/en-us/data-center/a100/)：BF16 312 TFLOPS（稠密）| 624 TFLOPS\*（稀疏），HBM 2039 GB/s
- [H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)：BF16 1979 TFLOPS\*（= 稠密 989.5），FP8 3958\*（= 稠密 1979），HBM 3.35 TB/s

注意两个页面的排版口径并不一致：A100 把稠密值放前面，H100 只给带 \* 的稀疏值。**照抄首页数字当分母，MFU 直接腰斩。**
"""),

        ("raw_py", "tools/profiling_bench/mfu_accounting.py",
         ["PEAKS", "GEN", "_cfg", "exp_denominator"], "exp_denominator()"),

        ("md", r"""
> **这一步的结论**：A100 那两行给出了 >100% 的「MFU」。这个数本身就是一个诊断——400 TFLOP/s/GPU 超过了 A100 稠密 BF16 的峰值 312，所以这行日志不可能来自 A100。**跨硬件套错分母，得到的是一个物理上不存在的利用率。**
"""),

        ("md", r"""
## 第 2 步 · 分子口径：MFU 不变而 HFU 变，差额就是重计算的代价

**这步在做什么**：分母固定成 H100 稠密 BF16，只改 recompute 策略。

关键在于两个指标的分子定义不同：

- **MFU** 的分子收尾在 Megatron `training.py:1007` 的 `return flops_fwd * 3`，**不含重计算系数**——所以开不开 recompute 这个数都不变。
- **HFU** 按实际发射量算，重算一遍就多一遍。

开满 full recompute 时每 token 从 $6P$ 变成 $8P$，比值恒为 $4/3$。**想评价「重计算划不划算」得自己换算成 HFU，拿 MFU 去问是问错了指标。**
"""),

        ("raw_py", "tools/profiling_bench/mfu_accounting.py",
         ["exp_numerator"], "exp_numerator()"),

        ("md", r"""
## 第 3 步 · 分子口径的来源：6ND 近似随序列变长而失真

**这步在做什么**：业界常用的 $6ND$ 近似（$N$ 取参与矩阵乘的参数量）在短序列上误差可忽略，长序列上就不行了。

原因是 $6ND$ 把 attention 的 $L^2$ 项当常数忽略了。序列越长，这一项越不能忽略，于是「同一个模型」在长序列下的 6ND-MFU 与精确-MFU 会系统性背离。

对比的两个口径都含 causal 折半，所以偏差纯粹来自 $L^2$ 项的处理方式。
"""),

        ("raw_py", "tools/profiling_bench/mfu_accounting.py",
         ["exp_seqlen_sweep"], "exp_seqlen_sweep()"),

        ("md", r"""
## 第 4 步 · 代际算术强度门槛：算力涨 3.2×，带宽只涨 1.6×

**这步在做什么**：换一代卡不只是分母变大，**分子能用到的那部分也不同比例地涨**。

机器平衡点 = 峰值算力 / 峰值带宽，单位 FLOP/Byte。一段算术强度固定为 $X$ 的子层，在 A100 上要 $X$ 大于平衡点才算计算受限，换到 H100 这个门槛更高。

门槛上升的直接后果：**「换新卡后吞吐上去了、但 log 里的 TFLOP/s/GPU 没同比例涨」是正常的**，不是日志坏了。
"""),

        ("raw_py", "tools/profiling_bench/mfu_accounting.py",
         ["exp_generation"], "exp_generation()"),

        ("md", r"""
## 第 5 步 · 一次完整 forward 的 FLOPs 构成：不是所有 FLOPs 都长成 GEMM 的样子

**这步在做什么**：把 Llama-3-8B 一次 forward 的每 token FLOPs 拆成五项，看比例。

这里有个容易写错的地方：`forward_flops_per_token()` 返回的是**单层**的量，但 lm_head 是每次 forward 只算一次。直接拿单层的项和 lm_head 相加会得到一个没有物理含义的比例——必须按 $L$ 折算。

值得注意的是 attention core 占的比例：它里面有一半是 softmax（elementwise）而不是 matmul，而 MFU 的分子只算 matmul。**这部分 FLOPs 一分都不计入分子，但真机上要花时间。** 序列越长，这个「计入分子但不完整」的区域越大。
"""),

        ("raw_py", "tools/profiling_bench/mfu_accounting.py",
         ["exp_layer_mix"], "exp_layer_mix()"),

        ("md", r"""
## 第 6 步 · 把 log 里的 TFLOP/s/GPU 换回 MFU

**这步在做什么**：把前面所有口径收成一个实用工具——给定一行 log 的读数，分别按稠密分母与稀疏分母换算。

结论很直白：同一行 log，稠密分母下 40%、稀疏分母下 20%。**报数的人如果不写分母口径，你无法判断他说的是哪个。**
"""),

        ("raw_py", "tools/profiling_bench/mfu_accounting.py",
         ["exp_official_log"], "exp_official_log()"),

        ("md", r"""
## 小结

MFU/HFU 之争不是定义之争，是口径之争。写清楚「分子含不含重计算、分母用哪种精度哪个稀疏假设」才算报了个数。

**报 MFU 不写分母等于没报。** 三个元组缺一个，这个数就没法与别人的数比较。

## 参考文献

- [NVIDIA A100 产品页](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) —— MFU 指标的出处
- Megatron-LM `training.py:802` / `:1007`（基准 commit `60e039626`，v0.20.0）
"""),
    ],
    "readme": r"""
## 对应文章

- 《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》
  https://lrypcy.github.io/2026/09/29/profiling-00-metrics/

## 如何运行

在 GitHub 上直接读（所有输出已固化）。想自己跑：

```bash
# 任意带 Python 3.9+ 的解释器即可，无第三方依赖
jupyter nbconvert --to notebook --execute profiling-00-mfu-accounting.ipynb
```

或者克隆仓库后用 JupyterLab / VS Code 打开。

## 代码来源

代码逐字摘自博客仓库 `lrypcy.github.io`：

- `tools/profiling_bench/mfu_accounting.py`
- `tools/megatron_bench/configs.py`（模型与硬件标称规格）
- `tools/megatron_bench/flops.py`（FLOPs 记账，精确 / 6ND 双口径）

由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取，不是手抄。

## 实验内容

- **分母口径**：分子固定，六种峰值规格换来换去 → 同一行 log 报出 20% 到 128%
- **分子口径**：recompute 策略变化 → MFU 不动、HFU 按 4/3 变
- **6ND 失真**：序列从 1K 到 128K，近似误差如何发散
- **代际门槛**：算力涨 3.2×、带宽涨 1.6×，机器平衡点抬高多少
- **forward 构成**：attention core 里有多少是 softmax 而非 matmul
- **反算**：把 TFLOP/s/GPU 换回 MFU（稠密 / 稀疏两个分母）

## 关键数字（实测输出见 `results/stdout.txt`）

- 400 TFLOP/s/GPU 在 H100 稠密 BF16 分母下 = 40.42%；在 A100 稠密分母下 = **128.21%**（物理上不存在）
- 每 token 57.91 GFLOP（llama3-8b，序列 8192，不含重计算）→ 折合 6907 tokens/s/GPU
- full recompute 的 HFU/MFU = 1.315（≈4/3，每 token 从 6P 变 8P）
- 6ND 近似在序列 8192 处偏低 22.91%，到 131072 处偏低 253.68%
- 机器平衡点：A100 153.0 → H100 295.4 FLOP/Byte，门槛上升 1.93×（算力 3.17×、带宽 1.64×）
- Llama-3-8B forward 构成：MLP 58.4%、attention core 22.2%、lm_head 5.4%
""",
}
