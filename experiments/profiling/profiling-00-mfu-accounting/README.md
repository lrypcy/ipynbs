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
