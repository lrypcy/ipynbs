## 对应文章

- 《QAT（00）：总览》§5 LLM-QAT 与 QLoRA https://lrypcy.github.io/2026/08/25/qat-00-overview/
- 相关：《20 蒸馏量化 QAT》§4.1 LLM-QAT https://lrypcy.github.io/2026/09/19/llm-quant-20-distillation-qat/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace llmqat_qlora.ipynb
```

纯 numpy + matplotlib，CPU 分钟级（smoke）；SEED=0 可复现。
NF4 码本照搬 bitsandbytes 的公开常数；**全部为合成探针任务，不是真实模型精度。**

## 实验内容

- **A. NF4 非均匀 4-bit + 双重量化**：把"每块 absmax 的元数据税"算清楚
- **B. NF4 vs 均匀 INT4**：高斯 / 拉普拉斯 / Student-t 三种分布上的 SQNR
- **C. QLoRA 的"4-bit 冻结基座 + fp16 低秩适配器"形态**：base-only / QLoRA / 全 fp16 微调三者对比
- **D1. LLM-QAT 数据生成式蒸馏**：teacher 自产数据（无任何真实语料），student 用 QAT vs PTQ
- **D2. 注意力 softmax outlier**：单个超大 logit 把 softmax 压成近 one-hot，低 bit 在尖峰处最脆弱

## 关键数字（SEED=0，smoke 实测）

| 实验 | 关键数字 |
|---|---|
| A | 元数据税 **0.2500 → 0.1260 bits/weight**，双重量化省 **49.6%** |
| B | 高斯上 NF4 **20.71 dB** vs 均匀 INT4 19.92（**+0.79 dB**）；拉普拉斯 **+2.30 dB**；Student-t **+3.97 dB** |
| C | QLoRA 比 base-only 低 **10.26 dB**；全 fp16 再低 **21.64 dB**；适配器恢复了 base-only 误差的 **90.6%** |
| D1 | 无真实语料的自蒸馏下，QAT 比 PTQ 低 **1.14 dB** |
| D2 | softmax 8-bit 量化 MSE **1.67e-08**；尖峰处（概率 0.989）量化绝对误差 0 |

## 预期输出

- `results/qlora_nf4_vs_int4.png`、`qlora_lowrank_recovery.png`、`qlora_llmqat_distill_and_softmax_outlier.png`
- `results/stdout.txt` / `results/results.json`
