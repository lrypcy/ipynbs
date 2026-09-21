## 对应文章

- 《大模型量化算法（22）：Reasoning LLM 低比特量化》 https://lrypcy.github.io/2026/09/19/llm-quant-22-reasoning-llm-lowbit/
- 相关：《20 蒸馏量化 QAT》 https://lrypcy.github.io/2026/09/19/llm-quant-20-distillation-qat/ （§3 KL 设计、§4.3 reward rectification）

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace reasoning_llm_lowbit.ipynb
```

纯 numpy + matplotlib，CPU 秒级（smoke）；SEED=0 可复现。
**合成探针（重尾激活 + 长链正确率模型），不是真实模型精度。**

## 实验内容

- **A. 长链误差放大**：单步错误率 ε=0.005（PPL 上完全看不出来）在 L=512 上全链正确率塌到多少
- **B. 两域形状差异 = 量化代价差异**：预训练域（近高斯）vs 推理域（重尾），per-tensor vs per-group
- **C. 混合域校准**：用 **MSE observer**（不是 absmax）求 scale，扫推理域占比 ρ，最小化**两域最差者**的误差
- **D. 渐进量化**：FP16→W8→W4→W2 vs 一步到 W2（总步数相同），per-channel 非对称网格 + STE + 夹在网格内
- **E. 教师 vs 学生 reweighting**（ReasoningQAT 的核心）
- **F. 诊断**：分域 + 分长度的掉点热力图，平均分会藏住什么

## 关键数字（SEED=0，smoke 实测）

| 实验 | 关键数字 |
|---|---|
| A | ε=0.005 恒定：L=128 全链正确率 **0.526**、L=512 **0.077**；累积模型 L=128 只剩 **0.042** |
| B | 4-bit per-tensor 下推理域误差是预训练域的 **3.96x**；换 per-group(g=32) 推理域降 **5.15x**，预训练域只降 **2.00x** |
| C | 最优 ρ* = **0.15~0.20**（正是文章说的 80/20 量级）；ρ=0 推理域误差 0.890，ρ=1 预训练域被压成 0 |
| D | 渐进比一步到位再降 **12.5%**（RTN 在本探针上已很强，单层线性层 RTN 本就近最优） |
| E | p_S=0.41 处 1/p_S 相对教师加权放大 **3.39x** |
| F | W4 平均分看着只掉一点，但长链 L=128 的全链正确率只剩约 **1/3** |

## 预期输出

- `results/rl_a_chain_amplification.png`、`rl_b_domain_shape.png`、`rl_c_mixed_domain_calibration.png`、`rl_d_progressive_quant.png`、`rl_e_reward_rectification.png`、`rl_f_length_scan_heatmap.png`
- `results/stdout.txt` / `results/results.json`
