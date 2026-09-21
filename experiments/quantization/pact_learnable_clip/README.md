## 对应文章

- 《大模型量化算法（18）：LSQ / PACT / DSQ——可学习的 scale 与 clip》（§4 全部）
  https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/
- 《大模型量化算法（11）：伪量化算子插入》
  https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace pact_learnable_clip.ipynb
# 或在 JLab 里打开后把第一个代码格的 MODE 改成 "full" 再 Run All（全量版）
```

纯 numpy + matplotlib，CPU 分钟级（smoke）/ 数分钟级（full）；随机种子固定 SEED=0，结果可复现。
`MODE = "smoke" | "full"` 一个开关控制全部网格/步数/arm 数（含 bits、SNR 扫描点数、MLP 步数、α0 列表、L2 λ 列表）。

## 实验内容

任务：18 篇 §4 的 PACT——把激活上界 `α = clip(x, 0, α)` 变成可学参数（ReLU = α=∞ 的特例）。
配套一个重尾正激活分布（Student-t(2.5)×0.6 后 clip，用于 SNR 实验）与一个 teacher–student 小 MLP
（teacher 重尾权重 + ReLU；student 隐层激活过 PACT 4-bit 量化，可学 α）。

- **A. 选对 α 的增益**：重尾激活上网格搜索使重建 SNR 最大的 `α*`，对比 `α=max(x)`（min-max 量化）。
  4-bit 下"选对 α"比 `α=max(x)` 高 **+5.50 dB**（2-bit +4.52 dB，8-bit +0.13 dB，几乎无差别）。
- **B. `∂ŷ/∂α` 三段式**：范围内 `≤ 1/(2M)`（=`1/30`，精确吻合）、截断区恒为 `1`；
  0.027% 的被截断元素承担了 **3.1%** 的梯度质量（不对称 ≈ 2M）。
- **C1. 损失地形对 α 很钝**：固定 α 只训权重，任务损失随 α 宽而慢变（无尖锐最优点）。
  文章在更大网络/更长训练下给出"12–40 宽高原"，本探针规模下为缓慢单调，结论一致。
- **C2. α 的慢动力学与路径依赖**：可学 α（lr×10），α0 从 2.0 到 46.09 最终收敛到 2.55 / 4.70 / 37.13，
  训练结束时仍未忘记初值——α 动力学远慢于权重。
- **C3. L2 正则极难标定**：λ 从 0→1e-2 把最终 α 从 35.4 单调压到 30.0；λ 是根极细的缰绳，
  论文在更大预算下 `λ=1e-2` 直接把 α 掐到 ~1.2，PACT 退化成 α≈max 的硬截断、A 的 5.5 dB 增益丢失。

## 关键数字（SEED=0，smoke 模式实测）

| 实验 | 指标 | 数值 |
|---|---|---|
| A（4-bit） | α* / SNR@α* / SNR@max(x) | 12.68 / 10.06 dB / 4.56 dB |
| A（4-bit） | 选对 α 相对 α=max(x) 的增益 | **+5.50 dB** |
| A（2/8-bit） | 增益 | +4.52 dB / +0.13 dB |
| B（4-bit） | 范围内 \|∂ŷ/∂α\| 上界 | 0.03333 = 1/(2M) |
| B（4-bit） | 截断区 \|∂ŷ/∂α\| | 恒为 1.0 |
| B（4-bit） | 0.027% 被截断元素承担的梯度质量 | 3.1% |
| C2 | α0→最终 α（2 / 15 / 46.09） | 2.55 / 4.70 / 37.13 |
| C3 | λ→最终 α（0 / 1e-3 / 1e-2） | 35.4 / 31.4 / 30.0 |

> 诚实标注：所有数字都是**合成探针任务**（重尾激活 + teacher–student MLP）跑出来的，不是真实模型精度；
> 18 篇 §4 里"α 高原/被掐死"等现象依赖更大的训练预算与网络深度，本探针用更小规模复现其机制方向与定量量级
> （如 B 的 1/(2M)、C2 的慢动力学与路径依赖、C3 的 λ 单调性），但绝对数字不具迁移性。

## 预期输出

- `results/pact_snr_vs_alpha.png`、`pact_dalpha_hist.png`、`pact_alpha_plateau.png`、
  `pact_alpha_trajectory.png`、`pact_l2_alpha.png`（5 张图）
- `results/stdout.txt`（全部打印日志） / `results/results.json`（结构化结果）
- 执行后的 `pact_learnable_clip.ipynb`（含全部输出）
