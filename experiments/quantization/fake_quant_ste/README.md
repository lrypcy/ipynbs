## 对应文章

- 《大模型量化算法（17）：伪量化算子插入——QAT 的地基》
  https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/
- 《QAT（00）：总览》 https://lrypcy.github.io/2026/08/25/qat-00-overview/ （§2 STE 地基）

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace fake_quant_ste.ipynb
# 或在 JLab 里打开后把第一个代码格的 MODE 改成 "full" 再 Run All（全量版）
```

纯 numpy + matplotlib，CPU 秒级（smoke）/ 分钟级（full）；随机种子固定 SEED=0，结果可复现。
`MODE = "smoke" | "full"` 一个开关控制全部采样数/步数/温度网格。
**所有任务均为合成探针任务（synthetic probe），不是真实模型精度**，仅定性复现文章的数量级与趋势。

## 实验内容

任务：17 篇 §7.1 的合成权重 `W ∈ R^{256×1024}`（基底 N(0,0.02²)，8 个 ×10 离群通道 + 0.1% ×30 极端元素）+ 合成 ReLU 重尾激活。

- **A. Round-trip 误差与上界**：三种粒度 × 三种位宽的 MSE/SNR/最大绝对误差；验证 `|ε_r|≤s/2`；散点+直方图展示网格吸附（值被拉到 15 个离散电平）
- **B. 对称 vs 非对称 / 粒度**：单边偏斜激活上非对称比对称高 ~6.6 dB；算子参数 `(s, z_p)` 的 shape 差异即粒度差异
- **C. clamp 饱和效应**：用校准百分位 scale（非 min-max）模拟部署定点范围，拆出舍入/饱和误差；仅 1% 被裁剪却贡献 ~32% 误差能量
- **D. STE 有偏性**：教学用软量化（每 bin 半平台半斜坡，平均斜率 0.5）vs STE（恒为 1）——偏差 +0.50；截断区两者梯度都→0（梯度截停）
- **E. 两层 MLP QAT**：手写前向/反向，固定 scale 激活伪量化；QAT（模拟 clamp）比 PTQ 低 1.38 dB，而"只模拟 round 不模拟 clamp"与 PTQ 一样差——直接验证 §4.3

## 关键数字（SEED=0，smoke 模式实测）

| 实验 | 关键数字 |
|---|---|
| A | 4-bit per-tensor SNR 1.68 dB → per-group(128) 14.71 dB，**白捡 13.02 dB**；误差上界 s/2=0.7665 |
| B | 单边激活上 非对称 vs 对称 **+6.62 dB**；per-channel 再 +4.08 dB |
| C | 仅 **1.00%** 元素被裁剪，却贡献 **32.2%** 误差能量（饱和效应） |
| D | STE 有效区均值 1.0 vs 软量化 **0.499**（偏差 **+0.50**）；截断区 STE=0（梯度截停）；方向一致性 100% |
| E | QAT 评估损失 2.92e0 vs PTQ 4.01e0（**+1.38 dB**）；round-only 与 PTQ 同样差 |

## 预期输出

- `results/fqs_roundtrip_snr_and_attraction.png`、`fqs_dequant_histogram.png`、`fqs_symmetry_granularity.png`、`fqs_saturation_effect.png`、`fqs_ste_bias.png`、`fqs_mlp_qat_vs_ptq.png`
- `results/stdout.txt` / `results/results.json`
- 执行后的 `fake_quant_ste.ipynb`（含全部输出）
