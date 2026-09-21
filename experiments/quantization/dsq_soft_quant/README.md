## 对应文章

- 《大模型量化算法（18）：LSQ / PACT / DSQ——可学习的 scale 与 clip》（§5 全部）
  https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/
- 《大模型量化算法（11）：伪量化算子插入》
  https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace dsq_soft_quant.ipynb
# 或在 JLab 里打开后把第一个代码格的 MODE 改成 "full" 再 Run All（全量版）
```

纯 numpy + matplotlib，CPU 分钟级（smoke）/ 数分钟级（full）；随机种子固定 SEED=0，结果可复现。
`MODE = "smoke" | "full"` 一个开关控制全部网格/步数/arm 数（含扫描点数、α 列表、训练步数网格、L2 λ 列表）。

## 实验内容

任务：18 篇 §5 的 DSQ——用 `tanh` 磨圆硬量化的阶梯，造一个真可导的量化函数 `Q_S(x)`，
通过相似度因子 `α` 退火（α→0 收敛硬量化，α→1 退化为恒等映射）。

- **A. tanh 构造与极限**：`Q_S(x)` 曲线（α=0.5/0.2/0.05/0.01）叠加硬量化与恒等映射；
  `max|Q_S(x)−x|` 随 α→1 趋近 0（恒等退化，文中 §5.2 的 α→0.5 为笔误，正确极限是 α→1）。
- **B. 梯度 vs STE**：在 `x∈[−1.2,1.2]` 扫描，`∂Q_S/∂x` 均值对全部 α 都 = **1.0000**（与 STE 完全相同）；
  区别只在质量分布——峰值随 α 减小从 1.16× 升到 2.67×（峰值更高、支撑更窄，趋近真实 Dirac 导数）。
- **C. 有限差分校验**：`∂Q_S/∂x`、`∂Q_S/∂α` 与中心差分最大绝对误差 ~1e-7 / ~1e-10（真导数，可用）。
- **D（致命不变量）**：`max|Q_S − hard|` 恒等于 `Δ/2 = 0.20000`，α 从 0.5 降到 0.005（两个数量级）一动不动；
  而 mean 偏差对数级收敛（0.0943→0.0446）。这是 DSQ 的拓扑死穴。
- **E（最致命）**：欠定线性回归 + DSQ 软量化训练。**必须同时报告两套数字**——训练 loss（软前向）与
  deploy loss（硬量化前向）。训练步数越多，train(软) 跌向机器精度、deploy(硬) 焊在 ~1e-2；
  gap 随训练预算单调放大（5.8→228.9 dB），固定 arm 在 500 步处约 **26–48 dB**。

## 关键数字（SEED=0，smoke 模式实测）

| 实验 | 指标 | 数值 |
|---|---|---|
| B（4-bit） | `∂Q_S/∂x` 均值（α=0.4/0.2/0.05/0.01） | 1.00003 / 1.00007 / 1.00019 / 1.00033（均 = 1.0000） |
| B（4-bit） | `∂Q_S/∂x` 峰值 | 1.1552 / 1.3733 / 1.9282 / 2.6734 |
| C | 有限差分 max 绝对误差 `∂Q_S/∂x` / `∂Q_S/∂α` | 1.23e-7 / 7.16e-10 |
| D | `max|Q_S − hard|`（α=0.5→0.005） | 恒 = 0.19999 = Δ/2 |
| E（500 步） | train(软) / deploy(硬) | 3.37e-7 / 7.55e-3 |
| E（500 步） | train→deploy gap | **43.5 dB** |
| E（arms） | 各 arm gap（固定 α=0.4/0.2/0.05、学 α、学 α+λ） | 47.8 / 43.5 / 25.8 / 45.4 / 40.5 dB |

> 诚实标注：所有数字都是**合成探针任务**（欠定线性回归 + DSQ 软量化）跑出来的，不是真实模型精度。
> 本文 §5.4 报告文章原文 33–43 dB 的 gap 来自更大网络/更长训练；本探针在 500 步处得到 43.5 dB，
> 且 gap 随训练继续放大，机制方向与定量量级一致。

## 预期输出

- `results/dsq_construction.png`、`dsq_grad_vs_ste.png`、`dsq_max_dev.png`、
  `dsq_train_deploy.png`、`dsq_arms_train_deploy.png`（5 张图）
- `results/stdout.txt`（全部打印日志） / `results/results.json`（结构化结果）
- 执行后的 `dsq_soft_quant.ipynb`（含全部输出）
