## 对应文章

- 《大模型量化算法（18）：LSQ / PACT / DSQ——可学习的 scale 与 clip》（§3 全部、§5.4 三方对比）
  https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/
- 《QAT（00）：总览》（§3.1 LSQ 的精确梯度）
  https://lrypcy.github.io/2026/08/25/qat-00-overview/
- 《大模型量化算法（11）：伪量化算子插入》
  https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace lsq_learned_step_size.ipynb
# 或在 JLab 里打开后把第一个代码格的 MODE 改成 "full" 再 Run All（全量版）
```

纯 numpy + matplotlib，CPU 分钟级（smoke）/ 十分钟级（full）；随机种子固定 SEED=0，结果可复现。
`MODE = "smoke" | "full"` 一个开关控制全部网格/步数/arm 数。

## 实验内容

任务：18 篇 §2 的过定线性回归（W∈R^{64×128}，2% ×3 离群元素，4-bit 为主）。

- **A. PTQ 基线**：scale 扫描澡盆曲线，min-max → MSE 最优白捡 4.64 dB（LSQ 的收益天花板）
- **B. 梯度分解**：∇_s L = 舍入项 − 截断项；**两项的过零点（二分定位）与离线最优 s*
  相对偏差 ~0.1%，过零点处残差精确为 0**；逐元素不对称 6.7× → 240× → 2747× 随位宽指数增长
- **C. 自我稳定**：s0 从 0.1× 到 10× LSQ init，大 lr 下收敛 s 离散仅 7.6%；
  但小 lr 下 10× 初值直接崩（5.2e-1）——**初始化宁小勿大**
- **D. 失衡比与 g**：实测 R = 论文 √(n_Q·Q_P) 估计的 9–15×（∇_s L 是相干和，30–80× 于随机和）；
  g 消融见学习率敏感度
- **E. 端到端对比**：LSQ(只学 s) 1.7495e-2 vs 离线网格最优 1.7522e-2 —— **只差 0.007 dB**；
  再叠 STE 式 W 训练反而 -0.19 dB（本任务的 FP 解已最优）
- **F. 粒度 × 可学习性**：per-channel 粒度 +5.19 dB >> 只学 s +0.27 dB；粒度和可学习性是两个正交收益

## 关键数字（SEED=0，smoke 模式实测）

| 方法（4-bit） | task loss | 较 min-max PTQ |
|---|---|---|
| min-max PTQ（不训练） | 5.102e-2 | 0 dB |
| per-channel min-max PTQ（不训练） | 1.543e-2 | +5.19 dB |
| 离线网格最优 s*（不训练） | 1.752e-2 | +4.64 dB |
| LSQ（只学 s，W 冻结） | 1.750e-2 | **+4.65 dB** |
| LSQ（s + W 同训） | 1.828e-2 | +4.46 dB |

## 预期输出

- `results/lsq_*.png`（6 张图）
- `results/stdout.txt` / `results/results.json`
- 执行后的 `lsq_learned_step_size.ipynb`（含全部输出）
