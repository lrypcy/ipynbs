## 对应文章

- 《大模型量化算法（02)：SmoothQuant W8A8》  
  https://lrypcy.github.io/2026/08/24/llm-quant-02-smoothquant-w8a8/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
亦覆盖 PTQ（06）：<https://lrypcy.github.io/2026/08/24/ptq-06-smoothquant-zeroquant/>

## 实验内容

- **A. alpha 扫描**：s_j = max|x_j|^alpha / max|w_j|^(1-alpha)，网格找最优 alpha
- **B. 困难守恒定律**：数值上验证 sum log(max|x|)+sum log(max|w|) 在迁移前后不变
  （偏差 < 1e-15），平滑只是搬动困难而非消灭困难
- **C. 统计量的校准估计**：用少量校准样本估计激活 channel 幅值的偏差

## 关键数字（SEED=0 实测）

- W8A8 朴素 rel err 0.0238 -> SmoothQuant 0.0073（**MSE 改善 10.7x**）
- 最优 alpha=0.50；守恒定律最大相对偏差 2.4e-16

## 预期输出

- `results/smoothquant_alpha_sweep_mse.png`
- `results/smoothquant_difficulty_conservation.png`
- `results/smoothquant_activation_stat_estimation.png`
- `results/stdout.txt` / `results/results.json`
