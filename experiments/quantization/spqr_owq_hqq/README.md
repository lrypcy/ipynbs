## 对应文章

- 《PTQ（04）：SpQR、OWQ 与 HQQ》  
  https://lrypcy.github.io/2026/08/24/ptq-04-spqr-owq-hqq/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **OWQ/SpQR 混合精度**：按激活显著度挑 1% 弱列保 FP16，其余 INT4 分组量化；
  对比三种选择的误差-位宽权衡
- **HQQ**：半二次优化 min ||w-q||^2 + lam||q||_1 的迭代求解演示
- 比例扫描（0%~5% 高精度占比）画误差-B_eff 曲线

## 关键数字（SEED=0 实测）

- INT4 全量化激活误差 0.1842；1% 混合精度后 OWQ 0.1461（**-20.7%**）、
  SpQR 0.1465（-20.5%），代价仅 B_eff 4.12 / 4.30 bit
- OWQ 命中 5/6 个人工植入弱列

## 预期输出

- `results/owq_mixed_precision.png`
- `results/hqq_half_quadratic.png`
- `results/stdout.txt` / `results/results.json`
