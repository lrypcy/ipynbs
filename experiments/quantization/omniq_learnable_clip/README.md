## 对应文章

- 《PTQ（03）：OmniQuant 可学习裁剪》  
  https://lrypcy.github.io/2026/08/24/ptq-03-awq-omniq/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **A. 澡盆曲线**：全局 clip 系数 alpha 扫描，MinMax(alpha=1) 不在盆底
- **B. 失效对照**：朴素 STE 梯度从 MinMax 出发几乎不动（内部舍入残差与
  r 不相关，一阶信号是二阶小量）——这正是 OmniQuant 用分位数初始化的原因
- **C. 可学习裁剪**：逐列黄金分割坐标下降（免梯度黑盒优化），逼近每列独立最优

## 关键数字（SEED=0 实测）

| 方法 | 输出 rel.MSE | 较 MinMax |
|---|---|---|
| MinMax (alpha=1) | 2.07e-2 | — |
| 全局网格搜索 | 1.28e-2 | -38.2% |
| 朴素 STE-GD | 2.02e-2 | -2.2%（失效对照） |
| 逐列可学习 | **1.26e-2** | **-38.9%** |

学到的逐列 alpha：min 0.55 / median 0.77 / max 0.94。

## 预期输出

- `results/omniq_clip_basin_and_alpha_hist.png`
- `results/stdout.txt` / `results/results.json`
