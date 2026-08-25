## 对应文章

- 《PTQ（05）：QuIP# 与 AQLM》  
  https://lrypcy.github.io/2026/08/24/ptq-05-quip-aqlm/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **A. incoherence 的必要性**：低秩骨架 + 2% 稀疏大值的权重直接做逐张量
  标量量化会灾难性失败；Hadamard 旋转（在线转激活、离线转权重）层数学等价地救回来
- **B. VQ vs 标量同预算对比**：v=8 维码字、256 码本、逐块 RMS 归一化
  （AQLM/QuIP# 风格），手写 kmeans++/Lloyd 训练码本

## 关键数字（SEED=0 实测）

| 方法 | raw 基 | 旋转基 |
|---|---|---|
| INT2 逐张量 | 4.50 | 0.71 |
| INT3 逐张量 | 2.02 | 0.34 |
| VQ v8 C256（~2bpw 含块尺度开销） | 0.39 | 0.48 |

旋转基下同预算 VQ 比 INT2 **好 32.3%**；标量量化则完全依赖旋转救命（-83%）。

## 预期输出

- `results/vq_vs_scalar_and_incoherence.png`
- `results/stdout.txt` / `results/results.json`
