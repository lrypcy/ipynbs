## 对应文章

- 《PTQ（13）：QServe 与 QQQ（W4A8）》  
  https://lrypcy.github.io/2026/08/24/ptq-13-qserve-qqq/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

本实验为纯 stdout 演示（无图）。三个 Demo：

- **DemoA SmoothAttention**：K 侧逐张量 int8 的 attention 分数重构误差
- **DemoB 低秩补偿**：W4(RTN) 层输出的 rank-r fp16 残差补偿
- **DemoC KV cache**：V 的逐通道对称 vs 非对称 4-bit

## 关键数字（实测，见 results/stdout.txt）

- attention 分数误差：直接量化 0.0135 -> SmoothAttention **0.0040**（-70.4%）
- W4 输出误差 0.1249 -> rank-16 补偿后 0.1103（存储开销 12.5%）
- KV cache int4：对称 0.0643 -> 非对称 **0.0387**（-39.7%）

## 预期输出

- `results/stdout.txt`（无 PNG，控制台实验）
