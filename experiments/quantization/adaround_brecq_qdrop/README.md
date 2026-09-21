## 对应文章

- 《大模型量化算法（19）：AdaRound / BRECQ / QDrop》 https://lrypcy.github.io/2026/09/19/llm-quant-19-adaround-brecq-qdrop/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace adaround_brecq_qdrop.ipynb
```

纯 numpy + matplotlib，CPU 秒级；SEED=0 可复现。**合成探针，不是真实模型精度。**

## 实验内容

- **A. 学舍入方向**：离散坐标下降（O(1) 解析翻转收益 + 早停）与可微松弛两条路线的对比
- **B. 端到端**：RTN / AdaRound / BRECQ（block-wise 重建）在不同 bit 上的层输出重构误差
- **C. QDrop**：训练时随机 drop 掉激活的量化，看它在本任务上到底值多少

## 关键数字（SEED=0，smoke 实测）

| 实验 | 关键数字 |
|---|---|
| A | 4-bit 下离散坐标下降 **+1.68 dB**、可微松弛 **+1.97 dB** |
| B | BRECQ block-wise 比 RTN **+1.65 dB** @4-bit |
| C | 本任务上 QDrop 收益微弱（**+0.003 dB**）——因为 8-bit 激活的量化误差本来就小，drop 掉的扰动不显著 |

> QDrop 的收益与"激活量化误差有多大"直接相关：激活 bit 越低（误差越大），QDrop 越有戏。

## 预期输出

- `results/adaround_vs_rtn.png`、`brecq_block_and_qdrop.png`
- `results/stdout.txt` / `results/results.json`
