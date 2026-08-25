## 对应文章

- 《PTQ（01）：RTN 基线与 LLM.int8()》  
  https://lrypcy.github.io/2026/08/24/ptq-01-rtn-llmint8/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- 合成含 6 条离群激活通道的数据（幅值约为正常的 12.9x）
- 三种方案对比：权重 per-channel RTN / 双路 per-tensor / LLM.int8() 混合精度
- 坍缩机制可视化：per-tensor scale 被离群值绑架后正常值的等效位宽只剩 4.2 bit，
  18.7% 的正常值被量化到 0
- 内存账：LLM.int8() 相对 FP16 压缩 1.88x

## 关键数字（SEED=0 实测）

| 方案 | 输出相对误差 |
|---|---|
| W4 per-channel RTN（激活FP16） | 0.71% |
| 双路 per-tensor INT8 | 2.40% |
| LLM.int8() 混合精度 | **0.15%** |

## 预期输出

- `results/synthetic_activation_outliers.png`
- `results/method_error_memory.png`
- `results/stdout.txt` / `results/results.json`
