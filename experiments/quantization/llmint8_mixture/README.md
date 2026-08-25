## 对应文章

- 《大模型量化算法（01）：LLM.int8() 混合精度分解》  
  https://lrypcy.github.io/2026/08/24/llm-quant-01-llmint8-outlier-mixture/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **A. 离群长尾实证**：0.4% 的离群列贡献了主要幅值（mag ratio ~198x）
- **B. 双路径分解**：Y = X_int8 W_int8 + X_out W_fp16 的数值验证，
  含阈值 tau 扫描
- **C. 带宽账**：INT8 GEMM 的输入字节 vs 有效位宽坍塌分析

## 关键数字（SEED=0 实测）

- rel-MSE(vs FP16)：naive per-tensor 2.03e-3 / token-wise 无拆分 1.39e-3 /
  LLM.int8() 混合 **2.56e-6**（改善 792x）
- 正常值有效位宽：无拆分 2.98 bit -> 拆分后 7.8 bit（恢复 2.61x）

## 预期输出

- `results/activation_outlier_tail.png`
- `results/mixed_decomposition_relmse.png`
- `results/gemm_bandwidth_breakdown.png`
- `results/stdout.txt` / `results/results.json`
