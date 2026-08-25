## 对应文章

- 《大模型量化算法（00）：量化器数学地基与 RTN 基线》  
  https://lrypcy.github.io/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
另覆盖 PTQ 系列总览：<https://lrypcy.github.io/2026/08/24/ptq-00-overview/>

## 实验内容

- **A. SNR-vs-bitwidth**：均匀分布严格贴合 6.02b 规律；高斯分布在 b=2 时
  比线性规律低约 2.5 dB（高分辨率假设失效 -> 极低比特必须换码本）
- **B. 粒度对比**（含离群列）：per-tensor sym 权重 SNR 仅 0.97 dB（输出相对误差 89.4%），
  per-channel 5.8 dB，per-group(128) 达 15.25 dB（误差 17.4%）
- **C. MSE 最优裁剪**：对高斯权重网格搜索 alpha*，验证解析解 M*≈2.5σ

## 预期输出

- `results/granularity_comparison.png` —— 各粒度的量化前后权重对比
- `results/snr_vs_bitwidth.png` —— 四种分布的 SNR 曲线与 6.02b 参考线
- `results/mse_vs_clipping_ratio.png` —— L(alpha) 盆形曲线
- `results/results.json` —— 全部实测数字（如 demo_b.rows 粒度表）
