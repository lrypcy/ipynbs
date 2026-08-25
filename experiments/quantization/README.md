# 量化系列配套实验

本目录是博客「大模型量化算法」系列的配套实验集：**llm-quant 主线 4 篇 + PTQ 专题 13 篇**，
每个子目录对应一篇文章的 core 论点，全部为纯 numpy 实现，CPU 几秒内跑完、结果可复现（SEED=0）。

## 运行环境

- Python 3.11+（开发环境：`/Users/congyuan/Software/miniconda3/bin/python`）
- 仅依赖 numpy + matplotlib（无 torch / 无 GPU）

```bash
cd <任一子目录> && python run.py
```

## 实验 × 文章索引

| 目录 | 覆盖文章 | 一句话结论 |
|---|---|---|
| `quantizer_granularity/` | [量化00：数学地基与RTN](https://lrypcy.github.io/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) · [PTQ 总览](https://lrypcy.github.io/2026/08/24/ptq-00-overview/) | 粒度决定一切：per-tensor SNR 0.97dB → per-group(128) 15.25dB |
| `rtn_llmint8/` | [PTQ01：RTN 与 LLM.int8()](https://lrypcy.github.io/2026/08/24/ptq-01-rtn-llmint8/) | 离群通道让 per-tensor INT8 的有效位宽坍塌到 4.2 bit |
| `gptq_obcq/` | [PTQ02：GPTQ](https://lrypcy.github.io/2026/08/24/ptq-02-gptq/) | 误差反馈补偿：INT4 下 GPTQ 比 RTN 低 35% MSE |
| `awq_scale_search/` · `omniq_learnable_clip/` | [PTQ03：AWQ 与 OmniQuant](https://lrypcy.github.io/2026/08/24/ptq-03-awq-omniq/) · [AWQ scale search](https://lrypcy.github.io/2026/08/24/llm-quant-03-awq-scale-search/) | 显著度排序 + 可学习裁剪各显其能 |
| `spqr_owq_hqq/` | [PTQ04：SpQR/OWQ/HQQ](https://lrypcy.github.io/2026/08/24/ptq-04-spqr-owq-hqq/) | 1% 敏感坐标保高精度，误差降 21% |
| `quip_aqlm/` | [PTQ05：QuIP#/AQLM](https://lrypcy.github.io/2026/08/24/ptq-05-quip-aqlm/) | incoherence 让逐张量 INT3 误差降 83%；VQ 同预算再胜标量 |
| `smoothquant_alpha_sweep/` | [SmoothQuant W8A8](https://lrypcy.github.io/2026/08/24/llm-quant-02-smoothquant-w8a8/) · [PTQ06](https://lrypcy.github.io/2026/08/24/ptq-06-smoothquant-zeroquant/) | α=0.5 时 W8A8 误差比朴素低 10.7x |
| `quarot_spinquant_rotation/` | [PTQ07：QuaRot/SpinQuant](https://lrypcy.github.io/2026/08/24/ptq-07-quarot-spinquant/) | Hadamard 旋转压平动态范围 4x，端到端 INT4 误差减半 |
| `fp8_mxfp4_formats/` | [PTQ08：GGUF/FP8/MXFP4](https://lrypcy.github.io/2026/08/24/ptq-08-gguf-fp8-mxfp4/) | E4M3 比 E5M2 高 6dB；MXFP4 18.8dB、NF4 20.7dB |
| `squeezellm_vptq_claq/` | [PTQ09：SqueezeLLM/VPTQ](https://lrypcy.github.io/2026/08/24/ptq-09-squeezellm-vptq-claq/) | 非均匀码本 -91%；0.2% 坐标保FP16 再降 61% |
| `outlier_suppression_plus/` | [PTQ10：Outlier Suppression+](https://lrypcy.github.io/2026/08/24/ptq-10-outlier-suppression/) | γ 迁移 -79%；shift+scale 有效电平波动 1.9→36 |
| `rptq_quik_atom/` | [PTQ11：RPTQ/QUIK/Atom](https://lrypcy.github.io/2026/08/24/ptq-11-rptq-quik-atom/) | 通道重排让组 absmax 从 40 收敛到 4 以内 |
| `olive_abfloat/` | [PTQ12：Olive/AbFloat](https://lrypcy.github.io/2026/08/24/ptq-12-olive-abfloat/) | abfloat 跳过正常区间专治离群：4.8% vs 91.7% |
| `qserve_qqq_w4a8/` | [PTQ13：QServe/QQQ](https://lrypcy.github.io/2026/08/24/ptq-13-qserve-qqq/) | SmoothAttention -70%；非对称 int4 KV -40% |
| `llmint8_mixture/` | [LLM.int8() 混合精度分解](https://lrypcy.github.io/2026/08/24/llm-quant-01-llmint8-outlier-mixture/) | 双路径分解 rel-MSE 2.6e-6，较 naive 好 792x |

> 注：部分实验同时覆盖主线与 PTQ 专题两篇文章（同一机制），已在上表合并。
