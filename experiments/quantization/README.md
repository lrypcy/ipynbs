# 量化系列配套实验

本目录是博客「大模型量化算法」系列的配套实验集：**PTQ 专题 13 篇 + llm-quant 主线（含 QAT 侧 17~25 篇）**，
每个子目录对应一篇文章的 core 论点，结果可复现（SEED=0）。

## 运行环境

- Python 3.11+（开发环境：`/Users/congyuan/Software/miniconda3/bin/python`）
- 手撸版：仅 numpy + matplotlib（无 torch / 无 GPU）
- 官方库版（`official_*`）：torch 2.10 + torchvision（conda env `torch`），CPU 后端

```bash
# 老目录（run.py 形式）
cd <子目录> && python run.py

# 新目录（ipynb 形式，顶部有 MODE="smoke"|"full" 开关）
cd <子目录> && jupyter nbconvert --to notebook --execute --inplace <name>.ipynb
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
| `lsq_learned_step_size/` | [QAT 18：LSQ 可学习步长](https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/)（QAT 侧首个） | 两项梯度过零点 = 离线最优 s*（差 0.1%）；只学 s 即得 +4.65 dB |

### QAT / 低比特部署侧（手撸版）

| 目录 | 覆盖文章 | 一句话结论 |
|---|---|---|
| `fake_quant_ste/` | [17：伪量化算子插入](https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/) · [QAT 总览](https://lrypcy.github.io/2026/08/25/qat-00-overview/) | 4-bit per-tensor → per-group 白捡 13.02 dB；STE 斜率恒为 1（有偏 +0.50） |
| `pact_learnable_clip/` | [QAT 18：PACT 可学习裁剪](https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/) | 4-bit 下搜出的 α* 比 α=max 高 +5.50 dB；截断区 ∂ŷ/∂α 恒为 1 |
| `dsq_soft_quant/` | [QAT 18：DSQ 软量化](https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/) | DSQ 梯度均值 = 1.0000；max\|Q_S−hard\| 恒 = Δ/2；train→deploy gap 43.5 dB |
| `adaround_brecq_qdrop/` | [19：AdaRound / BRECQ / QDrop](https://lrypcy.github.io/2026/09/19/llm-quant-19-adaround-brecq-qdrop/) | 学舍入方向 +1.68 dB @4bit；BRECQ block 比 RTN +1.65 dB |
| `distillation_qat/` | [20：蒸馏量化 QAT](https://lrypcy.github.io/2026/09/19/llm-quant-20-distillation-qat/) | 掉的不是 argmax 是分布形状；top-K 支撑集 > 全词表；高温度把量化学生训平 |
| `reasoning_llm_lowbit/` | [22：Reasoning LLM 低比特量化](https://lrypcy.github.io/2026/09/19/llm-quant-22-reasoning-llm-lowbit/) | ε=0.005 在 L=512 塌到 0.077；混合域校准最优 ρ*≈0.15~0.20（80/20 量级） |
| `unified_view/` | [23：LLM PTQ 统一视角](https://lrypcy.github.io/2026/08/29/llm-quant-23-unified-view/) | 同类自由度叠加效率 0.55，异类 0.96；F5 是恒等变换（等价性 2e-16） |
| `kv_cache_quant/` | [24：KV Cache 量化](https://lrypcy.github.io/2026/09/19/llm-quant-24-kv-cache/) | 4-bit KV per-channel 比 per-tensor +9.62 dB；batch=32/seq=32768 时 KV 是 W4 权重 256x |
| `mixed_precision/` | [25：混合精度](https://lrypcy.github.io/2026/09/19/llm-quant-25-mixed-precision/) | 混合精度把误差压到预算内统一方案的 27%；拉格朗日与精确 DP 只差 1.05% |
| `llmqat_qlora/` | [QAT 总览 §5：LLM-QAT / QLoRA](https://lrypcy.github.io/2026/08/25/qat-00-overview/) | 双重量化元数据税 0.25→0.126 bits/weight（省 49.6%）；QLoRA 比 base-only 低 10.26 dB |

### 官方库版（torch / torch.ao，验证真实体积 · 延迟 · API 默认行为）

| 目录 | 覆盖内容 | 一句话结论 |
|---|---|---|
| `official_torchao_qat/` | torch.ao.quantization 全流程 QAT（真实 resnet18 权重） | FP32→INT8 top-1 降 5.08pp；体积 2.56x、延迟 2.17x；per-channel 比 per-tensor +7.41 dB |
| `official_torchao_ptq/` | torch.ao PTQ（MinMax / Histogram observer、真实位打包） | 官方 MinMax(affine) 与手撸非对称 min-max **相对差 0.000%**；8-bit 真实压缩 4.00x（含元数据税） |
| `official_llm_ptq_torch/` | 真实 tiny-gpt2 上的 RTN / GPTQ / SmoothQuant / AWQ | GPTQ W4 rel-MSE 0.0014 vs RTN 0.034（低 24x）；int4 真实压缩 2.67x |

## 两类实验的分工

- **手撸版（纯 numpy）**：验证算法**机理与公式**——梯度长什么样、最优解在哪、收益来自哪个自由度。
  全部为合成探针，CPU 秒级，SEED=0 可复现。
- **官方库版（torch）**：验证**工程事实**——真实体积/延迟、API 的默认行为（observer、engine、
  支持的 dtype）、以及本机环境的能力边界（例如 torch 2.10 上 `quantize_dynamic(Linear)` 不可用、
  resnet18 的残差 add 没有量化后端实现）。

> 注：部分实验同时覆盖主线与 PTQ 专题两篇文章（同一机制），已在上表合并。
