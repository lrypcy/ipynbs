## 对应文章

- 《大模型量化算法（01）：量化器数学地基与 RTN 基线》
  https://lrypcy.github.io/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/
- 《E1：RTN 与 LLM.int8()》 https://lrypcy.github.io/2026/08/24/ptq-01-rtn-llmint8/
- 《17：伪量化算子插入》 https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/
- 《E3：部署侧视角》 https://lrypcy.github.io/2026/09/19/llm-quant-E3-deployment-support/

## 与本目录 sibling 的区别

| 目录 | 实现 | 回答什么 |
|---|---|---|
| `rtn_llmint8/`、`quantizer_granularity/` 等 | 纯 numpy 手撸 | 算法机理、公式、梯度、可控消融 |
| **本目录** | **PyTorch 官方栈**（`torch.ao.quantization`） | 官方 API 的默认行为、真实体积/延迟、**本机哪些 API 跑不通** |
| `official_torchao_qat/` | PyTorch 官方栈 | `prepare_qat` → `convert` 全流程 + 自定义 LSQ/PACT |
| `official_llm_ptq_torch/` | transformers + torch | 真实 LLM 上的 RTN / GPTQ / SmoothQuant / AWQ |

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/envs/torch/bin/jupyter nbconvert \
    --to notebook --execute --inplace official_torchao_ptq.ipynb
```

需要 `miniconda3/envs/torch`（torch 2.10.0 + torchvision 0.25）。
真实权重来自 `/tmp/resnet18.pth`（ImageNet 预训练 resnet18，46 MB）；若缺失会退回随机初始化并在
stdout 中标注。校准/评估用合成图像（无 ImageNet 数据），因此**体积/延迟/scale 是真实度量，
logits 只做 FP32 与 PTQ 的相对比较**。`MODE = "smoke" | "full"` 控制规模。

## 实验内容

- **A. 官方 observer 到底选什么 scale**：MinMax / Histogram / MovingAverage / 对称 vs 仿射，
  与手撸 min-max、MSE 网格最优同口径对比
- **B. 官方 FakeQuantize 全流程 PTQ**：逐层量化真实权重，8/6/4 bit × per-tensor / per-channel，
  用 logits rel-MSE 与 cosine 相似度评估
- **C. 真实存储与元数据税**：`quantize_per_*` + 4/6-bit 位打包；元数据税随粒度（per-tensor →
  per-channel → per-group(64/32/16)）的上升曲线
- **D. 真 int8 算子 + 延迟**：`prepare → calibrate → convert → int8 前向` 全流程实测；
  并记录本机（arm64 macOS, torch 2.10）官方量化栈的可用性清单

## 关键数字（smoke 实测，真实 resnet18 权重）

| 结论 | 数字 |
|---|---|
| 官方 MinMax(affine) vs 手撸非对称 min-max | scale 相对差 **0.000%**（官方 API 没有魔法） |
| 官方默认激活 observer（Histogram）vs min-max | rel.MSE 5.754e-05 vs 7.874e-05，**好 1.37 dB**，几乎等于 MSE 网格最优（5.742e-05） |
| 8-bit 权重 per-channel vs per-tensor | logits rel.MSE **好 3.91 dB**（6-bit 时 +9.57 dB） |
| 真实压缩比（含 scale 元数据） | 8-bit 4.00x / 6-bit 5.33x / 4-bit 8.00x（per-tensor） |
| 4-bit 元数据税 | per-tensor 0.02% → per-channel 1.37% → per-group(16) 更重 |
| 真 int8 算子（SmallCNN, CPU, qnnpack） | 体积 2.81x、延迟 1.09x |

## 本机真实边界（对 E3 部署篇有用）

- `torch.backends.quantized.supported_engines == ['qnnpack']`，**必须显式**
  `torch.backends.quantized.engine = 'qnnpack'`，否则 `convert` 报 `NoQEngine`
- **resnet18 静态量化 `convert` 后前向失败**：`quantized::conv2d.new` 无 CPU 后端实现
  （残差 add 也缺 `aten::add.out` 的 QuantizedCPU 实现，需 `FloatFunctional`）
- `torch.ao.quantization` 在 torch 2.10 已 **deprecated**，官方迁移目标是 **torchao**
- 经典栈的量化张量类型只有 `qint8/quint8`：**4/6-bit 必须自己做 bit-packing**

## 预期输出

- `results/official_observer_scale_and_error.png`、`official_ptq_weight_bits.png`、
  `official_storage_and_metadata_tax.png`、`official_int8_size_and_latency.png`
- `results/stdout.txt` / `results/results.json`
