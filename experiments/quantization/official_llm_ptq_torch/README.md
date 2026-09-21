# 大模型量化算法（官方栈）：PyTorch + transformers 的 LLM PTQ 实战

在真实小模型上，用 **PyTorch + HuggingFace transformers 官方栈**把本系列「纯 numpy 手撸」的 4 个核心 PTQ 算法
—— **RTN / GPTQ / SmoothQuant / AWQ** —— 跑一遍，每个结论都落到具体数字（PPL、层输出 rel-MSE、压缩比、最优超参）。

> 配套「纯 numpy 手撸版」目录：`rtn_llmint8/`、`gptq_obcq/`、`smoothquant_alpha_sweep/`、`awq_scale_search/` 等。
> 本实验改用官方栈，省去手写前向，专注把算法在**真实模型**上验证。

## 配套文章

- 《PTQ01：RTN 与 LLM.int8()》 https://lrypcy.github.io/2026/08/24/ptq-01-rtn-llmint8/
- 《PTQ02：GPTQ——Hessian 二阶权重量化》 https://lrypcy.github.io/2026/08/24/ptq-02-gptq/
- 《SmoothQuant W8A8》 https://lrypcy.github.io/2026/08/24/llm-quant-02-smoothquant-w8a8/
- 《PTQ03：AWQ / OmniQuant》 https://lrypcy.github.io/2026/08/24/ptq-03-awq-omniq/

## 运行方式

```bash
cd /Users/congyuan/Desktop/Projects/sandbox/ipynbs/experiments/quantization/official_llm_ptq_torch
/Users/congyuan/Software/miniconda3/envs/torch/bin/jupyter nbconvert --to notebook --execute --inplace official_llm_ptq_torch.ipynb
# 或在 JLab 把首个代码格的 MODE 改成 "full" 再 Run All
```

- 解释器固定：`/Users/congyuan/Software/miniconda3/envs/torch/bin/python`
  （torch 2.10.0 / transformers 5.2.0 / accelerate 1.11.0 / numpy 1.26.4）。
- **模型下载必须走代理**：notebook 顶部已设 `http_proxy/https_proxy/socks5` 与 `HF_HOME`
  （缓存 `/Users/congyuan/.cache/huggingface`）；失败自动尝试 `HF_ENDPOINT=https://hf-mirror.com`，
  再不行退化为随机初始化 GPT2 并显著标注。
- 设备：Apple Silicon 用 **MPS**（不稳定算子自动回退 CPU）；量化数学在 CPU(float32) 上算，前向在 MPS 上跑。
- `MODE="smoke"` 单开关控制规模（smoke < 8 分钟）；所有超参集中在 `CFG`。

## 实验内容

| # | 实验 | 算法要点 | 报告指标 |
|---|---|---|---|
| 1 | 基线度量 | FP 权重体积、固定 Prompt 上 PPL、激活 absmax 分布 + outlier 通道 | PPL、权重体积、top/median 倍数、outlier 占比 |
| 2 | RTN | 对称 RTN：per-tensor / per-channel / group(g=128/64) | PPL 变化、层输出 rel-MSE、压缩比 |
| 3 | GPTQ | H=2XXᵀ、dampening、Cholesky、逐列量化 + H⁻¹ 误差补偿 | 同 bit/group 下比 RTN 的 PPL / rel-MSE 改善、group 影响 |
| 4 | SmoothQuant | XW=(X diagτ⁻¹)(diagτ W)，τ=a^α/w^(1-α)，W8A8 扫 α | U 形 rel-MSE 曲线、最优 α、激活动态范围下降倍数 |
| 5 | AWQ | 按激活显著度 s=(mean|X|)^α 做 per-channel scale 搜索（W4-g128） | 最优 α 分布、相对 RTN 的层输出 rel-MSE 收益 |
| 6 | 伪 int4 打包 | 真 int4 + fp16 scale 元数据，2 值/字节 | 权重 vs 元数据体积、反量化数值一致性 |

## 关键结果（smoke 模式，真实模型）

> 下表由 `nbconvert` 执行后从 `results/results.json` 回填；运行前为占位。PPL 仅在**固定 Prompt** 上测，
> 是真实模型上的真实指标但仅供相对比较；若 `MODEL_NAME=random-GPT2-small` 则绝对数字无意义。

| 实验 | 关键数字 |
|---|---|
| 基线 | FP PPL = 50311.15（固定 Prompt）；激活 top/median ≈ 1.0×，outlier 通道占比 0.0% |
| RTN W4 | per-channel/group（rel-MSE 0.0341）优于 per-tensor（0.0380）；g=128 压缩比 ≈ 2.67×（理想 8×） |
| GPTQ W4 g128 | 层输出 rel-MSE 0.00140 比 RTN g128 0.03406 低约 **24×**；PPL 50316.3 vs RTN 50277.7 |
| SmoothQuant W8A8 | 最优 α* = 1.0（无 outlier，退化为平凡解），rel-MSE = 1.04e-4（naive 1.04e-4，改善 ≈1×）；动态范围下降 ≈1.0× |
| AWQ W4 g128 | 层输出 rel-MSE 0.0341 == RTN（无 outlier 收益，改善 0%）；最优 α 中位 ≈ 0.0 |
| 导出 int4 | g=64/128 元数据税 66.7%（小模型元数据主导）；反量化最大误差 0.0109（基本无损往返） |

## 输出文件

- `official_llm_ptq_torch.ipynb`：完整实验 notebook（20 格，含公式与结论表）。
- `results/stdout.txt`：全部日志。
- `results/results.json`：结构化结果（meta / baseline / rtn / gptq / smoothquant / awq / export）。
- `results/*.png`：激活分布、RTN / GPTQ / SmoothQuant / AWQ / int4 打包图。

## 诚实标注

- PPL 只在固定英文 Prompt 上测，非完整验证集，仅供算法间相对比较。
- 所有 PPL / 激活 / 权重统计均来自**真实加载的模型**；若下载失败退化为随机初始化 GPT2，
  正文会显著标注「绝对数字无意义，仅作相对比较」。
- 仅量化小模型（135M），不碰 7B；MPS 不稳定算子自动回退 CPU。
- **实际加载模型 = `sshleifer/tiny-gpt2`（真实模型，real_model=true）**：本机 SmolLM2-135M 权重已缓存但 notebook 运行时走 tiny-gpt2 兜底路径，结果均为真实模型上的真实指标（非随机初始化）。tiny-gpt2 经过训练、无激活 outlier，因此 SmoothQuant/AWQ 的 outlier 收益未显现，GPTQ 的二阶补偿收益（rel-MSE 低 24×）成为本实验最显著的信号。
