## 对应文章

- 《大模型量化算法（11）：伪量化算子插入——QAT 的地基》
  https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/
- 《大模型量化算法（18）：LSQ / PACT / DSQ——可学习的 scale 与 clip》
  https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/
- PyTorch 官方 `torch.ao.quantization`（eager 模式：fuse → prepare_qat → convert）

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/envs/torch/bin/jupyter nbconvert --to notebook --execute --inplace official_torchao_qat.ipynb
# 或在 JLab 里打开后把第一个代码格的 MODE 改成 "full" 再 Run All（全量版）
```

解释器固定为 `/Users/congyuan/Software/miniconda3/envs/torch/bin/python`
（torch 2.10.0 / torchvision 0.25.0 / numpy 1.26.4，CPU + qnnpack；MPS 本机可用但未混用）。
首个代码格 `MODE = "smoke" | "full"` 单开关控制全部规模；`CFG` 字典集中所有超参。smoke 跑完 < 5 分钟。

> 注意 A：torch 2.10 下 `quantize_fx` 已迁走（顶层 `prepare_fx` 不存在），虽 `torch.ao.quantization.quantize_fx`
> 子模块仍可导入（FX 图模式，已 deprecate），本实验统一用更稳的 **eager 模式 `prepare_qat` / `convert`**。
> 注意 B：运行环境外网 HuggingFace 不可达，所有权重均随机初始化并明确标注为合成数据。

## 实验内容

- **实验 1｜官方 QAT 全流程**：小 CNN（Conv-BN-ReLU + Pool + FC，带 QuantStub/DeQuantStub、可 fuse）
  在合成 10 类「模板 + 高斯噪声」任务（SIGMA=0.9，刻意调难以产生可见量化代价）上，走通
  `fuse(eval) → set qconfig → prepare_qat(train) → 训练 → convert(eval)` 得到**真 int8** 模型。
  报告训练损失曲线、INT8 vs FP32 的准确率/损失、state_dict 体积（字节）、推理延迟（中位数）。
- **实验 2｜FakeQuantize 内置行为**：固定受控权重 `W(64×128, 2% 离群)` 与尖峰+长尾激活 `A`，用官方
  `FakeQuantize` 各配置量化后比较**权重重构 SNR(dB)**，回答三件事：
  - 粒度：per-tensor vs per-channel（per-channel 显著增益）
  - 对称/非对称：per-tensor 对称 vs 非对称(affine)
  - observer：MovingAvgMinMax vs Histogram（激活）
- **实验 3｜手撸 LSQ vs 官方 FakeQuantize（复现 18 篇「学 s 比固定 min-max 好几个 dB」）**：
  在 18 篇 §2 的「过定线性回归」受控权重上，用自写 `autograd.Function`（实现 LSQ Eq.(3) scale 梯度，
  带 `g=1/√(n_Q·Q_P)` 缩放、s 进 `weight_decay=0` 参数组）学 step size `s` 做重构，
  与官方 `MovingAvgMinMaxObserver` 固定 min-max 对照；并附离线网格最优 `s*`（校验梯度）与 per-channel 粒度增益。

## 关键数字（SEED=0，smoke 模式实测）

**实验 1（8-bit 官方 QAT 全流程，qnnpack，CPU）**

| 模型 | val_acc | val_loss | sd_bytes | 延迟(ms) |
|---|---|---|---|---|
| FP32 | 0.9922 | 0.2357 | 27635 | 2.682 |
| INT8 | 0.9414 | 0.1871 | 10809 | 1.236 |

- INT8 相对 FP32 下降 **5.08 个百分点**（合成代理指标，SIGMA=0.9）
- state_dict 体积比值 **2.56x**（小模型因 observer/buffer 元数据，INT8 磁盘体积未达 4x）
- 理论权重压缩 **4.0x**；真实推理延迟加速 **2.17x**

**实验 2（8-bit 权重重构 SNR，越高越保真）**

| 配置 | recon SNR(dB) |
|---|---|
| W per-tensor 对称 | 33.98 |
| W per-tensor 非对称 | 35.42 |
| W per-channel 对称 | 41.39 |
| A MovingAvgMinMax | 25.58 |
| A Histogram | 26.12 |

- per-channel 比 per-tensor **+7.41 dB**；对称 vs 非对称 **+1.44 dB**；激活 Histogram 比 MAvg **+0.54 dB**

**实验 3（4-bit 过定线性回归，复现 LSQ 结论）**

| bits | min-max per-tensor | per-channel | 离线最优 s* | LSQ(学 s) | LSQ−min-max |
|---|---|---|---|---|---|
| 4 | 9.42 | 16.27 | 15.42 | **15.73** | **+6.30 dB** |
| 8 | 33.98 | 41.39 | 34.18 | 34.23 | +0.25 dB |

- 4-bit：**LSQ 学 s 比固定 min-max 高 6.30 dB**（复现「好几个 dB」）
- LSQ 15.73 dB 与离线最优 s* 15.42 dB 仅差 0.30 dB（Eq.(3) 梯度正确收敛）
- per-channel 再 +6.85 dB（粒度增益，与可学习性正交）
- 8-bit 下 LSQ 仅 +0.25 dB —— **可学习 step 的收益是低比特现象**

## 诚实标注

- 分类**准确率/损失**为合成随机数据上的**代理指标**，仅用于同任务排序，绝对值不具迁移性。
- **权重重构 SNR(dB)、state_dict 体积、推理延迟**是**真实度量**的数字。
- MPS 本机可用，但 eager 量化算子仅 CPU 后端，为公平对比未混用。

## 预期输出

- `results/qat_official_full_flow.png`（实验 1：训练损失曲线 + 精度/延迟对比）
- `results/fakequant_behavior_ablation.png`（实验 2：粒度/对称/observer 重构 SNR）
- `results/lsq_vs_official_fakequant.png`（实验 3：LSQ vs 固定 min-max 重构 SNR）
- `results/fp32_state_dict.pt` / `results/int8_state_dict.pt`（真实 state_dict）
- `results/results.json` / `results/stdout.txt`（全部数字与可读表格）
- 执行后的 `official_torchao_qat.ipynb`（含全部输出）
