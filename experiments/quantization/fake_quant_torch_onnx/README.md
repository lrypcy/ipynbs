## 对应文章

- 《大模型量化算法（17）：伪量化算子插入——QAT 的地基》
  https://lrypcy.github.io/2026-08-26/llm-quant-11-fake-quant-insertion/
- 《大模型量化算法（18）：LSQ / PACT / DSQ》
  https://lrypcy.github.io/2026-08/29/llm-quant-18-lsq-pact-dsq/

姊妹实验：`fake_quant_ste/`（纯 numpy，讲算子数学与插入位置）。**本实验讲真实框架的行为**，
所有结论都来自可执行断言，不来自文档转述。

## 如何运行

```bash
cd <本目录>
~/Software/miniconda3/envs/torch/bin/jupyter nbconvert --to notebook --execute --inplace fake_quant_torch_onnx.ipynb
```

- 解释器：`~/Software/miniconda3/envs/torch/bin/python`（需要 torch；base python 没有）
- **纯 CPU**，无 GPU 依赖；`SEED=0`，smoke 秒级 / 分钟级
- 环境版本（实测）：**torch 2.10.0 / onnx 1.20.1 / Python 3.11.14 / macOS**
- 本机**没有 onnxruntime**，因此第 4 组只做结构断言，不做 ONNX 数值前向

## 实验内容

### 1. FakeQuantize 的数值语义与 STE 梯度
- (a) 手写 round-trip `s·(clip(round(x/s)+z_p) − z_p)` 与 `torch.fake_quantize_per_tensor_affine`
  在 quint8 非对称 / qint8 对称 / zp=0 三组参数下一致（max|Δ| < 1e-6，差异只在 float32 舍入）；
- (b) **11 个探针点的梯度实测表**——这是本篇最重要的一组；
- (c) `MinMaxObserver` 在 `[-1,0,2,5]` 上的 qparams；
- (d) `PerChannelMinMaxObserver` 的 scale/zp 形状与逐通道值；
- (e) **int4 非对称下「漏写 `z_p`」的两种写法对照**：量化/解码两处都不写 `z_p`，
  与只在解码处漏写 `z_p`——两者偏差形态完全不同；
- (f) **放行域端点的可判定断言**（`scale=1.0` 整数格点，六个探针点可直接手算复核）。

### 2. Eager mode QAT
`get_default_qat_qconfig("x86")` 逐字段打印；BN 折叠的**两种产物**；训练 40 步；
`disable_observer` 后 scale 逐位不变而权重仍在更新。

### 3. FX graph mode QAT
`prepare_qat_fx` 后原样打印图；统计 `activation_post_process` 节点数与真实类型；
确认权重的伪量化**不在图上**（在 `Linear` 内部）。

### 4. PT2E → ONNX QDQ
两条导出路径都跑；QDQ 节点逐个列 scale 形状 / axis / 是否 initializer。

### 4b. 对照组：per-channel 配置能否导出带 axis 的 QDQ
同一模型换 `is_per_channel=True` 重走一遍，捕获异常链的**根因**。

### 5. 三层表示汇总表 + 放行域示意图

## 关键数字 / 事实（本机实测）

| 组 | 实测结论 |
|---|---|
| 1(a) | 三组参数下手写与 torch 一致，max\|Δ\| ≤ 2.48e-07（float32 舍入量级）；**漏减 `z_p` 使每个元素整体偏移 `s·z_p=0.84`**——未饱和元素 `x=0.0` 本应精确映到 0，漏减后变成 0.840，**zero-point 存在的意义被直接破坏** |
| 1(b) | **STE 放行判据不是「x 在可表示区间内」，而是 `round(x/s)+z_p ∈ [q_min, q_max]`**。实测：`x/s=127.49`→grad 1、`127.50`→**0**、`128.00`→0；`-128.50`→**1**、`-129.00`→0。两端半格点行为相反，因为 round 是 **ties-to-even**（127.5→128 越界，−128.5→−128 未越界） |
| 1(c) | **决定对称性的开关是 `qscheme`，不是码字范围**：`qint8` + 默认 `per_tensor_affine` → `z_p=−86`（非 0）；显式 `per_tensor_symmetric` → `z_p` 钉在码字中点（qint8 是 0、**quint8 是 127**）。两种情况下 `x=0` 都能被精确反量化为 0.0。附带：PyTorch 的 symmetric scale = `(x_max−x_min)/(q_max−q_min)`=10/255，**不是** 教科书的 `absmax/q_max`；`x=5.0` 在前者下被量化成 4.9804（absmax 不精确落格），后者下精确为 5.0 |
| 1(e) | `x=[-1,0,2,5]`、`s=5/7`、`z_p=-8`、`q∈[-8,7]`：torch 与手写 `s*(clip(round(x/s)+z_p)-z_p)` **逐元素严格相等** = `[0, 0, 2.1429, 5.0]`；**两处都不写 `z_p`** 的 `s*clip(round(x/s))` = `[-0.7143, 0, 2.1429, 5.0]`（偏差只在饱和元素上非零，**不是**常数 `s·z_p`）；**只在解码漏减**则整体平移 `s·z_p = -5.7143`（每个元素都偏）。正确区间 `[0, 10.7143]`，前者错成 `[-5.7143, 5.0]` |
| 1(f) | 对称 int8 `s=1.0, z_p=0, q∈[-128,127]` 下六个探针点梯度 = **`[1, 0, 0, 0, 1, 0]`**（`x/s=127.49, 127.50, 127.51, 128.0, -128.50, -128.51`）：切点恰在**每侧半格**，上界不含（`round(127.5)=128` 越界）、下界含（`round(-128.5)=-128` ties-to-even 未越界） |
| 2 | x86 QAT 默认激活 `MovingAverageMinMaxObserver(quant_min=0, quant_max=255, `**`reduce_range=True`**`)` → **名义 8-bit、实际 128 个码字**；权重 `MovingAveragePerChannelMinMaxObserver`，`per_channel_symmetric`，`reduce_range=False` |
| 2 | eval 版 `fuse_modules` 在 train 模式下**报错拒绝**（`Fusion only for eval!`）；QAT 要用 `fuse_modules_qat`，产物是 **`ConvBn2d`**（BN 作为可训练子模块留在里面），而不是把 BN 抹掉 |
| 2 | 训练 40 步后 `rel.MSE(W, FQ(W))` 从 3.7429e-03 涨到 **5.3566e-02** → 权重每步都被重新伪量化，不是「离线一次」 |
| 2 | `disable_observer` 后再训 5 步：scale 用 `torch.equal` 断言**逐位不变**，权重仍在变 |
| 3 | 图上节点叫 `activation_post_process_*`，**名字里没有 FakeQuant**；真实类型 `FusedMovingAvgObsFakeQuantize`。权重侧是 `fc*.weight_fake_quant` 子模块，**图上没有独立节点** |
| 3 | `weight_fake_quant.scale.shape` 在 `prepare` 刚结束时是占位的 **(1,)**，跑过一次前向才变成 **(32,)** |
| 4 | **x86 Inductor quantizer 给 `nn.Linear` 的 convert 后图里 quantize 调用 = 0 处**（裸 `aten.linear`）；要显式 QDQ 必须换 XNNPACK quantizer |
| 4 | 旧 TorchScript 导出器（`dynamo=False`）报 `UnsupportedOperatorError`：认不出 `quantized_decomposed` 命名空间 |
| 4 | `dynamo=True` 成功：**QuantizeLinear 3 个 / DequantizeLinear 5 个**，opset **20**（导出器自选，不是我传的 18）；全部 scale 是 initializer；`zero_point` dtype = **INT8**（与规范「y_zero_point 与 y 同类型」一致） |
| 4 | 权重侧的 `QuantizeLinear` **被折叠成 int8 initializer**，图上只剩 `DequantizeLinear` → 训练图/表示图/执行图三态差别的直接证据 |
| 4 | 8 个 QDQ 节点**全部带 `axis=1`，但 scale 全是标量** → axis 在这里是冗余属性。**判粒度必须看 scale 的 shape，不能看有没有 axis** |
| 4 | 2/2 个矩阵乘的输入都来自 `DequantizeLinear`（浮点）→ **QDQ 图 ≠ 整数 kernel** |
| 4b | per-channel 配置导出**失败**，根因：`DispatchError: No ONNX function found for quantized_decomposed.dequantize_per_channel`。ONNX 规范支持 per-axis，但**导出链路不支持** |

## 与常见说法不符的地方（本机实测）

1. **「对称量化 ⇒ `z_p = 0`」** → 两句话都要拆开看：只把码字范围设成 `(-128, 127)` **不等于**对称量化，
   默认 `qscheme=per_tensor_affine` 实测给 `z_p=−86`；显式 `per_tensor_symmetric` 下 `z_p` 被钉在
   **码字区间中点**——qint8 是 0，**quint8 是 127**（不是 0）。所以 `z_p=0` 既不是对称的必要条件也不是充分条件，
   判对称性看 `qscheme`；而两种情况下 `x=0` 都能精确反量化为 0.0，这才是 `z_p` 的定义性作用（1(c)）。
2. **「STE 就是把量化当作不存在」** → 放行判据作用在 round **之后**的整数码上：
   放行域 = `[q_min·s − s/2, q_max·s + s/2)`，**两端各宽半格**且上界不含、下界含
   （ties-to-even：`round(127.5)=128` 越界切断，`round(−128.5)=−128` 未越界仍放行）；
   写成 `1[x ∈ [q_min·s, q_max·s]]` 与 PyTorch 实际行为不等价（1(b) / 1(f)）。
3. **「x86 QAT 是 INT8 激活」** → 默认 `reduce_range=True`，实际只用 128 个码字。
4. **「quantize_fx() 一步完成」「prepare/convert 是一条通用流水线」** → PTQ 与 QAT、
   eager 与 FX、PT2E 是不同流程，`prepare_qat_fx` 之后必须训练、必须冻结 observer。
5. **「导出成 QDQ 就是量化部署完成」** → 导出图里矩阵乘的输入仍是浮点；
   per-channel 甚至根本导不出来。
6. **TVM 相关**（不在本实验，见文章 §7.5）：`qnn.simulated_quantize` 签名里没有
   `min_range`/`max_range`；annotate/calibrate/realize 属于 `relay.quantize` 而非 `relay.qnn`；
   **relay qnn 在 v0.20.0 就被删了**（commit `ccaa534b2`，2025-02-15），不是更晚。

## 预期输出

- `results/fq_g3_fx_graph.png`、`results/fq_g5_ste_domain.png`
- 1(e)/1(f) 的数值与断言在 `results/stdout.txt` 的 `[1e]` / `[1f]` 两段，
  并写入 `results/results.json` 的 `1e_A2_int4_zeropoint` / `1f_D2_halfstep`
- `results/fq_qdq_model.onnx`（可直接用 `onnx.load` 打开的 QDQ 图）
- `results/stdout.txt` / `results/results.json`
- 执行后的 `fake_quant_torch_onnx.ipynb`（含全部输出）

**断言总数：49**（全部通过）
