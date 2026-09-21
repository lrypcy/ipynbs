## 对应文章

- 《大模型量化算法（23）：LLM PTQ 统一视角》 https://lrypcy.github.io/2026/08/29/llm-quant-23-unified-view/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace unified_view.ipynb
```

纯 numpy + matplotlib，CPU 秒级（smoke）/ 分钟级（full）；SEED=0 可复现。
**合成单层探针（激活带离群通道、权重带重尾），不是真实模型精度。**

## 实验内容

在一个线性层 $y=XW^\top$（激活 per-tensor 8-bit、权重 per-channel $b$-bit）上，
把 F2 / F4 / F5 / F8 各自的边际收益量出来，指标是**测试集输出相对误差**（不是权重误差、也不是校准集误差）。

- **F2**：逐输出通道 scale 的网格搜索（LSQ / MSE-observer 那一族）
- **F4a**：GPTQ 误差补偿（$H^{-1}$ 把误差分摊到未量化的列）
- **F4b**：AdaRound 坐标下降（解析判据 $\Delta\mathcal L=-2\delta(DH)_{o,i}+\delta^2H_{i,i}$）
- **F5a**：SmoothQuant 的 $s_j=\max|X_j|^\alpha/\max|W_{:,j}|^{1-\alpha}$
- **F5b**：AWQ 式激活感知缩放 $s_j=(\overline{|X_j|}/\overline{\overline{|X|}})^\gamma$ —— **与 F5a 同一个自由度的另一种配方**
- **F8**：STE 训练权重去适配网格（夹在网格内防漂移 + 按校准损失早停）

三条判断：

- **判断一**：同类（F5a⊕F5b）叠加效率 << 1，异类（F5⊕F2）接近 1
- **判断二**：bit 下降时各自由度的收益塌缩 → 必须换赛道
- **判断四**：F5 是严格恒等变换，$X'W'^\top=XW^\top$ 到浮点舍入级，可折叠进相邻层

## 关键数字（SEED=0，smoke 实测，@W4A8，RTN 基线 0.3064）

| 项目 | 关键数字 |
|---|---|
| 单自由度增益 | F5a(SmoothQuant α=0.45) **51.6%**、F5b(AWQ γ=0.86) **42.0%**、F2 **29.5%**、F4a **-0.8%**、F4b **-1.8%**、F8 **-0.1%** |
| 判断一 | 同类 F5a⊕F5b 效率 **0.551**、F4a⊕F4b **0.539**；异类 F5a⊕F2 **0.899**、F5b⊕F2 **0.957** |
| 判断二 | F5a 增益 8-bit **81.7%** → 4-bit **51.6%** → 3-bit **33.7%** → 2-bit **10.3%**；F2 在 2-bit 也从 31.4% 掉到 **19.6%** |
| 判断四 | $\|XW^\top-(X/s)(Ws)^\top\|_{\max}$ = **1.8e-15**（相对输出量级 **2.4e-16**） |

> **诚实注释**：本探针上 F4（GPTQ/AdaRound）几乎没有收益——per-channel 网格上 RTN 已近最优落点，
> GPTQ/AdaRound 只能在 $H$ 的估计噪声里找便宜，泛化到测试集后归零甚至略负。
> F8 同理：单层线性层 + 校准与测试同分布时没有可"适应"的空间。

## 预期输出

- `results/uv_a_single_dof.png`、`uv_b_stacking_efficiency.png`、`uv_c_bit_migration.png`、`uv_d_deployability.png`
- `results/stdout.txt` / `results/results.json`
