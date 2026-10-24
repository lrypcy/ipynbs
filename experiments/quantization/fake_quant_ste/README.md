## 对应文章

- 《大模型量化算法（17）：伪量化算子插入——QAT 的地基》
  https://lrypcy.github.io/2026/08/26/llm-quant-11-fake-quant-insertion/
- 《大模型量化算法（18）：LSQ / PACT / DSQ》§1.1（本篇定义的三个自由度从这里被解锁）
  https://lrypcy.github.io/2026/08/29/llm-quant-18-lsq-pact-dsq/

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace fake_quant_ste.ipynb
# 或在 JLab 里打开后把第一个代码格的 MODE 改成 "full" 再 Run All
```

纯 numpy + matplotlib，CPU 秒级（smoke）/ 分钟级（full）；`SEED=0`，结果可复现。
`MODE = "smoke" | "full"` 一个开关控制全部采样数与位宽网格。
**所有任务均为合成探针（synthetic probe），不是真实模型精度**，只用于验证公式与量级。
notebook 末尾打印 `assertions passed: N`，全部通过才算跑通。

## 实验内容

### A. 伪量化的完整定义与两段误差

非对称参数 $$s=0.5,\;z_p=20,\;q\in[0,255]$$（可表示区间 $$[-10.0,\,117.5]$$）上，对 8 个元素
逐步打印 `x/s → round → +z_p → clip → 反量化`，覆盖格点 / 未饱和 / 上溢 / 下溢四种情形，
并验证三件事：

1. 手算结果与 `fq()` 逐元素相等；饱和元素被 clip 到 `q_min`/`q_max`；
2. `s/2` 只约束**未饱和**元素——饱和元素误差可达 $$s/2$$ 的 60～130 倍；
3. **漏减 `z_p` 的写法使每个元素整体偏移 $$s\,z_p$$**（`x=3.2` 处误差从 0.20 放大到 9.80，49 倍）。

### B. 粒度 × 位宽

$$W\in\mathbb{R}^{256\times128}$$（6 个离群列 ×12），比较 per-tensor / per-channel / per-group(32) × 对称/非对称 × 2/4/8 bit。
**per-group 在本实现下沿输入维切块、块内对整块（行×列）取 absmax**，比 per-channel 更粗：

### C. 激活侧的 clamp

重尾激活（$$|t(2.5)|\times0.8$$）上对比 min-max 不裁剪与分位裁剪。
结论**按位宽换向**：4-bit 裁剪是赚的，8-bit 是亏的——舍入误差 ∝ $$s^2$$ 随位宽指数下降，
而饱和误差与 $$s$$ 无关。

### D. STE 的判据、形状与直通比例

1. 逐点：`g_STE` 恒为 1（范围外也有一段），真实数值差分几乎处处为 0，跳变处的差商是
   $$s/(2\varepsilon)$$ 的**差分伪影**（有限值，$$\varepsilon\to0$$ 时发散）；
2. **放行判据是 `round(x/s)+z_p ∈ [q_min, q_max]`，比可表示区间宽一整格**（不是半格）；
3. 直通比例有闭式解 `min(1, (q_max+1)s/M)`（均匀分布），随 scale 收紧而下降。
   min-max scale 下直通率恰好 100%，但分辨率最粗——「梯度拿得多」与「量化误差小」方向相反。

### E. 插入位置端到端

两层 MLP（256→64→128），只改 `act_q` 传不传就得到三种部署方案；测试集尺度是校准集的 3 倍。
判断「某张量是否真的挂了 FQ」用**网格对齐率**（排除 ReLU 压出的精确 0），不能用 rel.MSE。

最后一组用两支 scale 相差 8.5× 的残差相加，检验「两支必须共享网格」这条规则——结论是不成立。

## 关键数字（SEED=0，smoke 模式实测）

| 组 | 关键数字 |
|---|---|
| A | 饱和元素误差：下溢 15.00、上溢 32.50（$$s/2=0.25$$）；未饱和 max 误差 0.007874 ≤ $$s/2$$；20000 样本中被裁剪 883 个 |
| B | 4-bit per-tensor 1.348e-01 → per-channel 1.598e-02（**+9.26 dB**）；per-group(32) 1.115e-01 更差（+0.83 dB） |
| B（反直觉格） | 2-bit：per-tensor 0.8688 / per-channel 0.7107 / **per-group 0.7700（比 per-channel 差 8.3%）** |
| C | 4-bit：不裁剪 3.0236e-01（9/16 码字）→ clip@p0.999 1.0346e-01（**+4.66 dB**）；8-bit：不裁剪 1.8780e-03 → 裁剪@p0.999 7.4811e-02（**−16.00 dB**） |
| D | 数值差分非零占比 0.37%，非零差商恒为 ±500（=$$s/2\varepsilon$$）；直通比例 1.0000 → 0.5715（随 $$M/(s\,q_{\max})$$ 从 0.25 到 2.0） |
| D（真实重尾） | min-max scale 直通 100.00%；p0.999 → 99.91%；p0.99 → 99.17%；p0.9 → 91.23% |
| E（网格对齐率） | W4A16：X 0.0000、a 0.0000；W8A8：X 1.0000、a 1.0000 |
| E（校准错配） | 静态 logits 误差 7.944e-04（同分布）→ 7.535e-02（尺度差 3×），**+19.8 dB**；动态 5.872e-04 不变 |
| E（共享网格反例） | 两支 scale 0.0559 / 0.4739（差 8.5×），A 支 col0 与 col1 相关系数 **−0.0106**，相加 rel.MSE 9.1e-05 |

## 与旧结论的差异（本版已更正）

| 旧说法 | 实测 |
|---|---|
| 「4-bit per-tensor → per-group 白捡 13.02 dB」 | 本网格上 per-group(32) 只有 +0.83 dB，且**比 per-channel 差**。"白捡 dB" 是另一个实验（`quantizer_granularity`）的 per-group(128) 数字，不该混用 |
| 「STE 梯度均值 1.0 vs 软量化 0.5，系统性高估一倍」 | 该对比口径不成立：STE 是**替代导数的假设**，不是软量化梯度的偏差项。改成"逐点 vs 一格积分"与"直通比例"两个可测口径 |
| 「残差两支必须量化到同一网格」 | 实测两支不同 scale 时输出**不退化**。共享 scale 来自整数域加法的执行约束，不是量化语义要求 |
| C 组「码字数」打印 | 旧版把 `2**4-1+1` 硬编码进日志，与实测 used 码字数不符；本版用实测变量 |

## 预期输出

- `results/fq_a_roundtrip.png`、`fq_b_granularity.png`、`fq_c_clip.png`、`fq_d_ste.png`、`fq_e_insertion.png`
- `results/stdout.txt` / `results/results.json`
- 执行后的 `fake_quant_ste.ipynb`（含全部输出）

**断言总数：29**（smoke 模式，全部通过）
