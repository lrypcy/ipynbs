## 对应文章

- 《大模型量化算法（25）：混合精度》 https://lrypcy.github.io/2026/09/19/llm-quant-25-mixed-precision/
- 相关：《LLM PTQ 统一视角》 https://lrypcy.github.io/2026/08/29/llm-quant-23-unified-view/ （F7 一维）

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace mixed_precision.ipynb
# 或在 JLab 里把第一个代码格的 MODE 改成 "full" 再 Run All
```

纯 numpy + matplotlib，CPU 秒级（smoke）/ 分钟级（full）；SEED=0 可复现。
**全部为合成探针（synthetic probe），不是真实模型精度。**

## 实验内容

形式化是**多重选择背包问题**：$\min \sum_\ell \mathcal{E}_\ell(b_\ell)\ \text{s.t.}\ \sum_\ell \mathcal{S}_\ell(b_\ell)\le B$。

- **A. 敏感度长尾**：Pareto 分布的 per-unit 敏感度，看"统一比特"为什么亏
- **B. 三种求解器**：边际收益贪心 / 拉格朗日扫描 / 精确 DP（多重选择背包），对照组是**预算内最好的统一方案**（不是"同 bit 的统一方案"）
- **C. λ 扫描 = 完整 Pareto 前沿**：顺手拿到精度-体积权衡曲线，并用精确 DP 校验
- **D. 敏感度度量选哪个**：在**实测**探针（误差表是量出来的，不是公式造的）上比较 Hessian 类 / 激活幅度类 / 权重幅度 / 重建误差类
- **E. MoE 专项**：路由动力学（router Δℓ2）vs 调用频率，300 轮 bootstrap 验证二者无关
- **F. 硬件可行性校验**：显存下降 vs 混合精度额外开销（10/20/30%）的净收益 + packing 效率

## 关键数字（SEED=0，smoke 实测）

| 实验 | 关键数字 |
|---|---|
| A | 敏感度 max/min = **84.6x**；top-4 单元占 **43.3%** 敏感度 |
| B | 预算 = 统一 4-bit 的 70% → 只能塞下统一 **2-bit**。贪心 **0.271**、拉格朗日 **0.273**、精确 DP **0.271**（相对统一 2-bit 基线） |
| C | 拉格朗日与精确 DP 只差 **+1.05%**；贪心差 **+0.07%** |
| D | 重建误差类与实测代价 Spearman **+0.424**，Hessian 类 **+0.051**、激活幅度类 **+0.018**、权重幅度 **-0.096** → "最朴素但最可靠" |
| E | 300 轮 bootstrap rho 均值 **+0.012**（95% 区间 [-0.513, +0.568]，覆盖 0）→ **频率与脆弱度无关**；router W4 时 token 改道率 **20.0%**、W2 时 **82.8%** |
| F | 相对统一 4-bit 显存下降 **30.3%**；额外开销 20% 时净收益仍 **+0.103**；3/6-bit 的 packing 效率 **93.8%** |

## 预期输出

- `results/mp_a_sensitivity_tail.png`、`mp_b_bit_allocation.png`、`mp_c_pareto_frontier.png`、`mp_d_metric_agreement.png`、`mp_e_moe_router.png`、`mp_f_hardware_check.png`
- `results/stdout.txt` / `results/results.json`
