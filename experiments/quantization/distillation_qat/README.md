## 对应文章

- 《大模型量化算法（20）：蒸馏量化 QAT》 https://lrypcy.github.io/2026/09/19/llm-quant-20-distillation-qat/
- 相关：《22 Reasoning LLM 低比特量化》 https://lrypcy.github.io/2026/09/19/llm-quant-22-reasoning-llm-lowbit/ （§5 reward rectification）

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace distillation_qat.ipynb
```

纯 numpy + matplotlib，CPU 秒级（smoke）；SEED=0 可复现。
**合成 logit 分布探针，不是真实 LM 精度。**

## 实验内容

关键设定：**学生是 fake-quantized 的**——前向走 4-bit 网格上的 logits，反传走 STE。这正是 QAT 的设定，
也是"蒸馏目标设计和普通 KD 不一样"的根本原因。

- **A. 量化到底损失了什么**：top-1（argmax）几乎不变，掉的是**分布形状**（尾部质量 + 熵）
- **B. 支撑集**：top-K 截断，扫 K 看 U 形曲线
- **C. 温度**：为什么"温度越高越好"在量化学生身上翻车
- **D. 方向**：mode-seeking KL(T‖S) / mode-covering KL(S‖T) / 对称 JSD
- **E. 广义 JSD（UPQ 目标）**：为什么 2-bit 指令模型必须换掉 KL——梯度范数的有界性
- **F. Reward rectification**：教师概率加权 vs 学生概率加权

## 关键数字（SEED=0，smoke 实测）

| 实验 | 关键数字 |
|---|---|
| A | 2-bit 时 top-1 仍不变，但 KL 单调恶化、尾部质量与熵持续漂移 → 分类指标看不见掉点 |
| B | 最佳 K 处 KL 明显低于全词表 KL；K 太小支撑集不足、K 太大长尾噪声回流（U 形） |
| C | 最佳 T 与最差 T 的 KL 差距显著；高温度把学生训练成**明显过平**（熵差大） |
| D | KL(S‖T) 会让学生"糊成一片"；KL(T‖S) 挑主峰贴，对量化学生更友好 |
| E | 2-bit 静态学生：KL 梯度范数 vs JSD 梯度范数低一个量级；**max JSD ≤ log2 = 0.693** |
| F | p_S=0.41 时 1/p_S 相对教师加权放大 **3.39x**；p_S=0.05 时放大 **27.78x** |

## 预期输出

- `results/dq_a_what_quantization_costs.png`、`dq_b_topk_support.png`、`dq_c_temperature.png`、`dq_d_direction.png`、`dq_e_generalized_jsd.png`、`dq_f_reward_rectification.png`
- `results/stdout.txt` / `results/results.json`
