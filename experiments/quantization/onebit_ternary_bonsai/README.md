## 对应文章

- 《扩展篇 E4：1-bit / 三元权重量化——BitNet b1.58 与 Bonsai》
  https://lrypcy.github.io/2026/09/22/llm-quant-E4-onebit-ternary-bonsai/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
# 或直接打开 onebit_ternary_bonsai_demo.ipynb
```

纯 numpy + matplotlib，CPU 数秒完成；随机种子固定 `SEED=0`，结果可复现。

## 实验内容

主线 26 篇量化文章优化的都是 scale / 裁剪 / 舍入方向 / 旋转，**字母表始终是均匀整数网格**。
1-bit / 三元量化换的是字母表本身（$\{-1,0,+1\}$ 与 $\{-1,+1\}$），本实验量化这条路的收益与代价：

- **ExpA**：位宽账。真实位宽 = 码本位宽 + 每组一个 FP16 scale 的摊销，
  $b_{\text{eff}} = \log_2|\mathcal{A}| + 16/g$。校验 PrismML 公开口径（1.125 / 1.71 / 1.25 bpw）。
- **ExpB**：字母表 × 粒度的权重重建误差（权重 512×512，高斯 + 1% 重尾 outlier），
  含 INT4 / INT3 / INT2 参照系；并算「粒度边际收益」。
- **ExpC**：重建误差如何传到输出（激活含 5% 幅度 ×10 的 outlier 通道）。
- **ExpD**：得到三元权重的三条路径对比，手写 STE（无 torch 依赖）：
  A 训完再量化（PTQ）/ B 随机初始化 + 量化在环（原生低比特）/ C 全精度解 + 量化在环微调（QAT）。

## 关键数字（实测，见 results/stdout.txt）

- **位宽账**：binary g128 = 1.125 bpw（14.22×）、ternary g128 = 1.710 bpw（9.36×）、
  MLX 因 scale+bias 打包退化为 1.25 bpw。27.3B 参数对应 3.84 GB / 5.84 GB，与官方 3.9 / 5.9 GB 吻合。
- **粒度边际收益随字母表变粗而坍塌**（per-tensor → g32）：

  | 字母表 | 误差变化 | 误差降幅 | 位宽涨幅 |
  |---|---|---|---|
  | INT4 | 72.74% → 14.97% | **79.4%** | +12.5% |
  | INT2 | 93.75% → 52.87% | 43.6% | +25.0% |
  | Ternary | 66.21% → 63.85% | **3.6%** | +31.5% |

  三元字母表上细化粒度几乎不产生收益——这是 Bonsai 敢用 g128 的真正原因。
- **三元码本比同 bpw 的均匀 INT2 更高效**：ternary g128（1.710 bpw，65.69%）优于 INT2 g128（2.125 bpw，69.23%）。
- **三元码本中 0 的占比 32.7%**（per-tensor absmean），即"自然剪枝"。
- **三条路径**（test MSE，全精度上界 0.00253）：
  A 训完再量化 **0.26194** / B 随机起+环内 **0.34660** / C 全精度解+环内微调 **0.23432**。
  B 反而输给 A（训练预算不足时 STE 起点太脏）；C 最优但只比 A 好 10.5%，
  距全精度上界仍有 **~100 倍** —— **低比特的瓶颈是表示容量，不是优化器**。
- **STE 是存在前提而非技巧**：若 `round()` 导数真的为 0，1200 步后 test MSE = 1.00320，等于初始值，训练完全停滞。

## 预期输出

- `results/stdout.txt`（四个实验的完整控制台输出）
- `results/results.json`（结构化数字，供文章引用）
- `results/alphabet_granularity_frontier.png` —— 字母表 × 粒度的误差-位宽前沿
- `results/granularity_diminishing_return.png` —— 粒度边际收益随字母表坍塌
- `results/three_routes_to_ternary.png` —— 三条路径的训练曲线与容量鸿沟
