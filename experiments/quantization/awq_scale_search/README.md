## 对应文章

- 《大模型量化算法（03）：AWQ 激活感知权重量化》  
  https://lrypcy.github.io/2026/08/24/llm-quant-03-awq-scale-search/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
亦覆盖 PTQ（03）中 AWQ 部分：<https://lrypcy.github.io/2026/08/24/ptq-03-awq-omniq/>

## 实验内容

- **A. 显著度证据**：top-1% 权重承担 57.2% 的量化误差；按激活幅值选保护列
  命中率远高于按权重幅值（remaining_ratio 0.44 vs 1.00）
- **B. gamma-scale 搜索**：s* = s·(max|x|/max|w|)^gamma 逐组搜索最优 gamma
- **C. 端到端流水线对比**：AWQ scale + 自动裁剪 vs 无保护基线

## 关键数字（SEED=0 实测）

- group=128 时 gamma=0.25 最优，输出误差相对无 scale 降低约 40%
- 激活幅值排序与真实显著列的重合度高（error share 57.2% vs 随机 57.2% 中的命中差）

## 预期输出

- `results/awq_saliency_selection_comparison.png`
- `results/awq_gamma_scale_search.png`
- `results/awq_clip_and_pipeline_comparison.png`
- `results/stdout.txt` / `results/results.json`
