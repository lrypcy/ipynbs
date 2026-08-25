## 对应文章

- 《PTQ（10）：Outlier Suppression+》  
  https://lrypcy.github.io/2026/08/24/ptq-10-outlier-suppression/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **A. gamma 迁移**：LayerNorm 后接放大 gamma 的通道把异常搬进权重侧，
  量化前做通道缩放迁移；对比直接量化的误差
- **B. shift+scale**：对均值漂移型离群通道先平移再缩放，
  展示"释放有效电平数"机制（OSC++ 核心）

## 关键数字（SEED=0 实测）

- gamma 放大 5x 场景：直接量化误差 0.1363 -> 迁移后 0.0291（**-78.6%**）
- shift 后 absmax 52.77 -> 4.86（范围收缩 10.9x），有效电平波动 1.9 -> 36，
  W8A8 误差 0.0179 -> 0.0026（scale only）/ 0.0013（shift+scale）

## 预期输出

- `results/gamma_migration.png`
- `results/shift_vs_scale_w8a8.png`
- `results/stdout.txt` / `results/results.json`
