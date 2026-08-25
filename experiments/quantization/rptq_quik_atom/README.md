## 对应文章

- 《PTQ（11）：RPTQ、QUIK 与 Atom》  
  https://lrypcy.github.io/2026/08/24/ptq-11-rptq-quik-atom/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **RPTQ 通道重排**：把离群通道集中到专属组，其余组 absmax 骤降；
  数值上验证置换等价性（偏差 ~1e-14）
- **变体阶梯**：naive per-group A4 -> RPTQ 重排 -> Atom 双路径分解，
  对照 W4A8 理论下限（weight-only floor）

## 关键数字（SEED=0 实测）

- 重排后各组 absmax：[35.8, 37.4, 36.8, 40.6] -> [3.35, 3.59, 3.90, 40.56]
- 相对误差（floor=0.1153）：naive 0.1741(1.51x) -> RPTQ 0.1385(1.20x)
  -> Atom 双路径 0.1174(**1.02x**)

## 预期输出

- `results/rptq_group_absmax_reorder.png`
- `results/variant_errors_vs_floor.png`
- `results/stdout.txt` / `results/results.json`
