## 对应文章

- 《PTQ（02）：GPTQ 与 OBQC》  
  https://lrypcy.github.io/2026/08/24/ptq-02-gptq/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- 手写 Hessian H=2XX^T、逐列消元的误差反馈补偿（GPTQ 核心）
- 主对比：RTN vs GPTQ @ INT4/3/2（校准/留出双评估）
- group size 扫描（32/64/128/256）
- dampening 对 Cholesky 数值稳定性的影响（低样本时 H 秩亏）

## 关键数字（SEED=0 实测）

- bits=4：GPTQ eval MSE 5.70e-2 vs RTN 8.78e-2（**降 35.1%**，SNR 12.44 vs 10.56 dB）
- bits=3：降 19.9%；bits=2：降 7.4%（越低比特补偿越力不从心）
- n_cal=48 < d_col=256 时 damp=0 直接 Cholesky 失败；damp_frac=1e-2 起稳定且更准

## 预期输出

- `results/gptq_vs_rtn.png` —— 主对比 + group size 扫描
- `results/dampening_cholesky.png` —— cond(H) 与 MSE 随 damp 变化
- `results/stdout.txt` / `results/results.json`
