## 对应文章

- 《PTQ（08）：GGUF、FP8 与 MXFP4》  
  https://lrypcy.github.io/2026/08/24/ptq-08-gguf-fp8-mxfp4/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- 通用浮点格式仿真器：frexp + 半偶舍入实现任意 (e,m,bias) 格式 round-trip，
  正确处理正规数/子正规数/溢出饱和
- 七种格式横评：INT8 sym / FP16 / BF16 / E5M2 / E4M3 / MXFP4(grp32) / NF4(blk64)
- E2M1 码本 + 组共享指数的 MXFP4 与 QLoRA 式 NF4 的 4-bit 对决
- 表示层级密度图：E4M3 vs E5M2 在 (0,2] 的台阶分布

## 关键数字（SEED=0 实测，SQNR dB）

| 格式 | gauss | laplace |
|---|---|---|
| INT8 sym | 39.3 | 33.3 |
| BF16 | 55.6 | 55.6 |
| E5M2 | 25.6 | 25.6 |
| E4M3 | **31.5** | **31.5** |
| MXFP4 grp32 | 18.8 | 18.0 |
| NF4 blk64 | 20.7 | 19.6 |

E4M3 比 E5M2 恰好高一个尾数位的 ~6 dB；i.i.d. 场景 NF4 略胜 MXFP4，
但后者硬件友好且对结构化数据更稳。

## 预期输出

- `results/format_sqnr_comparison.png`
- `results/e4m3_vs_e5m2_levels.png`
- `results/stdout.txt` / `results/results.json`
