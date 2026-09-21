## 对应文章

- 《大模型量化算法（24）：KV Cache 量化》 https://lrypcy.github.io/2026/09/19/llm-quant-24-kv-cache/
- 相关：《PTQ13：QServe/QQQ》 https://lrypcy.github.io/2026/08/24/ptq-13-qserve-qqq/ （KV 是非对称 int4 最友好的战场）

## 如何运行

```bash
cd <本目录>
jupyter nbconvert --to notebook --execute --inplace kv_cache_quant.ipynb
```

纯 numpy + matplotlib（合成注意力），CPU 秒级；SEED=0 可复现。

## 实验内容

- **A. KV 量化敏感度**：K 和 V 谁更敏感？per-tensor vs per-channel 差多少？
- **B. 序列长度累积**：KV 会被反复读取，误差随 seq 怎么累积
- **C. 显存/带宽账本**：KV cache 字节 vs 权重字节，随 batch / seq 谁先成为瓶颈

## 关键数字（SEED=0，smoke 实测）

| 实验 | 关键数字 |
|---|---|
| A | 4-bit 下 per-tensor **5.817e-01** → per-channel **6.344e-02**（**+9.62 dB**）；V 的误差是 K 的 **0.36 倍**（K 更敏感） |
| B | seq 128 → 512，4-bit 输出误差 **4.987e-02 → 6.344e-02** |
| C | batch=1 / seq=2048 时 KV 是 W4 权重的 **0.50x**；batch=32 / seq=32768 时是 **256x** → **长上下文下 KV 才是瓶颈** |

## 预期输出

- `results/kv_cache_sensitivity.png`、`kv_cache_sequence_scaling.png`、`kv_cache_memory_ledger.png`
- `results/stdout.txt` / `results/results.json`
