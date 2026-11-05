# 线性注意力的 KV cache 去哪了：状态账本 / 容量 / 分块 / 前缀复用

对应文章：《线性注意力的 KV Cache 去哪了——从递推状态、容量下界到混合架构的缓存账》，
<https://lrypcy.github.io/2026/11/05/linear-attn-01-kv-cache-state/>

纯 CPU numpy 合成探针，固定 seed `20261105`。**七个分组各自带断言，最后一个 cell 打印断言总数
（58/58）**。`linear_attn_kv_cache_lab.ipynb` 已跑出全部输出，点开 GitHub 就能看到结果；
`.py` 是同一份源码（按 `# %%` 切 cell 生成 notebook），`results/stdout.txt` 是它的完整输出，
`linear_attn_figures.py` 从 stdout 解析数字出三张 SVG。

## 怎么跑

```bash
~/Software/miniconda3/bin/python linear_attn_kv_cache_lab.py   # 约 3 秒，58 条断言
~/Software/miniconda3/bin/python make_notebook.py              # 重新生成并执行 .ipynb
~/Software/miniconda3/bin/python linear_attn_figures.py        # 出 3 张 SVG 到 figures/
```

## 七组账

| 组 | 问题 | 结论 |
| --- | --- | --- |
| 1 | parallel / recurrent / chunkwise 是不是同一个东西 | 与显式双重循环真值逐位一致（最大误差 2.220e-16），但与 softmax 输出平均差 0.2947 —— 换核不是等价变换 |
| 2 | 固定状态是不是免费 | 设定 n_h=32 / n_kv=8 / d=128 / bf16：softmax 每 token 4096 bytes，状态恒为 1048576 bytes（1 MiB），交叉点 **T\* = 256 token**；T<T\* 时混合反而更占显存 |
| 3 | 有限状态能装多少关联 | 正交 key 可逐位读回；随机 key 读回误差 0.9014 → 3.1916（N=8→128），rank(S) ≤ min(N,d_k,d_v)=6，N>d 时最小二乘残差 0.7203 ~ 0.9104；delta rule 能精确覆盖同一个 key（误差 0），加性外积不能（1.8337） |
| 4 | chunkwise 为什么必须分块 | 带衰减 α=0.97 的五种块大小与递推全部一致（2.220e-16）；C 翻倍，块内 FLOPs 翻倍、块间状态 IO 减半，乘积恒定 134.2177（M×MiB）；C=16 时传状态 8 MiB > 写一遍 softmax KV 的 2 MiB |
| 5 | 状态精度的坑在哪 | T=20000、α=0.999：fp32 递推相对误差 9.483e-06，fp16 是 5.649e-02（放大 5956.7 倍）；C=128 的衰减连乘在 fp16 下相对误差 325.7 倍，log-space 只有 8.729e-07 |
| 6 | 前缀复用为什么不能直接搬 PagedAttention | 无快照从 state(PA) 接后缀 B 的相对偏差 0.4240；快照 + copy-on-write 后逐位一致（0）；softmax 侧前缀 K/V 只读共享即可 |
| 7 | 官方「减少 75%」是哪个口径 | 3:1 的 KV 口径恒为 75.00%，但把线性层状态也算进显存，减少 = 0.75(1-T\*/T)：4K 时 70.3125%、1M 时 74.9817%；7:1 对应 87.50% / 87.4786% |

## 断言对照表

| 组 | 断言数 | 关键不变量 |
| --- | ---: | --- |
| 1 | 6 | 三种形式 = 朴素真值；与 softmax 明显不同；状态元素数 d_k d_v |
| 2 | 8 | T\*=256、T=T\* 三口径相等、减少随 T 单调增且恒 <75%、闭式 0.75(1-T\*/T) |
| 3 | 14 | rank 上界、N≤d 有精确解 / N>d 无精确解、串扰单调、delta 覆盖 |
| 4 | 10 | 五种 C 与递推一致、FLOPs×2 / IO÷2 / 乘积恒定、C=16 的 IO > KV 写入 |
| 5 | 4 | fp32 与闭式真值一致、fp16 显著更差、fp16 连乘失稳、log-space 稳定 |
| 6 | 5 | 无快照输出不同、COW 逐位一致、状态形状固定、只读前缀可共享 |
| 7 | 11 | 显存口径 ≤ 层型占比、4K 的 70.3125%、两种混合比例 |
| **合计** | **58** | `总计 58/58 条断言通过` |

## 目录

- `linear_attn_kv_cache_lab.ipynb` —— 代码与输出并排（10 个 cell）
- `linear_attn_kv_cache_lab.py` / `make_notebook.py` —— 可复现源码与 notebook 生成器
- `results/stdout.txt` —— 完整固化输出，正文与图上的数字都从这儿回填
- `linear_attn_figures.py` / `figures/*.svg` —— 三张自画 SVG（数值直接解析 stdout）
  1. `fig1_cache_ledger.svg` 缓存账本曲线与 T\*
  2. `fig2_state_vs_kv.svg` softmax KV 列表 vs 固定矩阵状态
  3. `fig3_capacity_chunk.svg` 容量碰撞与 chunk 两笔账

## 边界

这组实验是**合成探针**：验证的是代数恒等式、字节账本与数值量级，不是真实模型的精度或吞吐。
论文数字（Kimi Linear / Qwen3-Next / MiniMax-01 / GLA / Mamba-2 等）在正文里单独标出处，
不与本目录的实测数字混在同一张表。
