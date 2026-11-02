# ring-attention-lab：序列并行的四组可复现实验

对应文章：[分布式训练（05）序列并行——把注意力沿序列切开，Ring Attention 与长上下文](https://lrypcy.github.io/2026/08/31/dist-train-05-sequence-parallel/)

纯 numpy、纯 CPU、固定 seed（`SEED = 0`）。四组实验各自独立可复现，每组末尾带断言，
最后一个 cell 打出断言总数，**共 20 条**，与 `results/stdout.txt` 末尾一致。

## 文件

| 文件 | 内容 |
| --- | --- |
| `ring_attention_lab.ipynb` | 四组实验主体，**已带输出**，点开 GitHub 就能看到结果 |
| `ring_attention_lib.py` | 公共函数（注意力参考实现、环模拟、负载模型、通信量、算术强度） |
| `ring_attention_figures.py` | 三张配图的生成脚本（输出 SVG + PNG 到 `figures/`） |
| `results/stdout.txt` | notebook 的纯文本输出，便于 diff 与对账 |
| `figures/` | 图；正文用的是博客仓库的 `assets/img/dist-train/*.svg`，`figures/*.png` 只作校对 |

## 运行

```bash
# 生成并执行 notebook（约 2 秒，CPU 即可）
jupyter nbconvert --to notebook --execute --inplace ring_attention_lab.ipynb

# 出图
python ring_attention_figures.py
```

## 四组实验各回答什么

### A. 分块 online softmax 与全局 softmax 是否逐元素等价

模拟 ring 的 KV 轮转，每轮把新块的部分状态 $(m,\ell,O)$ 与本地累加器合并。结论：

* 任意分块、任意到达次序，与全局 softmax 的最大绝对误差 **$5\times10^{-16}$**（float64），是等价而不是近似；
* 反例「每块各自 softmax 再平均」在同一批数据上误差 **0.186**——差 14 个数量级，且不会 NaN；
* logits 放大 20 倍（全局 max 98.8）后，每步减 running max 的实现误差仍有 $3\times10^{-15}$。

### B. 因果掩码下的负载分布

按 token 对精确计数（query $q$ 能看 key $k$ 当且仅当 $q\ge k$），模拟三种切法的墙钟
（每轮取最慢的那台，$P$ 轮相加）：

| 序列 | 卡数 | 连续切分 | zigzag | striped |
| --- | --- | --- | --- | --- |
| 256 | 4 | 57.2% | 100.0% | 98.8% |
| 1024 | 8 | 53.4% | 100.0% | 99.3% |
| 4096 | 16 | 51.6% | 100.0% | 99.6% |

（数字是利用率 = 理想 / 墙钟。）连续切法随 $P$ 增大逼近 50%，这正是「理论上限 2×」的另一种写法；
zigzag 在块级模型下配平到严格相等；striped 残余不到 1%。

### C. 三种方案的通信量

| | 每卡字节数 | 随 $P$ 的趋势 |
| --- | --- | --- |
| ring | $2Sh\cdot\frac{P-1}{P}$ | 收敛到 $2Sh$，与 $P$ 无关 |
| all-gather | $2Sh\cdot\frac{P-1}{P}$ | 与 ring **逐字节相同** |
| Ulysses | $4Sh\cdot\frac{1}{P}$ | 随 $P$ 线性下降 |

比值 ring/Ulysses $=(P-1)/2$：$P=2$ 时 Ulysses 更贵，$P\ge4$ 之后 Ulysses 更便宜，交叉点在 $P=3$。
**ring 省的不是带宽，是可重叠性。**

### D. 通信藏进计算的算术强度门槛

由 $4dc^2/F\ge 4cd/B$ 得 $c\ge F/B$，再叠加 6 块驻留得 $s\ge 6c$：

| 硬件 | $c_{\min}$ | $s_{\min}=6c$ |
| --- | --- | --- |
| A100 + NVLink | 1 040 | 6.2K |
| A100 + InfiniBand | 24 960 | 149.8K |
| H100 + NVLink | 2 198 | 13.2K |
| H100 + IB 400G | 39 560 | 237.4K |

前两行与 Ring Attention 论文 Table 2 的 1.0K / 6.2K、24.5K / 149.5K 对得上。
同一代 GPU 换到 IB，门槛抬高一个数量级以上——这是「ring 不该跨机」的定量说法。

## 口径说明

* 负载模型按 **token 对**计数，隐含假设 tile 粒度足够细（块内被掩码的部分不计时间）。
  这是「理论上限」，真实 kernel 达不到——Striped Attention 实测端到端 1.45×/1.65×，
  低于 2× 的主因是注意力只占端到端时间的一部分，不是模型错了。
* 硬件表的算力用 BF16 dense 峰值、带宽用**单向**值，与论文 Table 2 的口径一致。
* 这些数字全部可以由 `ring_attention_lib.py` 单独算出，notebook 只是把它们组织成可断言的序列。
