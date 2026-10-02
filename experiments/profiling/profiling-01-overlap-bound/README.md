## 对应文章

- 《性能剖析（01）：训练栈 profiling》
  https://lrypcy.github.io/2026/09/29/profiling-01-training/

## 如何运行

```bash
# 纯标准库，无第三方依赖，秒级完成
jupyter nbconvert --to notebook --execute profiling-01-overlap-bound.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io`：

- `tools/profiling_bench/overlap_bound.py`
- `tools/megatron_bench/comm.py`（α-β 集合通信代价模型）
- `tools/megatron_bench/configs.py`（模型与硬件标称规格）

由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **第 1 节 · 交叉点 $D^*$**：$D^* = n \alpha BW$ 的全硬件 × 全链路表，
  区分「延迟主导」与「带宽主导」两个区间
- **第 2 节 · 重叠收益上界**：$C$ / $M$ 的解析计算，四种口径（不重叠 / 各级 $\eta$ / 完美重叠）
- **第 3 节 · algbw vs busbw**：换算因子表 + 排名反转的演示

## 关键结论

- **序列并行不改变 TP 通信总量**（比值恒 1.00×）——恒等式 $\text{AR} = \text{RS} \circ \text{AG}$ 决定总量守恒。
  SP 的收益在**激活显存**，提速来自可以关掉 activation recompute。
- **跨机通信藏不住**，第一个原因是它落在延迟主导区（IB 的 $D^*$ 只有几百 KB），第二个才是带宽窄。
- **DP 梯度 reduce 在带宽主导区**（可重叠），**TP 每层 AR 恰好落在 $D^*$ 附近**（这就是 `tp_comm_overlap` 收益不稳定的原因）。
- **N=2 的 AllReduce 没有 busbw 口径优势**；**busbw / 单向链路 > 2 时不能反推线速**（NVLS / CollNet / 分层算法）。
