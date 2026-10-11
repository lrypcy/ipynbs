# 流水线气泡率与 1F1B 时序验证 · 对应《深度剖析（06）》

## 对应文章

《Megatron-LM 深度剖析（06）：流水并行调度器——调度器的细节不在论文里》
https://lrypcy.github.io/2026/10/16/megatron-06-pipeline-parallel-schedules/

## 这是什么

本机**无 GPU**，所以这里只有两个级别：

| 脚本 | 级别 | 做什么 |
|---|---|---|
| `bubble.py` | **A 级**（解析脚本实跑） | 气泡率闭式解，与源码 warmup 公式逐组合对账 |
| `gantt.py` | **C 级**（CPU 微型复现） | 逐行复刻 `get_schedule_table`；离散事件模拟 1F1B 出甘特图区间，并**独立测出**气泡率 |

两个脚本**不含任何性能测量**，不给吞吐 / MFU 数字（那属 B 级真机实测范围）。
所有输出均为纯算术或离散调度，读者可自行复算。

源码基准：`megatron-core` 0.20.0 / commit `60e039626`，纯 Python，无第三方依赖。
用 `~/Software/miniconda3/bin/python3` 跑（managed python 3.13 无 numpy，本目录也不需要 numpy）。

## 如何运行

```bash
cd ~/Desktop/Projects/sandbox/ipynbs/experiments/megatron/pipeline_parallel

~/Software/miniconda3/bin/python3 bubble.py   # 写 results/bubble_stdout.txt
~/Software/miniconda3/bin/python3 gantt.py    # 写 results/gantt_stdout.txt
```

两个脚本退出码非 0 即为对账失败，可直接当 CI 用。
输出确定性——重跑与 `results/` 中固化文件**逐位一致**。

## 文件

| 文件 | 作用 |
| --- | --- |
| `bubble.py` | A 级：气泡率闭式解 `B1 = (p-1)/(m+p-1)`、`Bv = (p-1)/(m*v+p-1)`，扫 `p × m × v` 出表 |
| `gantt.py` | C 级：复刻 `get_schedule_table`（`schedules.py` L989）+ 1F1B 离散事件模拟 |
| `results/bubble_stdout.txt` | 固化的 A 级输出（47 行） |
| `results/gantt_stdout.txt` | 固化的 C 级输出（40 行，含两张甘特图） |

## 依据（均锁 commit `60e039626`）

| 事实 | 源码位置 |
| --- | --- |
| 非交错 warmup 步数 `= p - rank - 1` | `schedules.py` L2277 |
| warmup 再封顶 `min(·, num_microbatches)` | `schedules.py` L2278 |
| 交错 warmup `= (p - rank - 1) + (v - 1) * group_size` | `schedules.py` L964–L965 |
| 总 microbatch `= num_microbatches * num_model_chunks` | `schedules.py` L949 |
| `get_schedule_table` 的两分支构造 | `schedules.py` L989–L1016 |
| 官方 docstring 的 schedule 表（group=3, M=5） | `schedules.py` L1227–L1229 |
| 官方 docstring 的 schedule 表（group=2, M=4） | `model_parallel_config.py` L441–L444 |
| `microbatch_group_size_per_vp_stage` 默认取 PP 度 | `model_parallel_config.py` L567–L568 |

## 结论

### §3 气泡率闭式解

$$\text{非交错}\quad B_1 = \frac{p-1}{m+p-1}
\qquad\qquad
\text{交错}\quad B_v = \frac{p-1}{m\,v+p-1}$$

**两式的分子相同、分母只差一个 `v`。** 交错不改变气泡的「槽数」（恒为 `p-1`），
只把每个槽位拉长——所以 `v` 越大，气泡率趋于 `(p-1)/(m*v)`，即被 microbatch 数除。

固定 `p=4, m=8` 时（`results/bubble_stdout.txt` 表 3）：

| v | 总 microbatch | 气泡率 | 相对非交错 |
| --- | --- | --- | --- |
| 1 | 8 | 27.27% | 1.00× |
| 2 | 16 | 15.79% | 1.73× |
| 4 | 32 | 8.57% | 3.18× |
| 8 | 64 | 4.48% | 6.09× |
| 16 | 128 | 2.29% | 11.91× |

### 三组对账（A 级）

| 对账 | 判据 | 结果 |
| --- | --- | --- |
| A1 | 闭式解的分子/分母与源码 warmup 公式自洽（stage 0 warmup == `p-1`，末 stage == 0） | ok |
| A2 | `m` 增大时气泡率单调下降，且渐近 `bubble × m → p-1`（相对偏差 < 5%） | ok |
| A3 | 交错的改善倍数 == `(m*v+p-1)/(m+p-1)` | ok |

**A2 用渐近判据而非「趋近 0」**——有限 `m` 下气泡率不应为 0，
`p=16, m=1024` 时 1.44% 是正确行为，不是 bug。

### §2/§4 schedule table 逐位复刻（C1）

`get_schedule_table` 的两分支构造被逐行复刻，并与官方 docstring 的两张表**逐位对账**：

| 表 | 出处 | 实算 == 文档 |
| --- | --- | --- |
| group=2, M=4, VP=2 | `model_parallel_config.py` L441–L444 | ✓ |
| group=3, M=5, VP=2 | `schedules.py` L1227–L1229 | ✓ |

### §3/§4 时序模拟（C2/C3）

离散事件模拟遵守两条约束：每 stage 一次只做一个操作且按 1F1B 顺序出队；
`F(s,j)` 依赖 `F(s-1,j)`、`B(s,j)` 依赖 `B(s+1,j)`。

| 对账 | 判据 | 结果 |
| --- | --- | --- |
| C2 | 模拟 makespan == `2m + 2(p-1)` | ok（20 组配置全过） |
| C3 | 模拟测得的气泡率 == A 级闭式 | ok（20 组配置全过） |

**C3 是本目录最有价值的一组**：气泡率不是抄公式，而是从模拟的 makespan 里
独立测出 `(makespan - 2m)/makespan`，再与 A 级闭式相等。

### 甘特图（`results/gantt_stdout.txt`）

`p=4, m=4` 的输出（`F` 前向 / `B` 反向 / `i` 空闲，makespan=14）：

```
  stage 0: |F|F|F|F|i|i|i|B|i|B|i|B|i|B|
  stage 1: |i|F|F|F|i|i|B|F|B|i|B|i|B|i|
  stage 2: |i|i|F|F|i|B|F|B|F|B|i|i|i|i|
  stage 3: |i|i|i|F|B|F|B|F|B|F|B|i|i|i|
```

填充阶段的空闲三角（左下）与排空阶段的空闲三角（右上）**肉眼可见**，
气泡总量 `2(p-1) = 6` 格正好对应 makespan `14 - 2×4 = 6`。

## 两个实现上的观察

1. **`m < p` 合法靠的是 `min` 封顶。** 若照搬 `num_warmup = p - rank - 1`
   而不封顶，`p=6, m=4` 时 stage 0 的 warmup（5）会超过 microbatch 总数（4），
   调度直接死锁。源码 L2278 的 `min(num_warmup_microbatches, num_microbatches)`
   正是为此。
2. **模拟的 `done` 字典键必须带 stage。** 同一个 microbatch `j` 在每个 stage 上
   都有自己的 `F` 与 `B`；若只用 `(op, j)` 做键，各 stage 会互相覆盖完成时刻，
   makespan 会随 `p×m` 线性膨胀（实测 `p=4, m=8` 会得到 64 而非 22）。

## 它验证了本文哪些数字

- §3 的气泡率闭式解两张表（`bubble_stdout.txt` 表 1、表 2）
- §3 表 3 的 `p=4, m=8` 五个 `v` 值与改善倍数
- §2/§4 `get_schedule_table` 复刻出的 schedule 表
- §4 甘特图的填充/排空形状与气泡格数