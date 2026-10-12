# Refit / Resharding 的确定性与代价解析 · 对应《深度剖析（11）》

## 对应文章

《Megatron-LM 深度剖析（11）：Refit / Resharding——训练与推理之间的权重搬运》
https://lrypcy.github.io/2026/10/19/megatron-11-refit-resharding/

## 这是什么

本机**无 GPU**，两个脚本：

| 脚本 | 级别 | 做什么 |
|---|---|---|
| `planner_determinism.py` | **C 级** | **直接跑真实 `build_plan_from_rosters`**，验证五组确定性断言 |
| `reshard_cost.py` | **A 级** | 用真实 planner 产出的 op 列表算字节数与 op 数，给出同构/异构解析模型 |

**关键点：这两个脚本不重实现 planner。** 它们在运行时用
`git show 60e039626:megatron/core/resharding/{planner,utils}.py`
把源码抽到一个临时包再 import，所以测的就是仓库里那份代码。
`Megatron-LM` **全程只读**，脚本不会写入仓库任何文件。

之所以能这样做，是因为 `planner.py` 的依赖极浅——它只 import
`torch` / `torch.distributed` 与同目录的 `utils.py`，**不碰任何
`megatron.core.*` 模块**，因此可以整段抽出独立加载。
`build_plan_from_rosters` 的 docstring 也明说它是
"Pure and collective-free, so it can be tested or reused with preassembled
rosters without touching the process group"。

## 如何运行

```bash
cd ~/Desktop/Projects/sandbox/ipynbs/experiments/megatron/refit
~/Software/miniconda3/envs/torch/bin/python planner_determinism.py
~/Software/miniconda3/envs/torch/bin/python reshard_cost.py
```

**必须用 `envs/torch` 的解释器**（有 torch 2.10.0）；
`miniconda3/bin/python3` 无 torch，抽出来的 planner 导不进来。
退出码非 0 即为对账失败。输出确定性——重跑逐位一致。

## 文件

| 文件 | 作用 |
| --- | --- |
| `planner_determinism.py` | C 级：抽真实 planner，跑 C1–C5 五组断言 |
| `reshard_cost.py` | A 级：字节/op 数解析模型，跑 M1–M2 不变量 |
| `results/planner_determinism.stdout.txt` | 固化输出（23 行） |
| `results/reshard_cost.stdout.txt` | 固化输出（47 行） |

## 依据（均锁 commit `60e039626`）

| 事实 | 源码位置 |
| --- | --- |
| 排期迭代顺序只依赖 roster，故各 rank 重放得到同一 task_id | `planner.py` `L305–L356` |
| 只保留 `dst_rank == my_global_rank` / `src_rank == my_global_rank` 的 op | `planner.py` `L424–L445` |
| `build_plan_from_rosters` 是纯函数、无集合通信 | `planner.py` `L403–L413` |
| TP 维度的**精确**尺寸守恒校验 | `planner.py` `L56–L62`（`_build_descriptors_for_param`） |
| 一次 all-gather 取代 gather-to-0 + scatter | `planner.py` `L492–L495` |
| rank list 去重使 pickle 体积降约 75% | `planner.py` `L482–L484` |
| `build_centralized_reshard_plan` 已废弃 | `planner.py` `L503–L524` |
| NVSHMEM 要求每次传输有全局唯一 `task_id` | `copy_services/nvshmem_copy_service.py` `L16–L21`, `L54–L58` |
| `run()` 含集合操作，**所有 rank 都必须调用** | `copy_services/nvshmem_copy_service.py` `L151–L156` |
| Gloo / NVSHMEM 按 `task_id` 配对本rank拷贝 | `nccl_copy_service.py` `L48–L56`、`nvshmem_copy_service.py` `L124–L133` |
| MXFP8 sender 侧转换会破坏同 rank 配对 | `transforms.py` `L142–L146` |

## 结论

### C 级：五组确定性断言全过

| 对账 | 判据 | 结果 |
| --- | --- | --- |
| C1 | 同输入跑两次，`task_id -> op` 映射逐位相同 | ok |
| C2 | 所有 rank 的 `send_ops` 合起来，每个 `task_id` 恰好出现一次 | ok |
| C3 | 同一 `task_id` 上 sender 的 `my_slice` == receiver 的 `peer_slice` | ok |
| C4 | 打乱 roster 的 dict 插入顺序，`task_id` 分配不变 | ok |
| C5 | roster 的 key 换成非连续值，结果不变且为基准子集 | ok |

测试拓扑：训练侧 TP=8（源 rank 0–7）、推理侧 TP=4（目的 rank 100–103），
6 层 × 4 类权重，`OUT=1152 INP=1024` bf16。产出 **192 个 `task_id`**、
192 send + 192 recv = 384 个 op，**12 个 rank 全部有 op**。

**C3 是「每 rank 重放同一份排期」这一设计的直接证据**：
sender 与 receiver 从未通信，却能对同一个 `task_id` 得出互为镜像的切片，
唯一可能就是两边独立重放了同一个确定性迭代。

**C4 是最容易被实现错的一环**：源码用 `sorted(dst_param_metadata_by_rank)`
遍历，所以 dict 插入序不影响结果——实测确认。

### A 级：通信量与 op 数的解析模型

模型：24 个参数，`OUT=1152 INP=1024` bf16，**总量恒为 `56,623,104` 字节**
（`24 × 1152 × 1024 × 2`）。

异构（训练 TP=8 固定，推理侧 TP 变）：

| dst_tp | task_id | send | 远端字节 | 本地字节 | 远端占比 |
| --- | --- | --- | --- | --- | --- |
| 8 | 192 | 192 | 56,623,104 | 0 | 100.0% |
| 6 | 576 | 576 | 56,623,104 | 0 | 100.0% |
| 4 | 192 | 192 | 56,623,104 | 0 | 100.0% |
| 2 | 192 | 192 | 56,623,104 | 0 | 100.0% |
| 1 | 192 | 192 | 56,623,104 | 0 | 100.0% |

**字节总量与 TP 度无关**——异构并不搬运「更多」的数据，
而是**把同样的数据切成不同形状**。

真正的分野在「同构」二字：

| 情形 | 远端字节 | 本地字节 | 说明 |
| --- | --- | --- | --- |
| 同构 **且共址**（TP=8→8，rank 重合） | **0** | 56,623,104 | 全部走 `copy_()`，不过网络 |
| 同构 **不共址**（TP=8→8，rank 错开） | 56,623,104 | 0 | 布局相同，字节照样整份搬 |

**这就是 §5 的核心结论：决定代价的不是「同构与否」，而是「rank 是否重合」。**
共址时所有拷贝退化成进程内 `tensor.copy_()`，四种 copy service 里的
远端路径一条都用不上。

### op 数律

固定 `dst_tp=4`，改变 `src_tp`：

| src_tp | task_id | send | 每参数 op 数 |
| --- | --- | --- | --- |
| 1 | 96 | 96 | 4.00 |
| 2 | 96 | 96 | 4.00 |
| 4 | 96 | 96 | 4.00 |
| 8 | 192 | 192 | 8.00 |
| 16 | 384 | 384 | 16.00 |

**每参数 op 数 == `$$\max(\text{src\_tp}, \text{dst\_tp})$$`**（在两者互整除时）。
字节恒定而 op 数随 TP 度线性增长，意味着**提高 TP 度不增加通信量，
只把同样的字节切成更多次、更小的传输**。`dst_tp=6` 那行 task_id
跳到 576 是因为 4 不整除 6，分片边界不再对齐。

### 两个实测到的实现细节

1. **TP 维度必须精确整除。** 最初把 `OUT` 取 2048、`dst_tp` 取 6，
   真实 planner 直接抛错：`src_world*src_local=8*256=2048 != 6*341`。
   这正是 `_build_descriptors_for_param` 的尺寸守恒校验在起作用
   （`planner.py` `L56–L62`）。改用 `OUT=1152` 后 6 正常。
2. **共址必须在建 roster 时就编对 rank。** 事后只把 roster 的 dict key
   从 100 改成 0、而不同步改 `ParameterMetadata.tensor_parallel_group_ranks`，
   `_plan_tp` 会因为 `my_global_rank` 不在该 TP 组里而**静默返回空排期**——
   不报错、不抛异常，只是 `task_id` 变成 0。定位这个坑花了一轮。

## 它验证了本文哪些数字

- §2 的五组确定性断言（C1–C5）
- §2「12 个 rank 全部有 op」与 192 `task_id` / 384 op 的规模
- §3 本地 vs 远端字节的分野
- §5 表 1–表 5 的全部数字，以及「op 数 = `$$\max$$`」「字节恒定」两条律