# CUDA Graph launch 计数与边界模型 · 对应《深度剖析（10）》

## 对应文章

- 《Megatron-LM 深度剖析（10）：CUDA Graph——把 step 图化到什么程度》
  https://lrypcy.github.io/2026/10/18/megatron-10-cuda-graph/

## 这是什么

本机**无 GPU**，CUDA Graph 必须在真卡上才能 capture/replay，因此本篇没有真图化实验。
这里只放**解析类实验**：不测时间、不测吞吐，只把能被算术数出来的量数清楚。

| 部分 | 等级 | 验证对象 |
|---|---|---|
| launch 计数模型 | A（解析脚本实跑） | 单 layer / 整 step 的 CPU 侧 launch 数，图化前后对比 |
| 融合粒度敏感性 | A（解析脚本实跑） | pointwise 融合粒度假设对绝对值的影响 |
| optimizer step 计数 | A（解析脚本实跑） | Adam step 的 launch 数，图化前后对比 |

**不覆盖**：任何墙钟时间、吞吐、MFU、显存占用数字。那些需要真卡，本机没有。

## 核心诚实性约束：CPU 侧与 device 侧必须分开记

这是本实验最要紧的一条。CUDA Graph 常被描述成「减少 kernel 数量」，这是错的：

| 量 | 图化前 | 图化后 |
|---|---|---|
| CPU 侧 launch 次数 | `N` | `1` |
| device 侧 kernel 执行次数 | `N` | `N` |

图化**不会**让 GPU 少算一遍乘法。图里仍然有 N 个 kernel node，只是由 driver
一次性提交，省掉的是 CPU 反复调用 launch API 的开销。

`launch_count.py` 的输出把这两列并排放，就是为了让这个区分无法被误读。

## 算术的两条来源

1. **GEMM 个数** —— 从 Megatron 的 dense 层模块结构数出来：
   `linear_qkv` / `linear_proj` / `linear_fc1` / `linear_fc2`，共 4 个 / layer。
   这由源码结构决定，与 TE 版本无关，**可靠**。
2. **每类算子的 launch 数** —— TE 的融合粒度随版本变化
   （`megatron/core/fusions/` 里那批 `fused_*` 就是这件事，属另一编号）。
   所以 pointwise 算子拆成几个 kernel 是**显式假设**，脚本对它做敏感性扫描，
   **不把假设当结论**。

反向 launch 数 = 每个 GEMM 2 次（dgrad + wgrad）+ 每个 pointwise 1 次。

## 结论（对应正文 §1 与 §6）

配置 `num_layers=32`、`microbatches=8`、单 dense layer 前向 11 次 / 反向 15 次 launch：

| 量 | 图化前 | 图化后 |
|---|---|---|
| CPU 侧 launch 次数 / step | 6656 | 512 |
| device 侧 kernel 执行次数 / step | 6656 | 6656 |

即 CPU 侧省下 6144 次 / step，而 device 侧执行次数**一个不少**。

融合粒度敏感性（pointwise 拆成 k 个 kernel）：

| k | eager launch / step | 相对 k=1 |
|---|---|---|
| 1 | 6656 | 1.00x |
| 2 | 9728 | 1.46x |
| 3 | 12800 | 1.92x |

k 只改变绝对值，不改变「CPU 侧 N → 1」这个结构性事实。

optimizer step（对应正文 §6 的孤儿节）每参数组 launch 数：

| 实现 | launch 数 |
|---|---|
| Adam 逐元素（未融合上界） | 11 |
| Adam foreach（批成 5 类） | 5 |
| `_fused_adam`（单 kernel） | 1 |
| 图化后（`capturable`） | 1 |

`capturable` 切的是「优化器能否被 capture」这个**可用性边界**，
不是「给 step 加个图」的开关。

## 怎么跑

```bash
~/Software/miniconda3/bin/python3 launch_count.py             # 写 results/launch_count.txt
~/Software/miniconda3/bin/python3 launch_count.py --stdout    # 只打屏
~/Software/miniconda3/bin/python3 launch_count.py --check     # 与冻结结果比对，有漂移则非 0 退出
```

`--check` 是回归闸：正文里引用的每个数字都必须与本脚本输出按**数字字面量**一致，
改动 `FORWARD_OPS` 后忘记同步正文，`--check` 与 `check_post.py` 会一起报警。

## 文件清单

| 文件 | 等级 | 说明 |
|---|---|---|
| `launch_count.py` | A | launch 计数模型 + 敏感性扫描 + optimizer 计数，产出 `results/launch_count.txt` |
| `results/launch_count.txt` | A | 固化的输出（71 行），读者不必自己跑 |
| `README.md` | — | 本文件 |

## 断言结果

`results/launch_count.txt` 首行 `OK  10/10 断言通过`，覆盖：

1. GEMM 个数 / layer = 4
2. 前向 launch / layer = 11
3. 反向 launch / layer = 15
4. 每层合计 = 26
5. `eager = per_layer × L × M` = 6656
6. `graph = 2 × L × M` = 512
7. device 侧执行数不变 = 6656
8. 省下 = eager − graph = 6144
9. Adam 未融合上界 = 11
10. 敏感性 k=1 基线 = 6656

## 引用约定

正文引用本实验时链接到本目录的 `launch_count.py` 与 `results/launch_count.txt`。
本仓库推上 GitHub 后，按系列规范补 commit 锁定的 permalink。
