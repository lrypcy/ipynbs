# blockmask_lab：BlockMask 工程那一半的数值验证

对应文章：[《FlexAttention（02）：BlockMask 的工程与长上下文》]
（`lrypcy.github.io`，编译器技术分类，FlexAttention 系列第 2 篇）。
第 1 篇配套实验在 [`../flex_core/`](../flex_core/)。

## 这份脚本回答什么

01 那篇（与 `flex_core`）回答的是「为什么这个编程模型成立」——任意逐点 `score_mod`
为什么能和 online softmax 共存。但真正决定要不要上 FlexAttention 的是另一批问题，
而且官方材料对它们只给了定性建议：

1. `create_block_mask` 到底做了什么，它是不是真的会物化一份 $$Q\times KV$$ 的布尔张量？多大？
2. 「每步重建 / 每层重建」到底差多少？划不划算的阈值在哪？官方 FAQ 说「每层重建是最难的场景」——难在哪？
3. 确定性 dQ 到底是修什么？「预先算好的写入顺序」长什么样？
4. 上下文并行里，BlockMask 除了当掩码还能当什么？
5. 块大小 $$BS$$ 与 tile 大小 $$BLOCK\_M/N$$、GQA 比值怎么选，选错了浪费多少？

这份脚本用**纯 numpy / CPU** 把这五条摆到数字上。

## 如何运行

```bash
python3 run.py              # 跑全部五组，打印表格
python3 run.py 2 4          # 只跑第 2、4 组
python3 run.py --selftest   # 只跑断言（静默），适合挂 CI
```

依赖只有 numpy，固定 `seed = 20261018`，逐位可复现。共 19 条断言，全过。

## 五组检查与结论

| # | 检查内容 | 结论 |
| ---:|:---|:---|
| 1 | `create_block_mask` 的成本结构 | 构造过程确实会物化一份 $$B\cdot H\cdot Q\cdot KV$$ 的 bool 张量：8×16×8192² = **8192 MiB**；`B=None, H=None` 广播后降到 **64 MiB**（128×）；压缩后的 BlockMask 元数据只有 0.0159 MiB（516222×）。 $$BS$$ 从 64 提到 1024，元数据降 **249×** |
| 2 | 每步 / 每层重建划不划算 | 单次重建相对一次完整注意力的成本是 $$\kappa/(4D)$$：$$\kappa=1$$ 时 0.391%、$$\kappa=4$$ 时 1.562%。 **纯 FLOPs 口径下每层重建的 break-even 阈值不到 2%**，三种掩码的净收益都是 +45%~+73%。 真实成本不在 FLOPs，在(bool 峰值显存 / 独立 kernel 序列 / 动态 shape 重编译)这三处 |
| 3 | 确定性 dQ | float32 下同一组部分和换顺序就有差别：升/降序差 7.153e-07，64 个随机顺序的最大偏差 1.192e-06。 复现 PyTorch 的 rank 算法（$$\mathrm{rank}=\mathrm{cumsum}(\mathrm{dense})-\mathrm{dense}$$）后，按 rank 累加与串行累加**逐位相等**（0.000e+00），SPT 版本同样逐位等于降序串行 |
| 4 | PTRR 负载均衡 | 代价模型就是 BlockMask 的 `kv_num_blocks + full_kv_num_blocks`。 causal + CP=8：朴素连续切分 max/理想 = **1.875**（峰谷差 224），head-tail 与 PTRR 都是 **1.000**。 但 sliding / document 下 head-tail 退化到 1.143 / 1.333，PTRR 仍是 1.125 / 1.111 —— PTRR 是通用的，head-tail 只对单调代价曲线成立 |
| 5 | $$BS \times BLOCK \times $$ GQA 选型 | 浪费率随 $$BS$$ 涨：causal 0.77%→1.55%→3.11%（$$BS$$=64/128/256），sliding w=1024 6.24%→12.49%→**24.99%**。 **document 掩码在文档长度对齐块边界时浪费率恒为 0.00%**（元素密度 15.62% = 块级密度 15.62%）。 约束：$$(BS, BLOCK\_N)$$ 必须整除，否则 `raise_flex_kernel_options_error`；FlexDecoding 要求 GQA 比值是 2 的幂（3 和 6 直接出局） |

## 哪些结论是这份脚本**没有**主张的

- **不含任何计时**。第 2 组的 $$\kappa$$ 是「每次 mask_mod 求值的 op 数」的模型参数，
  不是实测；结论只主张**量级与阈值结构**（重建成本是注意力的千分之几），
  不主张某个具体的墙钟毫秒数。
- **第 1 组算的是 bool 张量的理论字节数**。实际峰值还取决于 allocator 的行为与
  `_convert_mask_to_block_mask` 的实现（`_compile=True` / `torch.compile` 之后可能不同）；
  这里给的是「不编译、按源码路径」的上界。
- **第 3 组是浮点非结合性的演示，不是 PyTorch 的实现**。PyTorch 用的是 semaphore
  而不是「先排序再累加」，两者在语义上等价（都把顺序钉死），但具体数值基准取决于
  `dq_kv_order` 的取值。这里只验证「固定顺序 ⇒ 逐位可复现」这件事本身。
- **第 4 组的 PTRR 是照源码算法在 numpy 里重写的**，与
  `torch.distributed.tensor.experimental._context_parallel._load_balancer._PTRRLoadBalancer`
  的 `ptrr_scheduling` 同构，但没有跑真实的多机通信，不含 all-gather / ring 的开销。
- **第 5 组的「浪费率」只算 QK^T 与 PV 两笔 matmul 的 MAC**，不含 softmax、mask 判定、
  访存与调度开销；它是一个「粒度损失」的下界。

## 文件

- `run.py` —— 全部五组检查，支持按编号分组跑与 `--selftest`
- `results/summary.md` —— 完整输出（`python3 run.py > results/summary.md`）
