# pd-disaggregation-transfer-cost：PD 分离的两笔账

对应文章：[推理引擎（05）：PD 分离与 KV 传输](https://lrypcy.github.io/2026/10/18/serving-05-pd-disaggregation-kv-transfer/)
（`lrypcy.github.io` 推理服务分类，serving 系列第 5 篇）。

## 这份脚本回答什么

PD 分离把 prefill 和 decode 拆到不同实例，中间隔着一根链路传 KV。两笔账决定它划不划得来：

1. **传输 vs 重算**：把 KV 从 P 传到 D，和 D 自己重算一遍 prefill，哪个快？
   流传的说法是「prompt 越长越值得传」。这份脚本说明**这个直觉是错的**：
   两个时间都正比于 $$L$$，比值与 $$L$$ 无关。
2. **租约 vs 超时**：P 算完之后 KV 块要驻留到 D 读走。D 崩溃了怎么办？
   vLLM 的答案是**短租约 + 心跳续租**（`docs/design/nixl_kv_cache_lease.md`），
   本脚本把它的状态机跑出来，和旧的「单一 480s 超时」对照。

纯 Python / CPU、确定性输出（无随机数），两组都做「闭式真值 ↔ 数值反算」自校验。

## 如何运行

```bash
python3 run.py              # 跑全部两组，打印表格
python3 run.py --selftest   # 只跑断言（静默通过），适合挂 CI
python3 run.py 1 2          # 只跑第 1、2 组
```

## 两组检查与结论

| # | 检查内容 | 结论 |
| --- | --- | --- |
| 1a | 每 token KV 字节数：闭式解 vs 逐层累加 | 四个模型全部一致（Llama-3-8B 131,072 B/token、Qwen3-32B 262,144、Llama-3-70B 327,680、DeepSeek-V3 MLA 70,272） |
| 1b | 比值随 prompt 长度 $$L$$ 的变化 | $$L\in[512,131072]$$ 五个数量级上比值恒为 0.08（PCIe Gen5，Llama-3-8B）。**传输与重算的时间比与 $$L$$ 无关** |
| 1c | 盈亏平衡带宽 $$BW^\ast = bF/(2N_{\rm active})$$ | 8B 4.04 GB/s、Qwen3-32B 1.98、70B 1.15、DeepSeek-V3(MLA) 0.47。10GbE TCP 上 8B 重算快 3.2 倍，70B 传输已反超（0.92） |
| 1d | 单请求 KV 体量 | BF16、$$L=4096$$ 时 0.27~1.25 GB；$$L=32768$$ 时 2.1~10 GB —— 这是 P 侧租约窗口内要钉住的显存 |
| 2 | lease + heartbeat 状态机（四场景） | 健康排队 100s：租约活到传输，白占 0s；D 崩溃：27.0s 回收（旧设计 480s，**17.8 倍差距**）；过载（心跳间隔 4.8s）：仍活到传输 |

**一句话结论**：该不该做 PD 分离，第一笔账由「模型每 token KV 字节数 ÷ 激活参数量」
与链路带宽决定，和 prompt 长度无关；第二笔账由故障模型决定 ——
单一超时无法同时满足「崩溃后快回收」和「排队别误杀」，短租约 + 心跳可以。

完整输出见 [`results/summary.md`](results/summary.md)，机读版见
[`results/summary.json`](results/summary.json)。

## 哪些结论是这份脚本**没有**主张的

- **不主张任何实测带宽 / 算力数字**。NVLink 400 GB/s、PCIe 50 GB/s、
  IB 50 GB/s、TCP 1.25 GB/s、H100 BF16 494.7 TFLOPS 全部是**标称值假设**，
  用来展示量级关系；真实有效带宽取决于拓扑、并发、协议栈。
- **不主张端到端延迟数字**。组 1 只算「传输一块 KV」与「重算这块 KV」的比值，
  没有 PD 分离带来的调度、路由、序列化开销，也没有 prefill 批处理对
  $$t_{\rm recompute}$$ 的摊薄。
- 模型参数取自公开 config（LLaMA-3、Qwen3、DeepSeek-V3 论文/仓库），
  MoE 用**激活参数量**（37B）而非总量（671B）—— 前向 FLOPs 只由激活参数决定，
  这一点脚本内部没有验证，是建模假设。
- 组 2 是**设计文档参数的状态机复现**，不是对 vLLM 运行时的测量；
  心跳「在 forward loop 里发、因此不精确」这一实现在仿真里表现为理想周期。
- 没有模拟 RDMA 传输本身失败（只模拟了 D 崩溃），也没有模拟 P 崩溃。

## 目录

```
run.py               两组检查（--selftest 只跑断言）
results/summary.md   固化输出（确定性，无 seed 依赖）
results/summary.json 机读版
```
