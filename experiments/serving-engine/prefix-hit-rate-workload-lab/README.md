# prefix-hit-rate-workload-lab：前缀命中率的工作负载形状依赖

对应文章：《推理引擎（08）：前缀命中率 —— 选型前最便宜的那次测量》
（`lrypcy.github.io` 推理服务分类，serving 系列第 8 篇）。

## 这份脚本回答什么

serving-07 的决策树把「前缀命中率」放在负载形状那一级：>60% 选 SGLang，
<20% 选 vLLM，中间先测。这份脚本把那一级的判断做实，回答三件事：

1. **三种典型负载各自能到多少命中率**：多轮对话 / RAG / Agent 循环，
   各按 token 级最长前缀（radix 理想形态）与 16 token 块对齐
   （vLLM APC 形态）两种粒度各测一遍，顺带量出**块对齐税**；
2. **缓存容量对命中率的塑造**：容量小于 working set 时命中率被淘汰
   压住，超过后饱和 —— 小显存副本上测出的命中率测的是淘汰曲线，
   不是负载曲线；
3. **口径陷阱**：token 加权 vs 请求加权、聚合窗口 vs 时间窗、
   分层命中，为什么两个引擎报出的同一个 `cache_hit_rate` 不可直接对比。

全程纯 Python / CPU、确定性输出（固定 seed）。合成的是 token 序列——
生产里先用自己的 tokenizer 把请求 tokenize 成序列，后面的算法完全相同。

## 如何运行

```bash
python3 run.py              # 跑三组，打印表格
python3 run.py --selftest   # 只跑断言（静默通过），适合挂 CI
```

## 三组检查与结论

| # | 检查内容 | 结论 |
| --- | --- | --- |
| 1a | 多轮对话（system 510 + 6 轮 × 130） | token 级 **86.3%**，块对齐 85.7%，税 0.58pp |
| 1b | RAG（system 250 + doc 1022 + query 61） | token 级 **91.6%**，块对齐 91.0%，税 0.61pp；闭式上界 1272/1333 = 95.4%（Zipf 流行度 + 冷启动压住） |
| 1c | Agent 循环（system+tools 1020 + 15 步 × 98） | token 级 **94.4%**，块对齐 94.0%，税 0.37pp |
| 1d | 块对齐税上界自校验 | 逐负载 ≤ 15 token / 平均 prompt 长度；prompt 越短税越重 |
| 2a | 容量曲线（RAG，working set 30,910 token ≈ 1932 块） | 64 块 2.8% → 256 块 40.0% → 1933 块 87.2% → 饱和 89.4%；**容量 < working set 时测到的是淘汰曲线** |
| 2b | 冷缓存假象 | 前 20 个请求 59.4% vs 全窗口 89.4%：短窗口开场系统性低估 |
| 3 | 口径陷阱 | token 加权 86.1% vs 请求加权 85.1%（相差 1.0pp）；vLLM 聚合最近 1000 请求 vs SGLang 15s 时间窗；SGLang 分 device/host/storage 层上报，vLLM 另有 connector 命中另一本账 |

**一句话结论**：命中率首先由负载形状决定（system prompt 长度、共享文档数、
轮数），块对齐税是次要修正；容量必须大于 working set；跨引擎对比前先对齐
上报口径。所以选型的第一步不是跑 benchmark，而是数自己请求里的公共前缀。

## 哪些结论是这份脚本**没有**主张的

- **合成负载不是线上流量**。三种形状的参数（system 长度、doc 数、轮数）
  是构造的，用来展示「形状 → 命中率」的单调关系与量级；
  上线前要用真实请求日志复算。
- tokenize 用均匀随机 id 代替真实 tokenizer：命中率只依赖 token 序列的
  相等关系，与词表无关；但真实 tokenizer 的合并边界会影响「同一段文本
  是否切成同一串 token」。
- 组 1 的块对齐仿真只按块淘汰/命中建模，没有建 vLLM 的 hash 冲突、
  evictable 优先级与 per-request 生命周期细节。
- 组 2 的 LRU 是近似：vLLM 按块维护引用计数与 FIFO/LRU 混合策略，
  曲线形状一致、具体数值会有偏差。
- 组 3 只打印两引擎的源码口径事实，不做加权口径差异的仿真
  （那需要建到达率模型）。

## 目录

```
run.py               三组检查（--selftest 只跑断言）
results/summary.md   固化输出（确定性，无 seed 依赖）
results/summary.json 机读版
```
