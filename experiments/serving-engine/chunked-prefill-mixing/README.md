# chunked-prefill-mixing：把 prefill 切碎塞进 decode 空隙

对应文章：[推理引擎（02）：调度与批处理](https://lrypcy.github.io/2026/10/18/serving-02-scheduling-and-batching/)

## 问题

continuous batching 之后，引擎在一个 batch 里同时装正在 decode 的请求和正在 prefill 的请求。
把长输入的 prefill **切碎**（chunked prefill）塞进 decode 之间的空隙，收益和代价各是什么？
chunk 大小 $$\tau$$ 该怎么选？

## 模型

成本模型与文章 §2 的推导同一套：

```
t_step = t_W + t_compute * prefill_tokens + t_kv * n_running_decode
```

- `t_W`：每步读一遍权重，与 batch 无关 → decode 可被 batch 摊薄（文章 §2 的
  $$t_{\rm step}(b)/b = t_W/b + \bar t_K$$）；
- `t_compute`：每个 prefill token 的计算成本（算力受限）；
- `t_kv`：每个**在飞 decode** 请求每步的 KV 读取成本（带宽受限）。

一步耗时跑两种口径，混批的收益假设完全取决于第二种：

| 口径 | 公式 | 含义 |
|:---|:---|:---|
| `sum`（保守） | `t_W + t_compute*pf + t_kv*b` | 计算与带宽串行 |
| `max`（重叠） | `max(t_W + t_compute*pf, t_W + t_kv*b)` | 计算单元与带宽单元并行 |

两种调度模式：

- `phase`（分相）：一步要么全是 prefill（整请求入场、可拼批），要么全是 decode。
  prefill 优先 —— 长 prefill 阻塞 decode（即关闭 chunked prefill 的引擎行为）；
- `mixed`（混批）：一步内 decode 各拿 1 token，剩余预算按 $$\tau$$ 切给等待中的 prefill。

负载：400（重载）/ 120（轻、中载）个请求，15% 是 8192 token 的长输入、其余 256 token，
输出固定 128 token，一步预算 2048 token（vLLM `max_num_batched_tokens` 默认值）。

## 运行

```bash
python3 run.py 轻载   # 或 中载 / 重载；沙箱有时长限制，分开跑
python3 run.py all    # 一次跑完（可能被 SIGTERM 掐断）
```

`results/summary-*.json` 是各 regime 的分文件输出，`results/summary.json` 是合并版，
`results/summary.md` 是全部 stdout。

## 自校验

1. **token 守恒**：全部请求完成，输出 token 总数 == `sum(L_out)`；
2. **摊薄极限**：每请求每步成本 $$= t_W/b + t_K$$，$$b=1$$ 时 102.0、$$b\to\infty$$ 时 $$\to 2.0$$；
3. **分相忙碌时间** <= 解析上界（prefill 全串行 + decode 全串行）；
4. **延迟方向**（两种成本口径下都成立）：$$\tau$$ 增大（切得更粗）时，长请求延迟不得上升；
5. **尾延迟方向**（中载）：$$\tau \ge 256$$ 的混批，短请求 p99 不高于分相（无队首阻塞）。

## 结论

| 负载 | makespan（吞吐） | 短请求尾延迟 | 长请求 p50 |
|:---|:---|:---|:---|
| 轻载（并发 <1） | 各模式基本相同（到达率受限） | p99 212–527，$$\tau=64$$ 最差 | $$\tau=64$$ 时 16.5× |
| 中载（并发 ~3） | 基本相同（差异 <0.1%） | **$$\tau=256$$ 把 p99 从 565 砍到 233（−59%）** | 889 → 13917（$$\tau=64$$，16×） |
| 重载（饱和） | 分相略快 2–7% | 排队主导，差异被淹没 | $$\tau=64$$ 时 2.2× |

1. **吞吐不是混批的收益点**：三种负载下 makespan 差异 1–7% 且方向不一致
   （中载混批不慢于分相，重载分相反而略快）。
2. **收益点在短请求尾延迟**：中载 $$\tau=256$$ 让短请求 p99 降 59% 而 makespan 不变 ——
   分相模式下短请求要等长 prefill 整个算完（队首阻塞）。
3. **代价恒定**：$$\tau$$ 越小长请求 p50 越差，且单调（中载 16×、重载 2.2×）。
4. **$$\tau$$ 调的是延迟分布不是吞吐**，且存在中间最优（本负载约 256）。
5. 两种成本口径只改变绝对耗时（重载 172k vs 146k），不改变任何结论的方向。
