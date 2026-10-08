# Megatron-LM 解析工具箱 · 对应《深度剖析（00）》四张表

## 对应文章

- 《Megatron-LM 深度剖析（00）：代码地图、配置系统与一次迭代的控制流全图》
  https://lrypcy.github.io/2026/09/28/megatron-00-foundation/

正文引用本目录的 `flops.py`，锁定 commit 的 permalink 见正文 §4 与参考节。

## 这是什么

从博客仓库 `tools/megatron_bench/` 迁入的**纯 CPU 解析工具箱**。无 GPU、无第三方依赖
（纯标准库），所有数字都是 A 级证据——可重跑、可改参数、可被他人复算。

源码基准：`megatron-core` 0.20.0 / commit `60e039626`。

## 如何运行

```bash
cd ~/Desktop/Projects/sandbox/ipynbs/experiments/megatron/foundation

# ① 逐格对账：把正文 §4.1 / §4.2 / §4.4 / §4.5 四张表的每个数字重跑一遍
~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏

# ② 工具箱 CLI（单模型全量报告）
~/Software/miniconda3/bin/python3 __main__.py --model llama-7b --flops
~/Software/miniconda3/bin/python3 __main__.py --model gpt3-175b --hardware a100-80g --mem --recompute full
```

`gen_stdout.py` 退出码非 0 表示有对账失败项，可直接当 CI 用。

> **与原仓库的差别**：原 `tools/megatron_bench/` 是可按 `python3 -m megatron_bench` 调用的包，
> 内部用相对导入（`from .configs import ...`）。本目录按 `experiments/profiling/_build/`
> 的既有约定改成**平铺 sibling 导入**（`sys.path.insert` + `from configs import ...`），
> 因此入口是 `python3 __main__.py` 而非 `python3 -m megatron_bench`。
> 计算逻辑逐字未改，只改了导入方式与运行方式。

## 文件

| 文件 | 作用 |
|---|---|
| `configs.py` | 模型规格（`ModelSpec`）与硬件规格、`BenchConfig` |
| `flops.py` | FLOPs 记账：逐算子精确累加 / Megatron 官方公式复刻 / 6ND 近似，三口径并列 |
| `mem.py` | 显存账本：权重 / 梯度 / 优化器状态 / 激活（逐张量清单）/ 碎片 |
| `comm.py` | 集合通信 α-β 代价模型，All-Reduce / RS / AG / All-to-All |
| `__main__.py` | CLI 入口，打印单模型的 flops / mem / comm 报告 |
| `gen_stdout.py` | 逐格对账本文四张表，产出 `results/stdout.txt` |
| `results/stdout.txt` | 固化的对账输出（71 行），读者不必自己跑 |

## 结论

### §4.1 参数量自检——GPT-3 算出 175.181 B，与公布的 175,181,291,520 一致

| 模型 | total | no-embed | gated |
|---|---|---|---|
| llama-7b | 6.738 B | 6.476 B | True |
| llama3-8b | 8.030 B | 6.979 B | True |
| qwen-14b | 14.769 B | 13.212 B | True |
| gpt3-175b | **175.181 B** | 173.946 B | **False** |

**这一步当场拦下过一个 bug**：初版 `ModelSpec` 不区分 MLP 结构，统一按 SwiGLU 的
gate/up/down 三个投影计。GPT-3 用的是 GELU，只有 up/down 两个——按 SwiGLU 会算出
233.2 B（差 33%）。修法是加 `gated_mlp` 标记。**每个模型规格都必须标明 MLP 是 gated
还是非 gated**，FLOPs 记账同样依赖这个区分（SwiGLU 是 $$6hI$$，GELU 是 $$4hI$$）。

### §4.2 交叉验证——两套独立实现在四种架构上完全吻合

| 模型 | 序列长度 | 独立实现（causal 折半） | Megatron 官方公式 | 偏差 |
|---|---|---|---|---|
| llama-7b | 4096 | 42.864 GFLOP | 42.864 GFLOP | +0.00% |
| llama3-8b | 8192 | 51.470 GFLOP | 51.470 GFLOP | +0.00% |
| qwen-14b | 8192 | 96.023 GFLOP | 96.023 GFLOP | -0.00% |
| gpt3-175b | 2048 | 1061.878 GFLOP | 1061.878 GFLOP | +0.00% |

### §4.4 官方 FLOPs 记账不含重计算

`training.py::num_floating_point_operations()` 收尾是 `return flops_fwd * 3`（L1007），
**不乘重计算系数**。所以开重计算后日志里的 `throughput per GPU` 不随之增大：

| llama-7b | none | selective | full |
|---|---|---|---|
| GFLOP/token | 46.08 | 48.23 | 61.18 |

实际算力增量 = 61.18 / 46.08 = **1.328x（+32.8%）**，计数器报的数不变。

### §4.5 6ND 与精确口径的偏差随序列长度放大

| 模型 | 序列长度 | 官方口径 | 6ND | 偏差 |
|---|---|---|---|---|
| llama-7b | 4096 | 42.864 | 38.856 | +10.3% |
| llama3-8b | 8192 | 51.470 | 41.876 | +22.9% |
| qwen-14b | 8192 | 96.023 | 79.272 | +21.1% |
| gpt3-175b | 2048 | 1061.878 | 1043.677 | +1.7% |

序列翻倍则偏差翻倍（未折半口径：llama-7b +18.6% → llama3-8b +38.3%）。原因是 6ND 的
$$N$$ 不含 attention core 的 $$4sh$$ 项——该项与参数量无关，只随序列长。gpt3-175b 只差
1.7%，因为它序列短（2048）而参数量巨大，attention core 占比被稀释。

## 它验证了本文哪个数字

`results/stdout.txt` 里 30 个断言全部 `ok`、0 个 `MISMATCH`，覆盖正文：

- §4.1 的 8 个参数量数字（4 模型 × total / no-embed）+ 4 个 `gated` 标记
- §4.2 的 8 个 GFLOP 数字 + 4 个偏差百分比
- §4.4 的 2 个 GFLOP 数字 + `full/none = 1.328` 这个比值
- §4.5 第一张表的 8 个 GFLOP + 4 个偏差百分比
- §4.5 第二张表的 1 个 GFLOP + 2 个偏差百分比

正文 §4.5 第二张表只给了 llama-7b / llama3-8b 的百分比，没给 llama3-8b 的绝对值
（脚本实跑 57.913 GFLOP）；对账脚本对此只报不判。
