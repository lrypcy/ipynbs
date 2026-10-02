# 性能剖析系列 · 可复算 notebook

配套博客：《性能剖析（00/01/02）》三篇
（[00 度量口径](https://lrypcy.github.io/2026/09/29/profiling-00-metrics/) ·
[01 训练栈](https://lrypcy.github.io/2026/09/29/profiling-01-training/) ·
[02 推理服务](https://lrypcy.github.io/2026/09/29/profiling-02-inference/)）

博客正文里引用的每一个数字都出自这里。**所有输出已固化，GitHub 上直接读即可，不需要自己跑。**

## 为什么是 notebook 而不是 .py

博客仓库（`lrypcy.github.io`）的 `tools/` 被 Jekyll 排除在构建之外，站点上访问不到；
GitHub 上虽然能翻到源码，但要读懂一个 1200 行的脚本得先下载再逐行读。
notebook 的好处是：**代码和输出并排**，读者一眼能看到「这个函数跑出来是什么」，
不用本地配环境。

## 七个 notebook

| Notebook | 回答什么问题 | 对应脚本 |
|:---|:---|:---|
| [profiling-00-mfu-accounting](profiling-00-mfu-accounting/) | 分子换一下、分母换一下，MFU 差多少 | `mfu_accounting.py` |
| [profiling-00-timer-semantics](profiling-00-timer-semantics/) | 内置 timer 测的是「阶段耗时」还是「排空耗时」 | `timer_semantics.py` |
| [profiling-00-goodput](profiling-00-goodput/) | goodput 和 MFU 为什么会给出相反的结论 | `goodput_calc.py` |
| [profiling-01-trace-agg](profiling-01-trace-agg/) | 时间分解 / idle 三分 / overlap 双分母 / 掉队卡 | `trace_agg.py` |
| [profiling-01-overlap-bound](profiling-01-overlap-bound/) | 重叠能藏多少、什么情况下根本藏不动 | `overlap_bound.py` |
| [profiling-02-serving-model](profiling-02-serving-model/) | TTFT/TPOT 分位、goodput、拐点由什么决定 | `serving_model.py` |
| [profiling-02-serving-schema](profiling-02-serving-schema/) | 字段对齐、模型给不出的量、ITL vs TPOT 的分母 | `serving_schema.py` |

全部纯 CPU 可跑（`serving_model.py` 用到 numpy，但缺了也能跑，stdlib 有回退）。

## 代码从哪来（重要）

**代码不是手抄的。** `_build/build.py` 用 `ast` 按顶层符号从博客仓库的 `.py` 里抽取，
所以 notebook 里的每一行都与 `lrypcy.github.io/tools/` 下的源码**逐字一致**。

这样做的理由：博客正文引用了这些脚本跑出来的具体数字。两份代码一旦分叉，正文就变成编的。
抽取而非手抄是这个保证。改动博客仓库的脚本后，重跑构建即可同步：

```bash
cd <ipynbs 仓库根>
python3 experiments/profiling/_build/build.py
python3 experiments/profiling/_build/check_numbers.py
```

构建需要一个带 `nbformat` / `nbclient` / `ipykernel` 的解释器（miniconda base 即可）。
它会在 `BLOG_REPO`（默认 `~/Desktop/Projects/github/lrypcy.github.io`）里读源码。

`check_numbers.py` 是**回填一致性校验**：把 markdown 与 README 里出现的每个小数
回 `results/stdout.txt` 里找，找不到就报错。写 README 时誊错数字是这套流程里
最容易犯也最难发现的错误，所以交给脚本查。

（这个校验器就是被自己抓出来的：第一次写 `mfu-accounting` 的 README 时，
把机器平衡点 153.0 / 295.4 / 1.93× 誊成了 494.5 / 277.9 / 1.78×。）

## 几个需要提前知道的坑

- **`timer-semantics` 实验一有抖动**：它用真线程 + `sleep` 模拟设备侧耗时，
  线程调度开销随机器负载变化，所以**膨胀率的绝对值每次运行都不同**（单调性是稳的）。
  脚本会自己把每 chunk 的开销打印出来。实验二、三、四是确定性计算，无抖动。
- **`serving-model` 的代价常数是示意值**：`DEFAULT_COST` 是一组自洽的默认参数，
  **绝对数值不代表任何具体硬件或模型**。该 notebook 只主张结构。
- **`trace-agg` 的 idle 三分判据是自定的**：与 HTA 基于 Kineto CPU 侧算子链的推断规则
  不完全一致，跨工具比数前必须先对齐判据。

## 目录结构

```
profiling/
├── profiling-00-mfu-accounting/
│   ├── profiling-00-mfu-accounting.ipynb   # 带固化输出
│   ├── README.md
│   └── results/stdout.txt                 # 纯文本输出，便于 grep
├── ...
└── _build/
    ├── build.py          # 抽取 + 执行 + 落盘
    ├── nbextract.py      # ast 抽取器
    ├── check_numbers.py  # 回填一致性校验
    └── spec_*.py         # 每个 notebook 一份 cell 定义
```
