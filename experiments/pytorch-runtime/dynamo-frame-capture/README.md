# dynamo_frame_capture：CPython 运行时与 TorchDynamo 图捕获的可观测证据

对应文章：[《从 Python 解释器到 PyTorch Dynamo：字节码、Frame 与图捕获》](https://lrypcy.github.io/2026/10/19/python-runtime-dynamo-deepdive/)
（`lrypcy.github.io`，系统实现分类，Python 运行时与图捕获系列第 1 篇）。

## 这份 notebook 回答什么

`torch.compile()` 为什么能在不动模型源码的前提下，把一段普通 Python 程序变成
Tensor 计算图？答案不在算子系统，而在 CPython 的执行机制里。但「执行机制」这四个字
很容易讲成空话——字节码长什么样、code object 和 frame 到底谁存什么、
guard 失不失败各会发生什么，都是**可以直接量出来的**。

这份实验把文章里那些「据称如此」的论断摆到数字上，分两个 notebook：

| 文件 | 依赖 | 解释器 | 内容 |
| --- | :--- | :--- | :--- |
| [`runtime_lab.ipynb`](runtime_lab.ipynb) | 仅标准库 | 3.11 / 3.12 / 3.13 | 编译流水线、名称解析、优化层次、code object 与 frame、求值栈、观察钩子（G1–G6） |
| [`dynamo_lab.ipynb`](dynamo_lab.ipynb) | torch 2.10.0（CPU） | 3.11.14 | 图捕获、graph break 触发源、guard 构成与重编译（G7–G9） |

## 如何运行

两个 notebook 里**已经嵌好输出**——直接点开 GitHub 就能看到每一组跑了什么、跑出什么，
不必自己执行。要复现就 Run All：

- `runtime_lab.ipynb`：任何 CPython 3.11+ 内核都行，只用标准库、无随机性、纯 CPU。
- `dynamo_lab.ipynb`：需要装了 `torch 2.10.0` 的内核（CPU 即可，不需要 GPU），
  固定 `seed = 20261019`。

每组末尾的 `check()` 就是断言，最后一个 cell 打印总数。
嵌在文件里的输出来自 **CPython 3.11.14 + torch 2.10.0**：45 + 21 = **66 条断言全过**。
`runtime_lab` 在 3.13.12 上跑是 46 条（多出的一条是 `sys.monitoring` 检查，
3.11 无此模块会自动跳过），那次的输出留档在 `results/runtime_lab.313.out`。

`results/` 下另外两份是纯文本留档：`runtime_lab.out` / `dynamo_lab.out`
（notebook 执行的 stdout 汇总），以及 `dynamo_bytecode.out`
（`TORCH_LOGS="bytecode"` 抓的 ORIGINAL / MODIFIED BYTECODE 对照——这个没法在 notebook 里自然产出）。

## 九组检查与结论

| # | 检查内容 | 结论 |
| ---:|:---|:---|
| 1 | 编译流水线四个阶段的产物类型 | 源码 → `ast.Module` → `SymbolTable` → `types.CodeType` → 字节码。函数体是**外层** code object 的 `co_consts` 之一；默认参数 `b=1` 不在函数自己的 consts 里，而在外层存成元组 `(1,)`，`MAKE_FUNCTION` 时才绑定 |
| 2 | 编译期作用域分类 → 运行时字节码 | 同一行 `a + b + c + x`，四个名字编译期分成 free / free / local / global，运行时分别落成 `LOAD_DEREF` / `LOAD_DEREF` / `LOAD_FAST` / `LOAD_GLOBAL`。**字节码是编译期分类的结果，不是运行时查表的产物** |
| 3 | 编译期常量折叠 vs 运行时专门化 | `2*3` 在 `co_consts` 里就是 `6`（编译期）；而 `a + b` 的专门化是**运行时**状态——三份源码相同的 code object，只喂 int 得到 `BINARY_OP_ADD_INT`，只喂 str 得到 `BINARY_OP_ADD_UNICODE`，没跑过的仍是通用 `BINARY_OP` |
| 4 | code object 复用 / frame 独立 | 工厂函数每次返回的函数对象 `is` 同一份 `__code__`；递归 4 层同时存活 4 个 frame，各自 `f_locals['n']` = 4/3/2/1，用 `f_back` 串成链。两个同源生成器的 `gi_frame` 独立，`n` 互不干扰 |
| 5 | 求值栈深度反算 | 用 `dis.stack_effect` 逐指令累加得到的峰值 **恰好等于** `co_stacksize`（本例 2 == 2）——编译器算的是最坏情况的精确值，不是拍脑袋的上界 |
| 6 | 观察钩子的能力边界 | 2000 次 `work(20)`：基线 1.00x，行级 `settrace` 约 5–8x，指令级约 37–41x。回调返回 `None` / `DISABLE`，只能「看」，**不能改写这一步该算出什么**——这正是它和 PEP 523 的差别 |
| 7 | 图捕获了什么 | `relu(x@w)*2` → 1 个图、3 个 `call_function` 节点（matmul → relu → mul）、2 个 placeholder。Python 的赋值与控制流一个都没进图，只剩数据依赖 |
| 8 | graph break 触发源 | 基线 0 断点；`print` 切成 2 段；`if y.sum() > 0` 数据依赖分支切成 2 段（原因 `generic_jump TensorVariable()`）；`.item()` / `.tolist()` 让图里一个算子都捕不到（ops=0）。`fullgraph=True` 时断点直接变成异常 |
| 9 | guard 构成与失效 | 最简单的一次编译也有 10 条 guard，分两层：Python 侧 `code_list`（`torch` 模块 id、`torch.relu` 函数 id、`hasattr(L['x'], '_dynamo_dynamic_indices')`）与 C++ 侧 `TENSOR_MATCH`（形状 / dtype / stride / device 不在 Python code 里）。4 个不同 batch 依次调用：默认（自动动态形状开）只编译 **2** 次，关掉则 **4** 次；显式 `mark_dynamic(t, 0)` 后 **1** 次。dtype f32→f64→f32 是 **2** 次——自动动态只救得了形状，救不了 dtype |

## 复用这些数字时要小心什么

- **第 3 组的第三格（int/float 混跑）跨版本不稳定**：3.11 实测退回
  `BINARY_OP_ADAPTIVE`，3.13 仍保留 `BINARY_OP_ADD_INT`。版本间稳定的是前两格，
  正文引用的是前两格。
- **第 6 组的倍数只是量级参考**：具体倍率随机器、Python 版本、`work` 的循环体而变，
  稳定的是「指令级 ≫ 行级 ≫ 基线」这个量级差。
- **第 8 组里 `.item()` / `.tolist()` 两行**：`torch._dynamo.explain` 报
  `graph_break_count = 0`，但 `break_reasons` 里仍有一条 `graph_break = True` 的记录。
  这两列口径不同，正文只引用 `break_reasons` 的原因文本。
- **guard 条数随输入个数变化**：1 个 Tensor 输入 10 条，2 个 11 条。正文引用的是构成，不是条数。
