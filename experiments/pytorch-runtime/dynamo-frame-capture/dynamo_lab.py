#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamo_lab：TorchDynamo 图捕获 / Graph Break / Guard 的实测证据

对应文章：《从 Python 解释器到 PyTorch Dynamo：字节码、Frame 与图捕获》
（lrypcy.github.io，系统实现分类，Python 运行时与图捕获系列第 1 篇）

依赖：torch 2.10.0（CPU 即可，不需要 GPU）。固定 seed，纯 CPU，无网络。
所有数字都来自本机实测，用来支撑正文里那些「为什么会这样」的判断。

用法：
    python3 dynamo_lab.py            # 跑全部三组
    python3 dynamo_lab.py 7 9        # 只跑第 7、9 组
    python3 dynamo_lab.py --selftest # 只跑断言（静默）
"""

import sys
import warnings

import torch

warnings.filterwarnings("ignore")

from torch._dynamo.testing import EagerAndRecordGraphs  # noqa: E402

OK = []
VERBOSE = True
SEED = 20261019


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)


def section(title):
    if VERBOSE:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)


def record(fn, *args):
    """用 EagerAndRecordGraphs 跑一次，返回 (backend, 输出)。"""
    backend = EagerAndRecordGraphs()
    compiled = torch.compile(fn, backend=backend)
    out = compiled(*args)
    return backend, out


# ----------------------------------------------------------------------------
# G7  从 Python 函数到 FX Graph：捕获了什么
# ----------------------------------------------------------------------------
def g7_capture():
    section("G7  从 Python 函数到 FX Graph：捕获了什么")

    torch.manual_seed(SEED)

    def model(x, w):
        y = x @ w
        y = torch.relu(y)
        return y * 2

    x = torch.randn(8, 16)
    w = torch.randn(16, 32)

    backend, out = record(model, x, w)
    n_graphs = len(backend.graphs)

    check(n_graphs == 1, "G7: 纯 Tensor 运算的函数被捕成 1 个图（实得 %d）" % n_graphs)

    gm = backend.graphs[0]
    nodes = [n for n in gm.graph.nodes]
    call_nodes = [n for n in nodes if n.op == "call_function"]
    placeholders = [n for n in nodes if n.op == "placeholder"]
    ops = [n.target.__name__ if hasattr(n.target, "__name__")
           else str(n.target) for n in call_nodes]

    check(ops == ["matmul", "relu", "mul"],
          "G7: 三个算子按数据流排列 matmul → relu → mul（实得 %s）" % ops)
    check(len(placeholders) == 2, "G7: 两个 placeholder 对应 x / w")
    check(torch.allclose(out, torch.relu(x @ w) * 2),
          "G7: 编译后的输出与 eager 一致")

    # 关键：Python 的控制流与赋值没有进图，只留数据依赖
    check(all(n.op != "call_method" for n in nodes),
          "G7: 图里没有 Python 方法调用 —— 只有 Tensor 数据流")

    if VERBOSE:
        print("捕获的图：")
        print(gm.print_readable(print_output=False))
        print("图数量      :", n_graphs)
        print("算子节点    :", ops)
        print("placeholder :", [p.name for p in placeholders])
        print("输出等于 eager:", torch.allclose(out, torch.relu(x @ w) * 2))

    # 带一段纯 Python 逻辑的同一个函数
    def noisy(x):
        y = torch.relu(x)
        print("shape:", tuple(y.shape))
        return y * 2

    backend2, _ = record(noisy, torch.randn(4, 8))
    check(len(backend2.graphs) == 2,
          "G7: 中间插一句 print 之后被切成 2 个图（实得 %d）" % len(backend2.graphs))

    if VERBOSE:
        print("\n带 print 的版本：图数量 =", len(backend2.graphs))
        for i, g in enumerate(backend2.graphs):
            print("  graph%d 算子:" % i,
                  [n.target.__name__ for n in g.graph.nodes if n.op == "call_function"])


# ----------------------------------------------------------------------------
# G8  Graph Break 触发源清单
# ----------------------------------------------------------------------------
def g8_break_sources():
    section("G8  Graph Break 触发源清单（torch._dynamo.explain 实测）")

    torch.manual_seed(SEED)
    x = torch.randn(8, 8, dtype=torch.float32)

    cases = []

    def clean(t):
        return torch.relu(t) * 2

    def with_print(t):
        y = torch.relu(t)
        print("s", tuple(y.shape))
        return y * 2

    def with_item(t):
        return t.sum().item() * 2

    def data_dep(t):
        y = t @ t
        if y.sum() > 0:
            return torch.relu(y)
        return torch.sigmoid(y)

    def with_tolist(t):
        return [v * 2 for v in t.sum(dim=-1).tolist()]

    def numpy_bridge(t):
        import numpy as np
        return torch.from_numpy(np.asarray(t) * 2)

    for name, fn in [("纯 Tensor 运算（基线）", clean),
                     ("内置 print", with_print),
                     (".item()", with_item),
                     ("数据依赖分支", data_dep),
                     (".tolist()", with_tolist),
                     ("numpy 往返", numpy_bridge)]:
        torch._dynamo.reset()
        exp = torch._dynamo.explain(fn)(x)
        if exp.break_reasons:
            r = exp.break_reasons[0]
            reason = r.reason.strip().splitlines()[0]
            flag = bool(getattr(r, "graph_break", None))
        else:
            reason, flag = "", False
        cases.append((name, exp.graph_count, exp.graph_break_count,
                      exp.op_count, reason, flag))

    # 断言：基线无断点；print 与数据依赖分支各切一刀
    check(cases[0][2] == 0, "G8: 纯 Tensor 基线 0 个断点")
    check(cases[1][1] == 2, "G8: print 把函数切成 2 段（实得 %d）" % cases[1][1])
    check(cases[3][1] == 2, "G8: 数据依赖分支把函数切成 2 段（实得 %d）" % cases[3][1])
    check("generic_jump" in cases[3][4],
          "G8: 数据依赖分支的断点原因是 generic_jump（%s）" % cases[3][4])

    if VERBOSE:
        print("%-22s %6s %6s %6s  %s" % ("用例", "图数", "断点", "算子", "首个断点原因"))
        print("-" * 96)
        for name, gc, bc, oc, reason, flag in cases:
            print("%-22s %6d %6d %6d  %s" % (name, gc, bc, oc, reason[:48]))
        print("\ngraph_break 标志位：", {c[0]: c[5] for c in cases})

    # fullgraph=True：把断点变成硬错误
    torch._dynamo.reset()
    compiled = torch.compile(with_print, fullgraph=True)
    raised = None
    try:
        compiled(torch.randn(4, 4))
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    check(raised is not None, "G8: fullgraph=True 时断点变成异常（%s）" % raised)
    if VERBOSE:
        print("\nfullgraph=True 遇到 print 时抛出：", raised)


# ----------------------------------------------------------------------------
# G9  Guard 的构成，以及失效之后会发生什么
# ----------------------------------------------------------------------------
def g9_guards():
    section("G9  Guard 的构成 / 失效之后的重编译")

    torch.manual_seed(SEED)

    def f(x):
        return torch.relu(x) * 2

    # --- guard 清单长什么样 ---
    torch._dynamo.reset()
    exp = torch._dynamo.explain(f)(torch.randn(8, 32))
    guards = exp.out_guards
    code_lines = []
    for g in guards:
        cl = getattr(g, "code_list", None) or []
        for line in cl:
            if line:
                code_lines.append(line.split(",")[0][:78])
    types = sorted({t for g in guards for t in (getattr(g, "guard_types", None) or [])})

    check(len(guards) >= 6, "G9: 一次最简单的编译也产生 %d 条 guard" % len(guards))
    check("TENSOR_MATCH" in types,
          "G9: Tensor 的形状/dtype 由 TENSOR_MATCH 这类 C++ 侧 guard 检查（%s）" % types)
    check(any("check_obj_id" in c and "torch" in c for c in code_lines),
          "G9: Python 侧 code_list 里有 `torch` 模块的 id 检查")

    if VERBOSE:
        print("guard 总数：%d，其中带 Python code 的 %d 条" % (len(guards), len(code_lines)))
        print("guard 类型：", types)
        print("\nPython 侧 code_list：")
        for c in code_lines:
            print("   ", c)

    # --- 形状变化 -> 重编译；automatic dynamic shapes 会救一次 ---
    def run_shapes(calls, **cfg):
        torch._dynamo.reset()
        for k, v in cfg.items():
            setattr(torch._dynamo.config, k, v)
        b = EagerAndRecordGraphs()
        c = torch.compile(f, backend=b)
        for t in calls:
            c(t)
        return len(b.graphs)

    four = [torch.randn(n, 32) for n in (4, 8, 16, 32)]
    same = [torch.randn(8, 32) for _ in range(4)]

    n_static = run_shapes(four)
    n_noauto = run_shapes(four, automatic_dynamic_shapes=False)
    n_same = run_shapes(same)

    check(n_same == 1, "G9: 形状不变的 4 次调用只编译 1 次（实得 %d）" % n_same)
    check(n_noauto == 4, "G9: 关掉自动动态后，4 个 batch 各编译一次（实得 %d）" % n_noauto)
    check(1 <= n_static < n_noauto,
          "G9: 默认开启自动动态形状后，4 个 batch 只编译 %d 次 < %d" % (n_static, n_noauto))

    if VERBOSE:
        print("\n4 个不同 batch（4/8/16/32）依次调用：")
        print("   automatic_dynamic_shapes=True （默认）: %d 个图" % n_static)
        print("   automatic_dynamic_shapes=False        : %d 个图" % n_noauto)
        print("4 个相同 batch（8）依次调用                 : %d 个图" % n_same)
        print("\n读法：第一次形状不匹配时 guard 失败触发重编译；")
        print("      自动动态形状会把这一维标记成动态，于是后续尺寸不再重编译。")

    # --- dtype 变化 -> 自动动态救不了 ---
    torch._dynamo.reset()
    b = EagerAndRecordGraphs()
    c = torch.compile(f, backend=b)
    c(torch.randn(8, 32))
    c(torch.randn(8, 32).to(torch.float64))
    c(torch.randn(8, 32))
    n_dtype = len(b.graphs)
    check(n_dtype == 2,
          "G9: dtype 从 f32 换成 f64 再换回来，共 %d 个图（自动动态救不了 dtype）" % n_dtype)
    if VERBOSE:
        print("\ndtype f32 → f64 → f32 三次调用：%d 个图" % n_dtype)

    # --- 动态形状标注 ---
    torch._dynamo.reset()
    b = EagerAndRecordGraphs()
    c = torch.compile(f, backend=b)
    for n in (4, 8, 16, 32):
        t = torch.randn(n, 32)
        torch._dynamo.mark_dynamic(t, 0)
        c(t)
    n_marked = len(b.graphs)
    check(n_marked == 1,
          "G9: 显式 mark_dynamic(t, 0) 后 4 个 batch 只编译 1 次（实得 %d）" % n_marked)
    if VERBOSE:
        print("显式 mark_dynamic(t, 0) 后 4 个 batch：%d 个图" % n_marked)


GROUPS = [
    (7, "从 Python 函数到 FX Graph：捕获了什么", g7_capture),
    (8, "Graph Break 触发源清单", g8_break_sources),
    (9, "Guard 的构成 / 失效之后的重编译", g9_guards),
]


def main(argv):
    global VERBOSE
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}

    VERBOSE = "--selftest" not in flags

    if args:
        want = {int(a) for a in args}
        groups = [g for g in GROUPS if g[0] in want]
    else:
        groups = GROUPS

    if VERBOSE:
        print("dynamo_lab — torch %s / CPython %s" % (
            torch.__version__, sys.version.split()[0]))
        print("组：%s" % ", ".join(str(g[0]) for g in groups))

    for num, desc, fn in groups:
        if VERBOSE:
            print("\n>>> G%d  %s" % (num, desc))
        fn()

    if VERBOSE:
        print("\n" + "=" * 78)
        print("全部通过：%d 条断言" % len(OK))
        print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
