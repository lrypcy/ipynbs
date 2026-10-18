#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_lab：CPython 运行时四件事的可观测证据

对应文章：《从 Python 解释器到 PyTorch Dynamo：字节码、Frame 与图捕获》
（lrypcy.github.io，系统实现分类，Python 运行时与图捕获系列第 1 篇）

只用标准库（dis / symtable / sys / ast / types），纯 CPU，无随机性。
在 CPython 3.11 / 3.12 / 3.13 上均可运行；实测 3.11.14 与 3.13.12。

用法：
    python3 runtime_lab.py            # 跑全部六组
    python3 runtime_lab.py 1 3        # 只跑第 1、3 组
    python3 runtime_lab.py --selftest # 只跑断言（静默）
"""

import ast
import dis
import io
import sys
import symtable
import textwrap
import time
import types

OK = []          # 断言计数
VERBOSE = True


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)


def section(title):
    if VERBOSE:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)


def dump_codeobj(co, indent=0):
    """把一个 code object 的字节码抓成字符串（不依赖 stdout）。"""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        dis.dis(co)
    finally:
        sys.stdout = old
    return buf.getvalue()


def mnemonics(co, **kw):
    return [i.opname for i in dis.get_instructions(co, **kw)]


# ----------------------------------------------------------------------------
# G1  编译流水线：源码 -> AST -> 符号表 -> code object -> 字节码
# ----------------------------------------------------------------------------
def g1_pipeline():
    section("G1  编译流水线：源码 → AST → 符号表 → Code Object → 字节码")

    src = textwrap.dedent("""
        def add(a, b=1):
            c = a + b
            return c
    """)

    # 四个阶段的产物各自是什么类型
    tree = ast.parse(src)
    st = symtable.symtable(src, "<lab>", "exec")
    co = compile(src, "<lab>", "exec")

    check(isinstance(tree, ast.Module), "G1: ast.parse 产出 ast.Module")
    check(isinstance(st, symtable.SymbolTable), "G1: symtable 产出 SymbolTable")
    check(isinstance(co, types.CodeType), "G1: compile 产出 code object")

    # 顶层 code object 里嵌着 add 的 code object
    sub = [c for c in co.co_consts if isinstance(c, types.CodeType)]
    check(len(sub) == 1 and sub[0].co_name == "add",
          "G1: 函数体是顶层 code object 的 const 之一（%s）" % sub[0].co_name)

    fn = sub[0]
    check(fn.co_varnames == ("a", "b", "c"), "G1: co_varnames == ('a','b','c')")
    check(fn.co_names == (), "G1: 这段没有全局名，co_names 为空")

    # 默认参数不在函数自己的 co_consts 里，而在外层 code object 里存成一个元组
    check(fn.co_consts == (None,), "G1: 函数体自己的 co_consts 只有 None")
    check((1,) in co.co_consts,
          "G1: 默认值 b=1 存在外层 code object 的 consts 里，形态是元组 (1,)"
          " —— MAKE_FUNCTION 时才绑定到函数对象")

    if VERBOSE:
        print("顶层 code object consts :", co.co_consts)
        print("内嵌 code object name   :", fn.co_name)
        print("co_varnames             :", fn.co_varnames)
        print("co_consts               :", fn.co_consts)
        print("co_names                :", fn.co_names)
        print("\n--- add 的字节码 ---")
        print(dump_codeobj(fn))


# ----------------------------------------------------------------------------
# G2  编译期的作用域分类决定运行时的字节码（核心因果链）
# ----------------------------------------------------------------------------
def g2_names():
    section("G2  编译期分类 → 运行时字节码")

    src = textwrap.dedent("""
        x = 10
        def outer(a):
            b = 20
            def inner(c):
                return a + b + c + x
            return inner
    """)
    st = symtable.symtable(src, "<lab>", "exec")
    outer_tb = [c for c in st.get_children() if c.get_name() == "outer"][0]
    inner_tb = [c for c in outer_tb.get_children() if c.get_name() == "inner"][0]

    def kind(name):
        s = inner_tb.lookup(name)
        return ("local" if s.is_local() else
                "free" if s.is_free() else
                "global" if s.is_global() else "?")

    kinds = {n: kind(n) for n in ("a", "b", "c", "x")}
    check(kinds == {"a": "free", "b": "free", "c": "local", "x": "global"},
          "G2: 编译期分类 a/b=free, c=local, x=global（%s）" % kinds)

    ns = {}
    exec(src, ns)
    inner = ns["outer"](0).__code__
    ops = mnemonics(inner)

    # 分类 -> 字节码的对应
    expect = {"a": "LOAD_DEREF", "b": "LOAD_DEREF",
              "c": "LOAD_FAST", "x": "LOAD_GLOBAL"}
    for n, op in expect.items():
        check(op in ops, "G2: %s（%s）编译成 %s" % (n, kinds[n], op))

    check("LOAD_NAME" not in ops, "G2: 函数体内不会出现 LOAD_NAME")

    # LOAD_NAME 只出现在模块体 / 类体这类非优化作用域
    mod_ops = mnemonics(compile("y = z + 1", "<lab>", "exec"))
    check("LOAD_NAME" in mod_ops and "STORE_NAME" in mod_ops,
          "G2: 模块体用的是 LOAD_NAME / STORE_NAME")

    if VERBOSE:
        print("%-6s %-8s %s" % ("名字", "编译期分类", "运行时字节码"))
        print("-" * 40)
        for n in ("a", "b", "c", "x"):
            print("%-6s %-8s %s" % (n, kinds[n], expect[n]))
        print("\n实际反汇编 inner：")
        print(dump_codeobj(inner))
        print("模块体 `y = z + 1`：", mod_ops)


# ----------------------------------------------------------------------------
# G3  编译期优化 vs 运行时专门化（PEP 659）
# ----------------------------------------------------------------------------
def g3_optimize():
    section("G3  编译期优化（常量折叠）vs 运行时专门化（PEP 659）")

    # --- 编译期：常量折叠 ---
    def k():
        return 2 * 3
    check(6 in k.__code__.co_consts, "G3: `2 * 3` 在编译期被折叠成常量 6")

    def s():
        return "ab" * 3
    check("ababab" in s.__code__.co_consts, "G3: `'ab' * 3` 被折叠成 'ababab'")

    # 折叠不了的：涉及运行时名字
    def dyn(a):
        return a * 3
    check(dyn.__code__.co_consts == (None, 3) and "BINARY_OP" in mnemonics(dyn.__code__),
          "G3: 涉及局部名 a 的运算不会被折叠，仍留 BINARY_OP")

    # --- 运行时：自适应专门化 ---
    # 关键：三份**源码完全相同**的 code object，专门化结果由跑过的数据决定
    def compile_add():
        ns = {}
        exec("def add(a, b): return a + b", ns)
        return ns["add"]

    add_int = compile_add()
    add_str = compile_add()
    add_mixed = compile_add()
    check(len({id(add_int.__code__), id(add_str.__code__),
               id(add_mixed.__code__)}) == 3,
          "G3: 三次 exec 得到三个彼此独立的 code object")

    for _ in range(40):
        add_int(1, 2)
    for _ in range(40):
        add_str("a", "b")
    for i in range(3000):
        add_mixed(1, 1.5 if i % 2 else 2)

    specialized = mnemonics(add_int.__code__, adaptive=True)
    str_ops = mnemonics(add_str.__code__, adaptive=True)
    mixed = mnemonics(add_mixed.__code__, adaptive=True)

    check("BINARY_OP_ADD_INT" in specialized,
          "G3: 只喂 int  ->  BINARY_OP_ADD_INT")
    check("BINARY_OP_ADD_UNICODE" in str_ops,
          "G3: 只喂 str  ->  BINARY_OP_ADD_UNICODE")
    check(specialized != str_ops,
          "G3: 同一份源码、不同的运行数据 -> 不同的运行时指令状态")

    # 关键对照：尚未执行的同种 code object 仍是通用形式
    cold = compile("def add(a, b): return a + b", "<lab>", "exec")
    cold_add = [c for c in cold.co_consts if isinstance(c, types.CodeType)][0]
    cold_ops = mnemonics(cold_add, adaptive=True)
    check("BINARY_OP" in cold_ops and "BINARY_OP_ADD_INT" not in cold_ops,
          "G3: 未执行的同种 code object 仍是通用 BINARY_OP"
          " —— 专门化是运行时状态，不写在源码里")

    if VERBOSE:
        print("编译期折叠：co_consts of `return 2*3`      ->", k.__code__.co_consts)
        print("编译期折叠：co_consts of `return 'ab'*3`   ->", s.__code__.co_consts)
        print("\n运行时专门化（三份**源码相同**的 code object）：")
        print("   只喂 int     ->", specialized)
        print("   只喂 str     ->", str_ops)
        print("   int/float 混 ->", mixed)
        print("   冷对照       ->", cold_ops)
        print("\n注：混跑那一格是否退回 BINARY_OP_ADAPTIVE 与解释器版本有关")
        print("    （3.11 实测退回 ADD_ADAPTIVE，3.13 仍保留 ADD_INT），")
        print("     版本间稳定的结论是前两格：专门化状态由运行时数据决定。")


# ----------------------------------------------------------------------------
# G4  Code Object 静态复用 vs Frame 每次独立
# ----------------------------------------------------------------------------
def g4_code_vs_frame():
    section("G4  Code Object 静态复用 / Frame 每次调用独立")

    def make_adder(x):
        def add(y):
            return x + y
        return add

    f1 = make_adder(10)
    f2 = make_adder(100)

    check(f1.__code__ is f2.__code__, "G4: 两个闭包共享同一份 code object")
    check(f1(1) == 11 and f2(1) == 101, "G4: 同样的代码给出不同结果 —— 差别在环境不在代码")
    check(f1.__closure__[0] is not f2.__closure__[0], "G4: 各自的 cell 不同")

    # 同一份 code object，同时存活的多个递归调用各有各的 frame
    def depth(n, acc):
        if n == 0:
            return acc
        return depth(n - 1, acc + [sys._getframe()])

    frames = depth(4, [])
    check(len({id(f) for f in frames}) == 4,
          "G4: 递归 4 层同时存活 4 个不同的 frame 对象")
    check(all(f.f_code is depth.__code__ for f in frames),
          "G4: 这 4 个 frame 都属于同一份 code object")
    check([f.f_locals["n"] for f in frames] == [4, 3, 2, 1],
          "G4: 各 frame 各自的局部变量不同（n = %s）—— "
          "同一份代码、四份状态" % [f.f_locals["n"] for f in frames])
    check(frames[1].f_back is frames[0] and frames[2].f_back is frames[1],
          "G4: frame 用 f_back 串成调用链（内层指向外层）")

    # 两个同源生成器：frame 各自独立，局部变量互不可见
    def counter():
        n = 0
        while True:
            n += 1
            yield n

    g1, g2 = counter(), counter()
    check(g1.gi_frame is not g2.gi_frame, "G4: 两个生成器各有独立的 gi_frame")
    check(g1.gi_frame.f_code is counter.__code__, "G4: 但共享同一份 code object")
    next(g1); next(g1); next(g2)
    check(g1.gi_frame.f_locals["n"] == 2 and g2.gi_frame.f_locals["n"] == 1,
          "G4: 同源生成器的局部变量互不干扰（g1.n=%d, g2.n=%d）"
          % (g1.gi_frame.f_locals["n"], g2.gi_frame.f_locals["n"]))

    # 闭包捕获的是变量（cell）而不是值
    def pair():
        v = 1

        def get():
            return v

        def put(nv):
            nonlocal v
            v = nv
        return get, put

    get, put = pair()
    before = get()
    put(42)
    after = get()
    check(before == 1 and after == 42,
          "G4: get/put 共享同一个 cell —— 闭包捕获变量不是值（%d → %d）" % (before, after))

    # 工厂函数每次返回的函数对象共享同一份 code object（专门化状态因此也共享）
    check(f1.__code__ is make_adder(7).__code__,
          "G4: 工厂函数每次调用返回的都是同一份 code object —— 代码只有一份，环境才有多份")

    if VERBOSE:
        print("f1.__code__ is f2.__code__  :", f1.__code__ is f2.__code__)
        print("f1(1), f2(1)                :", f1(1), f2(1))
        print("递归 4 层同时存活的 frame 数:", len({id(f) for f in frames}))
        print("  各自的局部变量 n          :", [f.f_locals["n"] for f in frames])
        print("  f_back 串成链             :",
              frames[1].f_back is frames[0] and frames[2].f_back is frames[1])
        print("两生成器 gi_frame 是否同一  :", g1.gi_frame is g2.gi_frame)
        print("  各自的 n                  :", g1.gi_frame.f_locals["n"],
              g2.gi_frame.f_locals["n"])
        print("共享 cell：get() 前后        :", before, "→", after)


# ----------------------------------------------------------------------------
# G5  Frame 的字段、求值栈深度与生成器 frame 的保留
# ----------------------------------------------------------------------------
def g5_frame():
    section("G5  求值栈深度可反算 / 生成器 frame 在 yield 处保留")

    # --- 求值栈：stack_effect 累加 == co_stacksize ---
    def calc(a, b):
        c = a + b
        return c * 2

    depth = mx = 0
    trace = []
    for ins in dis.get_instructions(calc.__code__):
        e = dis.stack_effect(ins.opcode, ins.arg, jump=False)
        depth += e
        mx = max(mx, depth)
        trace.append((ins.opname, e, depth))

    check(mx == calc.__code__.co_stacksize,
          "G5: 直线代码 stack_effect 累加峰值 %d == co_stacksize %d" % (
              mx, calc.__code__.co_stacksize))

    if VERBOSE:
        print("%-22s %6s %8s" % ("指令", "栈效应", "栈深"))
        print("-" * 40)
        for op, e, d in trace:
            print("%-22s %+6d %8d" % (op, e, d))
        print("峰值 %d，co_stacksize %d（恰好相等：编译器算出的是最坏情况的精确值）"
              % (mx, calc.__code__.co_stacksize))

    # --- 生成器：yield 处 frame 保留，f_lasti 随 next 推进 ---
    def gen():
        x = 10
        yield x
        x += 1
        yield x

    g = gen()
    fr = g.gi_frame
    check(fr is not None and fr.f_lasti >= 0, "G5: 生成器有独立的 gi_frame")
    snaps = [(fr.f_lasti, dict(fr.f_locals))]
    for _ in range(2):
        next(g)
        snaps.append((fr.f_lasti, dict(fr.f_locals)))
    try:
        next(g)
    except StopIteration:
        pass
    snaps.append((fr.f_lasti, dict(fr.f_locals)))

    check(snaps[1][1].get("x") == 10 and snaps[2][1].get("x") == 11,
          "G5: yield 之后 frame 的局部变量仍在（%s）" % [s[1].get("x") for s in snaps[1:3]])
    check(snaps[0][0] < snaps[1][0] < snaps[2][0],
          "G5: f_lasti 单调推进 %s" % [s[0] for s in snaps[:3]])

    # 普通函数返回后 frame 就没了
    def plain():
        return sys._getframe()

    pf = plain()
    check(pf.f_locals is not None, "G5: 普通函数也能拿到 frame 对象")

    if VERBOSE:
        print("\n生成器 gi_frame 快照（f_lasti, f_locals）：")
        for s in snaps:
            print("   ", s)


# ----------------------------------------------------------------------------
# G6  观察 vs 接管：settrace / monitoring 做不到什么
# ----------------------------------------------------------------------------
def g6_observation():
    section("G6  观察钩子（settrace / monitoring）的能力边界")

    def work(n):
        t = 0
        for i in range(n):
            t += i * i
        return t

    # --- 基线 ---
    N = 20
    R = 2000
    t0 = time.perf_counter()
    for _ in range(R):
        work(N)
    base = time.perf_counter() - t0

    # --- 行级 tracing ---
    line_hits = [0]

    def tr_line(frame, event, arg):
        if event == "call":
            line_hits[0] += 1
            return tr_line
        return None

    sys.settrace(tr_line)
    t0 = time.perf_counter()
    for _ in range(R):
        work(N)
    traced = time.perf_counter() - t0
    sys.settrace(None)
    n_calls = line_hits[0]

    # --- 指令级 tracing（f_trace_opcodes） ---
    op_hits = [0]

    def tr_op(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
            return tr_op
        if event == "opcode":
            op_hits[0] += 1
        return tr_op

    sys.settrace(tr_op)
    t0 = time.perf_counter()
    for _ in range(R):
        work(N)
    traced_op = time.perf_counter() - t0
    sys.settrace(None)
    n_ops = op_hits[0]

    check(n_calls == R, "G6: 行级 trace 命中 %d 次调用（应为 %d）" % (n_calls, R))
    check(n_ops > n_calls, "G6: 指令级 trace 统计到 %d 条指令 > %d 次调用" % (n_ops, n_calls))
    check(traced > base, "G6: 行级 trace 比基线慢（%.2fx）" % (traced / base))
    check(traced_op > traced, "G6: 指令级比行级更慢（%.2fx vs %.2fx）" % (
        traced_op / base, traced / base))

    # --- settrace 的返回值只能决定抛异常，不能替换求值结果 ---
    def tr_noop(frame, event, arg):
        return None
    sys.settrace(tr_noop)
    check(work(5) == 30, "G6: tracing 不改变求值结果（work(5) == 30）")
    sys.settrace(None)

    # --- sys.monitoring（3.12+）：同样是观察，不是接管 ---
    has_mon = hasattr(sys, "monitoring")
    if has_mon:
        mon_hits = [0]
        mon = sys.monitoring
        # 0=debugger / 1=coverage / 2=profiler / 5=optimizer 是保留 id，取 3 号
        tool = getattr(mon, "DEBUG_TOOL_ID", 3)

        def on_instr(code, offset):
            mon_hits[0] += 1
            return mon.DISABLE

        mon.use_tool_id(tool, "runtime_lab")
        mon.register_callback(tool, mon.events.INSTRUCTION, on_instr)
        mon.set_local_events(tool, work.__code__, mon.events.INSTRUCTION)
        work(N)
        mon.set_local_events(tool, work.__code__, 0)
        mon.register_callback(tool, mon.events.INSTRUCTION, None)
        mon.free_tool_id(tool)
        check(mon_hits[0] > 0, "G6: sys.monitoring 的 INSTRUCTION 事件观察到 %d 条指令"
              % mon_hits[0])
    else:
        if VERBOSE:
            print("（本解释器无 sys.monitoring，需 3.12+）")

    if VERBOSE:
        print("\n%d 次 work(%d) 的耗时：" % (R, N))
        print("   基线            %.4f s   (1.00x)" % base)
        print("   行级 settrace   %.4f s   (%.2fx)" % (traced, traced / base))
        print("   指令级 settrace %.4f s   (%.2fx)" % (traced_op, traced_op / base))
        print("   tracing 下结果不变：work(5) =", work(5))
        print("\n结论：settrace / monitoring 的回调返回 DISABLE 或 None，")
        print("      只能「看」，不能改写这一步该算出什么 —— 这正是 PEP 523 的差别。")


GROUPS = [
    (1, "编译流水线：源码 → AST → 符号表 → Code Object → 字节码", g1_pipeline),
    (2, "编译期作用域分类 → 运行时字节码", g2_names),
    (3, "编译期常量折叠 vs 运行时专门化（PEP 659）", g3_optimize),
    (4, "Code Object 静态复用 / Frame 每次调用独立", g4_code_vs_frame),
    (5, "求值栈深度反算 / 生成器 frame 保留", g5_frame),
    (6, "观察钩子（settrace / monitoring）的能力边界", g6_observation),
]


def main(argv):
    global VERBOSE
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}

    quiet = "--selftest" in flags
    VERBOSE = not quiet

    if args:
        want = {int(a) for a in args}
        groups = [g for g in GROUPS if g[0] in want]
    else:
        groups = GROUPS

    if VERBOSE:
        print("runtime_lab — CPython %s" % sys.version.split()[0])
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
