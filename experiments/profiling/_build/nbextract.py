"""notebook 构建器的公共库：从博客仓库的 .py 脚本里按顶层符号抽取源码。

设计要点
--------
notebook 里的代码必须是**博客仓库脚本的逐字切片**，不能手抄第二份。理由：

1. 博客正文引用了这些脚本跑出来的数字，两份代码一旦分叉，正文就变成编的；
2. 手抄 inevitably 会漏掉 import、边界条件、注释里的判据说明。

所以这里用 `ast` 按顶层符号名（函数 / 类 / 赋值）切出源码段，
构建时把它拼进 cell，执行后把真实输出固化进 .ipynb。

跨脚本依赖（`megatron_bench`）也走同一条路：把被依赖模块里用到的顶层符号
一并抽出来，在 prelude 里拼好，再用 `types.SimpleNamespace` 做一层模块名
垫片 —— 这样原脚本里 `flops.training_flops_per_token(...)` 这种带模块前缀的
写法可以原样保留，不需要改一个字。
"""

from __future__ import annotations

import ast
import os
from typing import Dict, List, Tuple

# 顶层节点 -> 源码段。key 是符号名。
Index = Dict[str, str]


def read_source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _segment(src: str, node) -> str:
    """取一个顶层节点的源码，**从装饰器开始**。

    `ast.get_source_segment` 从 `def` / `class` 那一行起算，会把 `@dataclass`
    这类装饰器甩在门外——抽出来的类不再是 dataclass，构造器直接报
    `takes no arguments`。所以这里把起始行手动前移到第一个装饰器。
    """
    lines = src.splitlines(keepends=True)
    start = node.lineno - 1
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, dec.lineno - 1)
    # 往前吃掉紧贴的装饰器与空行之间的注释（@ 上面那行可能是 # type: ignore）
    return "".join(lines[start:node.end_lineno])


def build_index(path: str) -> Index:
    """建立 `符号名 -> 源码文本` 的索引（只收顶层定义，import 单独处理）。"""
    src = read_source(path)
    tree = ast.parse(src)
    out: Index = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = _segment(src, node)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = _segment(src, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = _segment(src, node)
    return out


def module_imports(path: str, skip_modules: List[str] | None = None) -> List[str]:
    """抽出模块顶层的 import 语句。

    两类必须跳过：
      * **相对导入**（`from .xxx import`）—— 自包含 notebook 里没有包结构，不成立；
      * **兄弟 bench 包的绝对导入**（`from megatron_bench.comm import ...`）——
        那些符号由 prelude 的垫片提供，写成真 import 会直接 ModuleNotFoundError。
    """
    src = read_source(path)
    tree = ast.parse(src)
    lines = src.splitlines()
    skip = tuple(skip_modules or ())
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Try):
            # `try: import numpy as _np / except ImportError: _np = None` 这类
            # 软依赖写法：整块照搬，符号（_np）必须连 fallback 一起带过来，
            # 否则只抽 ImportFrom 会让 _np 未定义。
            block = "\n".join(lines[node.lineno - 1:node.end_lineno])
            for sub in node.body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    mod = ""
                    if isinstance(sub, ast.ImportFrom) and sub.module:
                        mod = sub.module
                    elif isinstance(sub, ast.Import) and sub.names:
                        mod = sub.names[0].name
                    if skip and mod.startswith(skip):
                        continue
                    break
            else:
                continue
            out.append(block)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                continue
            mod = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
            elif isinstance(node, ast.Import) and node.names:
                mod = node.names[0].name
            if skip and mod.startswith(skip):
                continue
            out.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return out


def slice_names(index: Index, names: List[str], origin: str) -> str:
    """按给定顺序取出若干顶层符号，拼成一段可执行源码。

    缺符号直接报错 —— 静默跳过会让人以为 notebook 跑通了就等于覆盖到了。
    """
    chunks: List[str] = []
    missing: List[str] = []
    for n in names:
        if n in index:
            chunks.append(index[n])
        else:
            missing.append(n)
    if missing:
        raise KeyError("%s 里找不到顶层符号: %s" % (origin, ", ".join(missing)))
    header = "# ---- 摘自 %s ----" % origin
    return header + "\n" + "\n\n\n".join(chunks) + "\n"


def check_names(index: Index, names: List[str], origin: str) -> Tuple[bool, List[str]]:
    missing = [n for n in names if n not in index]
    return (not missing), missing


def blog_repo_path() -> str:
    """博客仓库根目录。可用 BLOG_REPO 环境变量覆盖。"""
    env = os.environ.get("BLOG_REPO")
    if env:
        return os.path.abspath(env)
    default = os.path.expanduser("~/Desktop/Projects/github/lrypcy.github.io")
    if os.path.isdir(default):
        return default
    raise RuntimeError(
        "找不到博客仓库，请设置 BLOG_REPO 环境变量指向 lrypcy.github.io 的根目录")
