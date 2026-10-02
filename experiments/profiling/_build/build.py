"""把博客仓库的 profiling 脚本编译成自包含的 .ipynb（带真实运行输出）。

用法（在 ipynbs 仓库根目录）：

    python3 experiments/profiling/_build/build.py

需要一个带 nbformat / nbclient / ipykernel 的解释器（miniconda base 即可）。
源码从 BLOG_REPO 读，默认 ~/Desktop/Projects/github/lrypcy.github.io。

流程：
    1. 读 spec（每个 notebook 一份 markdown 描述 + 要抽的符号列表）
    2. 从博客仓库 tools/ 下按顶层符号抽源码，拼成 cell
    3. nbclient 逐 cell 执行，输出固化进 notebook
    4. 写 .ipynb + README.md

为什么要有这一步：博客正文引用了这些脚本输出的具体数字。notebook 里的代码
必须与仓库里的 .py 逐字一致，否则两边数字对不上，正文就变成编的。抽取而非
手抄就是这个保证。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbextract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)

KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.11.4",
}


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip("\n"))


def build_notebook(cells, title: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": dict(KERNELSPEC),
        "language_info": dict(LANGUAGE_INFO),
        "title": title,
    }
    return nb


def execute(nb: nbformat.NotebookNode, workdir: str, timeout: int = 1800) -> None:
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": workdir}},
    )
    client.execute()


def write_notebook(nb: nbformat.NotebookNode, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    nbformat.validate(nb)


def banner(spec_name: str, nb: nbformat.NotebookNode) -> None:
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_md = len(nb.cells) - n_code
    print("  %-34s %2d markdown + %2d code" % (spec_name, n_md, n_code))


def run_spec(spec: dict, blog: str, outdir: str, execute_it: bool) -> str:
    cells = []
    for c in spec["cells"]:
        kind = c[0]
        if kind == "md":
            cells.append(md(c[1]))
        elif kind == "code":
            cells.append(code(c[1]))
        elif kind == "raw_py":
            # 从博客仓库抽顶层符号
            _, rel, names, tail = c
            path = os.path.join(blog, rel)
            index = nbextract.build_index(path)
            src = nbextract.slice_names(index, names, rel)
            if tail:
                src = src + "\n\n" + tail.strip("\n") + "\n"
            cells.append(code(src))
        elif kind == "imports_py":
            # 只抽目标模块的顶层 import（跳过相对导入与兄弟 bench 包）
            # 注意别把外层的 spec 覆盖掉 —— 它后面还要用来取 title / readme
            imp = c[1]
            if isinstance(imp, tuple):
                rels, skip = imp
            else:
                rels, skip = imp, ()
            out = []
            for rel in rels:
                path = os.path.join(blog, rel)
                for line in nbextract.module_imports(path, skip_modules=skip):
                    out.append(line)
            cells.append(code("\n".join(out)))
        else:
            raise ValueError("未知 cell 类型: %s" % kind)

    nb = build_notebook(cells, spec["title"])

    slug = spec["slug"]
    dest_dir = os.path.join(outdir, slug)
    os.makedirs(dest_dir, exist_ok=True)
    nb_path = os.path.join(dest_dir, slug + ".ipynb")

    if execute_it:
        t0 = time.time()
        execute(nb, dest_dir, timeout=spec.get("timeout", 1800))
        banner(slug, nb)
        print("      执行 %.1f s" % (time.time() - t0))
        write_notebook(nb, nb_path)
    else:
        banner(slug, nb)
        print("      (--no-exec，跳过执行)")
        write_notebook(nb, nb_path)

    if "readme" in spec:
        with open(os.path.join(dest_dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(spec["readme"].strip("\n") + "\n")

    # 把 notebook 的 stream 输出落一份纯文本，方便 grep 某个数字
    results_dir = os.path.join(dest_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "stdout.txt"), "w", encoding="utf-8") as fh:
        for c in nb.cells:
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    fh.write(o.get("text", ""))
        fh.write("\n" + "=" * 74 + "\n")
        for c in nb.cells:
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    fh.write("\n[ERROR] %s\n%s\n"
                             % (o.get("ename"), o.get("evalue")))

    return nb_path


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 profiling 系列的 notebook")
    ap.add_argument("--only", action="append", default=[],
                    help="只构建指定 slug（可重复）")
    ap.add_argument("--no-exec", action="store_true", help="只拼 cell 不执行")
    ap.add_argument("--list", action="store_true", help="列出所有 slug")
    args = ap.parse_args()

    blog = nbextract.blog_repo_path()
    outdir = EXPERIMENTS

    spec_files = sorted(f for f in os.listdir(HERE)
                        if f.startswith("spec_") and f.endswith(".py"))
    specs = []
    for f in spec_files:
        modname = f[:-3]
        mod = __import__(modname)
        specs.append(mod.SPEC)

    if args.list:
        for s in specs:
            print(s["slug"])
        return 0

    if not os.path.isdir(os.path.join(blog, "tools", "profiling_bench")):
        print("[!] 博客仓库下找不到 tools/profiling_bench：BLOG_REPO=%s" % blog)
        return 2

    print("博客仓库: %s" % blog)
    print("输出目录: %s" % outdir)
    print("-" * 62)
    for spec in specs:
        if args.only and spec["slug"] not in args.only:
            continue
        run_spec(spec, blog, outdir, not args.no_exec)
    print("-" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
