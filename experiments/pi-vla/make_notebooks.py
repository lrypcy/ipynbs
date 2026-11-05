# -*- coding: utf-8 -*-
"""为 pi-06..pi-10 生成「代码 + 已固化输出」的 notebook 版。

与 05 篇的 `/tmp/mk_pi05_nb.py` 思路一致但通用化：按 lab.py 里的
`# ===================================================================== 组`
分节切成 code cells，在同一 globals 里逐段 exec，把每段 stdout 固化进 cell。
这样读者点开 GitHub 就能看到结果，不必自己跑脚本。

运行：~/Software/miniconda3/bin/python make_notebooks.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    ("pi-06", "rtc_lab.py", "rtc_lab.ipynb",
     "RTC：延迟下动作块怎么接",
     "把 ΠGDM guidance / Eq.(5) 软掩码放进线性高斯世界，比较 naive async、hard masking、"
     "soft masking、temporal ensembling 与 synchronous 在切换点的加速度。"),
    ("pi-07", "recap_lab.py", "recap_lab.ipynb",
     "RECAP / π*0.6：从经验与纠正里学",
     "advantage 二值化与条件化、on-policy 纠正、与 best-of-N 的等价性边界。"),
    ("pi-08", "rlt_lab.py", "rlt_lab.ipynb",
     "RL Token：冻结 VLA 上的在线 RL",
     "bottleneck 维度、双 Q 离策略 critic、chunk-level 信用分配、参考动作 dropout 与 BC anchor。"),
    ("pi-09", "mem_lab.py", "mem_lab.ipynb",
     "MEM：短时视频记忆与长时文本记忆",
     "固定 token 预算下的时间跨度、遮挡补偿、阶段判别与两类记忆的分工边界。"),
    ("pi-10", "pi07_lab.py", "pi07_lab.ipynb",
     "π0.7：diverse context conditioning",
     "多条件分量与 dropout、CFG 引导强度、元数据 steering、延迟下的条件化。"),
]

SECTION = re.compile(r"^# ={10,}")


def split_sections(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    marks = [i for i, ln in enumerate(lines) if SECTION.match(ln)]
    if len(marks) < 3:
        return [src]
    out = ["".join(lines[: marks[0]])]
    for a, b in zip(marks, marks[1:] + [len(lines)]):
        out.append("".join(lines[a:b]))
    return [s for s in out if s.strip()]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(src: str, out: str, n: int) -> dict:
    outputs = []
    if out:
        outputs.append({"output_type": "stream", "name": "stdout",
                        "text": out.splitlines(keepends=True)})
    return {"cell_type": "code", "execution_count": n, "metadata": {},
            "outputs": outputs, "source": src.splitlines(keepends=True)}


def build(folder: str, lab: str, nb: str, title: str, desc: str) -> tuple[int, int, int]:
    d = os.path.join(ROOT, folder)
    lab_path = os.path.join(d, lab)
    nb_path = os.path.join(d, nb)
    with open(lab_path, encoding="utf-8") as f:
        src = f.read()

    sections = split_sections(src)
    # 让每篇 lab 都能 import 上一级目录的 series_common.py
    sys.path.insert(0, ROOT)
    g = {"__name__": "__main__", "__file__": lab_path,
         "__builtins__": __builtins__, "__package__": None}

    cells = [md_cell(
        f"# {title}\n\n"
        f"{desc}\n\n"
        f"对应 `experiments/pi-vla/{folder}/`，与 `{lab}` 完全同源：本 notebook 按"
        f"分节切分同一份源码，逐段执行后把输出固化在这里。\n\n"
        f"复现：\n\n```bash\ncd ~/Desktop/Projects/sandbox/ipynbs/experiments/pi-vla/{folder}\n"
        f"~/Software/miniconda3/bin/python {lab} > results/stdout.txt\n```\n\n"
        f"纯 numpy / CPU，固定 seed。最后一个 cell 打出断言总数，好和 README / 正文对账。\n"
    )]

    for i, sec in enumerate(sections, start=1):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(sec, lab_path, "exec"), g)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{folder} 第 {i} 段执行失败：{type(exc).__name__}: {exc}") from exc
        cells.append(code_cell(sec, buf.getvalue(), i))

    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=1)

    n_code = len(cells) - 1
    n_out = sum(1 for c in cells[1:] if c["outputs"])
    n_err = sum(1 for c in cells[1:] for o in c["outputs"]
                if o.get("output_type") == "error")
    return n_code, n_out, n_err


if __name__ == "__main__":
    for job in JOBS:
        n_code, n_out, n_err = build(*job)
        print(f"  {job[0]}/{job[2]}: {n_code} code cells, {n_out} with output, "
              f"{n_err} errors")
        if n_err:
            raise SystemExit(f"{job[0]} has error outputs")
    print("all notebooks written")
