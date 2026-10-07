#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 results/stdout.txt：逐组跑两个脚本，按逻辑顺序拼接。

为什么要分 14 次跑：沙箱对长脚本有 CPU 时间上限，会 SIGTERM（exit 137）
掐断且输出截断。所以每组单独跑、单独校验（必须出现「总计」才算跑完），
最后拼成一份完整输出。**不要改成一次性跑完。**

用法（从本目录）：
    python3 gen_stdout.py              # 全部 14 组（串行约 3–5 分钟）
    python3 gen_stdout.py 1 2 3        # 只跑连续部分的第 1/2/3 组，追加到文件
    python3 gen_stdout.py --append     # 追加模式（不清空已有内容）

沙箱对单条命令有 CPU 上限，14 组串行会被 SIGTERM；
所以分组追加是正常用法，不是变通。
"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 解释器路径不写死：用当前这个 python 自己（纯 numpy 脚本，换环境也能跑）
PY = sys.executable

HEADER = [
    "# Flow Matching / Diffusion 建模核验 —— 全部 14 组输出",
    "#",
    "# 生成方式：python3 gen_stdout.py（每组单独跑，规避沙箱 SIGTERM）",
    "# 对应博客：",
    "#   https://lrypcy.github.io/2026/08/22/flow-matching-01-algorithm-evolution/",
    "#   https://lrypcy.github.io/2026/08/22/flow-matching-02-score-diffusion-sde/",
    "#   https://lrypcy.github.io/2026/08/22/flow-matching-03-guidance-discrete-latent/",
    "",
]


def run(script: str, n: int) -> str:
    r = subprocess.run(
        [PY, os.path.join(HERE, script), str(n)],
        capture_output=True, text=True,
    )
    if "总计" not in r.stdout:
        raise SystemExit(f"{script} {n} 未跑完（可能被沙箱掐断），不写入以免留下截断输出")
    return r.stdout.rstrip()


def main() -> None:
    args = sys.argv[1:]
    append = "--append" in args
    which = [a for a in args if not a.startswith("-")]
    path = os.path.join(HERE, "results", "stdout.txt")

    if append:
        out = io.open(path, encoding="utf-8").read().rstrip().split("\n")
    else:
        out = list(HEADER)

    if which:
        # 显式指定：连续组 1–8 用 a，离散组用 d 前缀（如 d3）
        for w in which:
            if w.startswith("d"):
                out.append(run("fm_discrete_lab.py", int(w[1:])))
            else:
                out.append(run("fm_modeling_lab.py", int(w)))
            out.append("")
    else:
        for n in range(1, 9):
            out.append(run("fm_modeling_lab.py", n))
            out.append("")
        for n in range(1, 7):
            out.append(run("fm_discrete_lab.py", n))
            out.append("")

    io.open(path, "w", encoding="utf-8").write("\n".join(out).rstrip() + "\n")
    n_blocks = sum(1 for l in out if l.startswith("实验 "))
    print(f"wrote {path}（共 {n_blocks} 组）")


if __name__ == "__main__":
    main()