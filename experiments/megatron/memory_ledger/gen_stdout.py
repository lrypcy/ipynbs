#!/usr/bin/env python3
"""依次运行本目录三个实验，把输出固化到 results/stdout.txt。

用法：
    ~/Software/miniconda3/envs/torch/bin/python gen_stdout.py            # 写入 results/stdout.txt
    ~/Software/miniconda3/envs/torch/bin/python gen_stdout.py --stdout   # 同时打印到终端

解释器按脚本分开选：`selective_equivalence.py` 需要 torch（base 环境没有），
另两个只需要 numpy。用哪个解释器跑本脚本都无所谓 —— 每个子进程都显式指定。

每个实验都以 exit code 0 表示全部断言通过。任一失败则整体返回非 0，
不覆盖 results/stdout.txt —— 避免把失败的输出固化成「基线」。
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

NUMPY_PY = str(Path("~/Software/miniconda3/bin/python3").expanduser())
TORCH_PY = str(Path("~/Software/miniconda3/envs/torch/bin/python").expanduser())

EXPERIMENTS = [
    (NUMPY_PY, "pareto.py",
     "显存账本激活侧帕累托前沿（重计算 / 分布保存 / 卸载三档）"),
    (TORCH_PY, "selective_equivalence.py",
     "normal 与 output-discarding 两种 checkpointing 的数值等价性"),
    (NUMPY_PY, "dsa_shapes.py",
     "distribute_saved_activations 的逐 rank 形状推导"),
]

BASELINE = """\
========================================================================================
本文件由 gen_stdout.py 自动生成，请勿手工编辑。
复现：~/Software/miniconda3/envs/torch/bin/python gen_stdout.py
源码基准：megatron-core 0.20.0 / commit 60e039626
========================================================================================
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    chunks: list[str] = [
        BASELINE,
        f"生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}\n",
    ]

    failed: list[str] = []
    for interp, script, desc in EXPERIMENTS:
        path = HERE / script
        if not path.exists():
            failed.append(f"{script} 不存在")
            continue
        proc = subprocess.run(
            [interp, str(path)],
            capture_output=True, text=True, cwd=HERE,
        )
        chunks.append("\n" + "=" * 88 + "\n")
        chunks.append(f"【{script}】{desc}\n")
        chunks.append(f"解释器：{interp}\n")
        chunks.append("=" * 88 + "\n")
        chunks.append(proc.stdout)
        if proc.stderr:
            chunks.append("--- stderr ---\n" + proc.stderr + "\n")
        chunks.append(f"--- exit code: {proc.returncode} ---\n")
        if proc.returncode != 0:
            failed.append(f"{script} exit={proc.returncode}")

    body = "".join(chunks)
    if failed:
        sys.stderr.write("以下实验未通过，未写入 results/stdout.txt：\n")
        for f in failed:
            sys.stderr.write(f"  - {f}\n")
        return 1

    (RESULTS / "stdout.txt").write_text(body, encoding="utf-8")
    n_lines = body.count("\n")
    sys.stderr.write(f"[ok] 3 个实验全部通过，results/stdout.txt 共 {n_lines} 行\n")
    if args.stdout:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())