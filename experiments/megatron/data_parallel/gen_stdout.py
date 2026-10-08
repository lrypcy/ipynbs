#!/usr/bin/env python3
"""依次运行本目录三个实验，把输出固化到 results/stdout.txt。

用法：
    python3 gen_stdout.py            # 写入 results/stdout.txt
    python3 gen_stdout.py --stdout  # 同时打印到终端

每个实验都以 exit code 0 表示全部断言通过。任一失败则整体返回非 0，
不覆盖 results/stdout.txt——避免把失败的输出固化成「基线」。
"""

from __future__ import annotations

import argparse
import datetime
import io
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

EXPERIMENTS = [
    ("pad_tail_waste.py", "bucket padding 的 tail 浪费（复刻 param_layout.py 两套装箱逻辑）"),
    ("rs_shard_equivalence.py", "reduce-scatter + 分片更新 与 非分片 AdamW 的等价性验证"),
    ("mem_ledger.py", "三套 DP 路径的显存账本"),
]

BASELINE = """\
========================================================================================
本文件由 gen_stdout.py 自动生成，请勿手工编辑。
复现：python3 gen_stdout.py
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
    for script, desc in EXPERIMENTS:
        path = HERE / script
        if not path.exists():
            failed.append(f"{script} 不存在")
            continue
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, cwd=HERE,
        )
        chunks.append("\n" + "=" * 88 + "\n")
        chunks.append(f"【{script}】{desc}\n")
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