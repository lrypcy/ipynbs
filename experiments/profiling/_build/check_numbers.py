"""回填一致性校验：README / markdown 里引用的数字必须能在实测输出里找到。

**为什么需要这一步**
博客正文引用了这些脚本跑出来的具体数字。notebook 的 markdown 与 README 又
复述了一遍这些数字。人工誊写必然出错 —— 本文件就是第一次誊写就编了三个数
（把 153.0/295.4/1.93× 写成 494.5/277.9/1.78×）之后写的。

判据：从 markdown / README 里抽出所有「像数字的记号」，逐个回 `results/stdout.txt`
里找。找不到就报错。注意只查**带小数点或百分号的量**（`153.0`、`1.93×`、`40.42%`），
整数（层数 32、GQA 组数 8）到处都有，校验没有意义。
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)

# 带小数点 / 百分号 / ratio 后缀的记号，才值得回查
NUM_RE = re.compile(r"\d+\.\d+")
# 这些量是配置参数或代码常量，输出里本来就不会以小数形式出现
ALLOW = {
    "0.5", "1.0", "2.0", "3.0", "4.0", "6.0", "0.0", "8.0", "12.0",
    "0.25", "0.75", "1.5", "0.9", "0.95", "0.80",
    # 环境与版本号，不是实验结果
    "3.9",            # Python 3.9+
    "0.20",           # Megatron v0.20.0 的版本号
    "3.35",           # H100 HBM 3.35 TB/s（产品页规格，非本 notebook 输出）
    "2.0",            # PyTorch 2.x / 配置默认值
    # arXiv / DOI 编号
    "2204.02311", "2407.21783", "2205.05198", "2311.02382",
}
# 小节编号（「第 3.5 步」），不是实验结论
SECTION_NUMS = {"3.5"}


def stdout_of(slug: str) -> str:
    path = os.path.join(EXPERIMENTS, slug, "results", "stdout.txt")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def texts_of(slug: str) -> List[Tuple[str, str]]:
    """返回 [(来源标签, 文本)]：notebook 的 markdown + README。"""
    out: List[Tuple[str, str]] = []
    rd = os.path.join(EXPERIMENTS, slug, "README.md")
    if os.path.isfile(rd):
        with open(rd, "r", encoding="utf-8") as fh:
            out.append(("README.md", fh.read()))
    import json
    nbp = os.path.join(EXPERIMENTS, slug, slug + ".ipynb")
    if os.path.isfile(nbp):
        with open(nbp, "r", encoding="utf-8") as fh:
            nb = json.load(fh)
        for c in nb.get("cells", []):
            if c.get("cell_type") == "markdown":
                out.append(("notebook markdown",
                            "".join(c.get("source", []))))
    return out


def check_slug(slug: str, sibling_out: str = "") -> List[str]:
    out = stdout_of(slug)
    if not out:
        return ["%s: 没有 results/stdout.txt，先跑 build.py" % slug]
    # 输出里的数字形态：把 153.0 也可能打成 153，统一按「去掉尾部零」比对
    out_nums = set()
    for m in NUM_RE.findall(out):
        out_nums.add(m)
        out_nums.add(m.rstrip("0").rstrip(".") if "." in m else m)
    # 跨 notebook 引用是合法的（trace_agg 的 README 会引 timer_semantics 的数），
    # 所以把兄弟 notebook 的输出也算进可命中集合。
    haystack = out + "\n" + sibling_out
    sibling_nums = set()
    for m in NUM_RE.findall(sibling_out):
        sibling_nums.add(m)
        sibling_nums.add(m.rstrip("0").rstrip(".") if "." in m else m)

    problems: List[str] = []
    for label, text in texts_of(slug):
        for m in NUM_RE.findall(text):
            if m in ALLOW or m in SECTION_NUMS:
                continue
            if m in out_nums:
                continue
            if m.rstrip("0").rstrip(".") in out_nums:
                continue
            if sibling_nums and (m in sibling_nums
                                 or m.rstrip("0").rstrip(".") in sibling_nums):
                continue
            problems.append("%s: %s 里出现 %s，实测输出里找不到" % (slug, label, m))
    return problems


def main() -> int:
    slugs = sorted(d for d in os.listdir(EXPERIMENTS)
                   if os.path.isdir(os.path.join(EXPERIMENTS, d))
                   and not d.startswith("_"))
    only = sys.argv[1:]
    if only:
        slugs = [s for s in slugs if s in only]
    all_problems: List[str] = []
    # 兄弟 notebook 的输出合起来，作为跨引用的可命中集合
    sibling = "\n".join(stdout_of(s) for s in slugs)
    for s in slugs:
        p = check_slug(s, sibling_out=sibling)
        all_problems.extend(p)
        print("  %-34s %s" % (s, "OK" if not p else "%d 处待核" % len(p)))
    if all_problems:
        print("\n以下数字在实测输出里找不到，请核对：")
        for p in all_problems:
            print("  " + p)
        return 1
    print("\n所有 markdown / README 里的数字都能在实测输出里找到。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
