#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图 4：正交 key 与相似 key 的读回串扰（结构示意，非论文原图）。

对应文章 §4。图中只有两个来自 results/stdout.txt 的数字：正交 key 的
读回最大误差 0.000e+00，随机单位 key 在 N=128 时的读回相对误差 3.1916；
Gram 矩阵的格子是结构示意，不标实验数值。
"""
from pathlib import Path

W, H = 1000, 430
FONT = "PingFang SC, Helvetica, Arial, sans-serif"
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    '<text x="28" y="34" font-size="20" font-weight="600" fill="#1f2937">'
    '同一份状态，读回差别来自 key 的 Gram 矩阵</text>',
    '<text x="28" y="56" font-size="12.5" fill="#6b7280">'
    '左：正交 key，G = I，读回无串扰　｜　右：相似 key，非对角元把其他 value 混进读回结果</text>',
]

# 面板
for x, title, color in (
    (24, '正交 key：G = I', '#1d4ed8'),
    (506, '相似 / 随机 key：G ≠ I', '#b45309'),
):
    parts.append(
        f'<rect x="{x}" y="76" width="470" height="290" rx="10" '
        'fill="#f8fafc" stroke="#e2e8f0"/>'
    )
    parts.append(
        f'<text x="{x + 20}" y="105" font-size="15.5" font-weight="600" '
        f'fill="{color}">{title}</text>'
    )


def cell(x, y, fill, text, fg, size=12):
    return (
        f'<rect x="{x}" y="{y}" width="40" height="40" rx="4" fill="{fill}"/>'
        f'<text x="{x + 20}" y="{y + 25}" font-size="{size}" text-anchor="middle" '
        f'fill="{fg}">{text}</text>'
    )


# 左：单位 Gram 矩阵
for r in range(4):
    for c in range(4):
        diag = r == c
        parts.append(cell(60 + c * 46, 130 + r * 46,
                          '#1d4ed8' if diag else '#eef2f7',
                          '1' if diag else '0',
                          '#ffffff' if diag else '#94a3b8'))
# 右：非对角非零的 Gram 矩阵（深浅只表示「非零」，不是数值）
shades = ('#dbeafe', '#bfdbfe', '#c7d2fe', '#e0e7ff')
for r in range(4):
    for c in range(4):
        diag = r == c
        parts.append(cell(542 + c * 46, 130 + r * 46,
                          '#1d4ed8' if diag else shades[(r + c) % 4],
                          '1' if diag else '≠0',
                          '#ffffff' if diag else '#1e3a8a', 11 if diag else 12))

parts.append('<text x="60" y="330" font-size="12" fill="#64748b">Gram 矩阵 G（结构示意）</text>')
parts.append('<text x="542" y="330" font-size="12" fill="#64748b">Gram 矩阵 G（结构示意）</text>')

# 读回框
for x in (268, 750):
    parts.append(
        f'<rect x="{x}" y="140" width="206" height="150" rx="8" '
        'fill="#ffffff" stroke="#cbd5e1"/>'
    )
    parts.append(
        f'<path d="M {x - 22} 210 L {x - 6} 210" stroke="#94a3b8" stroke-width="2"/>'
        f'<path d="M {x - 6} 210 l -7 -4 l 0 8 z" fill="#94a3b8"/>'
    )

left_lines = [
    (166, '正交 ⇒ 无串扰', 14.5, '#0f172a', '600'),
    (194, '读回值 = 原值', 14, '#1d4ed8', '400'),
    (228, '最大误差 0.000e+00', 12.5, '#475569', '400'),
    (252, '实验组 3：正交 key', 11.5, '#94a3b8', '400'),
    (278, 'G = I ⇒ 读回表 = V', 12, '#64748b', '400'),
]
right_lines = [
    (166, '非对角元 ⇒ 串扰', 14.5, '#0f172a', '600'),
    (194, '读回值混入其他 value', 13, '#b45309', '400'),
    (228, 'N=128 相对误差 3.1916', 12.5, '#475569', '400'),
    (252, '实验组 3：随机单位 key', 11.5, '#94a3b8', '400'),
    (278, 'G ≠ I ⇒ 读回表 = G V', 12, '#64748b', '400'),
]
for x, lines in ((268, left_lines), (750, right_lines)):
    for y, text, size, fill, weight in lines:
        parts.append(
            f'<text x="{x + 16}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{text}</text>'
        )

parts.append(
    '<rect x="24" y="382" width="952" height="34" rx="6" '
    'fill="#fff7ed" stroke="#fed7aa"/>'
    '<text x="40" y="404" font-size="13" fill="#9a3412">'
    '同一份 S，差别只在 key 之间的内积；delta rule 改的是写入端，让重复 key 不再累加旧值</text>'
)
parts.append('</svg>')

out = Path(__file__).resolve().parent / 'figures' / 'fig4_gram_interference.svg'
out.write_text('\n'.join(parts) + '\n', encoding='utf-8')
print(f'wrote {out} ({out.stat().st_size} bytes)')
