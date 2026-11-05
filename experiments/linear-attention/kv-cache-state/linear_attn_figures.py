"""三张自画 SVG：缓存账本曲线 / 状态结构对照 / 容量与分块两笔账。

图上的数字全部从 results/stdout.txt 解析（与固化输出逐位一致），
或由 lab 的解析式复算。字体 PingFang SC，SVG 带固有 width/height。
"""

import math
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
STDOUT = (HERE / "results" / "stdout.txt").read_text(encoding="utf-8")

FONT = "PingFang SC, Helvetica, Arial, sans-serif"
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
RED = "#dc2626"
BLUE = "#2563eb"
PURPLE = "#7c3aed"
TEAL = "#0f766e"


def head(w, h, title, sub=""):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
            f'<text x="28" y="34" font-size="20" font-weight="600" fill="{INK}">{title}</text>',
            f'<text x="28" y="56" font-size="12.5" fill="{MUTED}">{sub}</text>']


def txt(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def line(x1, y1, x2, y2, color=GRID, width=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')


def rect(x, y, w, h, fill, stroke="none", dash="", rx=4):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = f' stroke="{stroke}" stroke-width="1.4"{d}' if stroke != "none" else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}"{s}/>')


def parse_rows(header, n):
    ls = STDOUT.splitlines()
    i = next(j for j, l in enumerate(ls) if header in l)
    rows = []
    for l in ls[i + 1:i + 1 + n]:
        parts = [p.strip() for p in l.split("|")]
        rows.append(parts)
    return rows


# ---------------------------------------------------------------- 图 1
def fig1():
    rows = parse_rows("T(token) | softmax MiB", 7)
    W, H = 1000, 470
    o = head(W, H, "缓存账本：每层 / 每请求，三种口径随上下文长度的变化",
             "设定 n_h=32, n_kv=8 (GQA 4:1), d=128, bf16；状态 1.0000 MiB 恒定，softmax 每 token 4096 bytes")
    x0, x1, y0, y1 = 90, 950, 395, 80
    lx = lambda T: x0 + (math.log10(T) - math.log10(128)) / (math.log10(1048576) - math.log10(128)) * (x1 - x0)
    ly = lambda M: y0 - (math.log10(M) - math.log10(0.25)) / (math.log10(8192) - math.log10(0.25)) * (y0 - y1)
    for T in (128, 256, 1024, 4096, 32768, 131072, 1048576):
        x = lx(T)
        o.append(line(x, y0, x, y1, GRID, 1))
        o.append(txt(x, y0 + 20, {128: "128", 256: "256", 1024: "1K", 4096: "4K", 32768: "32K",
                                  131072: "128K", 1048576: "1M"}[T], 11, MUTED, "middle"))
    for M in (0.5, 1, 4, 16, 128, 512, 4096):
        y = ly(M)
        o.append(line(x0, y, x1, y, GRID, 1))
        o.append(txt(x0 - 10, y + 4, str(M), 11, MUTED, "end"))
    o.append(txt((x0 + x1) / 2, y0 + 36, "上下文长度 T (token，对数轴)", 12, MUTED, "middle"))

    pts = [(int(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows]
    for idx, color, name in ((1, RED, "softmax KV (∝T)"), (2, BLUE, "线性状态 (固定)"), (3, PURPLE, "3:1 混合")):
        path = " ".join(f"{'M' if j == 0 else 'L'}{lx(p[0]):.1f},{ly(p[idx]):.1f}"
                        for j, p in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        o.append(txt(x1 - 12, ly(pts[-1][idx]) - 10, name, 12, color, "end", "600"))

    xt = lx(256)
    o.append(line(xt, y0, xt, y1, TEAL, 1.6, "5,4"))
    o.append(rect(xt + 8, 100, 250, 46, "#f0fdfa", TEAL, "4,3"))
    o.append(txt(xt + 18, 120, "T* = 256：三者相等", 12.5, TEAL, weight="600"))
    o.append(txt(xt + 18, 138, "短于此，混合反而更占显存", 11.5, MUTED))
    o.append(txt(x1 - 12, 376, "1M 时显存口径减少 74.9817%，KV 口径 75.00%", 11.5, MUTED, "end"))
    o.append(txt(28, 458, "曲线来自 results/stdout.txt 组 2 的逐行输出；减少比例 = 0.75 (1 - T*/T)。", 11.5, MUTED))
    o.append("</svg>")
    (OUT / "fig1_cache_ledger.svg").write_text("\n".join(o), encoding="utf-8")


# ---------------------------------------------------------------- 图 2
def fig2():
    W, H = 1000, 500
    o = head(W, H, "同样叫「缓存」，两种对象的形状完全不同",
             "左：逐 token 的只读历史；右：固定形状、被就地覆盖的矩阵状态")
    o.append(line(500, 78, 500, 455, GRID, 1.5, "6,4"))

    # 左：softmax KV
    o.append(txt(28, 96, "softmax attention：KV cache 是历史记录", 15, RED, weight="600"))
    o.append(rect(40, 112, 420, 250, "#fef2f2", RED, "5,4"))
    for i in range(6):
        y = 126 + i * 36
        o.append(rect(62, y, 160, 27, "#fee2e2"))
        o.append(txt(72, y + 18, f"token {i + 1}:  k, v", 12, INK))
        o.append(txt(240, y + 18, "→ 追加，不覆盖", 11.5, MUTED))
    o.append(txt(62, 348, "… 一直到第 T 个 token", 12, MUTED))
    o.append(txt(40, 386, "2 n_kv d T b  字节（随 T 线性增长）", 13, RED, weight="600"))
    o.append(txt(40, 410, "前缀只读 → 按 block 哈希复用，可任意 fork", 12, INK))
    o.append(txt(40, 432, "DeepSeek-V2 Table 1：GQA 2 n_g d_h l，MLA (d_c+d_h^R) l", 11, MUTED))

    # 右：linear state
    o.append(txt(528, 96, "linear attention：状态是一份矩阵，不是历史", 15, BLUE, weight="600"))
    o.append(rect(540, 112, 420, 250, "#eff6ff", BLUE, "5,4"))
    # 矩阵网格
    for r in range(5):
        for c in range(8):
            o.append(rect(570 + c * 30, 140 + r * 28, 24, 22,
                          "#bfdbfe" if (r + c) % 3 else "#93c5fd"))
    o.append(txt(570, 292, "S_t = α S_(t-1) + φ(k_t) v_t^T", 13.5, INK))
    o.append(txt(570, 316, "每步就地覆盖同一块内存，没有第 t 份副本", 11.5, MUTED))
    o.append(txt(880, 180, "d_k × d_v", 13, BLUE, "middle", "600"))
    o.append(txt(880, 202, "× n_h 头", 12, MUTED, "middle"))
    # in-place 箭头
    o.append(f'<path d="M 790 120 A 40 40 0 0 1 830 160" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    o.append(txt(870, 128, "in-place", 11.5, BLUE))
    o.append(txt(540, 386, "n_h d² b  字节（与 T 无关，但仍占显存）", 13, BLUE, weight="600"))
    o.append(txt(540, 410, "不可回滚 → 整段 snapshot + copy-on-write + 尾部重放", 12, INK))
    o.append(txt(540, 432, "conv state / normalizer z / SSM state 同样是每层固定状态", 11, MUTED))
    o.append(txt(500, 478, "交叉点 T* = n_h d / (2 n_kv)：GQA 4:1、d=128 时是 256 token", 12, TEAL, "middle", "600"))
    o.append("</svg>")
    (OUT / "fig2_state_vs_kv.svg").write_text("\n".join(o), encoding="utf-8")


# ---------------------------------------------------------------- 图 3
def fig3():
    cap = parse_rows("N | rank(S)", 5)
    chk = parse_rows("C | 块内 FLOPs", 5)
    W, H = 1000, 480
    o = head(W, H, "两个不能省的代价：状态容量与分块权衡",
             "合成探针，数值来自 results/stdout.txt 组 3 / 组 4；不是真实模型精度")

    # 左面板：容量
    ax0, ax1, ay0, ay1 = 70, 470, 350, 105
    o.append(txt(70, 92, "① 随机 key 的读回误差（d=16, d_v=6）", 13.5, INK, weight="600"))
    for v in (0, 1, 2, 3, 4):
        y = ay0 - v / 4 * (ay0 - ay1)
        o.append(line(ax0, y, ax1, y, GRID, 1))
        o.append(txt(ax0 - 8, y + 4, str(v), 10.5, MUTED, "end"))
    xs = [8, 16, 32, 64, 128]
    vals = [float(r[2]) for r in cap]
    xp = lambda i: ax0 + 20 + i * (ax1 - ax0 - 40) / 4
    yp = lambda v: ay0 - v / 4 * (ay0 - ay1)
    cap_path = " ".join(f"{'M' if j == 0 else 'L'}{xp(j):.1f},{yp(v):.1f}"
                        for j, v in enumerate(vals))
    o.append(f'<path d="{cap_path}" fill="none" stroke="{RED}" stroke-width="2.4"/>')
    for j, (x, v) in enumerate(zip(xs, vals)):
        o.append(f'<circle cx="{xp(j):.1f}" cy="{yp(v):.1f}" r="4" fill="{RED}"/>')
        o.append(txt(xp(j), yp(v) - 10, f"{v:.4f}", 10.5, RED, "middle"))
        o.append(txt(xp(j), ay0 + 18, str(x), 11, MUTED, "middle"))
    o.append(txt((ax0 + ax1) / 2, ay0 + 32, "存进去的 key-value 对数 N", 11.5, MUTED, "middle"))
    o.append(txt(ax0 + 10, 315, "rank(S) ≤ min(N, d_k, d_v) = 6", 11, MUTED))
    o.append(txt(ax0 + 10, 333, "N > d 时最小二乘残差 > 0.72", 11, MUTED))
    o.append(txt(ax0 + 10, 412, "正交 key 可逐位读回；随机 key 串扰随 N 增长", 11.5, INK))

    # 右面板：chunk 两笔账
    bx0, bx1, by0, by1 = 570, 950, 350, 105
    o.append(txt(570, 92, "② 块大小 C：块内 FLOPs 与块间状态 IO", 13.5, INK, weight="600"))
    for v in range(5):
        y = by0 - v / 4 * (by0 - by1)
        o.append(line(bx0, y, bx1, y, GRID, 1))
        o.append(txt(bx0 - 8, y + 4, f"{v / 4:.2f}", 10.5, MUTED, "end"))
    cs = [int(r[0]) for r in chk]
    fl = [float(r[1]) for r in chk]
    io = [float(r[2]) for r in chk]
    xn = lambda i: bx0 + 20 + i * (bx1 - bx0 - 40) / 4
    yn = lambda v, m: by0 - v / m * (by0 - by1)
    for series, color, name in ((fl, PURPLE, "块内 FLOPs / 268.4355 M"), (io, TEAL, "块间状态 IO / 8 MiB")):
        m = max(series)
        path = " ".join(f"{'M' if j == 0 else 'L'}{xn(j):.1f},{yn(v, m):.1f}"
                        for j, v in enumerate(series))
        o.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for j, v in enumerate(series):
            o.append(f'<circle cx="{xn(j):.1f}" cy="{yn(v, m):.1f}" r="3.5" fill="{color}"/>')
    for j, c in enumerate(cs):
        o.append(txt(xn(j), by0 + 18, str(c), 11, MUTED, "middle"))
    o.append(txt((bx0 + bx1) / 2, by0 + 32, "chunk size C", 11.5, MUTED, "middle"))
    o.append(txt(bx0 + 10, 412, "GLA 论文实测最优 C=128（BF16 / batch 32 / dim 1024）", 11.5, MUTED))
    o.append(txt(bx0 + 10, 440, "两条曲线方向相反，乘积恒定 134.2177", 11, INK))
    o.append(txt(bx0 + 10, 460, "C=16 时传状态 8 MiB > 写一遍 KV 2 MiB", 11, MUTED))
    o.append(line(510, 78, 510, 420, GRID, 1.5, "6,4"))
    o.append("</svg>")
    (OUT / "fig3_capacity_chunk.svg").write_text("\n".join(o), encoding="utf-8")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    for p in sorted(OUT.glob("*.svg")):
        print(p.name, p.stat().st_size, "bytes")
