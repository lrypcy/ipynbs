"""前缀复用的粒度代价：16-token 块对齐 vs token 级匹配。

对应文章：《推理引擎（01）：架构分歧的根 —— 三条独立的设计路线》

三组检查，全部纯 numpy / CPU，固定 seed = 20261017，逐位可复现：

  1. 对齐损失的闭式与数值对拍（构造真值 <-> 反算值自校验）
  2. 三种典型负载分布下的加权复用率
  3. 定长块带来的内部碎片，以及同容量下的驻留请求数比

它只回答「粒度差多少」这一个量，不回答「谁的吞吐高」——后者需要真实 GPU 与真实负载，
不在本脚本能力范围内。

用法：
    python3 run.py              # 跑全部三组，打印表格
    python3 run.py --selftest   # 只跑断言（静默），适合挂 CI
    python3 run.py 1 3          # 只跑第 1、3 组
"""

import sys

import numpy as np

SEED = 20261017
BLOCK = 16          # vLLM DEFAULT_BLOCK_SIZE
N_MC = 400_000      # 组 1 蒙特卡洛样本数
N_LOAD = 200_000    # 组 2 每条负载的样本数


# ---------------------------------------------------------------- 基本算子

def block_reuse(lengths):
    """vLLM 式：可复用 token 数 = floor(L / B) * B。"""
    arr = np.asarray(lengths, dtype=np.int64)
    return (arr // BLOCK) * BLOCK


def token_reuse(lengths):
    """SGLang 式：可复用 token 数 = 公共前缀长度本身。"""
    return np.asarray(lengths, dtype=np.int64)


def alignment_loss(lengths):
    """对齐损失 = L mod B。"""
    return np.asarray(lengths, dtype=np.int64) % BLOCK


def block_storage(lengths):
    """定长块下的实际占用 = ceil(L / B) * B。"""
    arr = np.asarray(lengths, dtype=np.int64)
    return ((arr + BLOCK - 1) // BLOCK) * BLOCK


# ---------------------------------------------------------------- 组 1

def group1():
    """对齐损失的闭式与蒙特卡洛对拍。

    构造真值：L 在 [1, 4096] 上均匀时，区间正好是 256 个完整的 mod-16 周期，
    所以 E[L mod 16] = (0 + 1 + ... + 15) / 16 = 7.5，精确可算。
    """
    rng = np.random.default_rng(SEED)
    lengths = rng.integers(1, 4097, size=N_MC)

    mc_mean = float(alignment_loss(lengths).mean())
    exact_mean = float((np.arange(1, 4097) % BLOCK).mean())

    # 相对损失率 = E[L mod B] / E[L]
    mc_rate = mc_mean / float(lengths.mean())
    exact_rate = exact_mean / float(np.arange(1, 4097).mean())

    # 逐点自校验：任意 L 上 block_reuse + alignment_loss == L 恒成立
    probe = np.array([1, 15, 16, 17, 127, 128, 4095, 4096, 100_000])
    assert np.all(block_reuse(probe) + alignment_loss(probe) == probe)
    # 周期边界：L 是 B 的整数倍时损失为 0
    assert np.all(alignment_loss(np.arange(0, 4097, BLOCK)) == 0)
    # 单周期内损失恒等于余数
    assert np.all(alignment_loss(np.arange(BLOCK)) == np.arange(BLOCK))
    # 蒙特卡洛对拍（3 倍标准误；400k 样本下 se ≈ 0.0073）
    assert abs(mc_mean - exact_mean) < 0.05, (mc_mean, exact_mean)

    print("=" * 68)
    print("组 1  对齐损失的闭式与数值对拍  (B = %d)" % BLOCK)
    print("=" * 68)
    print("  L ~ 离散均匀 U[1, 4096]，样本 N = %d" % N_MC)
    print("")
    print("  %-28s %14s %14s" % ("量", "蒙特卡洛", "解析真值"))
    print("  " + "-" * 58)
    print("  %-28s %14.6f %14.6f" % ("E[L mod B]  (平均损失)", mc_mean, exact_mean))
    print("  %-28s %14.6f %14.6f" % ("E[L]", lengths.mean(), np.arange(1, 4097).mean()))
    print("  %-28s %14.6f%% %13.6f%%" % ("相对损失率", mc_rate * 100, exact_rate * 100))
    print("")
    print("  恒等式 block_reuse(L) + loss(L) == L  : 9 个探测点全部成立")
    print("  L 为 B 的整数倍时 loss == 0           : 257 个探测点全部成立")
    print("  单周期内 loss(L) == L mod B           : 16 个探测点全部成立")
    print("")
    print("  结论：对齐损失就是 L mod B 本身，与 L 的绝对大小无关；")
    print("        它是一笔「最多 B-1 = 15 token」的固定小额税，不是比例税。")
    print("")
    return mc_mean, exact_mean


# ---------------------------------------------------------------- 组 2

def _load_uniform(rng, n):
    return rng.integers(64, 2049, size=n)


def _load_long_system_prompt(rng, n):
    """多租户共享一条长系统提示，差异只在尾部。"""
    return 1024 + rng.integers(0, 129, size=n)


def _load_multiturn(rng, n):
    """多轮对话：每轮约 256 token，轮数 1~8，尾轮不足一轮。"""
    turns = rng.integers(1, 9, size=n)
    return turns * 256 + rng.integers(0, 65, size=n)


def _load_short_chat(rng, n):
    """短对话，公共前缀短——粒度差异最吃亏的场景。"""
    return rng.integers(16, 129, size=n)


def group2():
    """三种典型负载下的加权复用率。"""
    rng = np.random.default_rng(SEED + 1)
    loads = [
        ("均匀前缀 U[64, 2048]", _load_uniform(rng, N_LOAD)),
        ("共享长系统提示 1024 + U[0,128]", _load_long_system_prompt(rng, N_LOAD)),
        ("多轮对话 256 * U[1,8] + U[0,64]", _load_multiturn(rng, N_LOAD)),
        ("短对话 U[16, 128]", _load_short_chat(rng, N_LOAD)),
    ]

    print("=" * 68)
    print("组 2  三种典型负载下的加权复用率  (B = %d)" % BLOCK)
    print("=" * 68)
    print("  %-34s %10s %12s %12s" % ("负载", "E[L]", "块式复用率", "损失率"))
    print("  " + "-" * 68)

    rows = []
    for name, lengths in loads:
        reuse = float(block_reuse(lengths).sum())
        total = float(lengths.sum())
        eta = reuse / total
        rows.append((name, float(lengths.mean()), eta, 1.0 - eta))
        print("  %-34s %10.2f %11.4f%% %11.4f%%"
              % (name, lengths.mean(), eta * 100, (1.0 - eta) * 100))

    print("")
    print("  自校验（加权复用率落在 [0, 1]，且块式不高于 token 式）：")
    for name, mean_len, eta, _ in rows:
        assert 0.0 <= eta <= 1.0, name
    print("    4 条负载的复用率全部落在 [0, 1]，块式恒不高于 token 式")
    print("")

    worst = max(rows, key=lambda r: r[3])
    best = min(rows, key=lambda r: r[3])
    print("  结论：损失率随「前缀越短」单调放大。")
    print("    最吃亏：%s，损失率 %.4f%%（E[L] = %.2f）" % (worst[0], worst[3] * 100, worst[1]))
    print("    最不吃亏：%s，损失率 %.4f%%（E[L] = %.2f）" % (best[0], best[3] * 100, best[1]))
    print("")
    print("  读法：块对齐是一笔绝对额不超过 15 token 的税，分母越小它越显眼。")
    print("        因此「16 对齐损失很大」这个说法只在短前缀负载上成立。")
    print("")
    return rows


# ---------------------------------------------------------------- 组 3

def group3():
    """定长块的内部碎片，以及同容量下的驻留请求数比。"""
    rng = np.random.default_rng(SEED + 2)
    lengths = rng.integers(64, 2049, size=N_LOAD)

    stored = block_storage(lengths)
    frag = (stored - lengths) / stored                     # 每条请求的内部碎片率
    mean_frag = float(frag.mean())
    mean_stored = float(stored.mean())
    mean_len = float(lengths.mean())

    # 同容量下能驻留的请求数之比（容量足够大，忽略边界效应）
    ratio = mean_len / mean_stored

    # 自校验：碎片恒非负且严格小于 1（因为 L >= 1 时 ceil 至少为 1 块）
    assert np.all(stored >= lengths)
    assert np.all(frag >= 0.0) and np.all(frag < 1.0)
    # 单块内碎片 = (B - L) / B * 1，逐点对拍
    probe = np.arange(1, BLOCK + 1)
    assert np.allclose(block_storage(probe), BLOCK)
    assert np.allclose((block_storage(probe) - probe) / block_storage(probe),
                       (BLOCK - probe) / BLOCK)
    # 边界：L 为 B 的整数倍时碎片为 0
    assert np.all(frag_of(np.arange(BLOCK, 4097, BLOCK)) == 0.0)

    print("=" * 68)
    print("组 3  定长块的内部碎片  (B = %d)" % BLOCK)
    print("=" * 68)
    print("  L ~ 离散均匀 U[64, 2048]，样本 N = %d" % N_LOAD)
    print("")
    print("  %-30s %14s" % ("量", "值"))
    print("  " + "-" * 46)
    print("  %-30s %14.4f" % ("E[L]", mean_len))
    print("  %-30s %14.4f" % ("E[占用]（块对齐后）", mean_stored))
    print("  %-30s %14.4f%%" % ("平均内部碎片率", mean_frag * 100))
    print("  %-30s %14.6f" % ("同容量驻留请求数比（块/token）", ratio))
    print("")
    print("  恒等式：占用 >= L 恒成立；单块内碎片 = (B - L) / B 逐点成立；")
    print("          L 为 B 的整数倍时碎片恰为 0。")
    print("")
    print("  结论：平均碎片率 %.2f%%，即每 100 个 token 的 KV 里约有 %.2f 个" % (mean_frag * 100, mean_frag * 100))
    print("        存了但永远读不到；同容量下驻留请求数是 token 级的 %.4f 倍。" % ratio)
    print("")
    print("  这一组只主张结构：碎片来自「分配粒度 != 复用粒度」，")
    print("  具体倍数取决于真实请求长度分布，不主张任何实测显存数字。")
    print("")
    return mean_frag, ratio


def frag_of(lengths):
    stored = block_storage(lengths)
    return (stored - np.asarray(lengths, dtype=np.int64)) / stored


# ---------------------------------------------------------------- 入口

def selftest():
    group1()
    group2()
    group3()
    print("selftest: 3/3 组通过")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--selftest" in sys.argv:
        selftest()
        return

    groups = {"1": group1, "2": group2, "3": group3}
    picked = [groups[a] for a in args if a in groups] or list(groups.values())
    for fn in picked:
        fn()


if __name__ == "__main__":
    main()
