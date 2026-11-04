# -*- coding: utf-8 -*-
"""π0.5 companion lab：共训练 / 单模型两阶段的合成探针。

跑法（数字要用 miniconda python 生成，numpy 2.1.1；managed python 的 numpy
2.5.x 会给出略不同的数值）::

    ~/Software/miniconda3/bin/python pi05_lab.py

四组实验（断言都在组内，最后汇总断言总数）::

    自校验   手写反向 vs 有限差分；世界不变量
    A 组     共训练配方消融（WD / ME / CE）——对应论文 Fig.10/11
    B 组     环境数量的 scaling——对应论文 Fig.8 的 3→104 曲线
    C 组     joint vs implicit HL vs flat——对应论文 Fig.13
    D 组     状态进 prompt 的量化代价——对应 openpi tokenizer.py 的 π0.5 格式
"""
from __future__ import annotations

import numpy as np

import pi05_lib as L
from pi05_lib import (Adam, D_A, D_C, D_CTX, D_PHI, HID, K, M, N_ME_ENV,
                      N_MM_ENV, N_TEST_ENV, Pi05Net, S, T, World, bwd, fwd,
                      make_batch, mse_grad)

N_STEPS = 2500
BATCH = 128
N_SEEDS = 3

_ASSERTS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "[ok]" if ok else "[FAIL]"
    _ASSERTS.append(f"{tag} {name}" + (f"：{detail}" if detail else ""))
    print(f"  {tag} {name}" + (f"（{detail}）" if detail else ""))


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n{title}\n{line}")


# ---------------------------------------------------------------- 训练一步
def _zero_grads(net: Pi05Net):
    return [np.zeros_like(p) for p in net.params]


def _psi_ix():
    return slice(0, 4)


def _trunk_ix():
    return slice(4, 8)


def _act_ix():
    return slice(8, 12)


def _skill_ix(net: Pi05Net):
    return slice(12, 16) if net.skill_head is not None else slice(0, 0)


def train_batch_grads(net: Pi05Net, b, *, with_skill: bool, with_act: bool,
                      alpha: float = 1.0):
    """一个机器人数据批的反向。返回 (losses, grads)。"""
    g = _zero_grads(net)
    obs, done = b["obs"], b["done"]
    n, kk, _ = obs.shape
    cv, kap, s, a = b["cv"], b["kap"], b["s"], b["a"]

    # ---- psi 前向（每槽独立）
    X1 = obs.reshape(n * kk, D_PHI)
    H1, P1 = fwd(net.psi, X1)
    zh = np.tanh(P1).reshape(n, kk * M)
    xs = np.concatenate([zh, done], axis=1)

    # ---- trunk 前向
    Xg = np.concatenate([cv, xs], axis=1)
    H2, F2 = fwd(net.trunk, Xg)
    dF2 = np.zeros_like(F2)
    losses = {}

    # ---- 技能头（CE）
    if with_skill and net.skill_head is not None:
        H3, L3 = fwd(net.skill_head, F2)
        loss_ce, dL = L.softmax_ce(L3, s)
        losses["ce"] = loss_ce
        gk = L.bwd(net.skill_head, F2, H3, dL)
        for i in range(4):
            g[12 + i] += gk[i]
        # 经技能头隐藏层回传到 F2（漏掉 tanh' 与 W1 是上一版的 bug）
        dH3 = (dL @ net.skill_head[2].T) * (1.0 - H3 * H3)
        dF2 += dH3 @ net.skill_head[0].T

    # ---- 动作头（MSE，teacher forcing：训练时条件于真值技能）
    if with_act:
        zin = (np.concatenate([F2, L.onehot(s, S), kap], axis=1)
               if net.use_skill_input else
               np.concatenate([F2, cv, kap], axis=1))
        H4, A4 = fwd(net.act_head, zin)
        loss_mse, dA = L.mse_grad(A4, a)
        losses["mse"] = loss_mse
        gk = L.bwd(net.act_head, zin, H4, alpha * dA)
        for i in range(4):
            g[8 + i] += gk[i]
        # 经动作头隐藏层回传到 F2（cv / κ 是输入，不回传）
        dH4 = (alpha * dA @ net.act_head[2].T) * (1.0 - H4 * H4)
        dF2 += (dH4 @ net.act_head[0].T)[:, :HID]

    # ---- trunk 反向（参数梯度 + 输入梯度，后者继续传给 ψ）
    gk = L.bwd(net.trunk, Xg, H2, dF2)
    for i in range(4):
        g[4 + i] += gk[i]
    dH2 = (dF2 @ net.trunk[2].T) * (1.0 - H2 * H2)
    dxs = (dH2 @ net.trunk[0].T)[:, D_C:]   # done 列是输入，不回传

    # ---- psi 反向（经 tanh；done 列是输入，不回传）
    dz = dxs[:, :K * M].reshape(n, kk, M)
    dP1 = dz * (1.0 - np.tanh(P1.reshape(n, kk, M)) ** 2)
    gk = L.bwd(net.psi, X1, H1, dP1.reshape(n * kk, M))
    for i in range(4):
        g[0 + i] += gk[i]
    return losses, g


def wd_batch_grads(net: Pi05Net, world: World, rng, n: int = 64):
    """web 数据批：直接监督 ψ(φ_t) → z_t（含 web-only 类型）。"""
    idx = rng.integers(0, T, size=n)
    phi = world.phi_types[idx]
    zt = world.z_types[idx]
    H1, P1 = fwd(net.psi, phi)
    out = np.tanh(P1)
    loss, dA = L.mse_grad(out, zt)
    g = _zero_grads(net)
    dP1 = dA * (1.0 - out ** 2)
    gk = L.bwd(net.psi, phi, H1, dP1)
    for i in range(4):
        g[i] += gk[i]
    return loss, g


def wd_qa_grads(net: Pi05Net, world: World, rng, n: int = 96):
    """web 语义问答：对**全部**属性模式（含 web-only 类型）问「该选哪个技能」。

    这是纯语言/语义监督，不含任何机器人动作：它走 ψ → 共享 trunk → 技能头，
    对应 web 数据给 VLM 主干补语义、而不是补动作。复合约束仍不进训练集。
    """
    ty = rng.integers(0, T, size=(n, K))
    z = world.z_types[ty]
    done = -np.ones((n, K))
    x = world.semantic_vec(z, done)
    ci = rng.integers(0, len(world.train_pool), size=n)
    cv = world.cvec_batch([world.train_pool[i] for i in ci])
    s = world.skill_vec(x, cv)
    b = dict(cv=cv, obs=world.observe(ty), x=x, done=done, ty=ty,
             kap=np.zeros((n, D_CTX)), s=s,
             a=np.zeros((n, D_A)))
    return train_batch_grads(net, b, with_skill=True, with_act=False)[1]


def train(world: World, seed: int, *, use_wd=True, use_ce=True, use_me=True,
          use_skill_input=True, train_skill=True, n_steps=N_STEPS,
          kaps_mm=None, kaps_me=None, kaps_ce=None, verbose=False):
    """训练一个变体。环境池由调用方显式给出（保证各变体评的是同一批环境）。"""
    rng = np.random.default_rng(seed)
    net = Pi05Net(world, rng, use_skill_input=use_skill_input,
                  train_skill=train_skill)
    opt = Adam(net.params, lr=3e-3)
    if kaps_mm is None:
        kaps_mm = world.rng.normal(size=(N_MM_ENV, D_CTX))
    if kaps_me is None:
        kaps_me = world.rng.normal(size=(N_ME_ENV, D_CTX)) if use_me else \
            np.zeros((0, D_CTX))
    kaps_me = kaps_me if use_me else np.zeros((0, D_CTX))
    kaps_all = np.concatenate([kaps_mm, kaps_me], axis=0) if len(kaps_me) \
        else kaps_mm
    if kaps_ce is None:
        kaps_ce = world.rng.normal(size=(20, D_CTX))
    ce_cs = world.train_pool        # 复合约束始终留作测试，不进任何训练集
    task = rng.random(n_steps)
    for t in range(n_steps):
        g = _zero_grads(net)
        # MM 批（总是有）
        r = task[t]
        pool = kaps_all if r < 0.35 and use_me else kaps_mm
        b = make_batch(world, world.train_pool, world.seen_types, pool,
                       max(1, BATCH // len(world.train_pool)), rng)
        _, gmm = train_batch_grads(net, b, with_skill=train_skill,
                                   with_act=True)
        for i in range(len(g)):
            g[i] += gmm[i] * 0.5
        # CE 批（更多样的任务与环境，只喂技能头）
        if use_ce and 0.35 <= r < 0.65:
            b = make_batch(world, ce_cs, world.seen_types, kaps_ce,
                           max(1, BATCH // len(ce_cs)), rng)
            _, gce = train_batch_grads(net, b, with_skill=True, with_act=False)
            for i in range(len(g)):
                g[i] += gce[i] * 0.5
        # WD 批：一半直接监督 ψ(φ→z)，一半是覆盖全部属性模式的语义问答
        if use_wd and r >= 0.65:
            if r >= 0.825:
                _, gwd = wd_batch_grads(net, world, rng)
                for i in range(4):
                    g[i] += gwd[i]
            else:
                gqa = wd_qa_grads(net, world, rng)
                for i in range(len(g)):
                    g[i] += gqa[i]
        opt.step(net.params, g)
        if verbose and (t + 1) % 500 == 0:
            print(f"    step {t+1}")
    return net


# ---------------------------------------------------------------- 评测
def eval_split(net: Pi05Net, world: World, cs, types, kaps, n_per, rng,
               web_only_target: bool | None = None):
    """一个评测切分。返回 (技能准确率, 端到端相对误差, 真值技能相对误差, n)。

    web_only_target=True 时只统计「目标槽是 web-only 类型」的样本。
    """
    accs, errs, oerrs, tot = [], [], [], 0
    for c in cs:
        b = make_batch(world, [c], types, kaps, n_per, rng)
        xs_hat = net.scene_rep(b["obs"], b["done"])
        feat = net.features(b["cv"], xs_hat)
        has_skill = net.use_skill_input and net.skill_head is not None
        if has_skill:
            s_hat = np.argmax(net.skill_logits(feat), axis=1)
        else:
            s_hat = b["s"]          # implicit/flat 没有技能输出，准确率无意义
        a_hat = net.act_out(feat, b["cv"], xs_hat, b["kap"], s_hat)
        a_star = b["a"]
        idx = world.target(b["x"], *c)
        m = np.ones(len(idx), dtype=bool)
        if web_only_target is not None:
            tslot = np.where(idx >= 0, b["ty"][np.arange(len(idx)), idx], -1)
            is_web = np.isin(tslot, world.web_types)
            m = is_web if web_only_target else ~is_web
        if not m.any():
            continue
        accs.append(float(np.mean(s_hat[m] == b["s"][m])) if has_skill
                    else float("nan"))
        errs.append(world.rel_err(a_hat[m], a_star[m]))
        # 真值技能条件下的模型输出（隔离「做什么」的错误，只看「怎么做」）
        a_or = net.act_out(feat, b["cv"], xs_hat, b["kap"], b["s"])
        oerrs.append(world.rel_err(a_or[m], a_star[m]))
        tot += int(m.sum())
    if not accs:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(accs)), float(np.mean(errs)), float(np.mean(oerrs)), tot


def run_variant(world: World, seed: int, pools, **kw):
    """pools = dict(mm, me, ce, test)：训练与评测用同一批环境。"""
    net = train(world, seed, kaps_mm=pools["mm"], kaps_me=pools["me"],
                kaps_ce=pools["ce"], **kw)
    rng = np.random.default_rng(1000 + seed)
    out = {}
    out["id"] = eval_split(net, world, world.train_pool, world.seen_types,
                           pools["mm"], 24, rng)
    out["env"] = eval_split(net, world, world.train_pool, world.seen_types,
                            pools["test"], 24, rng)
    out["oodc"] = eval_split(net, world, world.composite, world.seen_types,
                             pools["mm"], 24, rng)
    out["oodt"] = eval_split(net, world, world.composite, np.arange(T),
                             pools["mm"], 120, rng, web_only_target=True)
    out["seen_t"] = eval_split(net, world, world.composite, np.arange(T),
                               pools["mm"], 120, rng, web_only_target=False)
    return out


def avg_over_seeds(world: World, seeds, pools, **kw):
    runs = [run_variant(world, s, pools, **kw) for s in seeds]
    keys = runs[0].keys()
    out = {}
    for k in keys:
        out[k] = tuple(float(np.mean([r[k][j] for r in runs]))
                       for j in range(3))
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    world = World(0)
    seeds = list(range(N_SEEDS))

    # ------------------------------------------------------------ 自校验
    banner("自校验：手写反向 vs 有限差分；世界不变量")
    rng = np.random.default_rng(7)
    net = Pi05Net(world, rng)
    b = make_batch(world, world.train_pool, world.seen_types,
                   world.rng.normal(size=(4, D_CTX)), 4, rng)
    # ---- 自校验统一用「随机方向的方向导数」：数值稳定，也免去逐元素差分
    def dir_deriv(net, P, j, with_skill, with_act, eps=1e-6, seed=3):
        d = np.random.default_rng(seed).normal(size=P[j].shape)
        d /= np.linalg.norm(d)
        # 必须就地写回：net.params[j] = ... 只换列表项，net.psi 等仍指旧数组
        np.copyto(net.params[j], P[j] + eps * d)
        lp, _ = train_batch_grads(net, b, with_skill=with_skill,
                                  with_act=with_act)
        np.copyto(net.params[j], P[j] - eps * d)
        lm, _ = train_batch_grads(net, b, with_skill=with_skill,
                                  with_act=with_act)
        np.copyto(net.params[j], P[j])
        tot = (lambda L: L.get("ce", 0.0) + L.get("mse", 0.0))
        return (tot(lp) - tot(lm)) / (2 * eps), d

    _, g_full = train_batch_grads(net, b, with_skill=True, with_act=True)
    _, g_mse = train_batch_grads(net, b, with_skill=False, with_act=True)
    P = [p.copy() for p in net.params]
    # 主干输出层权重：只开 MSE 时与有限差分包一一对应
    num, d = dir_deriv(net, P, 6, with_skill=False, with_act=True)
    pred = float(np.sum(g_mse[6] * d))
    rel = abs(pred - num) / max(abs(num), 1e-12)
    check("主干/动作头梯度与有限差分一致", rel < 1e-5,
          f"相对误差 {rel:.1e}")
    # ψ 的梯度经过 tanh + 共享 trunk（技能 CE 与动作 MSE 两路都汇进来）
    num, d = dir_deriv(net, P, 0, with_skill=True, with_act=True)
    pred = float(np.sum(g_full[0] * d))
    rel2 = abs(pred - num) / max(abs(num), 1e-12)
    check("槽编码器梯度（经 tanh 与共享路径）与有限差分一致", rel2 < 1e-5,
          f"相对误差 {rel2:.1e}")
    check("类型属性模式互不相同", len({tuple(z) for z in world.z_types}) == T)
    grid = L._all_z()
    cnt = np.bincount(np.argmax(grid @ world.W_g.T, axis=1), minlength=S - 1)
    p = cnt / cnt.sum()
    check("W_g 的 argmax 在属性模式上分布均衡", p.min() >= 0.12,
          f"min={p.min():.2f} max={p.max():.2f}")
    check("环境项与 Bx 项同量级（都标定到 0.25/D_A）",
          abs(world.oracle_env_gap(*_gap_args(world)) - 0.25 / D_A) < 0.02,
          f"env={world.oracle_env_gap(*_gap_args(world)):.4f}")

    # ------------------------------------------------------------ A 组
    banner("A 组：共训练配方消融（WD=web 数据 / ME=多环境 / CE=跨本体监督）")
    rngP = np.random.default_rng(500)
    pools = dict(mm=rngP.normal(size=(N_MM_ENV, D_CTX)),
                 me=rngP.normal(size=(N_ME_ENV, D_CTX)),
                 ce=rngP.normal(size=(20, D_CTX)),
                 test=rngP.normal(size=(N_TEST_ENV, D_CTX)))
    full = avg_over_seeds(world, seeds, pools)
    no_wd = avg_over_seeds(world, seeds, pools, use_wd=False)
    no_me = avg_over_seeds(world, seeds, pools, use_me=False)
    no_ce = avg_over_seeds(world, seeds, pools, use_ce=False)
    mm_only = avg_over_seeds(world, seeds, pools, use_wd=False, use_me=False,
                             use_ce=False)

    def row(name, r):
        print(f"  {name:<10}  ID技能 {r['id'][0]:.3f}  复合 {r['oodc'][0]:.3f}"
              f"  web目标 {r['oodt'][0]:.3f}  新环境真值技能误差 {r['env'][2]:.3f}"
              f"  ID端到端误差 {r['id'][1]:.3f}")

    for name, r in [("full", full), ("no WD", no_wd), ("no ME", no_me),
                    ("no CE", no_ce), ("MM only", mm_only)]:
        row(name, r)

    check("full 在 ID 技能选择上 ≥ 0.85", full["id"][0] > 0.85,
          f"{full['id'][0]:.3f}")
    check("web 语义监督把未见类别上的技能选择抬起来（> +0.10）",
          full["oodt"][0] - no_wd["oodt"][0] > 0.10,
          f"{no_wd['oodt'][0]:.3f} → {full['oodt'][0]:.3f}")
    check("WD 的收益集中在未见类别：相对降幅比 ID 大 0.15 以上",
          (1 - no_wd["oodt"][0] / full["oodt"][0])
          - (1 - no_wd["id"][0] / full["id"][0]) > 0.15,
          f"web目标相对降 {1 - no_wd['oodt'][0] / full['oodt'][0]:.2f}，"
          f"ID 相对降 {1 - no_wd['id'][0] / full['id'][0]:.2f}")
    check("去掉 ME 后新环境动作变差（真值技能条件，> +0.03）",
          no_me["env"][2] - full["env"][2] > 0.03,
          f"{full['env'][2]:.3f} → {no_me['env'][2]:.3f}")
    check("ME 的收益主要在新环境：新环境伤害 > ID 伤害 0.02 以上",
          (no_me["env"][2] - full["env"][2])
          - (no_me["id"][2] - full["id"][2]) > 0.02,
          f"新环境 +{no_me['env'][2] - full['env'][2]:.3f}，"
          f"ID +{no_me['id'][2] - full['id'][2]:.3f}")
    check("CE 在探针里的边际收益很小：拿掉它对复合约束影响 < 0.12",
          abs(full["oodc"][0] - no_ce["oodc"][0]) < 0.12,
          f"{full['oodc'][0]:.3f} vs {no_ce['oodc'][0]:.3f}"
          "（探针未建模跨本体动作空间差异，见 README 的边界说明）")
    check("全配方对 MM-only 的优势落在未见类别与新环境动作",
          full["oodt"][0] - mm_only["oodt"][0] > 0.07 and
          mm_only["env"][2] - full["env"][2] > 0.02,
          f"web目标 {mm_only['oodt'][0]:.3f}→{full['oodt'][0]:.3f}，"
          f"新环境误差 {mm_only['env'][2]:.3f}→{full['env'][2]:.3f}")
    check("共训练的收益在分布外：ID 端到端误差并不优于 MM-only（差 < 0.08）",
          abs(full["id"][1] - mm_only["id"][1]) < 0.08,
          f"{full['id'][1]:.3f} vs {mm_only['id'][1]:.3f}")

    # ------------------------------------------------------------ B 组
    banner("B 组：环境数量的 scaling（完整配方，只改 MM+ME 的环境池大小）")
    n_envs = [3, 6, 12, 25, 50, 100]
    rngB = np.random.default_rng(202)
    big_pool = rngB.normal(size=(106, D_CTX))
    scal = {}
    for n in n_envs:
        pool = dict(mm=big_pool[:min(n, N_MM_ENV)],
                    me=big_pool[N_MM_ENV:n] if n > N_MM_ENV
                    else big_pool[:0],
                    ce=pools["ce"], test=pools["test"])
        r = avg_over_seeds(world, seeds, pool)
        # 对照：把测试环境也放进训练池（对应论文 Fig.8 的 green control）
        ctrl_mm = np.concatenate([pool["mm"], pools["test"]], axis=0)
        ctrl_me = np.concatenate([pool["me"], pools["test"]], axis=0) \
            if len(pool["me"]) else pools["test"]
        rc = avg_over_seeds(world, seeds,
                            dict(mm=ctrl_mm, me=ctrl_me, ce=pools["ce"],
                                 test=pools["test"]))
        scal[n] = (r, rc)
        print(f"  N={n:>3}  新环境技能 {r['env'][0]:.3f}  "
              f"新环境真值技能误差 {r['env'][2]:.3f}   对照(含测试环境) "
              f"{rc['env'][0]:.3f} / {rc['env'][2]:.3f}")
    lo, hi = scal[3][0], scal[100][0]
    errs = [scal[n][0]["env"][2] for n in n_envs]
    accs = [scal[n][0]["env"][0] for n in n_envs]
    check("环境只影响「怎么做」不影响「做什么」：技能准确率对 N 几乎不动",
          max(accs) - min(accs) < 0.08,
          f"accs={[round(a, 3) for a in accs]}")
    check("环境数 3→100：新环境动作误差好转 > 0.08",
          errs[0] - errs[-1] > 0.08, f"{errs[0]:.3f} → {errs[-1]:.3f}")
    check("环境最少（N=3）是最差的一档", errs[0] == max(errs),
          f"errs={[round(e, 3) for e in errs]}")
    check("N=100 已接近「直接用测试环境训练」的对照（误差差 < 0.04）",
          scal[100][1]["env"][2] - scal[100][0]["env"][2] < 0.04,
          f"{scal[100][0]['env'][2]:.3f} vs 对照 {scal[100][1]['env'][2]:.3f}")

    # ------------------------------------------------------------ C 组
    banner("C 组：joint vs implicit HL vs flat（对应论文 Fig.13）")
    # full 已是 joint
    flat = avg_over_seeds(world, seeds, pools, use_skill_input=False,
                          train_skill=False)
    implicit = avg_over_seeds(world, seeds, pools, use_skill_input=False,
                              train_skill=True)

    def row2(name, r):
        print(f"  {name:<10}  ID端到端误差 {r['id'][1]:.3f}  "
              f"新环境端到端 {r['env'][1]:.3f}  复合端到端 {r['oodc'][1]:.3f}"
              f"  ID真值技能误差 {r['id'][2]:.3f}")

    row2("joint", full)
    row2("implicit", implicit)
    row2("flat", flat)
    check("两阶段推理在分布内更省：joint 的 ID 端到端误差比 flat 低 > 0.02",
          flat["id"][1] - full["id"][1] > 0.02,
          f"{full['id'][1]:.3f} vs {flat['id'][1]:.3f}")
    check("joint 也比 implicit 低 > 0.05：只共训练不跑高层拿不到分布内收益",
          implicit["id"][1] - full["id"][1] > 0.05,
          f"{full['id'][1]:.3f} vs {implicit['id'][1]:.3f}")
    check("只共训练不跑高层，分布内反而略差（技能头分走了容量）",
          implicit["id"][1] - flat["id"][1] > 0.01,
          f"flat {flat['id'][1]:.3f} vs implicit {implicit['id'][1]:.3f}")
    check("分布外（未见复合约束）上离散接口成了单点故障：joint 比 flat 差 > 0.15",
          full["oodc"][1] - flat["oodc"][1] > 0.15,
          f"joint {full['oodc'][1]:.3f} vs flat {flat['oodc'][1]:.3f}")
    check("implicit 落在 joint 与 flat 之间（复合约束端到端误差）",
          flat["oodc"][1] < implicit["oodc"][1] < full["oodc"][1],
          f"flat {flat['oodc'][1]:.3f} / implicit {implicit['oodc'][1]:.3f}"
          f" / joint {full['oodc'][1]:.3f}")
    check("joint 在新环境动作上明显最好（真值技能条件，比 flat 低 > 0.10）",
          flat["env"][2] - full["env"][2] > 0.10,
          f"{full['env'][2]:.3f} vs {flat['env'][2]:.3f}")

    # ------------------------------------------------------------ D 组
    banner("D 组：状态进 prompt 的量化代价（openpi 的 π0.5 格式）")
    q = state_quant_probe()
    print("  bins      相对误差（占噪声下界的倍数）   prompt token 数（18 维状态）")
    for bins in (8, 16, 32, 64, 256):
        digits = len(str(bins - 1))
        tok = 18 * (digits + 1)
        print(f"  {bins:>4}      {q[bins]:>8.4f}  ({q[bins] / q['floor']:.2f}×)"
              f"                    {tok}")
    check("量化误差随 bins 单调下降", all(
        q[a] > q[b] for a, b in zip((8, 16, 32, 64), (16, 32, 64, 256))))
    check("量化从来不是瓶颈：粗到 8 bins 也只占噪声下界的 30% 不到",
          q[8] < 0.3 * q["floor"], f"{q[8]:.4f} vs floor {q['floor']:.4f}")
    check("256 bins 的量化误差几乎为零（< 0.01× 噪声下界）",
          q[256] < 0.01 * q["floor"],
          f"{q[256]:.2e} vs floor {q['floor']:.4f}")
    check("256 bins × 18 维 ≈ 72 个 token：π0 的 48 token 槽装不下，"
          "π0.5 的 200 token 槽才装得下", 48 < 18 * 4 < 200)

    # ------------------------------------------------------------ 汇总
    banner("汇总")
    n_ok = sum(1 for a in _ASSERTS if a.startswith("[ok]"))
    for a in _ASSERTS:
        print("  " + a)
    print(f"\n  断言总数：{len(_ASSERTS)}（通过 {n_ok}）")
    if n_ok != len(_ASSERTS):
        raise SystemExit("有断言未通过")


def _gap_args(world: World):
    x = world.sample_semantic(8192, np.arange(T))[0]
    kap = world.rng.normal(size=(len(x), D_CTX))
    return x, kap


def state_quant_probe(bins_list=(8, 16, 32, 64, 256), n=200000, seed=11):
    """π0.5 把本体状态离散化进 prompt（openpi tokenizer.py L26-28）。

    状态 18 维（论文：18–19 DoF）、取值 [-1,1]；动作 a = μ_s + B_s x + η
    （μ 是正交的技能原型，同 04 篇的设定）。量化误差
    E‖B_s(x−x̂)‖² / Var(a) 随 bins 变化；噪声下界 = σ²/Var(a)。
    """
    rng = np.random.default_rng(seed)
    D_S = 18
    B = rng.normal(size=(D_A, D_S))
    x = rng.uniform(-1, 1, size=(n, D_S))
    cur = float(np.mean(np.square(x @ B.T)))
    B *= np.sqrt((0.25 / D_A) / max(cur, 1e-12))
    mu = L._orthonormal(rng, S, D_A)
    s = rng.integers(0, S, size=n)
    a = mu[s] + x @ B.T
    a_obs = a + 0.15 * rng.normal(size=(n, D_A))
    var = float(np.mean(np.square(a_obs - a_obs.mean(axis=0))))
    out = {"floor": 0.15 ** 2 / var}
    for bins in bins_list:
        xq = np.round((x + 1) / 2 * (bins - 1)) / (bins - 1) * 2 - 1
        aq = mu[s] + xq @ B.T
        out[bins] = float(np.mean(np.square(aq - a))) / var
    return out


if __name__ == "__main__":
    main()
