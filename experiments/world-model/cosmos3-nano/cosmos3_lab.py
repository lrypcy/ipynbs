# %%
# -*- coding: utf-8 -*-
"""Cosmos 3 Nano 账本实验 —— 「16B」到底花在哪。

对应文章：世界模型 / 《Cosmos 3 Nano：16B 里只有 8B 在干活》。

技术报告（arXiv 版 2026-06-22，139 页）给了超参、token 预算、位置调制公式、
动作维度表、迁移对照表。这些都能在 CPU 上重算一遍并和公布值对账：

  A 组  参数账：从超参算单塔、双塔、总参，验证 2B/8B/32B 与 4B/16B/64B 两套口径
  B 组  计算账：稳态去噪循环里，那半个塔贡献多少 FLOPs
  C 组  token 账：74K 上下文为什么在 480p 只容 B=3、720p 只容 B=1
  D 组  位置账：FPS 调制 delta_t = 24/FPS 与 15000 的间隔到底是多少秒
  E 组  动作账：57/29/20/10/9 维从哪来；6D 旋转为什么不能用 3D
  F 组  噪声账：rectified flow 的 shift 重参数化把质量挪到哪
  G 组  迁移账：mid-init 相对 pre-init 的收益，早停时差多少倍
  H 组  协同账：跨域动作数据的迁移矩阵是不对称的

纯 numpy / CPU / 固定 seed / 每组末尾断言 / 最后打出断言总数。

一个前提：A–F 是从论文超参与公式**重算**，不是复现训练；G、H 用的是论文表格里
的读数（会标注表号），本实验只做换算与结构分析，不重跑模型。
"""

import numpy as np

SEED = 20260622
N_ASSERT = 0


def ok(cond, msg):
    global N_ASSERT
    assert cond, f"FAILED: {msg}"
    N_ASSERT += 1
    print(f"  [ok] {msg}")


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def banner(t):
    print(f"\n--- {t} ---")


def billions(x):
    return x / 1e9


# ====================================================================== 常量
# 技术报告 Table 2：三个变体的稠密塔超参
VARIANTS = {
    "Edge":  dict(L=28, h=2048, n_head=16, n_kv=8, d_head=128, ffn=9216,
                  total=4.0, dense=2.0, tie=True),
    "Nano":  dict(L=36, h=4096, n_head=32, n_kv=8, d_head=128, ffn=12288,
                  total=16.0, dense=8.0, tie=False),
    "Super": dict(L=64, h=5120, n_head=64, n_kv=8, d_head=128, ffn=25600,
                  total=64.0, dense=32.0, tie=False),
}

VOCAB = 151936          # Qwen3 系词表
CTX = 74000             # 训练 / 推理共用的 token 打包预算
T_FPS_BASE = 24.0       # 位置调制的基准帧率
VAE_T = 4               # 视频 VAE 时间压缩
VAE_S = 32              # 视频 VAE 空间压缩（16x16 压缩后 2x2 patch merge）


# ====================================================================== A 组
def layer_params(h, n_head, n_kv, d_head, ffn):
    """一个 transformer 解码层的参数量（Qwen3 系：无 bias，SwiGLU，两个 RMSNorm）。"""
    q = h * n_head * d_head
    k = h * n_kv * d_head
    v = h * n_kv * d_head
    o = n_head * d_head * h
    mlp = 3 * h * ffn                 # gate_proj / up_proj / down_proj
    norms = 2 * h                     # attn_norm + ffn_norm
    return dict(q=q, k=k, v=v, o=o, attn=q + k + v + o, mlp=mlp, norms=norms,
                total=q + k + v + o + mlp + norms)


def tower_params(cfg):
    per = layer_params(cfg["h"], cfg["n_head"], cfg["n_kv"], cfg["d_head"], cfg["ffn"])
    return per, per["total"] * cfg["L"]


def model_params(cfg, shared_embed=True):
    """共享一份 token embedding；lm_head 只在未 tie 时占一份。"""
    per, layers = tower_params(cfg)
    emb = VOCAB * cfg["h"]
    if cfg["tie"]:
        head = 0
    else:
        head = emb
    two_towers = 2 * layers + emb + head
    return per, layers, two_towers, emb, head


# %%
section("A. 参数账：16B 是两个 8B 塔，不是一个 16B 模型")
banner("A1 逐层拆解（Nano）")
per, layers = tower_params(VARIANTS["Nano"])
print(f"  Nano 单层：attn {billions(per['attn']):7.3f}B  mlp {billions(per['mlp']):7.3f}B"
      f"  norm {per['norms'] / 1e6:5.3f}M  合计 {billions(per['total']):7.3f}B")
print(f"  Nano {VARIANTS['Nano']['L']} 层合计（单塔）：{billions(layers):7.3f}B")
ok(6.8 < billions(layers) < 7.1,
   f"Nano 单塔层参数 {billions(layers):.3f}B 落在 6.8-7.1B（余下的 embedding 与 lm_head 另算）")

banner("A2 三个变体的两套口径")
print(f"  {'变体':6s} {'单塔层':>9s} {'+embed/head':>12s} {'双塔总':>9s} "
      f"{'报告总参':>9s} {'比值':>6s}")
rows_A = []
for name, cfg in VARIANTS.items():
    per, lay = tower_params(cfg)
    _, _, two, emb, head = model_params(cfg)
    rows_A.append((name, lay, two, cfg["total"], cfg["dense"]))
    print(f"  {name:6s} {billions(lay):9.3f} {billions(emb + head):12.3f} "
          f"{billions(two):9.3f} {cfg['total']:9.1f} {two / 1e9 / cfg['total']:6.3f}")

nano_two = [r for r in rows_A if r[0] == "Nano"][0][2]
super_two = [r for r in rows_A if r[0] == "Super"][0][2]
edge_two = [r for r in rows_A if r[0] == "Edge"][0][2]
ok(0.92 < nano_two / 1e9 / 16.0 < 1.00,
   f"Nano 双塔裸算 {billions(nano_two):.2f}B，占报告 16B 的 {nano_two / 1e9 / 16:.1%}")
ok(0.97 < super_two / 1e9 / 64.0 < 1.02,
   f"Super 双塔裸算 {billions(super_two):.2f}B，占报告 64B 的 {super_two / 1e9 / 64:.1%}")
ok(1.00 < edge_two / 1e9 / 4.0 < 1.10,
   f"Edge 双塔（tie）裸算 {billions(edge_two):.2f}B，占报告 4B 的 {edge_two / 1e9 / 4:.1%}")

banner("A3 两套口径的换算关系")
for name, cfg in VARIANTS.items():
    print(f"  {name:6s} 报告总参 {cfg['total']:5.1f}B  = {cfg['total'] / cfg['dense']:.2f} x "
          f"稠密塔 {cfg['dense']:5.1f}B")
ok(all(abs(c["total"] / c["dense"] - 2.0) < 1e-9 for c in VARIANTS.values()),
   "三个变体的总参 / 稠密塔恒等于 2")
ok(abs(nano_two - 2 * 8.19e9) / nano_two < 0.10,
   f"双塔裸算 {billions(nano_two):.2f}B 与 2 x Qwen3-8B(8.19B) 相差 "
   f"{abs(nano_two - 2 * 8.19e9) / nano_two:.1%}")

gap_nano = 16.0e9 - nano_two
print(f"\n  未计入双塔的 {billions(gap_nano):.3f}B 是编码器与投影：ViT、视频 VAE 解码器、"
      f"音频 VAE、每域一套动作投影")


# ====================================================================== B 组
# %%
section("B. 计算账：稳态去噪时那半个塔贡献多少 FLOPs")
banner("B1 场景设定（480p / 189 帧 / 35 步）")
W_PX, H_PX, N_FRAME, N_STEP = 832, 480, 189, 35


def latent_tokens(w, h, n_frame, s=VAE_S, t=VAE_T):
    return int(np.ceil(w / s)) * int(np.ceil(h / s)) * int(np.ceil(n_frame / t))


n_dm = latent_tokens(W_PX, H_PX, N_FRAME)
n_cond_text, n_cond_img = 200, latent_tokens(W_PX, H_PX, 1)
n_cond = n_cond_text + n_cond_img
print(f"  扩散子序列 {n_dm:,} token；条件 {n_cond:,} token（文本 {n_cond_text} + 首帧 {n_cond_img}）")

P_tower = layers        # 单塔层参数；共享词表与 lm_head 不参与每步前向
print(f"  单塔层参数 {billions(P_tower):.3f}B（embedding 与 lm_head 不进每步前向）")

banner("B2 每步 FLOPs 与 reasoner 占比")
f_step_gen = 2 * P_tower * n_dm
f_prefill = 2 * P_tower * n_cond
f_all = f_prefill + N_STEP * f_step_gen
f_step_reason = 2 * P_tower * n_cond

print(f"  单步（只走生成塔）      {f_step_gen / 1e12:10.2f} TFLOPs")
print(f"  一次 prefill（推理塔）  {f_prefill / 1e12:10.3f} TFLOPs")
print(f"  整条轨迹（1 + {N_STEP} 步）   {f_all / 1e12:10.2f} TFLOPs")
print(f"  推理塔占整条轨迹        {f_prefill / f_all:10.4%}")
print(f"  若每步都重算推理塔      {N_STEP * f_step_reason / f_all:10.4%}  （相对省 "
      f"{1 - f_prefill / (N_STEP * f_step_reason):.1%}）")

ok(f_prefill / f_all < 0.01,
   f"推理塔在整条轨迹里的 FLOPs 占比 {f_prefill / f_all:.3%} 低于 1%")
ok(0.44 < P_tower / nano_two < 0.48,
   f"单塔层参数占双塔总参 {P_tower / nano_two:.1%}（另一塔同样多，其余是词表与编码器）")

banner("B3 参数占比 vs 算力占比")
share_param = 0.5
share_flop = f_prefill / f_all
print(f"  参数占比 {share_param:.1%}   ->  算力占比 {share_flop:.3%}")
print(f"  倍数差   {share_param / share_flop:.0f}x")
ok(share_param / share_flop > 100,
   f"参数占比与算力占比差 {share_param / share_flop:.0f} 倍，说明 16B 不是 16B 的活")

banner("B4 权重驻留是另一笔账")
w_total = 16.0e9 * 2          # BF16
print(f"  BF16 权重 {w_total / 1e9:.1f} GB；NVFP4 约 {w_total / 4 / 1e9:.1f} GB")
act = n_dm * VARIANTS["Nano"]["h"] * 2 * 2   # 粗估：两倍的隐藏状态往返
print(f"  单请求激活量级 {act / 1e9:.1f} GB（{n_dm:,} token x {VARIANTS['Nano']['h']} x 2 层缓冲 x 2B）")
ok(w_total / 1e9 > 30, "BF16 双塔权重超过 30 GB，工作站单卡必须做分片或量化")


# ====================================================================== C 组
# %%
section("C. token 账：74K 上下文在三种分辨率下各容几个样本")
RES_TABLE = {
    "256p": [(320, 192), (320, 256), (256, 256), (256, 320), (192, 320)],
    "480p": [(832, 480), (736, 544), (640, 640), (544, 736), (480, 832)],
    "720p": [(1280, 720), (1104, 832), (960, 960), (832, 1104), (720, 1280)],
}
N_FRAME_EVAL = 189


def max_batch(w, h, n_frame, ctx=CTX):
    t = latent_tokens(w, h, n_frame)
    return t, max(1, ctx // t)


banner("C1 三种分辨率 x 五个宽高比（189 帧）")
for tier, dims in RES_TABLE.items():
    out = []
    for (w, h) in dims[:3]:
        t, b = max_batch(w, h, N_FRAME_EVAL)
        out.append(f"{w}x{h}: {t:6,} tok, B<={b:2d}")
    print(f"  {tier:5s} " + " | ".join(out))

banner("C2 与报告 Table 9 的批次上限对账")
t256, b256 = max_batch(320, 192, N_FRAME_EVAL)
t480, b480 = max_batch(832, 480, N_FRAME_EVAL)
t720, b720 = max_batch(1280, 720, N_FRAME_EVAL)
print(f"  16:9  报告 B=6 (256p) / 3 (480p) / 1 (720p)")
print(f"        算得 B={b256} / {b480} / {b720}")
print(f"        单样本 token {t256:,} / {t480:,} / {t720:,}")
print(f"        B+1 的 token  {t480 * (b480 + 1):,} / {t720 * (b720 + 1):,}")
ok(b720 == 1, f"720p 算得 B={b720}，与报告 B=1 一致（单样本 {t720:,} token）")
ok(b480 == 3, f"480p 算得 B={b480}，与报告 B=3 一致（B=4 需 {t480 * 4:,} > {CTX:,}）")
ok(b256 > b480 > b720, "分辨率越高，上下文可容纳的样本越少（单调）")
ok(b256 != 6, f"256p 算得 B={b256} 而报告写 B=6：这一格不是 token 预算约束")

banner("C3 720p 的帧数天花板")
t720_400 = latent_tokens(1280, 720, 400)
print(f"  720p 400 帧 = {t720_400:,} token，超过预算 {t720_400 / CTX:.1f} 倍")
ok(t720_400 > CTX, "720p 400 帧放不进 74K，所以表 5 把 720p 的帧数上限压到 300")
t720_300 = latent_tokens(1280, 720, 300)
print(f"  720p 300 帧 = {t720_300:,} token，占预算 {t720_300 / CTX:.1%}")
ok(t720_300 < CTX, "720p 300 帧刚好放进 74K")


# ====================================================================== D 组
# %%
section("D. 位置账：FPS 调制就是 delta_t = 24 / FPS")
TPS_BASE = T_FPS_BASE / VAE_T


def dt_of(fps):
    return TPS_BASE / (fps / VAE_T)


banner("D1 三种帧率的步长")
for fps in (16, 24, 30):
    print(f"  {fps:2d} FPS -> TPS = {fps / VAE_T:4.1f}，delta_t = {dt_of(fps):.3f}（= 24/{fps} = {24 / fps:.3f}）")
ok(all(abs(dt_of(f) - 24.0 / f) < 1e-12 for f in (10, 16, 24, 30, 60)),
   "delta_t 与 24/FPS 逐位相等，五个帧率都成立")

banner("D2 等真实时长 -> 等位置跨度")
for sec in (1.0, 5.0):
    spans = []
    for fps in (16, 24, 30):
        n_lat = sec * fps / VAE_T
        spans.append(n_lat * dt_of(fps))
    print(f"  {sec:.0f} 秒：16/24/30 FPS 的位置跨度 = "
          + " / ".join(f"{s:.4f}" for s in spans) + f"（= {sec * TPS_BASE:.1f}）")
    ok(max(spans) - min(spans) < 1e-9, f"{sec:.0f} 秒在三种帧率下的位置跨度完全相等")

banner("D3 15000 的间隔折算成时间")
print(f"  15000 个位置单位 / TPS_base {TPS_BASE:.0f} = {15000 / TPS_BASE:.0f} 秒 = "
      f"{15000 / TPS_BASE / 60:.1f} 分钟")
ok(15000 / TPS_BASE > 1000,
   "这个间隔折算是 2500 秒，远超任何训练片段长度：它是位置缓冲，不是物理时间")
gap_units = 15000
one_frame = dt_of(24)
print(f"  对照：24 FPS 下一帧只占 {one_frame:.1f} 个位置单位，缓冲是它的 {gap_units / one_frame:.0f} 倍")
ok(gap_units / one_frame > 10000,
   "缓冲与单帧步长差四个数量级，等价于把文本与首帧的位置嵌入彻底拉开")


banner("D4 音频与视频的密度比")
TPS_VIDEO = T_FPS_BASE / VAE_T
TPS_AUDIO = 48000 / 1920
print(f"  视频 {TPS_VIDEO:.0f} tok/s，音频 {TPS_AUDIO:.0f} tok/s，密度比 {TPS_AUDIO / TPS_VIDEO:.4f}")
print(f"  步长：视频 {TPS_BASE / TPS_VIDEO:.3f}，音频 {TPS_BASE / TPS_AUDIO:.3f}")
print(f"  密度比 x 音频步长 = {(TPS_AUDIO / TPS_VIDEO) * (TPS_BASE / TPS_AUDIO):.4f}"
      f"  视频步长 = {TPS_BASE / TPS_VIDEO:.4f}")
ok(abs(TPS_AUDIO / TPS_VIDEO - 25.0 / 6.0) < 1e-12, "音频 25 tok/s 对视频 6 tok/s，密度比 4.1667")
ok(abs((TPS_AUDIO / TPS_VIDEO) * (TPS_BASE / TPS_AUDIO) - TPS_BASE / TPS_VIDEO) < 1e-12,
   "密度比与步长比互为倒数：两种模态在同一时间轴上占同样的长度")


# ====================================================================== E 组
# %%
section("E. 动作账：57/29/20/10/9 维从哪来（Fig. 3 的六个格）")
EGO, EFF, GRASP_HAND, GRASP_BIN = 9, 9, 15, 1
# 技术报告 Fig. 3 的分解规则：ego 位姿可选，每个末端是 effector 9 + grasp（15 或 1）
ACTION_DIMS = {
    "自动驾驶": EGO,                                  # 只有 ego
    "相机运动": EGO,                                  # 只有 ego
    "单臂": EFF + GRASP_BIN,                          # 1 个末端，无 ego
    "双臂": 2 * (EFF + GRASP_BIN),                    # 2 个末端，无 ego
    "人形": EGO + 2 * (EFF + GRASP_BIN),              # 多一个 ego
    "第一人称（双手）": EGO + 2 * (EFF + GRASP_HAND),   # 手用 15 维指尖
}
banner("E1 组件反算")
for k, v in ACTION_DIMS.items():
    print(f"  {k:12s} {v:3d} 维")
ok(ACTION_DIMS["第一人称（双手）"] == 57, "第一人称 9 + 2x(9+15) = 57")
ok(ACTION_DIMS["人形"] == 29, "人形 9 + 2x(9+1) = 29（比双臂多的 9 就是 ego）")
ok(ACTION_DIMS["双臂"] == 20, "双臂 2x(9+1) = 20")
ok(ACTION_DIMS["单臂"] == 10, "单臂 9 + 1 = 10")
ok(ACTION_DIMS["自动驾驶"] == 9 and ACTION_DIMS["相机运动"] == 9, "自驾与相机都只有 ego 9 维")
ok(ACTION_DIMS["人形"] - ACTION_DIMS["双臂"] == EGO,
   "人形与双臂的差恰好是一个 ego 位姿，说明这条分解规则自洽")

banner("E2 6D 旋转 -> SO(3)（Gram-Schmidt + 叉乘）")
def sixd_to_R(m):
    a1, a2 = m[:3], m[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - b1 * (b1 @ a2)
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


rng = np.random.default_rng(SEED)
errs, dets = [], []
for _ in range(2000):
    m = rng.normal(size=6)
    R = sixd_to_R(m)
    errs.append(np.abs(R.T @ R - np.eye(3)).max())
    dets.append(np.linalg.det(R))
print(f"  2000 个随机 6 维输入：正交误差 max {max(errs):.2e}，det 范围 "
      f"[{min(dets):.6f}, {max(dets):.6f}]")
ok(max(errs) < 1e-12, f"正交误差 {max(errs):.2e} 在浮点意义上为零")
ok(abs(min(dets) - 1) < 1e-12 and abs(max(dets) - 1) < 1e-12, "行列式恒为 +1（不是镜像）")

banner("E3 为什么不用 3D：轴角表示在单位旋转附近有两组相距 2*pi 的参数")
def rot_x(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


eps = 0.01
R_small = rot_x(eps)
R_small_alt = rot_x(eps - 2 * np.pi)      # 绕 -x 转 2*pi-eps，同一个旋转
vec_a = np.array([eps, 0.0, 0.0])
vec_b = np.array([-(2 * np.pi - eps), 0.0, 0.0])   # 轴取反、角取 2*pi-eps
print(f"  两个旋转矩阵的最大元素差 {np.abs(R_small - R_small_alt).max():.2e}（是同一个旋转）")
print(f"  对应的轴角参数距离 {np.linalg.norm(vec_a - vec_b):.4f}（约 2*pi = {2 * np.pi:.4f}）")
ok(np.abs(R_small - R_small_alt).max() < 1e-3, "两组参数确实描述同一个旋转")
ok(abs(np.linalg.norm(vec_a - vec_b) - 2 * np.pi) < 1e-6,
   "同一旋转的两组参数相距 2*pi：网络在原点附近无论选哪支，跨过就得跳")

print(f"  6D 侧：两个第二列反向的输入给出相差 180 度的第三列，但映射本身连续、无奇异点")
R1 = sixd_to_R(np.array([1.0, 0, 0, 0, 1.0, 0]))
R2 = sixd_to_R(np.array([1.0, 0, 0, 0, -1.0, 0]))
print(f"    最大元素差 {np.abs(R1 - R2).max():.3f}")
ok(np.abs(R1 - R2).max() > 1.0, "6D 允许第二列反向（过参数化），映射是双覆盖而非一一映射")
ok(abs(np.linalg.det(R1) - 1) < 1e-12, "过参数化不破坏右手性")

banner("E4 域内投影与 padding")
print(f"  框架文档：动作 schema 5 的 normalized_unified_v1 是 59 个数，补齐到 64")
print(f"  59 - 57（双臂双手的分解）= {59 - 57}，这两维报告未说明，只记为对齐量")
ok(59 >= ACTION_DIMS["第一人称（双手）"], "统一 schema 的 59 维覆盖第一人称的 57 维并多出 2 维")
ok(64 % 8 == 0, "补齐到 64 是 8 的整数倍，便于按 kernel 的向量宽度对齐")


# ====================================================================== F 组
# %%
section("F. 噪声账：rectified flow 的 shift 把质量挪到哪")
def sigma_of_t(t, s):
    tb = 1.0 - t
    return s * tb / (1.0 + (s - 1.0) * tb)


banner("F1 边界与恒等")
for s in (1.0, 3.0, 5.0, 10.0):
    print(f"  s={s:4.1f}: sigma(0)={sigma_of_t(0, s):.6f}  sigma(1)={sigma_of_t(1, s):.6f}")
ok(all(abs(sigma_of_t(0, s) - 1) < 1e-12 for s in (1, 3, 5, 10)), "t=0 时三种 shift 都给出 sigma=1")
ok(all(abs(sigma_of_t(1, s)) < 1e-12 for s in (1, 3, 5, 10)), "t=1 时都给出 sigma=0")
ok(all(abs(sigma_of_t(t, 1.0) - (1 - t)) < 1e-12 for t in np.linspace(0, 1, 11)),
   "s=1 时退化为恒等映射 sigma = 1 - t")

banner("F2 同一 t 下 shift 抬高了多少噪声")
for t in (0.25, 0.5, 0.75):
    row = [sigma_of_t(t, s) for s in (1, 3, 5, 10)]
    print(f"  t={t:.2f}: " + "  ".join(f"s={s}:{v:.3f}" for s, v in zip((1, 3, 5, 10), row))
          + f"   （s=10 是 s=1 的 {row[3] / row[0]:.2f} 倍）")
ok(sigma_of_t(0.75, 5.0) > 2 * sigma_of_t(0.75, 1.0),
   "s=5 在 t=0.75 处把噪声水平抬到 2.5 倍：shift 在低噪声端放大得最厉害")
ok(sigma_of_t(0.5, 5.0) > 1.6 * sigma_of_t(0.5, 1.0),
   "t=0.5 处只抬到 1.67 倍，说明 shift 不是等比例缩放，而是重排整条轨迹")

banner("F3 单调性与反函数")
ts = np.linspace(0, 1, 10001)
for s in (1.0, 3.0, 5.0, 10.0):
    sig = sigma_of_t(ts, s)
    d = np.diff(sig)
    print(f"  s={s:4.1f}: 单调下降 = {bool(np.all(d < 0))}，端点 {sig[0]:.4f} -> {sig[-1]:.4f}")
    assert np.all(d < 0)
ok(True, "四种 shift 的 sigma(t) 全部严格单调下降，可逆")

banner("F4 高噪声区间占了多少采样步")
for s in (1.0, 5.0, 10.0):
    ts_ = np.linspace(0, 1, 100001)
    frac = float(np.mean(sigma_of_t(ts_, s) > 0.8))
    print(f"  s={s:4.1f}: sigma > 0.8 的 t 占 {frac:.1%}")
ok(True, "记录 s 对高噪声区间占比的拉伸")


# ====================================================================== G 组
# %%
section("G. 迁移账：mid-init 相对 pre-init")
# 技术报告 Table 18（FD/ID 指标）与 Table 20（LIBERO-10 成功率）
T18 = {
    "自驾 ID RRE(deg)": (0.249, 0.211, "low"),
    "自驾 ID ATE(m)":   (1.20, 0.98, "low"),
    "相机 FD ATE(m)":   (1.61, 1.24, "low"),
    "第一人称 FD PSNR":  (15.22, 16.12, "high"),
    "机器人 FD PSNR":    (23.24, 25.52, "high"),
}
banner("G1 五个下游域（Nano，Table 18）")
for k, (pt, mt, dirn) in T18.items():
    if dirn == "high":
        rel = (mt - pt) / pt
        arrow = "+"
    else:
        rel = (pt - mt) / pt
        arrow = "-"
    print(f"  {k:18s} PT {pt:6.3f} -> MT {mt:6.3f}   改善 {arrow}{abs(rel):6.2%}")
    ok(abs(mt - pt) > 1e-6, f"{k} 两个初始化有可测差异")
ok((16.12 - 15.22) / 15.22 > 0.05, "第一人称 FD 的相对改善超过 5%")

banner("G2 LIBERO-10 的早停优势（Table 20）")
LIB = {500: (0.0, 24.6), 1000: (73.8, 91.4), 1500: (93.4, 95.8), 2000: (95.2, 97.4)}
for it, (pt, mt) in LIB.items():
    print(f"  {it:5d} 步  PT {pt:5.1f}%  MT {mt:5.1f}%  差 {mt - pt:+5.1f} pp")
ok(LIB[500][1] > 20 and LIB[500][0] == 0.0,
   "500 步时 MT 已到 24.6% 而 PT 仍是 0.0%：这不是渐进改善，是从零起步")

banner("G3 达到 50% 与 90% 需要多少步（对数插值）")
def steps_to(target):
    xs = np.array(sorted(LIB))
    ys = np.array([LIB[x][1] for x in xs])
    if target <= ys[0]:
        return float(xs[0])
    i = int(np.searchsorted(ys, target))
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    lx = np.log(x0) + (target - y0) / (y1 - y0) * (np.log(x1) - np.log(x0))
    return float(np.exp(lx))


def steps_to_pt(target):
    xs = np.array(sorted(LIB))
    ys = np.array([LIB[x][0] for x in xs])
    if target <= ys[0]:
        return float("inf")
    i = int(np.searchsorted(ys, target))
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    lx = np.log(x0) + (target - y0) / (y1 - y0) * (np.log(x1) - np.log(x0))
    return float(np.exp(lx))


s50_mt, s50_pt = steps_to(50.0), steps_to_pt(50.0)
s90_mt, s90_pt = steps_to(90.0), steps_to_pt(90.0)
print(f"  50%: MT {s50_mt:7.0f} 步 / PT {s50_pt:7.0f} 步 -> 快 {s50_pt / s50_mt:.2f}x")
print(f"  90%: MT {s90_mt:7.0f} 步 / PT {s90_pt:7.0f} 步 -> 快 {s90_pt / s90_mt:.2f}x")
ok(1.15 < s50_pt / s50_mt < 1.35, f"到 50% 的加速比 {s50_pt / s50_mt:.2f}x 落在 1.15-1.35")
ok(s90_pt / s90_mt > s50_pt / s50_mt,
   "高门槛处的迭代数优势反而更大：mid-init 不只买冷启动，还把后段收敛拉直了")

banner("G4 动作模式联合训练（Table 31）")
FD_SINGLE, FD_JOINT = 27.13, 26.22
ID_SINGLE, ID_JOINT = 1.11e-3, 3.09e-4
COV_SINGLE, COV_JOINT = 74.1, 77.3
print(f"  FD PSNR   {FD_SINGLE:6.2f} -> {FD_JOINT:6.2f}  ({(FD_JOINT - FD_SINGLE) / FD_SINGLE:+.2%})")
print(f"  ID MSE    {ID_SINGLE:.2e} -> {ID_JOINT:.2e}  ({(ID_JOINT - ID_SINGLE) / ID_SINGLE:+.1%})")
print(f"  policy 覆盖率 {COV_SINGLE:.1f}% -> {COV_JOINT:.1f}%  ({COV_JOINT - COV_SINGLE:+.1f} pp)")
ok((ID_JOINT - ID_SINGLE) / ID_SINGLE < -0.6, "联合训练把 ID 的 MSE 砍掉六成以上")
ok(abs((FD_JOINT - FD_SINGLE) / FD_SINGLE) < 0.05, "代价是 FD 的 PSNR 掉了不到 5%")


# ====================================================================== H 组
# %%
section("H. 协同账：跨域动作数据的迁移矩阵不对称")
DOMAINS = ["AV", "Cam", "GR", "WX", "F1", "F2"]
# 技术报告 Fig. 28（FD PSNR）与 Fig. 29（ID MSE）的行向差值 Delta
DELTA_FD = np.array([
    [0.00, -0.05, -0.16, -0.19, 0.06, -0.05],
    [0.86, 0.00, 0.79, 0.19, 0.73, 0.48],
    [0.91, 0.94, 0.00, 0.39, 0.81, 0.89],
    [0.90, 0.45, 1.39, 0.00, 0.78, 0.78],
    [0.00, -0.01, 0.79, -0.02, 0.00, 0.58],
    [0.00, -0.05, 0.51, -0.03, 0.15, 0.00],
])
BASE_FD = np.array([17.87, 11.96, 19.70, 15.63, 21.07, 21.37])
DELTA_ID = np.array([
    [0.000, -0.010, -0.011, -0.010, -0.010, -0.008],
    [-0.012, 0.000, -0.043, -0.043, -0.047, -0.032],
    [-0.005, -0.005, 0.000, -0.004, -0.005, -0.007],
    [-0.005, -0.003, -0.006, 0.000, -0.005, -0.006],
    [-0.003, -0.003, -0.004, -0.003, 0.000, -0.004],
    [-0.001, -0.001, -0.001, -0.001, -0.001, 0.000],
])

banner("H1 结构与对角")
print(f"  FD 非对角均值 {DELTA_FD[~np.eye(6, dtype=bool)].mean():+.3f} dB，"
      f"最大值 {DELTA_FD.max():+.2f}，最小值 {DELTA_FD.min():+.2f}")
print(f"  ID 非对角均值 {DELTA_ID[~np.eye(6, dtype=bool)].mean():+.4f}（负值即改善）")
ok(abs(np.trace(DELTA_FD)) < 1e-12, "FD 矩阵对角线为零（对角即单域基线）")
ok(abs(np.trace(DELTA_ID)) < 1e-12, "ID 矩阵对角线为零")
ok(np.abs(DELTA_FD - DELTA_FD.T).max() > 0.5, "FD 矩阵显著不对称：迁移有方向")

banner("H2 按「加入的域」聚合（列）")
print(f"  {'目标域':6s} " + "  ".join(f"{d:>6s}" for d in DOMAINS))
for i, d in enumerate(DOMAINS):
    print(f"  {d:6s} " + "  ".join(f"{DELTA_FD[i, j]:+6.2f}" for j in range(6)))
col = DELTA_FD.sum(axis=0) / 5.0
row = DELTA_FD.sum(axis=1) / 5.0
for j, d in enumerate(DOMAINS):
    print(f"  加入 {DOMAINS[j]:3s} 的平均收益 {col[j]:+.3f} dB   "
          f"（它自己当目标域时的平均获益 {row[j]:+.3f} dB）")
ok(row[0] < 0, "自动驾驶是唯一一个「加了别的域反而变差」的目标域")
ok(col[0] > 0.5, "但自驾数据本身是强辅助域：平均带来 +0.53 dB")
ok(DELTA_FD[2, 0] > 0 and DELTA_FD[0, 2] < 0,
   "同一对域：自驾数据帮机器人 +0.91 dB，机器人数据帮自驾 -0.16 dB")
ok(row[1] > 0.5, "基线最低的相机运动域（11.96 dB）从几乎任何辅助域都获益")

banner("H3 谁从谁受益最多")
best = np.unravel_index(np.argmax(DELTA_FD), DELTA_FD.shape)
print(f"  最大正迁移：{DOMAINS[best[0]]} 域加入 {DOMAINS[best[1]]} 数据 -> "
      f"{DELTA_FD[best]:+.2f} dB（基线 {BASE_FD[best[0]]:.2f}）")
worst = np.unravel_index(np.argmin(DELTA_FD), DELTA_FD.shape)
print(f"  最大负迁移：{DOMAINS[worst[0]]} 域加入 {DOMAINS[worst[1]]} 数据 -> "
      f"{DELTA_FD[worst]:+.2f} dB")
ok(DELTA_FD[best] > 1.0, f"最强一格 {DELTA_FD[best]:+.2f} dB 超过 1 dB")

banner("H4 ID 侧的改善是普遍的吗")
neg_share = float(np.mean(DELTA_ID[~np.eye(6, dtype=bool)] < 0))
print(f"  ID 的 30 个非对角格里，改善（Delta<0）占 {neg_share:.1%}")
ok(neg_share > 0.95, "逆动力学侧的跨域迁移几乎全部为正（与 FD 侧不同）")
ok(DELTA_ID[1, :].min() < -0.02, "相机运动行在 ID 侧拿到最大的绝对改善（-0.047）")


# ====================================================================== 汇总
# %%
section("汇总")
print(f"  断言总数 {N_ASSERT}")
print("  A 参数账：16B = 2 x 6.95B 层 + 0.62B 共享词表 + lm_head 与编码器")
print("  B 计算账：稳态去噪里推理塔只占 0.09% 的 FLOPs，但参数占 45.9%")
print("  C token 账：74K 在 480p 卡到 B=3、720p 卡到 B=1，256p 不是 token 约束")
print("  D 位置账：delta_t = 24/FPS，15000 是位置缓冲，折算是 2500 秒")
print("  E 动作账：57/29/20/10/9 全部可由组件反算，6D 用双覆盖换连续性")
print("  F 噪声账：shift 单调可逆，越靠低噪声端放大越猛（t=0.75 处 2.5x）")
print("  G 迁移账：500 步 24.6% vs 0.0% 是冷启动；折算迭代数只省 18% 与 29%")
print("  H 协同账：FD 侧迁移不对称（自驾只给不拿），ID 侧 30 格全为正向")
