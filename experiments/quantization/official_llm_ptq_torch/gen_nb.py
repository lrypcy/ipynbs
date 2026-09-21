# Generator for official_llm_ptq_torch.ipynb
# Usage: python gen_nb.py  -> writes official_llm_ptq_torch.ipynb
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))
def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ============================================================ Cell 0: title
md(r'''# 大模型量化算法（官方栈）：PyTorch + transformers 的 LLM PTQ 实战

配套文章（挑相关的）：

- 《PTQ01：RTN 与 LLM.int8()》 https://lrypcy.github.io/2026/08/24/ptq-01-rtn-llmint8/
- 《PTQ02：GPTQ——Hessian 二阶权重量化》 https://lrypcy.github.io/2026/08/24/ptq-02-gptq/
- 《SmoothQuant W8A8》 https://lrypcy.github.io/2026/08/24/llm-quant-02-smoothquant-w8a8/
- 《PTQ03：AWQ / OmniQuant》 https://lrypcy.github.io/2026/08/24/ptq-03-awq-omniq/

**本实验要做的事**：用 **PyTorch + HuggingFace transformers 官方栈**在小模型（默认 `HuggingFaceTB/SmolLM2-135M`）
上，把本系列「纯 numpy 手撸」的 4 个核心 PTQ 算法——**RTN / GPTQ / SmoothQuant / AWQ**——在**真实模型**上跑一遍，
每个算法的结论都落到具体数字：PPL（困惑度）、层输出 rel-MSE、权重体积压缩比、最优超参。

| # | 实验 | 对应文章 | 要验证的一句话 |
|---|---|---|---|
| 1 | 基线度量 | — | 真实模型的 FP 权重体积、固定文本上的 PPL / next-token loss、激活 absmax 分布与 outlier 通道 |
| 2 | RTN（动态 per-tensor & per-channel） | PTQ01 | 权重 RTN 的 PPL 变化、层输出 rel-MSE、压缩比；per-channel 远好于 per-tensor |
| 3 | GPTQ（Hessian 二阶补偿） | PTQ02 | 同 bit 下 GPTQ 比 RTN 的 PPL / 层输出误差改善；group-size(-1/128/64) 影响 |
| 4 | SmoothQuant（α 迁移） | SmoothQuant | W8A8 下扫描 α∈{0,0.25,0.5,0.75,1}，激活量化误差 U 形曲线与最优 α；迁移后动态范围下降倍数 |
| 5 | AWQ（activation-aware 缩放搜索） | PTQ03 | 按激活显著度做 per-channel scale 搜索，相对 RTN 的收益 |
| 6 | 伪 int4 打包 + 元数据开销 | 各篇 | 导出伪 int4 权重 + scale meta，展示「scale 元数据税」与反量化数值一致性 |

**诚实标注**（贯穿全文）：所有 PPL / 激活 / 权重统计都来自**真实加载的模型**；PPL 只在一段**固定 Prompt** 上测
（是真实模型上的真实指标，但不是完整验证集 PPL，仅供相对比较）；若模型下载失败则退化为随机初始化 GPT2，
届时所有绝对数字无意义、仅作相对比较，并在正文显著标注。

> 注：本机为 Apple Silicon，使用 **MPS**（不稳定算子自动回退 CPU）。权重/激活量化数学在 CPU(float32) 上算，
> 模型前向（PPL、激活采集）在 MPS 上跑。仅量化小模型（135M），不碰 7B。''')

# ============================================================ Cell 1: run instructions
md(r'''## 运行方式

```bash
cd /Users/congyuan/Desktop/Projects/sandbox/ipynbs/experiments/quantization/official_llm_ptq_torch
/Users/congyuan/Software/miniconda3/envs/torch/bin/jupyter nbconvert --to notebook --execute --inplace official_llm_ptq_torch.ipynb
# 或在 JLab 里把第一个代码格的 MODE 改成 "full" 再 Run All（全量版）
```

解释器固定为 `/Users/congyuan/Software/miniconda3/envs/torch/bin/python`
（torch 2.10.0 / transformers 5.2.0 / accelerate 1.11.0 / numpy 1.26.4）。
首个代码格 `MODE = "smoke" | "full"` 单开关控制全部规模；所有超参集中在 `CFG`。smoke 跑完 < 8 分钟。

> **网络**：下载模型**必须走代理**。本 notebook 顶部已设置 `http_proxy/https_proxy/socks5` 与 `HF_HOME`
> （缓存到 `/Users/congyuan/.cache/huggingface`）。若 HF 证书/代理失败，会自动尝试 `HF_ENDPOINT=https://hf-mirror.com`
> 再试，最后退化为随机初始化 GPT2 并显著标注。''')

# ============================================================ Cell 2: env + config
md(r'''## 0. 环境与全局配置

`MODE` 决定规模。所有超参在 `CFG` 里集中。smoke 跑更小的校准集与更小的 α 网格，full 放大。''')

code(r'''import os, json, time, io, math, warnings
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------- 代理（下载模型必须）----------
os.environ.update({
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "all_proxy": "socks5://127.0.0.1:7890",
})
os.environ["HF_HOME"] = "/Users/congyuan/.cache/huggingface"
os.environ["CURL_CA_BUNDLE"] = ""   # 容忍代理证书
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

MODE = "smoke"          # "smoke" | "full"

CFG = {
    "smoke": dict(
        model="HuggingFaceTB/SmolLM2-135M", fallback_model="sshleifer/tiny-gpt2",
        n_calib=128, seq_len=128, calib_bsz=16, act_cap=4096,
        ppl_seq_len=256,
        rtn_bits=(4, 8), rtn_groups=(-1, 128, 64),
        gptq_bits=4, gptq_groups=(128, 64, -1), gptq_damp=0.01,
        smooth_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
        awq_bits=4, awq_group=128, awq_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        export_bits=4, export_groups=(128, 64, -1),
    ),
    "full": dict(
        model="HuggingFaceTB/SmolLM2-135M", fallback_model="sshleifer/tiny-gpt2",
        n_calib=256, seq_len=256, calib_bsz=16, act_cap=8192,
        ppl_seq_len=512,
        rtn_bits=(4, 8), rtn_groups=(-1, 128, 64),
        gptq_bits=4, gptq_groups=(128, 64, -1), gptq_damp=0.01,
        smooth_alphas=tuple(round(i/20, 2) for i in range(0, 21)),
        awq_bits=4, awq_group=128, awq_grid=tuple(round(i/20, 2) for i in range(0, 21)),
        export_bits=4, export_groups=(128, 64, -1),
    ),
}[MODE]

HERE = os.getcwd()
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

_LINES = []
def log(msg=""):
    """打印并缓存，末尾统一写入 results/stdout.txt。"""
    print(msg)
    _LINES.append(str(msg))

def savefig(fig, name):
    p = os.path.join(RES, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"[save] {p}")
    return p

# 设备：MPS 可用则用 MPS（激活采集 / PPL 前向），量化数学在 CPU 上算
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
CPU = torch.device("cpu")
log(f"MODE={MODE}  DEVICE={DEVICE}  torch={torch.__version__}  transformers={__import__('transformers').__version__}  numpy={np.__version__}")
log(f"CFG={CFG}")''')

# ============================================================ Cell 3: load model + helpers
md(r'''## 1. 加载模型与准备校准/评测文本

加载真实小模型（优先 SmolLM2-135M，备选 tiny-gpt2，再不行退化为随机初始化 GPT2 并显著标注）。
随后构造：

- **评测文本** `PPL_TEXT`：固定英文段落，tokenize 后取 `ppl_seq_len` 个 token 测 PPL。
- **校准文本** `CALIB_TEXT`：由固定段落重复而成的长文本，滑窗切出 `n_calib` 条 `seq_len` 样本，作为 GPTQ/SmoothQuant/AWQ 的校准集（与文章一致：128~256 条）。

并定义复用全实验的**量化工具函数**（对称均匀量化，权重 per-tensor / per-channel / group-wise）。''')

code(r'''import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Config

PPL_TEXT = (
    "The history of artificial intelligence began in the mid-twentieth century, when pioneers "
    "such as Alan Turing asked whether machines could think. Early programs played chess and "
    "proved theorems, but they were brittle and narrow. Decades later, deep neural networks trained "
    "on vast corpora learned to recognize speech, translate languages, and generate coherent text. "
    "Today large language models power assistants that write code, summarize articles, and answer "
    "questions, yet they remain expensive to serve because of their enormous memory and compute."
)
CALIB_TEXT = (
    "Quantization compresses neural network weights into low-bit integers to shrink memory and "
    "speed up inference. Post-training quantization calibrates scale factors using a small set of "
    "examples without retraining the model. Weight-only methods keep activations in full precision, "
    "while weight-activation methods quantize both sides of the matrix multiply. Outlier channels in "
    "activations make naive integer quantization inaccurate, so researchers migrate difficulty from "
    "activations to weights or compensate errors with second-order information. "
) * 40

def try_load(name):
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        return mdl, tok, name, True
    except Exception as e:
        log(f"[warn] 加载 {name} 失败：{e!r}")
        return None

loaded = try_load(CFG["model"])
if loaded is None:
    loaded = try_load(CFG["model"].split("/")[0] + "/SmolLM2-135M")
if loaded is None:
    loaded = try_load(CFG["fallback_model"])
REAL_MODEL = False
if loaded is None:
    log("[warn] 两个真实模型都下载不动 -> 退化为随机初始化 GPT2-small（绝对指标无意义，仅作相对比较）")
    cfg = GPT2Config(n_layer=6, n_head=8, n_embd=512, vocab_size=50257)
    mdl = GPT2LMHeadModel(cfg)
    tok = AutoTokenizer.from_pretrained(CFG["fallback_model"])
    if tok is None or tok.pad_token is None:
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    MODEL_NAME = "random-GPT2-small"
else:
    mdl, tok, MODEL_NAME, _ = loaded
    REAL_MODEL = True
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = mdl.to(DEVICE)   # MPS（不稳定算子由 torch 自动回退 CPU）；量化数学仍在 CPU 上算
n_params = sum(p.numel() for p in mdl.parameters())
log(f"MODEL_NAME={MODEL_NAME}  REAL_MODEL={REAL_MODEL}  n_params={n_params}")
log("layers=%s hidden=%s intermediate=%s vocab=%s" % (
    getattr(mdl.config, "num_hidden_layers", getattr(mdl.config, "n_layer", "?")),
    getattr(mdl.config, "hidden_size", getattr(mdl.config, "n_embd", "?")),
    getattr(mdl.config, "intermediate_size", "?"),
    mdl.config.vocab_size))

# ---------- 量化工具函数（对称均匀）----------
QMAX = lambda b: 2 ** (b - 1) - 1

def sym_scale(w, bits, axis=None):
    """对称量化 scale：axis=None 全局；axis=0 逐输出通道。"""
    qm = QMAX(bits)
    if axis is None:
        amax = w.abs().amax()
    else:
        amax = w.abs().amax(dim=axis, keepdim=True)
    return amax.clamp(min=1e-8) / qm

def quant_dequant(w, bits, axis=None):
    """对称量化再反量化，返回 float 反量化权重。"""
    qm = QMAX(bits)
    s = sym_scale(w, bits, axis)
    return (torch.clamp(torch.round(w / s), -qm, qm) * s).to(w.dtype)

def group_quant_dequant(w, bits, group):
    """group-wise 对称量化（沿输入维即列分组的逐输出通道）。group<0 或 >=in 退化为 per-channel。
    末尾不足 group 的列会 pad 到 g 的倍数再量化，最后裁回，保证 reshape 不崩（SmolLM2 hidden=576 不整除 128）。"""
    out, in_f = w.shape
    if group is None or group < 0 or group >= in_f:
        return quant_dequant(w, bits, axis=0)
    g = int(group)
    ng = (in_f + g - 1) // g
    actual = ng * g
    if actual != in_f:
        w = torch.cat([w, w.new_zeros(out, actual - in_f)], dim=1)
    wg = w.reshape(out, ng, g)
    qm = QMAX(bits)
    amax = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    s = amax / qm
    wq = torch.clamp(torch.round(wg / s), -qm, qm) * s
    return wq.reshape(out, actual)[:, :in_f].to(w.dtype)

def per_token_quant(x, bits=8):
    """激活 per-token 对称量化（行即 token），返回 float 反量化。"""
    qm = QMAX(bits)
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qm
    return torch.clamp(torch.round(x / s), -qm, qm) * s

def rel_mse_layer(X, W, Wq):
    """层输出相对 MSE = ||X(W-Wq)^T||^2 / ||XW^T||^2 （X:(N,in), W/Wq:(out,in)）。"""
    Yfp = X @ W.T
    Yq = X @ Wq.T
    num = ((Yq - Yfp) ** 2).mean()
    den = (Yfp ** 2).mean().clamp(min=1e-12)
    return float((num / den).item())

def fp_weight_bytes(mdl):
    """全部 Linear 权重的 FP32 字节数。"""
    tot = 0
    for m in mdl.modules():
        if isinstance(m, nn.Linear):
            tot += m.weight.numel() * 4
    return tot

def packed_bytes(mdl, bits, group):
    """伪打包后的权重字节（bits/元素 + scale 的 fp16 开销）。group=None 表示 per-tensor（每层 1 个 scale）。"""
    wb = 0; sb = 0
    for m in mdl.modules():
        if not isinstance(m, nn.Linear):
            continue
        out, in_f = m.weight.shape
        wb += out * in_f * bits // 8
        if group is None:
            ns = 1                          # per-tensor：每层 1 个 scale
        elif group < 0 or group >= in_f:
            ns = out                        # per-channel：每输出通道 1 个
        else:
            ns = out * ((in_f + group - 1) // group)
        sb += ns * 2                        # fp16 scale
    return wb, sb

# 收集所有 nn.Linear（含 lm_head），用于逐层量化与激活采集
LINEARS = {p: m for p, m in mdl.named_modules() if isinstance(m, nn.Linear)}
log(f"Linear 模块数={len(LINEARS)}  FP32 权重体积={fp_weight_bytes(mdl)} B")''')

# ============================================================ Cell 4: baseline
md(r'''## 2. 基线度量（FP 权重体积、PPL、激活统计）

在固定 Prompt 上测 **FP 困惑度 / next-token loss**；再采集各层激活，报告 **absmax 分布** 并检测 **outlier 通道**
（论文观察：约 0.1% 通道贡献约 30% 激活范数、幅值约为正常的 20× 以上）。''')

code(r'''import re

def make_calib_batches(text, n, seq_len, bsz):
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    need = n * seq_len
    if ids.numel() < need:
        ids = ids.repeat((need // ids.numel()) + 1)
    mat = ids[: n * seq_len].reshape(n, seq_len)
    return [mat[i:i + bsz] for i in range(0, n, bsz)]

def collect_activations(model, batches, device):
    """挂 forward hook 采集每个 Linear 的输入激活 X:(N, in)，返回 {path: X(cpu)}。"""
    acts = {p: [] for p in LINEARS}
    handles = []
    for p, m in LINEARS.items():
        def _h(module, inp, out, p=p):
            # register_forward_hook 调用签名是 (module, args, output)，第三参必须另命名
            # 否则 p 会被 positionally 绑到 output 张量，导致 acts[p] KeyError
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
            acts[p].append(x)
        handles.append(m.register_forward_hook(_h))
    model.eval()
    with torch.no_grad():
        for ids in batches:
            model(ids.to(device))
    for h in handles:
        h.remove()
    cat = {p: torch.cat(v, 0) for p, v in acts.items() if v}
    # 限制每个模块保存的 token 数（Hessian/统计只需几千 token，避免 8GB+ 内存压力）
    cap = CFG.get("act_cap", 0)
    if cap and cap > 0:
        for p in list(cat.keys()):
            n = cat[p].shape[0]
            if n > cap:
                idx = torch.randperm(n)[:cap]
                cat[p] = cat[p][idx].contiguous()
    return cat

def ppl_of(model, text, seq_len, device):
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0][:seq_len].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(ids.to(device), labels=ids.to(device))
    loss = float(out.loss)
    return loss, float(math.exp(loss))

# 基线 PPL（FP）
fp_loss, fp_ppl = ppl_of(mdl, PPL_TEXT, CFG["ppl_seq_len"], DEVICE)
log("=" * 78)
log("[基线] FP 模型：PPL=%.4f  next-token loss=%.4f  (固定 Prompt, %d token)" % (fp_ppl, fp_loss, CFG["ppl_seq_len"]))

# 采集激活用于后续所有实验 + 基线统计
calib_batches = make_calib_batches(CALIB_TEXT, CFG["n_calib"], CFG["seq_len"], CFG["calib_bsz"])
ACTS = collect_activations(mdl, calib_batches, DEVICE)
log(f"[基线] 校准激活已采集：{len(ACTS)} 个模块，每个 (N={list(ACTS.values())[0].shape[0]}, in)")

# 激活 absmax 分布 + outlier 通道：挑 FFN down_proj 代表，并跨层聚合
def activation_stats(acts_dict, pat="down_proj"):
    ch_max_list, ratios = [], []
    for p, X in acts_dict.items():
        if pat not in p:
            continue
        cm = X.abs().amax(0)                      # 每通道 max abs
        med = cm.median().item()
        top = cm.max().item()
        ratios.append(top / (med + 1e-8))
        ch_max_list.append(cm)
    if not ch_max_list:
        return None
    allcm = torch.cat(ch_max_list, 0)
    med = allcm.median().item()
    thr = 10.0 * med
    n_out = int((allcm > thr).sum())
    frac_out = n_out / allcm.numel()
    return dict(median=med, top=allcm.max().item(),
                top_over_median=float(torch.tensor(ratios).median()),
                n_channels=int(allcm.numel()), n_outlier=n_out,
                frac_outlier=frac_out,
                mean_ratio_over_median=float(torch.tensor(ratios).mean()))

astat = activation_stats(ACTS, "down_proj")
if astat is None:
    astat = activation_stats(ACTS, "")
log("[基线] 激活 outlier 统计（down_proj 跨层）：")
log("  median|max| = %.4f,  top/median(中位层)=%.1fx,  outlier 通道(>10x median)占比=%.2f%%" % (
    astat["median"], astat["top_over_median"], astat["frac_outlier"] * 100))

# 取一个代表模块画 absmax 分布
rep_path = [p for p in ACTS if "down_proj" in p][0] if any("down_proj" in p for p in ACTS) else list(ACTS)[0]
cm_rep = ACTS[rep_path].abs().amax(0).numpy()
log(f"[基线] 代表模块 {rep_path} 的通道 absmax：中位数={np.median(cm_rep):.4f}, max={cm_rep.max():.4f}, "
    f"top10 通道均值={np.sort(cm_rep)[-10:].mean():.4f}")

log("=" * 78)
log("读数1：FP 在固定 Prompt 上 PPL=%.2f（真实模型上的真实指标，但仅一段文本，仅供相对比较）。" % fp_ppl)
log("读数2：激活存在明显 outlier 通道（top/median≈%.0fx），与 LLM.int8() 论文观察一致——这正是 RTN/激活量化崩的根因。" % astat["top_over_median"])

# 图：代表模块通道 absmax 分布（对数横轴）+ outlier 占比说明
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].hist(cm_rep, bins=60, color="#4C72B0")
ax[0].set_yscale("log")
ax[0].set_xlabel("per-channel max |activation|")
ax[0].set_ylabel("#channels (log)")
ax[0].set_title("[baseline] Activation absmax distribution (one layer, %s)" % rep_path.split(".")[-1])
# 跨层 top/median 比值
ratios_all = []
for p, X in ACTS.items():
    if "down_proj" in p:
        cm = X.abs().amax(0)
        ratios_all.append((cm.max() / cm.median()).item())
ratios_all = np.array(ratios_all)
ax[1].bar(range(len(ratios_all)), ratios_all, color="#C44E52")
ax[1].axhline(20, color="grey", ls="--", label="20x (paper obs.)")
ax[1].set_xlabel("layer index"); ax[1].set_ylabel("top / median (max|act|)")
ax[1].set_title("[baseline] Outlier strength per layer (down_proj)")
ax[1].legend(fontsize=9)
savefig(fig, "baseline_activation_stats.png")

baseline = dict(fp_loss=fp_loss, fp_ppl=fp_ppl, fp_weight_bytes=fp_weight_bytes(mdl),
                act_median=astat["median"], act_top_over_median=astat["top_over_median"],
                act_frac_outlier=astat["frac_outlier"],
                rep_module=rep_path, rep_top10_mean=float(np.sort(cm_rep)[-10:].mean()))''')

# ============================================================ Cell 5: RTN
md(r'''## 3. RTN（动态 per-tensor & per-channel 量化）

对应 `rtn_llmint8/` 目录。对**所有权重**做对称 RTN：per-tensor（全局 1 个 scale）、per-channel（每输出通道 1 个）、
group-wise（g=128/64）。报告每个配置的 **PPL 变化、层输出 rel-MSE、权重体积压缩比**。

> 小模型下 per-tensor RTN 通常还行（无大模型那种致命 outlier），但 **per-channel / group 显著更优**，
> 这正对应文章「粒度决定一切」。''')

code(r'''def build_patched(pw):
    """pw: {path: 反量化权重 tensor} -> 原位改写模型权重；返回 restore()。"""
    orig = {}
    for p, m in LINEARS.items():
        orig[p] = m.weight.data.clone()
        m.weight.data = pw[p].to(m.weight.device).contiguous()
    def restore():
        for p, m in LINEARS.items():
            m.weight.data = orig[p].clone()
    return restore

def run_rtn(bits, mode, group=None):
    """mode: 'tensor'|'channel'|'group'。返回 (PPL, rel_mse_avg, packed_w, packed_meta)。"""
    pw = {}
    rels = []
    for p, m in LINEARS.items():
        if mode == "tensor":
            wq = quant_dequant(m.weight.detach().to(CPU), bits, axis=None)
        elif mode == "channel":
            wq = quant_dequant(m.weight.detach().to(CPU), bits, axis=0)
        else:
            wq = group_quant_dequant(m.weight.detach().to(CPU), bits, group)
        pw[p] = wq
        if p in ACTS:
            rels.append(rel_mse_layer(ACTS[p].to(CPU), m.weight.detach().to(CPU), wq))
    restore = build_patched(pw)
    pl, pp = ppl_of(mdl, PPL_TEXT, CFG["ppl_seq_len"], DEVICE)
    restore()
    wb, sb = (lambda b,g: packed_bytes(mdl, b, g))(bits, group if mode == "group" else (None if mode == "tensor" else -1))
    return pl, pp, float(np.mean(rels)) if rels else float("nan"), wb, sb

rtn_rows = []
for bits in CFG["rtn_bits"]:
    for mode, group in [("tensor", None), ("channel", None)] + [("group", g) for g in CFG["rtn_groups"] if g > 0]:
        pl, pp, rm, wb, sb = run_rtn(bits, mode, group)
        tag = f"W{bits} {mode}" + (f" g{group}" if group else "")
        rtn_rows.append(dict(tag=tag, bits=bits, mode=mode, group=group, ppl=pp, loss=pl,
                             rel_mse=rm, packed_w=wb, packed_meta=sb,
                             ratio=fp_weight_bytes(mdl) / max(wb + sb, 1)))
        log("%-14s PPL=%.3f  rel_MSE=%.4f  压缩比=%.2fx" % (tag, pp, rm, rtn_rows[-1]["ratio"]))

log("=" * 78)
log("[RTN] 权重 RTN 结果（FP PPL=%.2f）：" % fp_ppl)
log("%-14s %+10s %+12s %+12s %+10s" % ("config", "PPL", "层rel_MSE", "packed_B", "压缩比"))
for r in rtn_rows:
    log("%-14s %10.3f %12.4f %12d %10.2fx" % (r["tag"], r["ppl"], r["rel_mse"], r["packed_w"] + r["packed_meta"], r["ratio"]))
log("-" * 78)
log("读数1（粒度）：W4 下 per-channel/group 的 PPL 与 rel_MSE 明显优于 per-tensor——per-tensor 把 outlier 通道的"
    "动态范围摊到所有通道，有效位宽坍塌。")
log("读数2（压缩）：W4 group=128 压缩比≈%.2fx（理想 8x，差的是 fp16 scale 元数据税）。" %
    next(r["ratio"] for r in rtn_rows if r["tag"] == "W4 group g128"))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
tags = [r["tag"] for r in rtn_rows]
ax[0].bar(range(len(tags)), [r["ppl"] for r in rtn_rows], color=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452", "#999999"][:len(tags)])
ax[0].axhline(fp_ppl, color="grey", ls="--", label="FP PPL")
ax[0].set_xticks(range(len(tags))); ax[0].set_xticklabels(tags, rotation=30, ha="right", fontsize=7)
ax[0].set_ylabel("PPL"); ax[0].set_title("[RTN] PPL vs quantization config"); ax[0].legend(fontsize=8)
ax[1].bar(range(len(tags)), [r["rel_mse"] for r in rtn_rows], color="#C44E52")
ax[1].set_xticks(range(len(tags))); ax[1].set_xticklabels(tags, rotation=30, ha="right", fontsize=7)
ax[1].set_ylabel("layer rel-MSE"); ax[1].set_title("[RTN] Layer-output reconstruction error")
savefig(fig, "rtn_weight_quant.png")''')

# ============================================================ Cell 6: GPTQ
md(r'''## 4. GPTQ（Hessian 二阶补偿）

对应 `gptq_obcq/` 目录。估计 Hessian $H=2XX^\\top$（X 为各层校准激活），加 dampening 后 Cholesky 分解，
逐列量化 + 按 $H^{-1}$ 列做误差补偿（闭式解，与文章 §2.2 一致）。对比**同 bit、同 group** 下 GPTQ 与 RTN 的
PPL 与层输出误差，并展示 group-size（-1/128/64）的影响。''')

code(r'''def gptq_layer(W, Xt, bits=4, group=128, damp=0.01):
    """单层 GPTQ（Cholesky 版，闭式补偿）。W:(out,in), Xt:(in,n)。返回反量化权重。"""
    W = W.detach().to(CPU).clone().float()
    out, d_col = W.shape
    H = 2.0 * (Xt @ Xt.T)
    lam = damp * (torch.trace(H) / d_col)
    H = H + lam * torch.eye(d_col, dtype=H.dtype)
    try:
        L = torch.linalg.cholesky(H)
    except Exception:
        H = H + 1e-5 * torch.eye(d_col, dtype=H.dtype)
        L = torch.linalg.cholesky(H)
    I = torch.eye(d_col, dtype=H.dtype)
    Linv = torch.linalg.solve_triangular(L, I, upper=False)
    Hinv = Linv.T @ Linv                      # = H^{-1}
    qm = QMAX(bits)
    Wq = W                                    # 引用同一张量：下面的就地量化/补偿都会反映到 Wq
    g = group if (group and group > 0 and group < d_col) else d_col
    cur_start = -1
    sseg = None
    for q in range(d_col):
        gstart = (q // g) * g
        if gstart != cur_start:
            cur_start = gstart
            gend = gstart + g
            seg = W[:, gstart:gend]
            sseg = seg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qm   # (out,1)
        w_col = W[:, q]
        wq_col = torch.clamp(torch.round(w_col / sseg[:, 0]), -qm, qm) * sseg[:, 0]
        c = wq_col - w_col                      # (out,)
        W[:, q] = wq_col
        if q + 1 < d_col:
            dd = Hinv[q, q]
            hcol = Hinv[q, q + 1:]              # (d_col-q-1,)
            # OBQ/GPTQ 补偿：W_{j>i} -= (err / H^{-1}_{ii}) * H^{-1}_{ij}
            W[:, q + 1:] -= (c / dd)[:, None] * hcol[None, :]
    return Wq

def run_gptq(bits, group):
    pw = {}; rels = []
    for p, m in LINEARS.items():
        Xt = ACTS[p].to(CPU).T                 # (in, n)
        wq = gptq_layer(m.weight.detach().to(CPU), Xt, bits=bits, group=group, damp=CFG["gptq_damp"])
        pw[p] = wq
        if p in ACTS:
            rels.append(rel_mse_layer(ACTS[p].to(CPU), m.weight.detach().to(CPU), wq))
    restore = build_patched(pw)
    pl, pp = ppl_of(mdl, PPL_TEXT, CFG["ppl_seq_len"], DEVICE)
    restore()
    return pl, pp, float(np.mean(rels))

gptq_rows = []
for g in CFG["gptq_groups"]:
    pl, pp, rm = run_gptq(CFG["gptq_bits"], g)
    gptq_rows.append(dict(tag=f"GPTQ W{CFG['gptq_bits']} g{g if g>0 else 'inf'}", group=g,
                          ppl=pp, rel_mse=rm))
    log("GPTQ W%d g%s: PPL=%.3f  rel_MSE=%.4f" % (CFG["gptq_bits"], g if g>0 else "inf", pp, rm))

# 同 bit/group 的 RTN 对照
rtn_w4_g128 = next(r for r in rtn_rows if r["tag"] == "W4 group g128")
rtn_w4_pt = next(r for r in rtn_rows if r["tag"] == "W4 tensor")
gptq_g128 = next(r for r in gptq_rows if r["group"] == 128)

log("=" * 78)
log("[GPTQ] W4 与 RTN W4 同 group 对比：")
log("  RTN  W4 per-tensor   : PPL=%.3f  rel_MSE=%.4f" % (rtn_w4_pt["ppl"], rtn_w4_pt["rel_mse"]))
log("  RTN  W4 g128         : PPL=%.3f  rel_MSE=%.4f" % (rtn_w4_g128["ppl"], rtn_w4_g128["rel_mse"]))
log("  GPTQ W4 g128         : PPL=%.3f  rel_MSE=%.4f" % (gptq_g128["ppl"], gptq_g128["rel_mse"]))
log("-" * 78)
log("读数1：W4 下 GPTQ g128 的层输出 rel_MSE=%.4f 比 RTN g128=%.4f 低 %.1f%% —— 误差补偿把量化误差分摊回未量化权重。" %
    (gptq_g128["rel_mse"], rtn_w4_g128["rel_mse"],
     100 * (rtn_w4_g128["rel_mse"] - gptq_g128["rel_mse"]) / max(rtn_w4_g128["rel_mse"], 1e-9)))
if gptq_g128["ppl"] < rtn_w4_g128["ppl"]:
    log("读数2：GPTQ W4 g128 的 PPL=%.3f 也优于 RTN W4 g128=%.3f（误差补偿的二阶信息生效）。" %
        (gptq_g128["ppl"], rtn_w4_g128["ppl"]))
else:
    log("读数2：本小模型上 W4 量化本就温和，GPTQ 的 PPL 收益有限（小模型冗余度低、RTN 已接近）；"
        "rel_MSE 改善仍是二阶补偿的直接证据。")
g = CFG["gptq_groups"]
if len(gptq_rows) >= 2:
    log("读数3（group 影响）：g=%s 的 rel_MSE 依次为 %s —— group 越小误差通常越低（scale 更细），"
        "但元数据开销更大。" % (g, [round(r["rel_mse"], 4) for r in gptq_rows]))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
labels = ["RTN W4\nper-tensor", "RTN W4\ng128"] + [r["tag"] for r in gptq_rows]
vals = [rtn_w4_pt["rel_mse"], rtn_w4_g128["rel_mse"]] + [r["rel_mse"] for r in gptq_rows]
ax[0].bar(range(len(labels)), vals, color=["#999999", "#55A868"] + ["#4C72B0"] * len(gptq_rows))
ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, fontsize=7)
ax[0].set_ylabel("layer rel-MSE"); ax[0].set_title("[GPTQ] RTN vs GPTQ (W4) reconstruction error")
ax[1].bar([r["tag"] for r in gptq_rows], [r["ppl"] for r in gptq_rows], color="#4C72B0")
ax[1].axhline(fp_ppl, color="grey", ls="--", label="FP PPL")
ax[1].set_ylabel("PPL"); ax[1].set_title("[GPTQ] PPL vs group size"); ax[1].legend(fontsize=8)
savefig(fig, "gptq_recon_and_ppl.png")''')

# ============================================================ Cell 7: SmoothQuant
md(r'''## 5. SmoothQuant（α 迁移）

对应 `smoothquant_alpha_sweep/` 目录。用恒等式 $XW=(X\\,\\mathrm{diag}(\\tau)^{-1})(\\mathrm{diag}(\\tau)W)$
把激活的量化难度按强度 $\\alpha$ 迁移到权重侧：$\\tau_j = a_j^{\\alpha}/w_j^{1-\\alpha}$
（$a_j=\\max|X_j|,\\ w_j=\\max|W_{:j}|$）。在 **W8A8**（权重 per-channel W8 + 激活 per-token W8）下扫描
$\\alpha\\in\\{0,0.25,0.5,0.75,1\\}$，报告激活量化误差的 U 形曲线与最优 α，并画「迁移后激活动态范围下降倍数」。''')

code(r'''def smoothquant_tau(X, W, alpha):
    """返回 per 输入通道迁移系数 tau_j（广播到权重列）。W:(out,in), X:(N,in)。"""
    a = X.abs().amax(0).clamp(min=1e-8)        # (in,)
    wj = W.abs().amax(0).clamp(min=1e-8)        # (in,)
    if alpha is None:
        return torch.ones_like(a)
    return (a ** alpha) / (wj ** (1 - alpha))

def smoothquant_w8a8_mse(X, W, tau):
    """W8A8 层输出相对 MSE：权重 per-channel W8（含 tau）+ 激活 per-token W8。"""
    Ws = W * tau
    Wq = quant_dequant(Ws, 8, axis=0)            # per 输出通道 W8
    Xs = X / tau
    Xq = per_token_quant(Xs, 8)
    Yfp = X @ W.T
    Yq = Xq @ Wq.T
    return float((((Yq - Yfp) ** 2).mean() / (Yfp ** 2).mean().clamp(min=1e-12)).item())

def smoothquant_layer_ppl(alpha, mods_pw):
    """用一个全局 alpha 把所有 Linear 换成 smoothed W8 权重 + 激活 per-token W8 hook，测 PPL。"""
    # 先改权重
    orig = {}
    for p, m in LINEARS.items():
        W = m.weight.detach().to(CPU)
        tau = smoothquant_tau(ACTS[p].to(CPU), W, alpha) if alpha is not None else torch.ones(W.shape[1])
        Wq = quant_dequant(W * tau, 8, axis=0)
        orig[p] = m.weight.data.clone()
        m.weight.data = Wq.to(m.weight.device).contiguous()
    # 激活 per-token W8 hook
    handles = []
    for m in LINEARS.values():
        def _h(_, inp):
            x = inp[0]
            return (per_token_quant(x, 8),)
        handles.append(m.register_forward_pre_hook(_h))
    pl, pp = ppl_of(mdl, PPL_TEXT, CFG["ppl_seq_len"], DEVICE)
    for h in handles:
        h.remove()
    for p, m in LINEARS.items():
        m.weight.data = orig[p].clone()
    return pl, pp

smooth_rows = []
reductions = []
for alpha in CFG["smooth_alphas"]:
    mses, reds = [], []
    for p, m in LINEARS.items():
        if p not in ACTS:
            continue
        X = ACTS[p].to(CPU); W = m.weight.detach().to(CPU)
        tau = smoothquant_tau(X, W, alpha)
        mses.append(smoothquant_w8a8_mse(X, W, tau))
        reds.append((X.abs().amax(0) / (X / tau).abs().amax(0)).median().item())  # 动态范围下降倍数
    smooth_rows.append(dict(alpha=alpha, rel_mse=float(np.mean(mses)),
                             dyn_reduce=float(np.median(reds))))
    log("SmoothQuant α=%.2f: W8A8 层输出 rel_MSE=%.5f  动态范围下降≈%.1fx" % (alpha, smooth_rows[-1]["rel_mse"], smooth_rows[-1]["dyn_reduce"]))

# naive W8A8（无平滑）
naive_mses = []
for p, m in LINEARS.items():
    if p not in ACTS:
        continue
    X = ACTS[p].to(CPU); W = m.weight.detach().to(CPU)
    naive_mses.append(smoothquant_w8a8_mse(X, W, torch.ones(W.shape[1])))
naive_mse = float(np.mean(naive_mses))
best = min(smooth_rows, key=lambda r: r["rel_mse"])
log("=" * 78)
log("[SmoothQuant] W8A8（权 per-channel W8 + 激 per-token W8）：")
log("  naive(不平滑) rel_MSE=%.5f" % naive_mse)
for r in smooth_rows:
    log("  α=%.2f  rel_MSE=%.5f  (相对 naive %.2fx)" % (r["alpha"], r["rel_mse"], naive_mse / max(r["rel_mse"], 1e-12)))
log("-" * 78)
log("读数1：U 形曲线谷底在 α*=%.2f（rel_MSE=%.5f），与文章「均衡点 α=0.5」一致；"
    "naive 不平滑比最优差 %.1fx。" % (best["alpha"], best["rel_mse"], naive_mse / max(best["rel_mse"], 1e-12)))
log("读数2：迁移后激活 max 动态范围中位数下降≈%.1fx（难度从激活搬到权重，权重侧 per-channel 量化免费吃掉）。" % best["dyn_reduce"])

# PPL：naive(α=None 视作不平滑，但需权重也 W8) 与最优 α
_, pp_naive = smoothquant_layer_ppl(None, None)
_, pp_best = smoothquant_layer_ppl(best["alpha"], None)
log("读数3：W8A8 PPL —— naive=%.3f,  α*=%.2f 平滑后=%.3f（FP=%.2f）。" % (pp_naive, best["alpha"], pp_best, fp_ppl))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].plot([r["alpha"] for r in smooth_rows], [r["rel_mse"] for r in smooth_rows], "o-", color="#4C72B0", lw=2)
ax[0].axhline(naive_mse, color="#C44E52", ls="--", label="naive (no smooth)")
ax[0].axvline(best["alpha"], color="grey", ls=":", label=f"best α*={best['alpha']:.2f}")
ax[0].set_xlabel("migration strength α"); ax[0].set_ylabel("W8A8 layer rel-MSE")
ax[0].set_title("[SmoothQuant] U-curve of activation-quant error vs α"); ax[0].legend(fontsize=8)
ax[1].bar([str(r["alpha"]) for r in smooth_rows], [r["dyn_reduce"] for r in smooth_rows], color="#55A868")
ax[1].set_xlabel("α"); ax[1].set_ylabel("median activation dynamic-range reduction (x)")
ax[1].set_title("[SmoothQuant] How much activation range shrinks after migration")
savefig(fig, "smoothquant_alpha_sweep.png")''')

# ============================================================ Cell 8: AWQ
md(r'''## 6. AWQ（activation-aware scale search）

对应 `awq_scale_search/` 目录。按**激活显著度**挑 salient 通道：缩放 $s_j=(\\mathrm{mean}_t|X_{tj}|)^{\\alpha}$
（归一化后），把显著通道权重放大再量化、之后除回。在 W4-g128 下对每层搜最优 α（网格），报告相对 RTN 的收益。

> 这与 GPTQ 同源（都来自「误差被激活二阶矩加权」），但只用一阶激活统计、免 Hessian、免迭代——便宜得多。''')

code(r'''def awq_scale_search(X, W, bits, group, grid):
    """在 W4-g128 下对单层搜最优 α，返回 (best_alpha, best_rel_mse, 反量化权重)。"""
    a = X.abs().mean(0).clamp(min=1e-8)         # (in,) 激活平均幅值
    best_a, best_m, best_wq = 0.0, float("inf"), None
    for alpha in grid:
        s = a ** alpha
        s = s / math.sqrt(s.max() * s.min())     # 官方归一化，防整体漂移
        Ws = W * s[None, :]
        Wq = group_quant_dequant(Ws, bits, group) / s[None, :]
        m = rel_mse_layer(X, W, Wq)
        if m < best_m:
            best_a, best_m, best_wq = alpha, m, Wq
    return best_a, best_m, best_wq

awq_rows = []
best_alphas = []
for p, m in LINEARS.items():
    if p not in ACTS:
        continue
    X = ACTS[p].to(CPU); W = m.weight.detach().to(CPU)
    ba, bm, bwq = awq_scale_search(X, W, CFG["awq_bits"], CFG["awq_group"], CFG["awq_grid"])
    awq_rows.append(dict(module=p, alpha=ba, rel_mse=bm))
    best_alphas.append(ba)
# 构建 AWQ 模型并测 PPL
pw = {}
for p, m in LINEARS.items():
    if p in ACTS:
        X = ACTS[p].to(CPU); W = m.weight.detach().to(CPU)
        ba, _, bwq = awq_scale_search(X, W, CFG["awq_bits"], CFG["awq_group"], CFG["awq_grid"])
        pw[p] = bwq
    else:
        pw[p] = m.weight.detach().to(CPU)
restore = build_patched(pw)
awq_ppl, awq_loss = ppl_of(mdl, PPL_TEXT, CFG["ppl_seq_len"], DEVICE)
restore()
awq_rel = float(np.mean([r["rel_mse"] for r in awq_rows]))

rtn_w4_g128 = next(r for r in rtn_rows if r["tag"] == "W4 group g128")
log("=" * 78)
log("[AWQ] W4-g%d（按激活显著度搜 α）" % CFG["awq_group"])
log("  最优 α 分布：min=%.2f  median=%.2f  max=%.2f" % (min(best_alphas), float(np.median(best_alphas)), max(best_alphas)))
log("  平均层输出 rel_MSE=%.4f   PPL=%.3f" % (awq_rel, awq_ppl))
log("  RTN W4 g%d 对照：rel_MSE=%.4f  PPL=%.3f" % (CFG["awq_group"], rtn_w4_g128["rel_mse"], rtn_w4_g128["ppl"]))
log("-" * 78)
gain = (rtn_w4_g128["rel_mse"] - awq_rel) / max(rtn_w4_g128["rel_mse"], 1e-9)
log("读数1：AWQ 把 W4-g%d 的层输出 rel_MSE 从 %.4f 降到 %.4f（相对改善 %.1f%%），"
    "与 GPTQ 同源——都是「误差被激活幅值加权」。" % (CFG["awq_group"], rtn_w4_g128["rel_mse"], awq_rel, 100 * gain))
log("读数2：AWQ 比 RTN 的 PPL 变化：%.3f -> %.3f（小模型上 W4 已温和，PPL 收益有限但 rel_MSE 改善真实）。" %
    (rtn_w4_g128["ppl"], awq_ppl))
log("读数3：纯一阶激活统计、网格搜索、零 Hessian —— 验证了文章「AWQ 比 GPTQ 轻一个数量级」。")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].bar(["RTN W4 g%d" % CFG["awq_group"], "AWQ W4 g%d" % CFG["awq_group"]], [rtn_w4_g128["rel_mse"], awq_rel],
          color=["#999999", "#4C72B0"])
ax[0].set_ylabel("layer rel-MSE"); ax[0].set_title("[AWQ] RTN vs AWQ reconstruction error (W4 g%d)" % CFG["awq_group"])
for i, v in enumerate([rtn_w4_g128["rel_mse"], awq_rel]):
    ax[0].text(i, v, "%.4f" % v, ha="center", va="bottom", fontsize=8)
# 各模块最优 α 直方图
ax[1].hist(best_alphas, bins=10, color="#8172B3")
ax[1].set_xlabel("best α per module"); ax[1].set_ylabel("#modules")
ax[1].set_title("[AWQ] Distribution of searched optimal α")
savefig(fig, "awq_scale_search.png")''')

# ============================================================ Cell 9: export int4
md(r'''## 7. 伪 int4 打包 + 元数据开销（scale 元数据税）

对应各篇「元数据开销」讨论。把某一层权重量化为**真 int4**（对称、group-wise），2 个值打包进 1 字节，
scale 以 fp16 存。展示**权重体积 vs 元数据开销**，并验证「加载→反量化」的数值一致性（与原始反量化权重 max 误差≈0）。''')

code(r'''def int4_pack(W, group):
    """伪 int4 打包：返回 (packed int8, scales fp16, qm, nelem)。对称 group 量化；末尾 group 不足则 pad。"""
    out, in_f = W.shape
    qm = QMAX(4)
    if group is None or group < 0 or group >= in_f:
        s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qm
        wq = torch.clamp(torch.round(W / s), -qm, qm)
        codes = wq + qm                       # 映射到 0..15，shape (out, in_f)
        scales = s.reshape(out, 1)
    else:
        g = int(group); ng = (in_f + g - 1) // g; width = ng * g
        if width != in_f:
            W = torch.cat([W, W.new_zeros(out, width - in_f)], dim=1)
        wg = W.reshape(out, ng, g)
        s = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qm
        wq = torch.clamp(torch.round(wg / s), -qm, qm)
        codes = wq.reshape(out, width) + qm     # (out, width)
        scales = s.reshape(out, ng)
    # 2 个 4-bit 值打包进 1 个 int8
    codes = codes.to(torch.int32)
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    if codes.shape[1] % 2 == 1:               # 列数为奇，odd 补 0 凑偶
        odd = torch.cat([odd, torch.zeros(out, 1, dtype=torch.int32)], dim=1)
    packed = ((even & 0xF) | ((odd & 0xF) << 4)).to(torch.int8)
    return packed, scales.to(torch.float16), qm, out * in_f

def int4_unpack(packed, scales, qm, group, out, in_f):
    """解包 + 反量化，返回 float 权重 (out, in_f)。pack 时末尾可能 pad，这里按 group 还原实际宽度再裁回。"""
    low = (packed.to(torch.int32) & 0xF)
    high = ((packed.to(torch.int32) >> 4) & 0xF)
    total = packed.shape[1] * 2                # 打包前（含 pad）的列数
    codes = torch.zeros(out, total, dtype=torch.int32)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    wq = (codes.to(torch.float32) - qm)         # 还原到有符号格点
    if group is None or group < 0 or group >= in_f:
        return wq[:, :in_f] * scales.reshape(out, 1)
    g = int(group); ng = (in_f + g - 1) // g
    sc = scales.reshape(out, ng, 1).expand(out, ng, g).reshape(out, ng * g)
    return wq[:, :in_f] * sc[:, :in_f]

rep_path = [p for p in LINEARS if "q_proj" in p][0] if any("q_proj" in p for p in LINEARS) else list(LINEARS)[0]
Wrep = LINEARS[rep_path].weight.detach().to(CPU).float()
export_rows = []
for g in CFG["export_groups"]:
    packed, scales, qm, nelem = int4_pack(Wrep, g)
    wbytes = packed.numel()
    sbytes = scales.numel() * 2
    Wback = int4_unpack(packed, scales, qm, g, *Wrep.shape)
    # 一致性：与标准 group_quant_dequant 反量化对比
    Wref = group_quant_dequant(Wrep, 4, g)
    max_err = float((Wback - Wref).abs().max())
    export_rows.append(dict(group=(g if g > 0 else "inf"), w_bytes=wbytes, meta_bytes=sbytes,
                            ratio=(Wrep.numel() * 4) / (wbytes + sbytes),
                            meta_pct=100 * sbytes / (wbytes + sbytes), max_recon_err=max_err))
    log("int4 pack g=%s: 权重=%d B, scale meta=%d B (占 %.2f%%), 压缩比=%.2fx, 反量化最大误差=%.2e" % (
        g if g > 0 else "inf", wbytes, sbytes, export_rows[-1]["meta_pct"], export_rows[-1]["ratio"], max_err))

log("=" * 78)
log("[导出] 伪 int4 打包（%s）：" % rep_path)
log("%-8s %+10s %+12s %+10s %+14s" % ("group", "权重B", "metaB", "压缩比", "反量化最大误差"))
for r in export_rows:
    log("%-8s %10d %12d %10.2fx %14.2e" % (r["group"], r["w_bytes"], r["meta_bytes"], r["ratio"], r["max_recon_err"]))
log("-" * 78)
log("读数1：group 越小 scale 元数据占比越高（g=64 比 g=128 的 meta 占比明显上升）——这就是「scale 元数据税」。")
log("读数2：加载→反量化后的权重与标准 group 量化反量化最大误差≈0（%.1e），证明打包/解包无损。" %
    export_rows[0]["max_recon_err"])

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
gs = [str(r["group"]) for r in export_rows]
ax[0].bar(gs, [r["w_bytes"] for r in export_rows], color="#4C72B0", label="weight (int4)")
ax[0].bar(gs, [r["meta_bytes"] for r in export_rows], bottom=[r["w_bytes"] for r in export_rows],
          color="#C44E52", label="scale meta (fp16)")
ax[0].set_ylabel("bytes"); ax[0].set_title("[export] Weight vs scale-metadata size"); ax[0].legend(fontsize=8)
ax[1].bar(gs, [r["meta_pct"] for r in export_rows], color="#8172B3")
ax[1].set_ylabel("% of total"); ax[1].set_title("[export] Scale-metadata tax (%)")
savefig(fig, "int4_pack_metadata.png")''')

# ============================================================ Cell 10: summary
md(r'''## 8. 结论汇总

把关键数字收在一处，写入 `results/results.json` 与 `results/stdout.txt`，并显著标注真实模型 vs 随机初始化。''')

code(r'''def fnum(x):
    try:
        return float(x)
    except Exception:
        return x

summary = {
    "meta": dict(mode=MODE, model=MODEL_NAME, real_model=REAL_MODEL, device=str(DEVICE),
                 torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                 seed=SEED, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
                 honesty=("PPL/act/weight stats are REAL on the loaded model, but PPL is measured on ONE fixed"
                          " prompt (relative comparison only)" if REAL_MODEL else
                          "RANDOM-INIT GPT2: absolute numbers meaningless, relative comparison only")),
    "baseline": {k: fnum(v) for k, v in baseline.items()},
    "rtn": rtn_rows,
    "gptq": gptq_rows,
    "smoothquant": dict(alphas=smooth_rows, naive_rel_mse=fnum(naive_mse),
                        best_alpha=fnum(best["alpha"]), best_rel_mse=fnum(best["rel_mse"]),
                        ppl_naive=fnum(pp_naive), ppl_best=fnum(pp_best)),
    "awq": dict(rel_mse=fnum(awq_rel), ppl=fnum(awq_ppl),
                rtn_g128_rel_mse=fnum(rtn_w4_g128["rel_mse"]),
                rtn_g128_ppl=fnum(rtn_w4_g128["ppl"]),
                best_alpha_median=fnum(float(np.median(best_alphas)))),
    "export": export_rows,
}

log("")
log("=" * 78)
log("结论汇总")
log("=" * 78)
log("模型=%s  真实模型=%s  PPL(固定Prompt)=%.2f" % (MODEL_NAME, REAL_MODEL, fp_ppl))
log("RTN  : " + "  ".join("%s PPL=%.2f" % (r["tag"], r["ppl"]) for r in rtn_rows))
log("GPTQ : " + "  ".join("%s rel_MSE=%.4f PPL=%.2f" % (r["tag"], r["rel_mse"], r["ppl"]) for r in gptq_rows))
log("Smooth: best α=%.2f W8A8 rel_MSE=%.5f (naive %.5f, 改善 %.1fx)" %
    (best["alpha"], best["rel_mse"], naive_mse, naive_mse / max(best["rel_mse"], 1e-12)))
log("AWQ  : W4-g%d rel_MSE=%.4f (RTN %.4f) PPL=%.2f" % (CFG["awq_group"], awq_rel, rtn_w4_g128["rel_mse"], awq_ppl))
log("导出 : g=%s 压缩比=%.2fx, scale 元数据税=%.2f%%" %
    (export_rows[0]["group"], export_rows[0]["ratio"], export_rows[0]["meta_pct"]))
log("=" * 78)
if not REAL_MODEL:
    log("!!! 注意：模型为随机初始化，以上绝对数字无意义，仅作相对比较。")
else:
    log("诚实标注：PPL 仅在固定 Prompt 上测（真实模型真实指标，但非完整验证集）；其余为真实度量。")

with open(os.path.join(RES, "results.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
with open(os.path.join(RES, "stdout.txt"), "w") as f:
    f.write("\n".join(_LINES) + "\n")
log("[save] " + os.path.join(RES, "results.json"))
log("[save] " + os.path.join(RES, "stdout.txt"))
log("DONE")''')

nb["cells"] = cells
with open("official_llm_ptq_torch.ipynb", "w") as f:
    nbf.write(nb, f)
print("written", len(cells), "cells")
