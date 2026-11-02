# π 系列架构图（手工排版 SVG）

`pi_architecture_figures.py` 出 4 张**结构图**，与同系列另外两个实验目录里的
**数据曲线**分工不同：

| 目录 | 出的是什么 | 用到实验真值吗 |
| --- | --- | --- |
| `pi0-flow-matching/` | 数据曲线（边际速度场、欧拉步数、时间采样、KV 掩码） | 是，`import pi0_flow_lab` |
| `fast-action-tokenizer/` | 数据曲线（token 数、γ 权衡、展平顺序、率失真） | 是，`import fast_tokenizer_lib` |
| `architecture/`（本目录） | 结构：模块层次、token 形状、参数量、信息流 | 否，数字全部取自论文 / openpi 源码常量 |

## 为什么不用 Mermaid

Mermaid 只能表达拓扑（`A --> B`），画不出：

- **层级**（哪个模块包着哪个模块）；
- **尺寸**（主干 2048 宽 vs action expert 1024 宽，参数量 3B vs 300M）；
- **形状**（542–798 个 prefix token、50 × 32 的动作块）；
- **复用 / 缓存**（哪一段只算一次、哪一段走十遍）。

这些正是「网络结构」里真正要传达的信息，所以本目录的图全部手工排版。

## 出图

```bash
~/Software/miniconda3/bin/python pi_architecture_figures.py
```

产出写在 `figures/`：SVG 进仓库、也复制进博客；PNG 同时写一份用于人眼校对，
被 `figures/.gitignore` 挡在仓库外。
要同步到博客：

```bash
cp figures/*.svg ~/Desktop/Projects/github/lrypcy.github.io/assets/img/pi-vla/
```

## 两张排版坑（都踩过）

1. **字体必须是 PingFang SC**。Hiragino Sans GB 有缺字形，中文会渲染成方框，
   而且 matplotlib **不报任何警告**——只有打开 PNG 才看得出来。`−`(U+2212) 也只在新字体里有。
2. **画布坐标就是英寸**（`figsize=(W,H)` 配 `xlim(0,W)/ylim(0,H)`），一行字高约 `字号/72`。
   盒内多行文字要按字号算行距再垂直居中；写 `(h - 常数)/(n-1)` 在矮盒子里会退化成 0.02 英寸、
   两行字直接叠在一起。另外 `bbox_inches="tight"` 会把画到画布外的元素悄悄扩进画布，
   所以「元素排到 x=16 而画布只有 12.6 宽」不会报错——**每次都要看 PNG**。

## 四张图分别对应正文哪里

| 文件 | 博客位置 |
| --- | --- |
| `pi0_architecture.svg` | 01 篇 §3 网络：一个 transformer，两套权重 |
| `pi0_inference_loop.svg` | 01 篇 §9 工程账：KV cache 与一秒 chunk 的时延 |
| `fast_pipeline.svg` | 02 篇 §3 六步流水线 |
| `pi0_vs_pi0_fast.svg` | 02 篇 §10 两个头：连续 vs 自回归 |
