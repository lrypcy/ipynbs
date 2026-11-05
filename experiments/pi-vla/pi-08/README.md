# pi-08：RLT 把「冻结 VLA 上挂一个小 actor-critic」拆成了哪几件可解析的事

对应文章：π 系列具身模型（08）RLT / RL Token（arXiv:2604.23073v2，*RL Token:
Bootstrapping Online RL with Vision-Language-Action Models*，Physical Intelligence，
30 Apr 2026）。

论文的做法是：给预训练 π₀.₆ 的**最后层 token 序列**追加一个可学习 embedding，用轻量
encoder transformer 读出一个紧凑向量 **RL token z_rl**；再用 decoder 自回归重建原
embedding，逼这个向量成为**信息瓶颈**。之后 VLA 与 RL token 表示**全部冻结**，只在
z_rl 上在线训练一个轻量 actor-critic（TD3 式双 Q），把 VLA 提出的动作块当作
「局部编辑」的起点，用稀疏二值 reward 做样本高效在线 RL。

本实验在一个**线性高斯世界**里把最容易含混过去的四件事钉死：

1. **RL token 的瓶颈维度不是越大越好，够用就饱和。** 重建误差 L_ro 随维度 b 单调降，
   但下游控制精度在 b=内在秩 r=6 处**完全饱和**（0.5169，与 b=60 全维 oracle 同值）；
   b>6 后 L_ro 仍每维降 σ²=0.09，那只是拟合噪声。
2. **双 Q 抑制的是过估计，代价是转成小幅低估。** 单 Q 贪心单步高估 +1.6363、迭代
   40 步 TD 复合到 21.62（真值 10.00）；TD3 的 clipped min(Q1,Q2) 是 −0.5684 的
   小幅低估，迭代后 3.98，偏差远小于单 Q。冻结特征只要含任务信息，**小线性 critic
   就能学**（Q-MSE 0.0120）；换成无任务信息的通用特征则涨到 0.0739。
3. **chunk 化缩短的是信用分配的决策数，不是单个目标的噪声。** 决策数 50→5，
   REINFORCE 梯度方差 4987.9→50.1（99.6×）；但 Eq.(3) 的 C 步奖励和方差反而是
   单步的 4.62×。
4. **β anchor 是倒 U 形，不是单调。** β=0（论文「最大单项下降」）动作误差 2.0400，
   比满锚定 β=200 的 0.8321 还差；解析最优 β*=2.5 处 0.6114。参考动作 dropout
   太低会退化成复制 BC：p=0 时无参考推理误差 2.6504、参考被污染时 7.6551，
   p=0.5 分别降到 0.6405 与 1.4102。

**只验机制，不重算论文成绩**：20%→65%、最高 3× 提速、15 min–5 h 真机数据、
400–1000 episodes 等全部只是引用，见下方「边界」。

## 合成世界

单套冻结线性映射 + 高斯噪声，全部量可解析；采样只用于多 seed 平均与构造真值↔MC 反算。

- **VLA 最后层 token 序列** x ∈ ℝ^D（D=60=12 token×5 维）：由低秩**任务码**
  v ∈ ℝ^r（r=6，v~N(0,I)）经冻结正交映射生成，x = A v + η，η~N(0,σ²I)，σ²=0.09；
  信号方差 λ=[4.0,3.0,2.0,1.2,0.7,0.4]。Cov(x) 在正交基下对角，前 r 个特征值
  λ_j+σ²、其余 σ²——这正是 encoder–decoder L2 重建损失（论文 Eq.2，z̄=sg(z) 为
  stop-gradient 目标）的 Bayes 最优解 = **秩-b PCA 投影**成立的理由。
- **RL token / bottleneck**：z_rl = Q_bᵀ x ∈ ℝ^b（b 为瓶颈维度探针，论文**未给出**）。
  重建误差 = 丢弃的协方差特征值之和；下游控制用 z_rl 对 v 的 Wiener 估计产生动作块。
  动作块 a_{1:C}∈ℝ^{140}（C=10,d=14）投影到其 6 维任务相关子空间（列重要性
  w=[3.0,2.4,1.8,1.3,0.9,0.6]），其余 134 维是轨迹平滑分量，探针里不建模。
- **actor-critic**：critic Q(x,a) 用线性特征（含 z_rl 或通用特征）由岭回归拟合；
  双 Q 过估计用 K=12 候选动作 + τ=1 估计噪声的标准离散抽象；actor 是 Eq.5 的
  闭式最优局部编辑 a = ã + κ(â*−ã)，κ=1/(1+β)，ã 是带系统偏置 δ=0.8 的 VLA
  参考块、â* 是带噪 τ_Q²=2.0 的 critic 最优估计。
- **稀疏二值 reward**：episode 末尾成功当且仅当全部决策「好」（单决策概率 p_good），
  P(succ)=p_good^N，N 为决策数（step=50 / chunk=5）。
- **参考动作 dropout**：状态通路的「有效训练度」抽象成 s(p)=p/(p+0.15)、对参考的
  复制依赖 ρ=1−s；测试分「参考正常 / 参考隐藏 / 参考污染」三种推理。

## 每个实验组对应论文的哪个机制

| 组 | 对应机制（出处） | 本探针里的操作 |
| --- | --- | --- |
| 自校验 | §IV-A Eq.(2) L_ro 的 stop-gradient 自回归重建、VLA 对 L_ro 冻结 | 手写 encoder/decoder 重建梯度 ↔ 有限差分；解析 PCA ↔ MC 反算 |
| A | §IV-A Eq.(1)(2)、Fig.2：encoder–decoder bottleneck、z_rl 作为 RL state | b=1..12 扫描重建误差/控制误差，找饱和拐点 |
| B | §IV-B Eq.(3) chunk-level 回报 Q̂=Σ_{t'<C}γ^{t'−1}r_{t'}+γ^C Q'、TD3 双 Q 取 min（§III/App.-B）、稀疏二值 reward（§III）、§V Alg.1 | 单 Q/解耦双 Q/clipped 双 Q 偏差 + 迭代 TD；冻结特征下小 critic 的 Q-MSE；决策数 50→5 的方差 |
| C | §IV-B Eq.(4) 高斯 actor π=N(μ,σ²I) 条件于 ã、Eq.(5) L_π=−Q+β‖a−ã‖²、reference action dropout（§IV-B/App.-B 50%）、§VI-C Q3 的 β=0 / w/o Pass-Through 消融 | β 倒 U 扫描；dropout 扫描的 masked / corrupted / normal 推理误差 |
| D | §VI-C Q3 w/o Chunk（C=1 学不动）、§VI-B/App.-B 稀疏 reward、§V replay buffer（VLA warmup + 在线 rollout + 人类干预替换 ã） | b=1..3 控制误差；p_good=0.5 的 32 episode 才见一次成功；VLA-only buffer 的 Q 外推误差 |

## 要验证的论断（数字逐位来自 `results/stdout.txt`）

| 组 | 论断 | 实测 |
| --- | --- | --- |
| 自校验 | 手写重建梯度 ↔ 有限差分 | 相对误差 1.10e-09；解析↔MC 重建 0.002 / 控制 0.001 |
| A | 重建误差随 b 单调降 | 12.6100 → 4.8600（b=6）→ 4.3200（b=12） |
| A | 控制精度在内在秩处饱和 | 拐点 b=6=r，CTRL_floor=0.5169，b=6=b=12=b=60 oracle |
| A | b>r 后 L_ro 只是拟合噪声 | b=6→12 重建仍降 0.5400=6σ²，控制恒 0.5169 |
| A | b=1 维度过小 | 控制误差 7.0660（饱和值的 13.7×） |
| B | 单 Q 高估、双 Q 抑制 | 单步 +1.6363 / −0.0061 / −0.5684；迭代 TD 21.62 / 3.98（V*=10.00） |
| B | 冻结好特征 + 小 critic 可学 | Q-MSE 0.0120（多 40 随机特征 0.0121；buffer=50 时 0.0348） |
| B | w/o RL token（通用特征）显著更差 | 0.0739 / 0.0753（对应去 RL token→ResNet-10 吞吐降 50%） |
| B | chunk 化降低信用分配方差 | Var 4987.9 → 50.1（99.6×），决策 50→5 |
| B | 但 C 步奖励和方差更大 | 单步的 4.62× |
| B | 稀疏 reward 下 chunk 使任务可学 | P(succ|chunk) 0.5905/0.1681/0.0312 vs step 5.15e-03/1.80e-08/8.88e-16 |
| C | β 倒 U 形、解析 β*=V/D=2.5 | β=0 2.0400 / β* 0.6114 / β=200 0.8321 |
| C | β=0 是最差单项（印证论文） | 2.0400 > β=200 的 0.8321 |
| C | 成功率也倒 U（峰值 β=2） | 0.039 → 0.407 → 0.135 |
| C | dropout 太低退化成复制 BC | masked 2.6504→0.6405（4.1×）；corrupted 7.6551→1.4102 |
| C | dropout 不损害正常推理 | normal 0.6119→0.6002 基本持平 |
| D | token 过小 / 稀疏 / 离线偏差失效 | b1=7.0660→b6=0.5169；p_good=0.5 需 32.0 episode；Q-MSE 0.4477→0.0048 |

## 反直觉结论（写正文须照此口径）

- **别拿重建损失继续堆 RL token 维度**：b≥r 后 L_ro 仍因拟合噪声每维降 σ²=0.09，
  但下游控制精度已完全饱和（0.5169）。b=6 与 60 维全维 oracle 同值，压缩 10×。
- **TD3 的 clipped 双 Q 不是无偏**：它是「小幅低估」（−0.5684）换掉单 Q 的
  「大幅高估」（+1.6363）；迭代引导下单 Q 复合到 21.62、clipped 只到 3.98。
- **chunk 化减小的不是单个 TD 目标的噪声**：Eq.(3) 的 C 步奖励和方差比单步大
  4.62×；真正变小的是决策数 50→5 与引导传播次数，即信用分配。
- **β=0 比满锚定还差**（critic 噪声 τ_Q²=2.0 > 先验误差 D=0.8），印证论文
  「w/o BC Regularizer 是最大单项下降」；但 β≫β* 也会把动作粘在带偏置 ã 上、
  压住 RL 改进，是倒 U 形而非单调。
- **阈值化成功率与 MSE 不必逐点同峰**：成功率峰值在 β=2（0.407），β*=2.5 为
  0.403，差 0.004；倒 U 形结论不变。
- **参考动作 dropout 的 50% 并非严格饱和**：p=0.5 的 masked 误差 0.6405，升到
  0.9 还再降 2.7%（0.6231）；50% 兑现的是绝大部分收益，不是全部。
- **回报过稀疏时 chunk 化也不凭空造信息**：p_good=0.5 仍需 32 个 episode 才见
  一次成功。

## 运行

```bash
~/Software/miniconda3/bin/python rlt_lab.py       # 自校验 + A–D 四组，约 3 秒
~/Software/miniconda3/bin/python rlt_figures.py   # ≥5 张图（SVG + PNG）
```

解释器**必须锁 `~/Software/miniconda3/bin/python`（numpy 2.1.1 / matplotlib 3.11.1，
CPU）**：`results/stdout.txt` 是用它跑出来的；换成别的 numpy 版本会因为随机数流 /
浮点实现差异让数字漂几个千分位，正文数字就和固化输出对不上了。纯 numpy、固定
seed（20261108）、多 seed 平均（16 seed，双 Q 单步 4000 seed）、每组末尾断言、
最后打印断言总数（**45 条全绿**）。`rlt_figures.py` 的数据图数字逐位解析自
`results/stdout.txt` 末尾 `[MACHINE-BACKFILL-*]` 块，与「正文回填表」同源。

## 边界

- **只验机制，不重算论文成绩。** 论文里的成功率 **20%→65%**（§I）、**最高 3×
  提速**（Abstract/§I）、**15 min–5 h 真机数据**（§VI-B）、**400–1000 episodes**
  （§VI-B）、1–10 h 遥操作示范、2000–10000 gradient steps、去 RL token 吞吐降 50%
  （§VI-C Q3）、w/o Chunk 学不动、β=0 最大单项下降等，全部是**引用**；本探针一个
  都没有重算。合成世界的控制误差、动作误差、Q-MSE、成功率都是无量纲的探针读数，
  与 successes/hour、真机成功率不是同一量纲，**改进幅度不可与真机直接比**。
  论文无表格（`ltx_table`=0），逐任务精确成功率/吞吐只在柱状图里、正文未给出，
  因此也**不能目测估数**。
- **β（BC anchor 系数）的数值论文未给出**（全文只有符号 β 与消融里的 β=0）。
  合成探针里的 **β*=V/D=2.5、扫描 0…200 全部是自设**，只用来说明「倒 U 形 /
  太强压住改进 / β=0 最差」这一机制，**不能冒充论文数值**。
- **RL token 的瓶颈维度论文未给出**（只称 compact / small vector / bottleneck）。
  探针的 **b=1..12、内在秩 r=6、D=60 全部是自设**；论文里 256/512 是 actor/critic
  MLP hidden、140 是 chunked action 维度，都不是瓶颈维数，切勿混用。
- **γ、σ、warmup 步数、α、update-to-data 比值等论文未给出**：探针的 γ=0.9、
  σ=0.2、σ_feat=0.20 同样是自设。论文明确的 C=10、d=14、H=50、50 Hz、
  chunk=140、dropout 50%、稀疏二值 reward、2 腕+1 基座相机均按事实清单引用。
- **结构被大幅抽象**：encoder–decoder transformer 的自回归教师强制重建在这里退化成
  秩-b 线性 PCA（线性高斯世界下 L2 重建损失的 Bayes 最优解）；VLM backbone、
  diffusion action expert、flow matching、多峰动作分布、图像编码器全部没有建模。
  动作块被压到 6 维任务相关子空间。actor 用 Eq.5 的**闭式最优局部编辑**代替可训练
  Gaussian MLP；critic 用线性岭回归代替 2/3 层 MLP；双 Q 过估计用离散 K 动作 +
  估计噪声的标准抽象。参考动作 dropout 抽象成状态通路有效训练度 s(p)=p/(p+0.15)。
- **信用分配方差用的是线性高斯 reward 下的 REINFORCE 估计量**（Var ∝ N²）；论文是
  稀疏二值 reward + TD3 的 C 步引导。结论「决策数按 C 缩短、方差按 C² 降」是这个世界
  里的结论，稀疏二值部分另用 P(succ)=p_good^N 解析对照，二者不能混成一个量纲。
- **许可证**：RLT / arXiv:2604.23073 是 `arXiv.org perpetual non-exclusive license`
  （与 06 篇 RTC 的 CC BY 4.0 不同）。所有配图按论文**文字重画**，不复用任何官方
  图 / 截图；结构图标注的是论文写明的常量（H=50、C=10、d=14、50 Hz、chunk=140、
  z_rl 维数未给出、β 未给出），不是从图上量出来的。
