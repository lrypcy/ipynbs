"""生成 fm_principles.ipynb：把 fm_principles_lab.py 的实验按 section 拆成 notebook 单元。"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md(r"""# Flow Matching / 扩散模型「原理详解」· 核验 notebook

配套博文：**《Flow Matching 与扩散模型：一份讲到能自己推的原理详解》**
（<https://lrypcy.github.io/>，本次交付后文内会给出 permalink）

底本：Peter Holderrieth & Ezra Erives, *An Introduction to Flow Matching and
Diffusion Models*, MIT 6.S184 (2026) — <https://diffusion.csail.mit.edu/>

**这个 notebook 在做什么。** 把博文里每一条“看起来显然”的恒等式都拿数值算一遍，
让它们的误差落到机器精度，而不是停在“推一遍就相信”。

**为什么残差能做到 $10^{-13}$ 而不是 $10^{-10}$。** 关键在选择 $p_{\rm data}$：
这里把它取成**有限个原子的加权混合**。于是 $p_t$、$\nabla\log p_t$、$u_t$、
$\mathrm{div}(u_t)$、$\Delta p_t$ 全部有解析闭式，偏微分方程的残差不需要任何
数值差分——差分格式本身的截断误差被彻底绕开。

依赖：numpy（纯 CPU，约 2 分钟跑完）。每个 section 末尾自断言，最后汇总。""")

code("""import fm_principles_lab as L
import numpy as np

rng = np.random.default_rng(20261020)
print("numpy", np.__version__)
print("组：A 闭式三兄弟 / B CE+FPE / C Thm12 / D 四参数化 / E Thm17 / F CFG / G 端到端训练")""")

md(r"""## A. 高斯概率路径的三张面孔

讲义 Example 10 给出条件向量场的闭式

$$u_t(x\mid z)=\Big(\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t\Big)z+\frac{\dot\beta_t}{\beta_t}x,$$

它由条件流 $\psi_t(x\mid z)=\alpha_t z+\beta_t x$ 反解得到。Prop 1 再说它等于 score 的仿射组合

$$u_t(x)=a_t\nabla\log p_t(x)+b_tx,\qquad a_t=\beta_t^2\Big(\frac{\dot\alpha_t}{\alpha_t}-\frac{\dot\beta_t}{\beta_t}\Big),\quad b_t=\frac{\dot\alpha_t}{\alpha_t},$$

denoiser（后验均值 $\mathbb E[z\mid x]$）则是同一件事的第三种写法（Eq. 43）。

**要验的是**：这三件事 *逐点* 相等，而不是“理论上等价”。""")

code("n_ok = L.exp_A(rng)")

md(r"""## B. 连续性方程与 Fokker–Planck：分布层面的契约

Thm 11（连续性方程）与 Thm 19（Fokker–Planck）回答同一个问题：
**什么样的动力学能配得上这条概率路径**。

$$\partial_t p_t=-\mathrm{div}(p_tu_t),\qquad
\partial_t p_t=-\mathrm{div}(p_tu_t)+\frac{\sigma_t^2}{2}\Delta p_t .$$

注意第二个式子：**它会对「加了噪声的同一个 $u_t$」判死刑**。
因为 $\partial_t p_t=-\mathrm{div}(p_tu_t)$ 已经由连续性方程钉死，
只要 $\sigma_t\neq0$，右端多出来的 $\frac{\sigma^2}{2}\Delta p_t$ 就必须由漂移 absorbed 掉。
这就是 Thm 17 那个奇怪的修正项 $\frac{\sigma_t^2}{2}\nabla\log p_t$ 的来源——它不是技巧，是必需品。

下表把这个“必需品”的成本直接量化出来。""")

code("n_ok = L.exp_B(rng)")

md(r"""## C. Thm 12：为什么训练目标可以换成条件版的 MSE

flow matching 的全部可扩展性都压在这条定理上：

$$\mathcal L_{\rm FM}(\theta)=\mathcal L_{\rm CFM}(\theta)+C,\qquad C\ \text{与}\ \theta\ \text{无关}.$$

证明只用到「平方展开 + 期望线性化」，没有任何近似。
**但这条定理特别容易被“验错”**：$\mathcal L_{\rm FM}$ 里出现的是边际向量场，
它的重尾（CondOT 下含 $1/(1-t)$ 因子）会让蒙特卡洛估计量的方差大到吞掉常数本身。

判据很简单：**残差必须随网格/样本量收敛**，不收敛就说明是估计量不可用，不是定理错。""")

code("n_ok = L.exp_C(rng)")

md(r"""## D. velocity / score / denoiser / noise：四个名字，一个解

Prop 1 说 $u_t$ 与 $\nabla\log p_t$ 互为仿射重组；Remark 16 进一步说，
任何能反解出 $\mathbb E[z\mid x]$ 的量都行。于是四个参数化
（velocity $u_t$、score $s_t$、denoiser $D_t$、噪声 $\epsilon$）**最优解完全相同**，
差别只在损失函数里那条形如 $a_t^2$ 的**有效时间权重**。

这一组顺带给出两个工程结论：

1. VP 路径下 $a_t\equiv-\gamma/2$ 与 $t$ 无关 ⇒ CFM 与去噪 score matching 只差全局常数；
   CondOT 下 $a_t^2$ 跨越 4 个数量级 ⇒ 两者的时间权重天差地别。
2. 把 velocity 网络当 score 用时必须写成 $s_\theta=(v_\theta-b_tx)/a_t$，
   **不能**写成 $s_\theta=v_\theta/b_t$——丢掉 $b_tx$ 那项的代价见下表。""")

code("n_ok = L.exp_D(rng)")

md(r"""## E. Thm 17：训练完之后，$\sigma_t$ 是一个免费的旋钮

$$dX_t=\Big[u_t(X_t)+\frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\Big]dt+\sigma_t\,dW_t
\ \Longrightarrow\ X_t\sim p_t\quad(\forall\,\sigma_t\ge0).$$

也就是说：ODE 采样（$\sigma\equiv0$）和任意噪声水平的 SDE 采样，边际分布一样。
这张表用 Euler–Maruyama 把这条结论跑出来——**一直到 $\sigma$ 大到步长撑不住为止**。

讲义特别强调：**“最优 $\sigma_t$”是不完美模型 + 有限算力的产物**，
不是连续极限下动力学的性质。最后一行的 $\sigma=2.0$ 就是这句话的实测版本。""")

code("n_ok = L.exp_E(rng)")

md(r"""## F. Classifier-free guidance：从 Bayes 到“锐化”

三步推完：

1. Bayes 分解：$\nabla\log p_t(x\mid y)=\nabla\log p_t(x)+\nabla\log p_t(y\mid x)$，
   代入 Prop 1 得 $u_t(x\mid y)=u_t(x)+a_t\nabla\log p_t(y\mid x)$（classifier guidance）。
2. 把第二项放大 $w$ 倍，再把 $\nabla\log p_t(y\mid x)$ 反解成两个场的差，得到
   $\tilde u_t(x\mid y)=(1-w)u_t(x\mid\emptyset)+wu_t(x\mid y)$（CFG）。
3. 于是 CFG 场恰好是**倾斜分布** $\tilde p_t(x)\propto p_t(x)\,p_t(y\mid x)^w$ 的 Prop-1 速度场。

第 3 步是理解 CFG 的关键：当 $w=1$ 时 $\tilde p=p(\cdot\mid y)$，推断正确；
$w>1$ 时它采的是被 $p_t(y\mid x)^w$ 锐化过的分布——**CFG 是启发式，不是后验推断**。""")

code("n_ok = L.exp_F(rng)")

md(r"""## G. 端到端：不借 torch，用 numpy 手训两个头

最后把整条流水线跑一遍：同一个数据、同一套超参，训练

* **velocity 头**：回归 $u_t(x\mid z)=A_tz+B_tx$（CondOT）
* **$\epsilon$ 头**：回归噪声 $\epsilon$（DDPM 形态），再用 $z=(x-\beta\epsilon)/\alpha$ 换回 velocity

如果「四个参数化同解」是真的，两个头换算回 velocity 之后应该给出同一个函数。
真的差别在**换算里的 $1/\alpha_t$**：$t$ 越小，同样的 $\epsilon$ 预测误差被放得越大——
这才是 CondOT 下工业界一律用 velocity 头的理由（不是表达力更强）。""")

code("n_ok = L.exp_G()")

md(r"""## 结论速记

| 结论 | 实测 |
| :--- | :--- |
| 向量场 / score / denoiser 逐点相等 | 误差 $\le 5\times10^{-14}$，5 条路径一致 |
| 连续性方程是路径无关的契约 | 相对残差 $\le 6\times10^{-15}$ |
| 裸 $u_t$ + 噪声 ≠ 同一条 $p_t$ | $\sigma=0.8$ 时 FPE 残差 $2.43$ |
| 加上 $\frac{\sigma^2}{2}\nabla\log p_t$ 之后 | 残差回到 $5\times10^{-14}$ |
| $\mathcal L_{\rm FM}-\mathcal L_{\rm CFM}$ 与 $\theta$ 无关 | 求积相对跨度 $\le 10^{-12}$；MC 完全淹没 |
| 四种参数化同解 | 来回换算误差 $\le 2\times10^{-14}$ |
| $\sigma_t$ 任意可选 | $\sigma\in[0,1]$ 四档终点 W1 极差 $0.02$ |
| CFG ≡ 倾斜分布的真实速度场 | 逐点误差 $\le 8\times10^{-14}$ |
| velocity 头 vs $\epsilon$ 头 | $t=0.05$ 处后者 RMSE 高一个数量级 |

想复跑：`python3 fm_principles_lab.py`（默认 A–F；加参数可选组，如 `fm_principles_lab.py C E`，
所有输出已固化在 `results/stdout.txt`）。""")

nb["cells"] = c
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.11"}
nbf.write(nb, "fm_principles.ipynb")
print("written", len(c), "cells")
