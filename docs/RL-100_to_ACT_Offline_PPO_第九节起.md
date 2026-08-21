# 九、现在换成 ACT：绝大部分 Offline RL 都不用改

这是对 ARBiM 最重要的结论。

可以把 RL-100 分成：

### A. 与 Diffusion 无关的部分

这些都可以直接迁移：

\[
\boxed{
Dataset
}
\]

\[
\boxed{
Reward
}
\]

\[
\boxed{
IQL\ Q/V
}
\]

\[
\boxed{
A^{off}=Q-V
}
\]

\[
\boxed{
PPO clipping
}
\]

\[
\boxed{
Behavior/reference policy
}
\]

\[
\boxed{
BC regularization
}
\]

\[
\boxed{
OPE
}
\]

\[
\boxed{
Offline\rightarrow Online
}
\]

\[
\boxed{
Dataset expansion
}
\]

这些恰恰才是 RL-100 最值得你迁移的部分。

---

# 十、ACT 真正缺的只有一个核心东西

普通 ACT：

\[
s
\xrightarrow{ACT}
A_t
\]

其中：

\[
A_t
=
[a_t,a_{t+1},...,a_{t+H-1}]
\]

但你的 ACT 推理通常是 deterministic：

\[
\boxed{
A_t=\mu_\theta(s_t)
}
\]

于是：

\[
\pi_\theta(A_t|s_t)
\]

其实是 Dirac delta。

那么：

\[
\log\pi_\theta(A_t|s_t)
\]

不能像 PPO 那样正常使用。

因此：

# ACT → PPO 的真正桥梁不是 IQL。

而是：

\[
\boxed{
\text{把 ACT 变成 stochastic policy}
}
\]

---

# 十一、最直接的 ACT-PPO 数学形式

ACT 原来输出：

\[
\mu_\theta(s)
\in
\mathbb R^{H\times d}
\]

我们增加：

\[
\sigma_\theta(s)
\]

于是定义：

\[
\boxed{
\pi_\theta(A|s)
=
\mathcal N
(
\mu_\theta(s),
\operatorname{diag}(\sigma_\theta^2(s))
)
}
\]

那么：

\[
A=
[a_1,\ldots,a_H]
\]

的 log probability：

\[
\boxed{
\log\pi_\theta(A|s)
=
-\frac12
\sum_{j=1}^{H}
\sum_{d=1}^{D}
\left[
\frac{
(A_{j,d}-\mu_{j,d})^2
}{
\sigma_{j,d}^2
}
+
2\log\sigma_{j,d}
+
\log2\pi
\right]
}
\]

这时候 ACT 就正式拥有了 PPO 所需的：

\[
\boxed{
\log\pi_\theta(A|s)
}
\]

---

# 十二、这时候 ACT 的 Offline PPO 反而比 Diffusion 简单很多

Diffusion：

\[
s
\rightarrow
x_K
\rightarrow...
\rightarrow
x_0=A
\]

有：

\[
K
\]

个内部 PPO ratio。

ACT：

\[
s
\rightarrow
A
\]

没有 inner denoising MDP。

所以直接：

\[
A^{old}
\sim
\pi_{\rm ACT,old}(\cdot|s)
\]

Critic：

\[
\mathcal A_t
=
Q(s,A^{old})-V(s)
\]

ratio：

\[
\rho_t=
\exp[
\log\pi_\theta(A^{old}|s)
-
\log\pi_{\rm old}(A^{old}|s)
]
\]

loss：

\[
\boxed{
L^{ACT}_{PPO}
=
-
\min
[
\rho_t\mathcal A_t,\;
clip(\rho_t)\mathcal A_t
]
}
\]

**没有：**

\[
\sum_{k=1}^K
\]

因为 ACT 根本没有 diffusion timestep。

这一点你可以理解成：

> RL-100 为 Diffusion 做了一大堆“把生成过程改造成 PPO policy”的工程。
>
> ACT 如果直接拥有 Gaussian action head，反而不需要这些复杂东西。

---

# 十三、ACT 天然是 Action Chunk，所以这里还有一层非常关键

ACT：

\[
A_t=
[a_t,\ldots,a_{t+H-1}]
\]

你必须决定：

> 一个 chunk 到底是不是一个 RL action？

我认为第一版 ARBiM 最值得采用的是：

\[
\boxed{\text{Chunk as one macro-action}}
\]

这其实 RL-100 已经提供了数学和代码支持。

---

## Chunk reward

如果实际执行 \(H\) 步：

\[
\boxed{
R_t^{chunk}
=
\sum_{j=0}^{H-1}
\gamma^j r_{t+j}
}
\]

下一决策状态：

\[
s_{t+H}
\]

于是：

\[
Q(s_t,A_t)
\]

的 Bellman target：

\[
\boxed{
y_t=
R_t^{chunk}
+
\gamma^H
(1-d)
V(s_{t+H})
}
\]

RL-100 的 critic 代码在 `chunk_as_single_action` 情况下确实使用：

\[
\gamma^{n_{\rm action\ steps}}
\]

而不是普通 \(\gamma\)。

论文 supplementary 也明确写：

\[
R^{chunk}
=
\sum_{j=0}^{H-1}
\gamma^jR_{t+j}
\]

以及 chunk 间：

\[
\boxed{\gamma^{H}}
\]

这和 ACT 非常契合。

---

# 十四、于是你的 ACT Offline RL 可以得到一个非常干净的公式体系

我建议你以后把这一套当作 **ARBiM Offline PPO baseline**。

---

## ① Dataset

\[
\boxed{
\mathcal D=
\{
(s_t,A_t,R_t^{chunk},s_{t+H},d_t)
\}
}
\]

---

## ② IQL Q

\[
\boxed{
y_t=
R_t^{chunk}
+
\gamma^H(1-d_t)V(s_{t+H})
}
\]

\[
\boxed{
L_Q=
\mathbb E
[
(Q(s_t,A_t)-y_t)^2
]
}
\]

---

## ③ IQL V

\[
\delta_t
=
Q(s_t,A_t)-V(s_t)
\]

\[
\boxed{
L_V
=
\mathbb E[
|\tau-\mathbf 1(\delta_t<0)|
\delta_t^2
]
}
\]

---

## ④ Offline Advantage

现在 PPO 不一定使用 dataset action。

从 reference ACT：

\[
A_t^{old}
\sim
\pi_{\rm old}(A|s_t)
\]

计算：

\[
\boxed{
\hat A_t^{off}
=
Q(s_t,A_t^{old})
-
V(s_t)
}
\]

然后 batch normalize：

\[
\tilde A_t
=
\frac{
A_t-\mu_A
}{
\sigma_A+\epsilon
}
\]

RL-100 当前实现也对 IQL advantage 做 batch normalization。

---

## ⑤ ACT PPO ratio

\[
\boxed{
\rho_t
=
\exp
[
\log\pi_\theta(A_t^{old}|s_t)
-
\log\pi_{\rm old}(A_t^{old}|s_t)
]
}
\]

---

## ⑥ Actor Loss

\[
\boxed{
L_{\rm actor}
=
-
\mathbb E
\left[
\min(
\rho_t\tilde A_t,\;
clip(\rho_t,1-\epsilon,1+\epsilon)
\tilde A_t
)
\right]
}
\]

---

## ⑦ 再加一个 IL anchor

实际机器人我不会建议只有 PPO：

\[
\boxed{
L_{\rm total}
=
L_{\rm PPO}
+
\lambda_{BC}L_{BC}
}
\]

例如：

\[
L_{BC}
=
\|
\mu_\theta(s)-A^{dataset}
\|^2
\]

这样：

\[
\text{PPO}
\]

负责：

> 往高 value 行为移动。

而

\[
BC
\]

负责：

> 不要把已有 ACT 行为先验毁掉。

RL-100 本身的 offline PPO 实现也留了 optional BC loss。

---

# 十五、这里我反而不推荐 ACT 第一版直接用整个 Chunk 的 joint ratio

原因非常实际。

假设：

\[
H=50
\]

双臂：

\[
D=14
\]

整个 chunk 是：

\[
700
\]

维。

那么：

\[
\log\pi(A|s)
=
\sum_{700\ dims}
\log p(a_i)
\]

于是：

\[
\rho
=
e^{\Delta\log p}
\]

非常容易：

\[
\rho\gg1
\]

或者：

\[
\rho\approx0
\]

PPO 会疯狂 clipping。

RL-100 当前代码已经显式支持：

- chunk-level joint ratio
- per-action-step ratio

两种处理，而且代码里专门增加了 ratio diagnostics。

所以 ACT 第一版我更倾向：

\[
\boxed{
\rho_{t,j}
=
\exp
\left[
\sum_d
\log\pi_\theta(a_{t,j,d}|s)
-
\sum_d
\log\pi_{\rm old}(a_{t,j,d}|s)
\right]
}
\]

然后让整个 chunk 的：

\[
A_t
\]

共享给 \(H\) 个 action positions：

\[
\boxed{
L
=
-\frac1H
\sum_{j=1}^{H}
\min(
\rho_{t,j}A_t,
clip(\rho_{t,j})A_t
)
}
\]

这个结构其实和 RL-100：

> 一个 environment advantage 共享给多个 denoising steps

非常相似。

只是现在变成：

> 一个 chunk advantage 共享给 chunk 内多个 action positions。

这是我认为非常值得实验的 ACT 版本。

---

# 十六、还有一个你一定会想到的问题：ACT 不是本来就有 CVAE 吗？

有。

但是千万不要因此直接认为：

> ACT 本来就是 stochastic policy。

这不完全对。

ACT 的 CVAE 通常是：

\[
z\sim N(0,I)
\]

然后：

\[
A=f_\theta(s,z)
\]

但是 PPO 要的是：

\[
\boxed{
\log p(A|s)
}
\]

而：

\[
p(A|s)
=
\int
p(A|s,z)p(z)dz
\]

通常并不能从 ACT 原来的 decoder 里方便地算出来。

所以：

\[
\boxed{
\text{ACT 有 latent}
\neq
\text{ACT 已经有 PPO-ready log probability}
}
\]

这是你之后设计时千万不能混淆的点。

---

# 十七、ACT 可以有三种改法

按照我建议的优先级：

### 方案 A：Gaussian ACT head

最直接：

\[
ACT(s)
\rightarrow
(\mu,\log\sigma)
\]

然后：

\[
A\sim N(\mu,\sigma^2)
\]

优点：

- PPO 最标准；
- log_prob 清楚；
- 代码最容易检查；
- 和你现在学的 continuous PPO 完全一致。

**最推荐作为第一版。**

---

### 方案 B：ACT + stochastic residual

保持原 ACT：

\[
A_{\rm base}=ACT(s)
\]

RL 只输出：

\[
\Delta A
\sim
N(
\mu_\phi(s),
\sigma_\phi(s)^2
)
\]

执行：

\[
\boxed{
A=A_{\rm base}+\Delta A
}
\]

然后 PPO 实际优化：

\[
\pi_\phi(\Delta A|s)
\]

这对 ARBiM 很有吸引力。

因为可以：

\[
\Delta A_R=0
\]

只允许：

\[
\Delta A_L
\]

改变。

那么天然变成：

\[
\boxed{
Base\ ACT
+
Left\ stochastic\ residual
}
\]

这甚至比直接 PPO 整个 ACT 更符合你“只强化弱侧，不破坏强侧”的研究目标。

不过这是 **我们为 ACT / ARBiM 提出的设计**，不是 RL-100 论文的方法。

---

### 方案 C：对 ACT latent \(z\) 做 PPO

定义：

\[
z\sim\pi_\phi(z|s)
\]

然后：

\[
A=ACT(s,z)
\]

PPO ratio 定义在：

\[
\pi_\phi(z|s)
\]

上。

优点是不用直接给 700 维 action chunk 做 probability。

但：

\[
Q(s,ACT(s,z))
\]

以及 latent exploration 的语义会复杂很多。

所以我不建议第一版从这里开始。

---

# 十八、RL-100 里面还有哪些东西几乎可以原封不动迁到 ACT？

我会把它们分成三档。

| RL-100 组件 | ACT 能否迁移 | 怎么处理 |
|---|---|---|
| Offline Dataset | ✅ | 不变 |
| Reward | ✅ | 不变 |
| IQL Q/V | ✅ | Q 输入换成 ACT chunk |
| \(A=Q-V\) | ✅ | 不变 |
| PPO clipping | ✅ | 不变 |
| Reference old policy | ✅ | 不变 |
| BC anchor | ✅ | 强烈建议 |
| Dataset expansion | ✅ | 不变 |
| Offline → Online | ✅ | 不变 |
| Dynamics / OPE | ✅ | 可以继续研究 |
| encoder freeze | ✅ | 很适合 ACT |
| Action chunk MDP | ✅✅ | 与 ACT 天然匹配 |
| Denoising MDP | ❌ | ACT 不需要 |
| DDIM stochastic transition | ❌ | 换 Gaussian ACT |
| Flow SDE/CPS | ❌ | ACT 不需要 |
| denoise-step shared A | ❌ | 可研究 chunk-step shared A |
| CM distillation | ❌ | ACT 本身已经单次前向 |

这张表基本就是：

> **RL-100 → ACT 应该拆什么、保留什么。**

---

# 十九、OPE 这里还有一个论文和仓库实际实现的差别

这个值得单独告诉你。

论文描述得更像：

\[
J_{\rm OPE}(\pi_{\rm new})
-
J_{\rm OPE}(\pi_{\rm old})
\geq \delta
\]

才“accept” candidate。

但我们现在看的公开仓库实际代码里，是：

```python
if current_mean_qs > best_mean_qs:
    best_mean_qs = current_mean_qs
    self.unio4.set_old_policy()
```

也就是说，当前实现更准确地说是：

> **OPE 提高 → 刷新 PPO reference / behavior policy。**

不是严格：

> **这次 gradient 不好 → rollback 整个 candidate。**

这正是我们之前代码阅读笔记里强调过的区别。

如果 ARBiM 后面写论文，你要明确区分：

- RL-100 paper 的 OPE-gated narrative；
- RL-100 repository 当前实现；
- 我们自己的 ACT OPE gate 设计。

---

# 二十、把整个 ACT Offline PPO 收缩成你现在应该记住的一条链

你之前 Online PPO 记的是：

\[
\boxed{
Reward
\rightarrow
Return/Value
\rightarrow
GAE
\rightarrow
Advantage
\rightarrow
PPO
}
\]

现在 Offline ACT 你可以记成：

\[
\boxed{
Offline\ trajectories
}
\]

↓

\[
\boxed{
Reward\ relabel
}
\]

↓

\[
\boxed{
IQL:
Q(s,A),V(s)
}
\]

↓

\[
\boxed{
A^{off}(s,A)=Q(s,A)-V(s)
}
\]

↓

\[
\boxed{
Stochastic\ ACT
}
\]

↓

\[
\boxed{
\log\pi_{\rm new}
-
\log\pi_{\rm old}
}
\]

↓

\[
\boxed{
PPO\ Ratio
}
\]

↓

\[
\boxed{
Clipped\ Actor\ Update
}
\]

↓

\[
\boxed{
BC/KL\ anchor
}
\]

↓

\[
\boxed{
Improved\ ACT
}
\]

这就是我认为 **RL-100 迁移到 ACT 的最核心数学骨架**。

---

# 二十一、对 ARBiM，我现在会把第一版设计收敛到这个结构

不是一开始复制 Diffusion 的 two-level MDP，而是：

\[
\boxed{
\text{Base ACT}
\rightarrow
\text{Stochastic ACT / Residual ACT}
}
\]

数据：

\[
\mathcal D=
(s,A,R^{chunk},s')
\]

Critic：

\[
Q_\psi(s,A),\quad V_\phi(s)
\]

Advantage：

\[
A^{off}=Q-V
\]

Actor：

\[
\pi_\theta(A|s)
\]

优化：

\[
L
=
L_{\rm PPO}
+
\lambda_{BC}L_{\rm BC}
\]

如果进入非对称阶段，再进一步：

\[
\boxed{
L
=
L^{L}_{PPO}
+
\lambda_R
L^{R}_{preserve}
+
\lambda_{BC}L_{BC}
}
\]

或者更加干净地：

\[
\boxed{
A^{execute}
=
A^{BaseACT}
+
M_L\odot\Delta A^{RL}
}
\]

其中：

\[
M_L
\]

只开放左手维度。

这样你实际上就把：

> **RL-100 的 Conservative IL → Offline RL Post-training**

迁移成了：

> **ACT prior → IQL critic → stochastic/residual PPO → asymmetric update**

而不是硬把 Diffusion 的 denoising machinery 搬进 ACT。

这两者应该明确区分。

---

你现在接下来最值得继续深挖的其实就是 **IQL 本身**。因为从这里开始，Offline PPO 唯一真正新增、而你在 Online PPO 里没有系统学过的核心模块，就是：

\[
\boxed{
\text{Dataset}
\rightarrow Q
\rightarrow V
\rightarrow Q-V
}
\]

尤其是三个问题：

\[
\text{为什么 Offline 不直接用 Return？}
\]

\[
\text{为什么 IQL 要单独训练 Q 和 V？}
\]

\[
\text{为什么 }Q-V\text{ 能够指导一个“没有执行过的新动作”？}
\]

这三个搞明白之后，后面的 ACT Offline PPO 基本就会彻底串起来了。
