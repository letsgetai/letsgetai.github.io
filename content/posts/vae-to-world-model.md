---
title: "从 VAE 到世界模型：从概率建模到 Sequential Latent World Model（图解版）"
date: 2026-09-27
draft: false
categories: ["Machine Learning"]
---

> 目标：用一条连续的逻辑链，把“世界模型是什么、为什么需要 latent state、为什么会出现 posterior、ELBO 从哪里来、世界模型的 loss 为什么长成 reconstruction + dynamics KL”讲清楚。  
> 这不是论文综述，而是一份偏“推导 + 直觉”的入门笔记。

---

## 1. 世界模型到底是什么

最核心的直觉是：

> **世界模型学习一个内部世界状态，并预测这个世界在动作作用下如何变化。**

最简形式：

$$
(z_t, a_t) \rightarrow z_{t+1}
$$

其中：

- $o_t$：当前观测（图像、深度、传感器等）
- $s_t$：真实世界状态，通常不可直接观测
- $z_t$：模型内部学习到的 latent state
- $a_t$：动作

可以把世界模型拆成三个模块：

$$
\boxed{
\text{Representation}
+
\text{Dynamics}
+
\text{Decision Interface}
}
$$

对应：

$$
o_t \rightarrow z_t
$$

$$
(z_t,a_t)\rightarrow z_{t+1}
$$

$$
z_t \rightarrow a_t
$$

其中第三项不一定直接内置在 world model 本体里，也可能由 policy / planner 负责。

![世界模型的最小结构](/images/vae-to-world-model/fig01_world_model_architecture.svg)

*图 1：世界模型最小结构。表示模块把观测压缩为内部状态，动力学模块预测动作作用后的状态变化，决策接口利用内部状态产生动作。*

---

## 2. 为什么会自然进入 POMDP

现实里，agent 通常看不到真正的状态 $s_t$，只能看到观测 $o_t$。

因此我们可以写：

$$
s_t \rightarrow o_t
$$

而状态在动作作用下变化：

$$
p(s_t|s_{t-1},a_{t-1})
$$

于是，一个最基本的生成过程是：

$$
s_{t-1},a_{t-1}\rightarrow s_t \rightarrow o_t
$$

整个联合分布可以写成：

$$
p(s_{0:T},o_{1:T}|a_{0:T-1}) =
p(s_0)
\prod_{t=1}^{T}
p(s_t|s_{t-1},a_{t-1})
p(o_t|s_t)
$$

这个式子来自两件事：

1. 联合概率的 chain rule
2. POMDP / Markov 假设带来的条件独立性

---

## 3. 为什么要引入 latent state

真实 $s_t$ 通常不可见。

比如机器人真正的世界状态可能包含：

- 所有物体的 3D 位置
- 速度
- 遮挡物体
- 接触状态
- 摩擦系数
- 材质
- 其他 agent 的内部状态

但 agent 只能看到 RGB、Depth、Proprioception 等观测。

因此我们引入一个学习得到的内部状态：

$$
z_t
$$

希望它成为：

$$
z_t \approx \text{history 的有效压缩}
$$

也就是：

$$
z_t \approx f(o_{\le t},a_{\lt t})
$$

并且它应该满足两个要求：

$$
\boxed{
\text{能解释当前世界}
}
$$

以及：

$$
\boxed{
\text{能从过去预测出来}
}
$$

后面 world model 的 loss，本质上就是在同时约束这两件事。

---

## 4. 先从静态 VAE 开始

世界模型的 variational learning 很像 VAE，所以先把 VAE 的逻辑推清楚。

### 4.1 VAE 的生成假设

假设一张图片 $x$ 是由 latent variable $z$ 生成的：

$$
z \sim p(z)
$$

$$
x \sim p_\theta(x|z)
$$

其中：

- $p(z)$：latent prior，通常取 $N(0,I)$
- $p_\theta(x|z)$：decoder / likelihood
- $\theta$：decoder 的参数

所以联合分布：

$$
p_\theta(x,z) =
p(z)p_\theta(x|z)
$$

![VAE 的生成方向与推断方向](/images/vae-to-world-model/fig02_vae_generative_inference.svg)

*图 2：VAE 同时包含生成方向（$z\rightarrow x$）和推断方向（$x\rightarrow q_\phi(z|x)$）。*

---

## 5. VAE 真正想优化什么

我们的目标不是“重建图片”本身，而是 maximum likelihood：

$$
\boxed{
\max_\theta \log p_\theta(x)
}
$$

也就是说：

> 希望训练数据 $x$ 在模型下出现的概率尽可能大。

由于 $z$ 没有被观测到：

$$
p_\theta(x) =
\int p_\theta(x,z)\,dz
$$

展开：

$$
p_\theta(x) =
\int p(z)p_\theta(x|z)\,dz
$$

这个积分的含义是：

> 把所有可能生成当前 $x$ 的 latent $z$ 的贡献全部加起来。

---

## 6. 为什么这个积分难算

难点不是 $p_\theta(x|z)$ 本身不能算。

给定一个具体 $x,z$，通常可以直接计算：

$$
p_\theta(x|z)
$$

真正难的是：

$$
p_\theta(x) =
\int p(z)p_\theta(x|z)\,dz
$$

原因主要有两个。

### 6.1 $z$ 是高维连续变量

例如：

$$
z\in \mathbb R^{128}
$$

这就是一个 128 维积分。

### 6.2 Decoder 关于 $z$ 是复杂非线性函数

例如：

$$
p_\theta(x|z) =
N(x;f_\theta(z),\sigma^2I)
$$

其中：

$$
f_\theta(z)
$$

是一个深度神经网络。

于是：

$$
p_\theta(x) =
\int
N(z;0,I)
N(x;f_\theta(z),\sigma^2I)
dz
$$

一般没有解析解。

---

## 7. Monte Carlo 能不能近似这个积分

可以。

因为：

$$
p_\theta(x) =
E_{z\sim p(z)}
[p_\theta(x|z)]
$$

因此可以从 prior 中采样：

$$
z^{(1)},\dots,z^{(K)}
\sim p(z)
$$

然后：

$$
p_\theta(x)
\approx
\frac{1}{K}
\sum_{k=1}^{K}
p_\theta(x|z^{(k)})
$$

这就是 Monte Carlo estimation。

问题是：

> 对当前特定的 $x$，从整个 prior $p(z)$ 乱采，大部分 $z$ 都不能很好解释它。

所以估计效率低、方差大。

---

## 8. 我们真正希望知道的是 posterior

给定当前 $x$，我们更希望知道：

$$
p_\theta(z|x)
$$

它回答：

> 哪些 latent $z$ 最可能解释当前 $x$？

Bayes 公式：

$$
p_\theta(z|x) =
\frac{
p_\theta(x|z)p(z)
}{
p_\theta(x)
}
$$

问题又回来了：

$$
p_\theta(x) =
\int p(z)p_\theta(x|z)dz
$$

难算。

因此：

$$
p_\theta(z|x)
$$

也通常难算。

---

## 9. 引入 Encoder：approximate posterior

所以引入：

$$
q_\phi(z|x)
$$

去近似：

$$
p_\theta(z|x)
$$

也就是：

$$
q_\phi(z|x)
\approx
p_\theta(z|x)
$$

在标准 VAE 中，Encoder 常输出：

$$
\mu_\phi(x),\sigma_\phi(x)
$$

定义：

$$
q_\phi(z|x) =
N(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
)
$$

这意味着：

> Encoder 不是把 $x$ 编码成唯一一个 $z$，而是编码成一个关于 $z$ 的分布。

---

## 10. ELBO 是怎么出现的

真正目标仍然是：

$$
\log p_\theta(x)
$$

先从：

$$
p_\theta(x) =
\int p_\theta(x,z)dz
$$

开始。

乘一个 1：

$$
\frac{q_\phi(z|x)}{q_\phi(z|x)}
$$

得到：

$$
p_\theta(x) =
\int
q_\phi(z|x)
\frac{
p_\theta(x,z)
}{
q_\phi(z|x)
}
dz
$$

根据期望定义：

$$
E_q[f(z)] =
\int q(z)f(z)dz
$$

所以：

$$
p_\theta(x) =
E_{q_\phi(z|x)}
\left[
\frac{
p_\theta(x,z)
}{
q_\phi(z|x)
}
\right]
$$

注意：到这里全部还是等号。

取 log：

$$
\log p_\theta(x) =
\log
E_q
\left[
\frac{
p_\theta(x,z)
}{
q_\phi(z|x)
}
\right]
$$

由于 $\log$ 是凹函数，Jensen 不等式：

$$
\log E[Y]
\ge
E[\log Y]
$$

于是：

$$
\log p_\theta(x)
\ge
E_q
\left[
\log
\frac{
p_\theta(x,z)
}{
q_\phi(z|x)
}
\right]
$$

右边定义为 ELBO：

$$
\boxed{
\mathcal L_{\text{ELBO}} =
E_q
\left[
\log
\frac{
p_\theta(x,z)
}{
q_\phi(z|x)
}
\right]
}
$$

![ELBO 为什么出现](/images/vae-to-world-model/fig03_elbo_derivation.svg)

*图 3：Jensen 不等式把难算的 marginal likelihood 变成可优化的下界（ELBO），latent 积分由此进入 variational inference 框架。*

---

## 11. 把 ELBO 展开成 VAE loss

联合分布：

$$
p_\theta(x,z) =
p(z)p_\theta(x|z)
$$

代入：

$$
\mathcal L_{\text{ELBO}} =
E_q
[
\log p_\theta(x|z)
+
\log p(z) -
\log q_\phi(z|x)
]
$$

整理：

$$
\boxed{
\mathcal L_{\text{ELBO}} =
E_q[
\log p_\theta(x|z)
] -
KL(
q_\phi(z|x)
\|
p(z)
)
}
$$

因此如果用 loss 最小化形式：

$$
\boxed{
L_{\text{VAE}} =
-\mathcal L_{\text{ELBO}}
}
$$

即：

$$
\boxed{
L_{\text{VAE}} =
L_{\text{reconstruction}}
+
L_{\text{KL}}
}
$$

---

## 12. VAE 训练时代码里到底算什么

给定一张图 $x$。

### 12.1 Encoder

$$
x
\rightarrow
\mu,\sigma
$$

得到：

$$
q_\phi(z|x)
$$

### 12.2 Reparameterization

采：

$$
\epsilon\sim N(0,I)
$$

然后：

$$
z =
\mu+\sigma\odot\epsilon
$$

### 12.3 Decoder

$$
z\rightarrow \hat x
$$

### 12.4 Reconstruction loss

如果：

$$
p_\theta(x|z) =
N(\hat x,\sigma_x^2I)
$$

那么：

$$
-\log p_\theta(x|z)
\propto
\|x-\hat x\|^2
$$

所以常见实现就是 MSE。

如果采用 Bernoulli likelihood，则常对应 BCE。

### 12.5 KL loss

如果：

$$
q_\phi(z|x)=N(\mu,\sigma^2)
$$

且：

$$
p(z)=N(0,I)
$$

那么 KL 有闭式：

$$
KL(q||p) =
\frac12
\sum_j
\left(
\mu_j^2+\sigma_j^2-\log \sigma_j^2-1
\right)
$$

---

## 13. 从 VAE 转向世界模型：不要直接“替换符号”

现在重新从环境本身开始推。

假设我们已经维护一个 latent state：

$$
z_{t-1}
$$

执行动作：

$$
a_{t-1}
$$

那么模型需要预测：

$$
\boxed{
p_\theta(z_t|z_{t-1},a_{t-1})
}
$$

这就是 dynamics prior。

它回答：

> 不看当前真实 observation，只根据过去和动作，我认为当前 latent state 应该是什么。

---

## 14. 为什么只有 dynamics prior 不够

因为预测会有误差。

原因包括：

- 环境随机性
- actuator noise
- hidden disturbance
- model error
- 上一时刻 latent 本身的不确定性

于是当前真实 observation：

$$
o_t
$$

来了以后，我们希望利用它修正对 $z_t$ 的判断。

因此需要：

$$
\boxed{
q_\phi(
z_t
|
z_{t-1},a_{t-1},o_t
)
}
$$

这就是 filtered posterior / inference model。

![Sequential world model 的预测与修正](/images/vae-to-world-model/fig04_sequential_world_model.svg)

*图 4：Dynamics prior 只基于过去预测当前状态；filtered posterior 额外利用当前观测 $o_t$ 修正状态估计。*

它对应：

$$
\boxed{
\text{predict}
\rightarrow
\text{observe}
\rightarrow
\text{correct}
}
$$

---

## 15. 为什么还需要 observation model

只定义一个 latent $z_t$ 还不够。

否则模型完全可以学出一个毫无意义的内部编码。

所以需要：

$$
\boxed{
p_\theta(o_t|z_t)
}
$$

要求：

> 当前 latent state 必须能解释真实 observation。

这样 $z_t$ 才被迫包含和真实世界相关的信息。

---

## 16. Sequential Latent World Model 的三个核心模块

于是自然得到：

### 16.1 Dynamics Prior

$$
p_\theta(z_t|z_{t-1},a_{t-1})
$$

不看 $o_t$，只做预测。

### 16.2 Filtered Posterior

$$
q_\phi(z_t|z_{t-1},a_{t-1},o_t)
$$

看到了真实 $o_t$，对预测进行修正。

### 16.3 Observation Model

$$
p_\theta(o_t|z_t)
$$

保证 latent 能解释真实世界。

---

## 17. 整个生成模型怎么写

初始状态：

$$
z_0\sim p(z_0)
$$

每一步：

$$
z_t
\sim
p_\theta(z_t|z_{t-1},a_{t-1})
$$

然后：

$$
o_t
\sim
p_\theta(o_t|z_t)
$$

所以：

$$
\boxed{
p_\theta(
o_{1:T},z_{0:T}
|
a_{0:T-1}
) =
p_\theta(z_0)
\prod_{t=1}^{T}
p_\theta(z_t|z_{t-1},a_{t-1})
p_\theta(o_t|z_t)
}
$$

---

## 18. 世界模型真正想最大化什么

训练数据里真正观测到的是：

$$
o_{1:T},a_{0:T-1}
$$

而不是：

$$
z_{0:T}
$$

所以真正想最大化的是：

$$
\boxed{
\log p_\theta(
o_{1:T}
|
a_{0:T-1}
)
}
$$

由于 latent trajectory 没有观测：

$$
p_\theta(o_{1:T}|a) =
\int
p_\theta(o_{1:T},z_{0:T}|a)
dz_{0:T}
$$

这个积分一般仍然难算。

---

## 19. 为什么真实 posterior 也难算

真实 posterior 是：

$$
p_\theta(
z_{0:T}
|
o_{1:T},a_{0:T-1}
)
$$

即：

> 已经看到整段 observation 和 action 后，整条 latent trajectory 最可能是什么。

即使只看单步：

$$
p_\theta(
z_t
|
z_{t-1},a_{t-1},o_t
)
$$

根据 Bayes：

$$
p_\theta(
z_t
|
z_{t-1},a_{t-1},o_t
) =
\frac{
p_\theta(o_t|z_t)
p_\theta(z_t|z_{t-1},a_{t-1})
}{
p_\theta(o_t|z_{t-1},a_{t-1})
}
$$

分母：

$$
p_\theta(o_t|z_{t-1},a_{t-1}) =
\int
p_\theta(o_t|z_t)
p_\theta(z_t|z_{t-1},a_{t-1})
dz_t
$$

又出现了 nonlinear high-dimensional integral。

所以一般需要 approximate posterior：

$$
q_\phi(
z_t
|
z_{t-1},a_{t-1},o_t
)
$$

---

## 20. 世界模型 ELBO 的推导

目标：

$$
\log p_\theta(o_{1:T}|a_{0:T-1})
$$

引入：

$$
q_\phi(z_{0:T}|o_{1:T},a_{0:T-1})
$$

和 VAE 一样：

$$
p_\theta(o|a) =
\int
q_\phi(z|o,a)
\frac{
p_\theta(o,z|a)
}{
q_\phi(z|o,a)
}
dz
$$

写成期望：

$$
p_\theta(o|a) =
E_q
\left[
\frac{
p_\theta(o,z|a)
}{
q_\phi(z|o,a)
}
\right]
$$

取 log，再用 Jensen：

$$
\log p_\theta(o|a)
\ge
E_q
\left[
\log
\frac{
p_\theta(o,z|a)
}{
q_\phi(z|o,a)
}
\right]
$$

这就是 sequential ELBO。

---

## 21. 利用 Markov factorization 展开

生成模型：

$$
p_\theta(o,z|a) =
p_\theta(z_0)
\prod_{t=1}^{T}
p_\theta(z_t|z_{t-1},a_{t-1})
p_\theta(o_t|z_t)
$$

近似 posterior：

$$
q_\phi(z|o,a) =
q_\phi(z_0|o_0)
\prod_{t=1}^{T}
q_\phi(
z_t
|
z_{t-1},a_{t-1},o_t
)
$$

代入并整理以后：

$$
\boxed{
\mathcal L_{\text{WM}} =
\sum_{t=1}^{T}
E_q[
\log p_\theta(o_t|z_t)
] -
\sum_{t=1}^{T}
KL(q_t||p_t) -
KL(q(z_0|o_0)||p(z_0))
}
$$

其中：

$$
q_t =
q_\phi(
z_t
|
z_{t-1},a_{t-1},o_t
)
$$

$$
p_t =
p_\theta(
z_t
|
z_{t-1},a_{t-1}
)
$$

---

## 22. 世界模型的 loss

训练时通常最小化 negative ELBO：

$$
\boxed{
L_{\text{WM}} =
-\mathcal L_{\text{WM}}
}
$$

所以：

$$
\boxed{
L_{\text{WM}} =
\sum_t
L_{\text{reconstruction},t}
+
\sum_t
L_{\text{dynamics KL},t}
+
L_{\text{initial KL}}
}
$$

如果暂时忽略初始状态：

$$
\boxed{
L_t =
-\mathbb E_{q_t}
[
\log p_\theta(o_t|z_t)
]
+
KL(q_t||p_t)
}
$$

---

## 23. 两项 loss 分别约束什么

### 23.1 Reconstruction / Observation Loss

$$
-\log p_\theta(o_t|z_t)
$$

要求：

$$
\boxed{
z_t \text{ 能解释当前真实世界}
}
$$

也就是：

$$
z_t
\rightarrow
\hat o_t
\approx
o_t
$$

---

### 23.2 Dynamics KL

$$
KL(q_t||p_t)
$$

其中：

$$
q_t =
q(z_t|z_{t-1},a_{t-1},o_t)
$$

看过真实 observation。

而：

$$
p_t =
p(z_t|z_{t-1},a_{t-1})
$$

没看当前 observation。

所以这一项要求：

$$
\boxed{
\text{仅靠过去预测出的 state}
\approx
\text{看过真实世界后修正出的 state}
}
$$

换句话说：

$$
\boxed{
z_t \text{ 必须能从过去和动作预测出来}
}
$$

---

## 24. 最核心的理解

world model 没有 latent state 的 ground-truth label。

所以不能直接监督：

$$
\hat z_t \approx z_t^*
$$

而是利用两个约束把 latent state “夹”出来：

$$
\boxed{
\text{可解释当前 observation}
}
$$

以及：

$$
\boxed{
\text{可由过去和 action 预测}
}
$$

因此一个好的 world state 应同时满足：

$$
\boxed{
\text{observable}
+
\text{predictable}
}
$$

这就是经典 latent world model 最核心的 inductive bias。

![世界模型 loss 的两类约束](/images/vae-to-world-model/fig05_world_model_loss.svg)

*图 5：由于没有 latent ground truth，模型通过“可解释当前观测”和“可由过去预测”两类约束共同塑造 $z_t$。*

---

## 25. 和 VAE 的关系到底是什么

不要把它理解成：

> “世界模型就是把 VAE 的 $x,z$ 加个时间下标。”

更准确的是：

> **世界模型从 POMDP / state-space model 出发，而 VAE 提供了训练 latent generative model 的 variational inference 工具。**

静态 VAE：

$$
p(z),\quad
q(z|x),\quad
p(x|z)
$$

Sequential latent world model：

$$
p(z_t|z_{t-1},a_{t-1})
$$

$$
q(z_t|z_{t-1},a_{t-1},o_t)
$$

$$
p(o_t|z_t)
$$

真正的变化是：

$$
\boxed{
\text{static prior}
\rightarrow
\text{action-conditioned dynamic prior}
}
$$

于是：

$$
\boxed{
\text{VAE-style variational inference}
+
\text{state-space dynamics} =
\text{经典 latent world model 的训练骨架}
}
$$

![VAE 与 Sequential World Model 的结构对应](/images/vae-to-world-model/fig06_vae_to_world_model.svg)

*图 6：从结构上看，最核心的变化是 static prior 变成 action-conditioned dynamic prior；其余 variational inference 逻辑保持一致。*

---

## 26. 一张总逻辑图

```text
现实世界部分可观测
        │
        ▼
真实 state s_t 看不到，只能看到 observation o_t
        │
        ▼
引入 latent state z_t
        │
        ├───────────────┐
        ▼               ▼
能解释当前世界        能预测未来世界
p(o_t | z_t)          p(z_t | z_{t-1}, a_{t-1})
        ▲               │
        │               │
        └──── q(z_t | z_{t-1}, a_{t-1}, o_t)
             用真实 observation 修正 state
                         │
                         ▼
              latent 没有 ground truth
                         │
                         ▼
                  variational inference
                         │
                         ▼
                       ELBO
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 reconstruction                     dynamics KL
解释当前 observation            prior 接近 posterior
          │                             │
          └──────────────┬──────────────┘
                         ▼
             学到“可解释 + 可预测”的 latent world state
```

---

## 27. 如果继续往 PlaNet / Dreamer 看

到这里以后，再读经典方法会容易很多：

- **World Models**：encoder → latent dynamics → controller
- **PlaNet**：把 latent dynamics 做成 RSSM，并在 latent space 规划
- **Dreamer**：不只在 latent 里规划，而是在 imagined latent rollout 中训练 actor/value

之后的研究虽然表示形式、backbone、训练目标越来越复杂，但很多工作仍然围绕几个老问题：

1. latent 里应该保留什么？
2. dynamics 应该如何建模？
3. reconstruction 是否必要？
4. prior / posterior 应该如何对齐？
5. world model 是服务某个 policy，还是做通用 simulator？
6. 是否需要 pixel、token、BEV、occupancy、3DGS 等更显式的空间结构？

---

## 28. 最终压缩

如果只记一句：

$$
\boxed{
\text{World Model} =
\text{State Representation}
+
\text{Dynamics}
+
\text{Decision Interface}
}
$$

如果只记它的训练逻辑：

$$
\boxed{
\text{没有 latent label}
\Rightarrow
\text{让 latent 同时“解释现在”并“可预测未来”}
}
$$

如果只记最经典的 sequential latent world model loss：

$$
\boxed{
L_t =
L_{\text{reconstruction}}
+
KL(
q(z_t|z_{t-1},a_{t-1},o_t)
\|
p(z_t|z_{t-1},a_{t-1})
)
}
$$

这三句话基本就是从 VAE 到经典 latent world model 的主干。
