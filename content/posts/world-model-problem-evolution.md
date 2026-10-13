---
title: "World Model：从“学一个世界”到“学对决策有用的未来”——一次问题演进记录"
date: 2026-10-13
draft: false
categories: ["Machine Learning"]
---

> 1. 世界应该被表示成什么？
> 2. 未来到底应该预测什么？
> 3. 预测未来以后，怎么把它变成决策？
> 4. 一个模型需要多像真实世界，才足够支持 planning / control？

---

## 0. 最开始：我对 World Model 的朴素理解

最开始其实可以把 World Model 写得非常简单：

$$
\text{observation}
\rightarrow
\text{state representation}
\rightarrow
\text{dynamics}
\rightarrow
\text{decision}
$$

如果是真实状态完全可见的 MDP，我们会写：

$$
p(s_{t+1}\mid s_t,a_t)
$$

然后 policy 根据当前状态做决策：

$$
\pi(a_t\mid s_t)
$$

但真实机器人、视觉控制等问题里，我们通常只能看到 observation：

$$
o_t
$$

并不知道真正的环境状态 $s_t$。于是一个自然的问题是：

> 能不能从观测中学出一个内部状态 $z_t$，再在这个内部状态上学习世界如何变化？

于是最基本的形式变成：

$$
o_t\rightarrow z_t
$$

$$
(z_t,a_t)\rightarrow z_{t+1}
$$

$$
z_t\rightarrow a_t
$$

现在回头看，后面的很多论文其实都是在修改这三个箭头。

---

## 1. World Models（2018）：先学一个可以“做梦”的世界

**论文：David Ha, Jürgen Schmidhuber, _World Models_, 2018**

这是最适合拿来当起点的一篇。

它把整个系统拆成三个非常直观的模块：

$$
\boxed{
\text{VAE}
+
\text{MDN-RNN}
+
\text{Controller}
}
$$

也就是论文里常说的：

$$
V + M + C
$$

### 1.1 VAE：先把视觉世界压缩掉

原始 observation 是图像：

$$
o_t
$$

VAE 把它压成一个低维 latent：

$$
o_t\xrightarrow{\text{VAE}}z_t
$$

这个 $z_t$ 负责描述“当前画面大概是什么状态”。

这里很容易产生一个误解：  
$z_t$ 并没有人工标签。VAE 是先在环境采集到的图像上自己训练的，然后后续 dynamics 使用这个已经学好的 latent coordinate system。

所以数据流更像：

$$
o_t\rightarrow z_t
$$

$$
o_{t+1}\rightarrow z_{t+1}
$$

然后 dynamics 得到：

$$
(z_t,a_t,z_{t+1})
$$

这样的训练样本。

### 1.2 MDN-RNN：世界接下来会怎么变？

世界的动态由 MDN-RNN 学：

$$
p(z_{t+1}\mid z_t,a_t,h_t)
$$

其中 $h_t$ 是 RNN 的 hidden state，用来记住单帧看不到的历史信息，例如速度、运动趋势等。

为什么不是直接输出一个：

$$
\hat z_{t+1}
$$

而是输出一个 mixture distribution？

因为未来可能有不确定性，因此用 mixture density network 表达：

$$
p(z_{t+1}\mid z_t,a_t,h_t)
=
\sum_k\pi_k
\mathcal N(z_{t+1};\mu_k,\sigma_k^2)
$$

于是这篇论文第一次非常直观地把一个想法摆出来：

$$
\boxed{
\text{世界模型不是记住现在，而是能够产生未来}
}
$$

### 1.3 Controller：在学出来的世界里行动

Controller 反而很简单：

$$
(z_t,h_t)\rightarrow a_t
$$

如果 VAE 和 dynamics 已经给出了足够好的内部世界，那么 policy 本身不一定需要很复杂。

最有意思的实验思想是：  
可以让 agent 在 MDN-RNN 产生的“dream”里训练，再把 controller 放回真实环境执行。

论文真正留下来的概念，并不是某个具体 VAE 或 RNN 结构，而是：

$$
\boxed{
\text{learn a world}
\rightarrow
\text{act inside the learned world}
}
$$

### 1.4 当时留下的第一个疑问

这里很快会发现：

> VAE 学 $z_t$ 的目标只是 reconstruction。  
> 一个“适合重建当前图像”的 latent，真的也是一个“适合预测未来和控制”的 latent 吗？

而且 World Models 的训练基本是：

$$
\text{先学 VAE}
\rightarrow
\text{固定 latent space}
\rightarrow
\text{再学 dynamics}
$$

representation 和 dynamics 并没有真正一起塑造 latent。

这个问题自然引出了 PlaNet。

---

## 2. PlaNet（2019）：representation 和 dynamics 为什么要分开学？

**论文：Danijar Hafner et al., _Learning Latent Dynamics for Planning from Pixels_, ICML 2019**

PlaNet 的变化并不是简单“换一个更好的 RNN”。

它真正做的是：

$$
\boxed{
\text{把 representation learning 和 dynamics learning 联合起来}
}
$$

核心结构是 RSSM（Recurrent State-Space Model）。

---

### 2.1 RSSM：确定性记忆 + 随机状态

PlaNet 的 latent state 不再只是一个 $z_t$，而是可以理解为两部分：

$$
(h_t,z_t)
$$

其中：

- $h_t$：deterministic recurrent state，负责保留历史记忆；
- $z_t$：stochastic latent state，负责表示当前状态中的不确定性。

状态首先根据过去和动作推进：

$$
h_t=f(h_{t-1},z_{t-1},a_{t-1})
$$

然后在没有看到当前 observation 之前，world model 有一个 predictive prior：

$$
p(z_t\mid h_t)
$$

真正看到 $o_t$ 后，再通过 posterior 修正：

$$
q(z_t\mid h_t,o_t)
$$

这其实就是一个非常自然的：

$$
\boxed{
\text{predict}
\rightarrow
\text{observe}
\rightarrow
\text{correct}
}
$$

过程。

---

### 2.2 为什么 prior / posterior 这么关键？

训练的时候，真实 $o_t$ 已经存在，所以 posterior 可以利用它：

$$
q(z_t\mid h_t,o_t)
$$

但如果现在要想象未来：

$$
t+1,t+2,\ldots
$$

未来的 observation 根本还不存在。

所以 rollout 时只能依靠：

$$
p(z_{t+1}\mid h_{t+1})
$$

这解释了为什么 PlaNet / Dreamer 里 prior 和 posterior 的一致性那么重要。

训练目标可以粗略理解成两部分：

$$
\mathcal L
=
\underbrace{-\log p(o_t\mid h_t,z_t)}_{\text{解释当前观测}}
+
\underbrace{
KL\left(
q(z_t\mid h_t,o_t)
\Vert
p(z_t\mid h_t)
\right)
}_{\text{让预测状态接近观测修正后的状态}}
$$

于是一个好的 latent state 同时需要：

$$
\boxed{
\text{observable}
+
\text{predictable}
}
$$

这和 World Models 的“先 VAE、再 dynamics”已经是不同的思想。

---

### 2.3 PlaNet 还特别关心多步预测

如果模型只学好：

$$
z_t\rightarrow z_{t+1}
$$

并不能保证：

$$
z_t
\rightarrow z_{t+1}
\rightarrow z_{t+2}
\rightarrow\cdots
$$

长期 rollout 仍然准确。

PlaNet 为此提出了 **latent overshooting**：  
除了要求一步 transition 对，还额外让从更早状态多步 rollout 后的预测，与未来 observation 推断出的 posterior 保持一致。

这个细节很重要，因为它已经暴露了 world model 中一个一直延续到今天的问题：

$$
\boxed{
\text{one-step accuracy}
\neq
\text{long-horizon consistency}
}
$$

PlaNet 的论文结论也明确强调：一个用于 planning 的 dynamics model，需要能够较准确地预测未来多步 reward；RSSM 和 multi-step variational objective 都是围绕这个目标设计的。

---

## 3. PlaNet 的第二个推进：预测未来以后，怎么做决策？

这里我们一开始容易把两件事混在一起：

$$
\text{预测未来}
$$

和：

$$
\text{规划}
$$

其实并不是一回事。

如果给定一串动作：

$$
a_{t:t+H}
$$

world model 可以 rollout：

$$
s_t
\rightarrow
s_{t+1}
\rightarrow
\cdots
\rightarrow
s_{t+H}
$$

这是 **prediction / simulation**。

但 planning 还需要：

$$
\boxed{
\text{generate candidates}
\rightarrow
\text{predict consequences}
\rightarrow
\text{evaluate}
\rightarrow
\text{select}
}
$$

PlaNet 使用 CEM（Cross-Entropy Method）在 latent space 搜索 action sequence。

粗略地说：

$$
a_{t:t+H}^{(1)},
a_{t:t+H}^{(2)},
\ldots
$$

每条动作序列都通过 world model rollout，并预测未来 reward：

$$
\hat r_t,\hat r_{t+1},\ldots
$$

再寻找：

$$
a^*_{t:t+H}
=
\arg\max
\sum_{\tau=t}^{t+H}\hat r_\tau
$$

然后只执行第一个动作，再得到真实的新 observation，重新推断 state、重新 planning。

这就是 MPC / receding horizon 的味道：

$$
\boxed{
\text{plan}
\rightarrow
\text{execute one step}
\rightarrow
\text{observe}
\rightarrow
\text{replan}
}
$$

这里我们第一次比较明确地区分了：

$$
\boxed{
\text{World Model}
\neq
\text{Planner}
}
$$

world model 回答：

> 如果这样做，会发生什么？

planner 回答：

> 那么我现在应该做什么？

---

## 4. Dreamer（2020）：为什么每一步都要重新搜？

**论文：Danijar Hafner et al., _Dream to Control: Learning Behaviors by Latent Imagination_, ICLR 2020**

PlaNet 有一个很明显的计算模式：

$$
\text{每来到一个 state}
\rightarrow
\text{现场搜索很多 action sequences}
$$

也就是说，很多计算发生在 decision time。

Dreamer 的问题非常自然：

> world model 已经能产生 imagined trajectories，为什么不直接在这些 trajectory 上把 policy 学出来？

于是决策方式从：

$$
\arg\max_{a_{t:t+H}}
$$

逐渐变成：

$$
a_t\sim\pi_\psi(a_t\mid s_t)
$$

---

### 4.1 在 imagination 里训练 actor / value

Dreamer 基本保留了 RSSM world model。

从一个真实数据推断出的 latent state 出发：

$$
s_t=(h_t,z_t)
$$

actor 给动作：

$$
a_t\sim\pi_\psi(a_t\mid s_t)
$$

world model 产生 imagined next state：

$$
s_t,a_t
\rightarrow
\hat s_{t+1}
$$

同时 reward model 给出：

$$
\hat r_t
$$

然后继续：

$$
\hat s_{t+1}
\rightarrow
a_{t+1}
\rightarrow
\hat s_{t+2}
\rightarrow\cdots
$$

于是得到了 imagined trajectory。

actor 学习最大化 imagined return，value network 学习估计长期 return。

Dreamer 的一个关键技术点，是利用 learned differentiable dynamics，让 value / reward 的梯度沿 imagined trajectory 反向传播来改进行为。

---

### 4.2 PlaNet → Dreamer：搜索和“内化搜索”

我们当时觉得一个很自然的类比是：

$$
\boxed{
\text{PlaNet : Dreamer}
\approx
\text{search / BoN : learned policy / RL}
}
$$

当然这不是严格算法等价。

PlaNet 更像：

$$
\text{每次现场生成候选}
\rightarrow
\text{比较}
\rightarrow
\text{选一个}
$$

Dreamer 则把大量 imagined experience 逐渐吸收到 actor 参数里：

$$
\boxed{
\text{online search}
\rightarrow
\text{amortized decision}
}
$$

于是执行时只需要：

$$
s_t\rightarrow\pi(a_t\mid s_t)
$$

这也是为什么 Dreamer 相比 PlaNet 更接近一个可以直接部署的 policy。

---

### 4.3 “一条真实轨迹为什么能反复用”？

这里我们也有过一个疑问。

真实环境里只有：

$$
(o_t,a_t,r_t,o_{t+1})
$$

这些数据。

那 world model 凭什么从一个真实 state 出发，去尝试数据里没出现过的动作？

答案是：  
它并没有获得新的真实 label。

world model 只是根据已经从真实数据学到的 dynamics，给出 synthetic prediction：

$$
(s_t,a_t')
\rightarrow
(\hat s_{t+1},\hat r_t)
$$

同一个真实 state 可以被反复当作 imagination 的起点，并在不同版本的 actor 下展开不同的未来。

所以 model-based RL 的“数据杠杆”不是凭空创造真实信息，而是：

$$
\boxed{
\text{少量昂贵 real interaction}
\rightarrow
\text{learn model}
\rightarrow
\text{大量便宜 synthetic policy updates}
}
$$

但这也直接带来风险：

$$
\boxed{
\text{model bias}
}
$$

如果：

$$
\hat s_{t+1}\neq s_{t+1}^{real}
$$

甚至模型错误会被 policy 主动利用，就会出现 model exploitation。

所以 imagination 从来不是“免费真数据”。

---

## 5. MuZero（2020）：World Model 一定要还原世界吗？

**论文：Julian Schrittwieser et al., _Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model_, Nature 2020**

看到 PlaNet / Dreamer 后，一个更根本的问题出现了：

> 为什么 world model 一定要能把 observation 重建出来？

MuZero 的回答非常激进：

$$
\boxed{
\text{不需要}
}
$$

---

### 5.1 只学习 planning 真正需要的东西

MuZero 学三类模型：

### Representation

$$
h(o_{1:t})\rightarrow s_t
$$

### Dynamics

$$
g(s_t,a_t)\rightarrow(s_{t+1},r_t)
$$

### Prediction

$$
f(s_t)\rightarrow(\pi_t,v_t)
$$

它并不要求：

$$
s_t\rightarrow o_t
$$

也就是说，latent state 不需要还原 pixel，也不需要对应一个人类可解释的物理状态。

它只需要能够支持：

$$
\boxed{
\text{reward}
+
\text{value}
+
\text{policy}
}
$$

预测和 MCTS planning。

Nature 论文对 MuZero 的概括非常直接：  
模型学习的是 **predictions relevant to planning**，包括 policy、value 和 reward，而不是完整复原环境 dynamics。

---

### 5.2 AlphaZero → MuZero

这个区别也很清楚。

AlphaZero 做 tree search，但棋类规则本身是已知的：

$$
(s_t,a_t)\rightarrow s_{t+1}
$$

环境 simulator 是现成的。

MuZero 则把 simulator 本身也学出来：

$$
\boxed{
\text{AlphaZero}
=
\text{known dynamics}
+
\text{search}
}
$$

$$
\boxed{
\text{MuZero}
=
\text{learned latent dynamics}
+
\text{search}
}
$$

所以 MuZero 给 world model 留下了一个很重要的认识：

$$
\boxed{
\text{good world model}
\neq
\text{perfect reconstruction of the world}
}
$$

更准确地说，对于 planning：

$$
\boxed{
\text{decision-sufficient}
}
$$

可能比：

$$
\text{observation-complete}
$$

更重要。

这条思想后面会重新出现在 JEPA / feature prediction 等路线里，只是训练目标不同。

---

## 6. 到这里为止：我们主要看的其实只是 Dec / Seq / GLV

后来借助 embodied world model survey，才发现前面这条经典线在整个现代 taxonomy 里其实只是一个象限：

$$
\boxed{
\text{Decision-Coupled}
+
\text{Sequential}
+
\text{Global Latent Vector}
}
$$

PlaNet、Dreamer 系列都属于这一支。

这里的默认假设大概是：

$$
\text{世界}
\rightarrow
\text{一个 compact latent state}
$$

然后：

$$
s_t\rightarrow s_{t+1}
$$

按时间一步一步 rollout，并直接服务 planning / control。

接下来我们开始逐一问：

> 这些假设真的必须成立吗？

---

## 7. IRIS / TFS：世界一定要压成一个 latent vector 吗？

**代表论文：Vincent Micheli, Eloi Alonso, François Fleuret, _Transformers are Sample-Efficient World Models (IRIS)_, ICLR 2023**

Dreamer-style GLV 的思维还是：

$$
o_t\rightarrow z_t\in\mathbb R^d
$$

也就是整个世界被压缩成一个全局状态。

Token Feature Sequence（TFS）则变成：

$$
o_t
\rightarrow
X_t=[x_t^1,x_t^2,\ldots,x_t^N]
$$

世界不再只是一个 vector，而是一组 token / feature。

---

### 7.1 IRIS：把环境变成一种 sequence modeling 问题

IRIS 使用：

$$
\boxed{
\text{discrete autoencoder}
+
\text{autoregressive Transformer}
}
$$

先把 observation 编成离散 visual tokens，然后 Transformer 建模 token 和 action 的时序关系。

于是 world dynamics 越来越像：

$$
p(X_{t+1}\mid X_{\leq t},a_{\leq t})
$$

而不是传统 RSSM 的：

$$
p(z_{t+1}\mid z_t,a_t)
$$

论文在 Atari 100k 上强调的重点就是 **sample-efficient world-model learning**：通过 learned world model 中的 imagined experience 来训练 agent，同时让 Transformer 承担环境序列建模。

---

### 7.2 representation philosophy 发生了变化

这不是简单：

> RNN 换成 Transformer。

更重要的是：

$$
\boxed{
\text{world = one compact state}
}
$$

开始变成：

$$
\boxed{
\text{world = structured collection of tokens}
}
$$

这样 vision、language、proprioception、action 等不同 modality 都更容易进入同一个 sequence framework：

$$
[
\text{vision tokens},
\text{language tokens},
\text{action tokens},
\text{proprio tokens}
]
$$

于是这条路线很自然会继续走向：

$$
\boxed{
\text{World Model}
+
\text{VLA}
\rightarrow
\text{World-Action Model}
}
$$

这里 world prediction 和 action prediction 的边界开始变得没有过去那么清楚。

---

## 8. OccWorld / SLG：token 够不够？空间结构要不要显式保留？

**论文：Wenzhao Zheng et al., _OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving_, ECCV 2024**

继续往 embodied / robot 方向走，会碰到另一个问题：

> 如果机器人真正关心的是位置、距离、碰撞和可达性，仅仅把世界表示成 token 是否太隐式了？

于是 Spatial Latent Grid（SLG）把空间结构直接放进 representation。

---

### 8.1 世界直接表示成 occupancy

可以粗略写成：

$$
O_t(x,y,z)
$$

描述三维空间中不同位置：

- 是否被占据；
- 是什么语义；
- 哪些区域是 free space。

相比 bounding box，occupancy 能保存更细粒度的 3D scene structure。

OccWorld 先学习一个 reconstruction-based scene tokenizer，把 3D occupancy 压成离散 scene tokens，再使用 GPT-like spatial-temporal generative Transformer 同时预测：

$$
\text{future scene tokens}
$$

和：

$$
\text{future ego tokens}
$$

最后 decode 成：

$$
\hat O_{t+1:t+H}
$$

以及 ego trajectory。

所以它很有意思：

$$
\boxed{
\text{representation 是 spatial}
\quad
\text{但 dynamics 内部又可以 token 化}
}
$$

taxonomy 并不是互斥的网络结构标签，而是在描述模型最终保留了什么世界结构。

---

### 8.2 为什么这对 planning 很自然？

如果 future state 本身就是 occupancy：

$$
O_{t+1:t+H}
$$

那么规划器很容易直接评价：

> 这条 trajectory 会不会穿过未来被占据的位置？

因此 world representation 和 downstream cost 的接口非常直接：

$$
\boxed{
\text{future occupancy}
\rightarrow
\text{trajectory evaluation}
}
$$

OccWorld 的实验也显示，它不仅能做 3D scene forecasting，还能在不使用 instance 和 map supervision 的条件下得到有竞争力的 planning 结果。

这里的 inductive bias 很明确：

$$
\boxed{
\text{真实世界具有需要显式保留的空间拓扑}
}
$$

代价则是更高的计算和存储成本。

---

## 9. JEPA / V-JEPA：为什么一定要预测 pixel？

走到这里，又出现一个和 MuZero 有点相似、但来源不同的问题：

> 如果我们真正关心的是稳定、抽象、可预测的规律，为什么要让模型预测所有 pixel？

例如：

- 树叶具体怎么晃；
- 微小光照变化；
- 纹理细节；
- sensor noise。

这些东西很难预测，但未必是理解世界和做决策最重要的部分。

JEPA 的回答是：

$$
\boxed{
\text{不要预测 pixel，预测 representation}
}
$$

---

### 9.1 JEPA 的基本结构

最核心的三部分是：

$$
\text{context encoder}
+
\text{target encoder}
+
\text{predictor}
$$

context encoder 看到可见区域：

$$
x_c\rightarrow z_c
$$

target encoder 为被遮住的真实区域产生表示：

$$
x_{\text{target}}\rightarrow z_{\text{target}}
$$

predictor 只根据 context representation 去预测：

$$
\hat z_{\text{target}}
=
P(z_c,\text{target position})
$$

训练目标是：

$$
\hat z_{\text{target}}
\approx
z_{\text{target}}
$$

而不是：

$$
\hat x_{\text{target}}
\approx
x_{\text{target}}
$$

V-JEPA 把这一思想放到视频中，通过 masked spatiotemporal feature prediction 学习 motion 和 appearance representation。论文特别强调：它只使用 feature prediction objective，不依赖 pixel reconstruction、text supervision 或 negative examples。

---

### 9.2 它和 MuZero像在哪里，又不一样在哪里？

两者都在反对：

$$
\boxed{
\text{world model 必须完整复刻 observation}
}
$$

但它们的 inductive bias 不一样。

MuZero 用：

$$
\text{reward / value / policy}
$$

来定义“什么信息值得保留”。

所以它非常 task-coupled。

JEPA 则使用：

$$
\text{representation predictability}
$$

来定义值得学习的结构，希望先得到更 general-purpose 的 representation。

因此可以粗略理解：

$$
\boxed{
\text{MuZero：对决策 sufficient}
}
$$

$$
\boxed{
\text{JEPA：对世界的可预测结构 sufficient}
}
$$

这也是为什么后者更容易被放到 general-purpose world model 这一侧。

---

## 10. 回头看：这些论文其实不是在争同一个 architecture

如果只按模型名字看，很容易把历史理解成：

$$
\text{RNN}
\rightarrow
\text{Transformer}
\rightarrow
\text{Diffusion}
$$

但我们一路看下来，更像是在不断修改四个问题。

---

### 10.1 State 应该是什么？

最开始：

$$
\boxed{
\text{global latent vector}
}
$$

World Models / PlaNet / Dreamer 都基本属于这种思路。

然后：

$$
\boxed{
\text{token feature sequence}
}
$$

世界变成可以组合、交互的一组 token。

再到：

$$
\boxed{
\text{spatial / 3D representation}
}
$$

显式保存几何和 topology。

所以这里的演进不是“表示越来越大”这么简单，而是：

$$
\text{我愿意给模型什么结构先验？}
$$

GLV 假设一个 compact state 足够；

TFS 强调 compositional / relational structure；

SLG 则进一步认为 geometry 本身应该被显式编码。

---

### 10.2 Future 应该预测什么？

最早很自然会想到：

$$
\text{未来 observation / pixel}
$$

然后变成：

$$
\text{future latent state}
$$

MuZero 又进一步说：

$$
\text{只预测 planning relevant quantities}
$$

JEPA 则说：

$$
\text{预测 abstract representation}
$$

于是出现一个我觉得很重要的变化：

$$
\boxed{
\text{“把未来复制出来”}
\rightarrow
\text{“把未来真正有用的结构预测出来”}
}
$$

但“有用”到底如何定义，不同路线的答案并不一样。

---

### 10.3 时间应该怎么建模？

World Models / PlaNet / Dreamer 都有很强的 sequential flavor：

$$
s_t
\rightarrow
s_{t+1}
\rightarrow
s_{t+2}
\rightarrow\cdots
$$

它的问题很明显：

$$
\text{error accumulation}
$$

以及：

$$
\text{long-horizon consistency}
$$

后来 Transformer/token world model 把 temporal modeling 重新写成 sequence modeling。

再往现代 generative / global prediction 的方向走，又会出现：

$$
\boxed{
\text{不一定逐步 autoregressive rollout，
而是直接预测更长 future block}
}
$$

我们目前只建立了这个分支的概念，还没有深入 diffusion / DiT 细节。

---

### 10.4 Prediction 最后怎么变成 Decision？

这里的路线也非常有意思。

### World Models

$$
\text{world model}
\rightarrow
\text{controller}
$$

### PlaNet

$$
\text{world model}
\rightarrow
\text{CEM online planning}
$$

### Dreamer

$$
\text{world model}
\rightarrow
\text{imagined actor/value learning}
$$

### MuZero

$$
\text{learned latent model}
\rightarrow
\text{MCTS}
$$

再往 World-Action Model / VLA 方向走：

$$
\text{world prediction}
+
\text{action prediction}
$$

可能逐渐进入同一个 model。

于是我们反复碰到一个很通用的 trade-off：

$$
\boxed{
\text{inference-time search}
\leftrightarrow
\text{amortized policy}
}
$$

PlaNet 把很多计算留到现场；

Dreamer 把更多计算放进训练并沉淀到 policy；

MuZero 又保留了强 search；

现代 VLA / world-action model 则可能进一步把 planning pattern 吸进大规模 sequence model。

---

## 11. 暂时形成的理解：World Model 不等于“未来视频生成器”

最开始很容易把 world model 理解成：

> 给过去几帧，生成未来几帧。

但一路看下来，这个定义明显太窄。

现在我更愿意把它写成：

$$
\boxed{
\text{学习一个能够表达环境状态，并支持关于未来进行推演的内部模型}
}
$$

关键不在于这个内部模型是不是：

- pixel；
- token；
- vector；
- occupancy；
- 3D field。

真正的问题是：

$$
\boxed{
\text{它有没有保留进行未来预测和决策所需要的结构？}
}
$$

这也解释了为什么不同 world model 看起来差异越来越大：

- Dreamer 的 world 很小，是 compact latent；
- MuZero 的 world 甚至不用还原 observation；
- OccWorld 的 world 很几何，是 occupancy；
- JEPA 的 world 更像一个抽象 feature space。

它们并不是在逼近同一个“唯一正确的世界表示”。

---

## 12. 一个暂时没有答案的问题：什么才是 sufficient representation？

现在回头看，这可能是整条线里最核心的问题：

$$
\boxed{
\text{什么信息对未来预测和控制是 sufficient 的？}
}
$$

太弱的 representation：

$$
\Rightarrow
\text{预测不了真正重要的 dynamics}
$$

太强、太完整的 representation：

$$
\Rightarrow
\text{大量容量浪费在不可预测或与任务无关的细节}
$$

所以不同路线其实在不同地方押注：

$$
\text{Dreamer}
\rightarrow
\text{compact predictive latent}
$$

$$
\text{MuZero}
\rightarrow
\text{decision-relevant latent}
$$

$$
\text{TFS}
\rightarrow
\text{compositional token state}
$$

$$
\text{OccWorld}
\rightarrow
\text{geometry-aware spatial state}
$$

$$
\text{JEPA}
\rightarrow
\text{predictable abstract representation}
$$

到这里，我不太想直接下结论说其中哪一个更“正确”。

更合理的问题是：

> **对于具体的 downstream problem，哪些信息必须显式保存，哪些可以压掉？**

---

## 13. 目前这段阅读可以压成一条问题演进线

最后不按年份，而按我们遇到问题的顺序记：

$$
\boxed{\text{World Models}}
$$

> 能不能先学一个世界，再在里面行动？

$$
\downarrow
$$

$$
\boxed{\text{PlaNet}}
$$

> representation 和 dynamics 为什么要分开学？  
> 能不能直接在 latent world 里 planning？

$$
\downarrow
$$

$$
\boxed{\text{Dreamer}}
$$

> 每次都在线搜索是不是太贵？  
> 能不能把 imagined experience 内化进 policy？

$$
\downarrow
$$

$$
\boxed{\text{MuZero}}
$$

> world model 一定要还原真实 observation 吗？

$$
\downarrow
$$

$$
\boxed{\text{IRIS / TFS}}
$$

> 世界一定要压成一个 global latent vector 吗？

$$
\downarrow
$$

$$
\boxed{\text{OccWorld / SLG}}
$$

> embodied system 所需要的 geometry 是否应该显式保留？

$$
\downarrow
$$

$$
\boxed{\text{JEPA / V-JEPA}}
$$

> 为什么还要预测所有 pixel？  
> 能不能只预测抽象、稳定、可预测的 representation？

然后问题又回到了最开始：

$$
\boxed{
\text{到底什么才是一个“足够好的内部世界”？}
}
$$

现在的暂时答案不是某种具体 architecture，而更像：

$$
\boxed{
\text{一个好的 World Model，
应该保留对未来推演和下游决策真正 sufficient 的结构。}
}
$$

问题在于，**什么叫 sufficient，取决于我们最终到底想让 agent 做什么。**

---

## 参考论文

1. Ha, D., & Schmidhuber, J. **World Models**. 2018.
2. Hafner, D., et al. **Learning Latent Dynamics for Planning from Pixels (PlaNet)**. ICML 2019.
3. Hafner, D., et al. **Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)**. ICLR 2020.
4. Schrittwieser, J., et al. **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)**. Nature, 2020.
5. Micheli, V., Alonso, E., & Fleuret, F. **Transformers are Sample-Efficient World Models (IRIS)**. ICLR 2023.
6. Zheng, W., et al. **OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving**. ECCV 2024.
7. Bardes, A., et al. **Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)**. 2024.
8. **A Comprehensive Survey on World Models for Embodied AI**. 用于本文中 Dec/Seq/GLV、TFS、SLG、General-Purpose 等 taxonomy 的定位。
