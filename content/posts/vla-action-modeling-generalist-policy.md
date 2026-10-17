---
title: "VLA：动作建模与通用机器人策略"
date: 2026-10-17
draft: false
categories: ["Robotics"]
---

> VLA 的发展有两条线。
>
> 一条来自机器人领域本身：**动作到底应该怎么表示、怎么生成？**
>
> 另一条来自大模型：**已经非常成功的 [Transformer](https://arxiv.org/abs/1706.03762) / VLM，能不能直接变成一个通用机器人 policy？**
>
> 前一条路线从 [Behavior Cloning](https://publications.ri.cmu.edu/alvinn-an-autonomous-land-vehicle-in-a-neural-network)、[Action Chunking](https://arxiv.org/abs/2304.13705) 一直走到 [Diffusion Policy](https://arxiv.org/abs/2303.04137)；后一条路线则从 [RT-1](https://arxiv.org/abs/2212.06817)、[RT-2](https://arxiv.org/abs/2307.15818)、[OpenVLA](https://arxiv.org/abs/2406.09246) 逐渐把 vision、language、action 都塞进统一的 token / autoregressive 范式。
>
> 到 [$\pi_0$](https://arxiv.org/abs/2410.24164) 附近，这两条路线又重新汇合：保留 VLM 擅长的视觉语言表示，但不再要求连续机器人动作也必须按照语言 token 的方式输出，而是重新使用面向连续 action chunk 的生成式建模。

---

## 1. VLA 到底在解决什么问题？

最简单的机器人 policy 可以写成：

$$
(o_t,l)\rightarrow a_t
$$

其中：

- $o_t$：当前 observation，例如 RGB 图像、机器人关节状态；
- $l$：语言任务描述；
- $a_t$：当前机器人动作。

如果只看这个公式，VLA 好像只是：

$$
\text{Vision}
+
\text{Language}
\rightarrow
\text{Action}
$$

但真正往下拆，会发现至少有两个问题混在一起。

第一个问题是：

$$
\boxed{
\text{Action 到底应该是什么样的输出对象？}
}
$$

是一个确定的连续向量？

是一串未来动作？

是一个概率分布？

还是一串离散 token？

第二个问题是：

$$
\boxed{
\text{如何把大规模视觉语言模型已经学到的知识，变成机器人行为？}
}
$$

前一个问题在 VLA 出现之前，机器人 imitation learning 领域已经讨论了很久。

后一个问题则是在 Transformer、LLM、VLM scaling 成功之后才变得非常突出。

因此，与其把 VLA 看成一条简单的：

$$
\text{RT-1}\rightarrow\text{RT-2}\rightarrow\text{OpenVLA}\rightarrow\pi_0
$$

不如先分开看这两条路线。

---

## Part I：机器人领域本来就在研究 Action 应该怎么输出

## 2. Behavior Cloning：最直接的起点

最简单的 Behavior Cloning（BC）就是监督学习，经典实例可参考 [ALVINN（Pomerleau, 1989）](https://publications.ri.cmu.edu/alvinn-an-autonomous-land-vehicle-in-a-neural-network)。

给一批 expert demonstration：

$$
\mathcal D =
\{(o_t,a_t^{expert})\}
$$

学习：

$$
\pi_\theta(o_t)
\approx
a_t^{expert}
$$

例如机械臂的 action 可以是：

$$
a_t=
[
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z,
g
]
$$

其中 $g$ 表示夹爪开合。

如果直接做回归，可以使用类似：

$$
\mathcal L =
\|\pi_\theta(o_t)-a_t^{expert}\|^2
$$

这样的目标。

这件事情很直接：

> Expert 在这种 observation 下怎么做，我就学着怎么做。

但它很快会暴露两个并不相同的问题。

---

### 2.1 第一个问题：训练状态分布和执行状态分布不同

训练数据来自 expert：

$$
s_t\sim d_{\text{expert}}
$$

但真正执行的时候，未来 observation 是由模型自己的动作决定的：

$$
a_t\sim\pi_\theta
$$

$$
s_{t+1}\sim p(s_{t+1}|s_t,a_t)
$$

只要某一步产生一点误差，就可能进入 expert demonstration 中很少出现甚至没有出现的状态：

$$
s_t
\rightarrow
a_t^{wrong}
\rightarrow
s_{t+1}'
$$

然后模型在：

$$
s_{t+1}'
$$

上又没有足够 supervision，于是可能继续犯错：

$$
\text{small error}
\rightarrow
\text{unseen state}
\rightarrow
\text{larger error}
$$

这就是 imitation learning 里经典的 distribution shift / compounding error 问题。

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) 的思路可以看成一个很直接的修补：

$$
\text{让 learner 自己执行}
$$

然后在 learner 真正访问到的 state 上，请 expert 给出正确动作：

$$
(s',a^{expert}(s'))
$$

再把这些数据加回训练集。

它解决的核心不是“模型结构不够强”，而是：

> policy 的动作会改变自己未来看到的数据分布，因此 imitation learning 并不是普通的 i.i.d. supervised learning。

---

### 2.2 第二个问题：BC 本身没有长期 return

BC 优化的是：

$$
\pi_\theta(o_t)
\approx
a_t^{expert}
$$

它并没有直接优化：

$$
\max_\pi
\mathbb E
\left[
\sum_t r_t
\right]
$$

也就是说，训练目标本身没有问：

> 当前这个动作，几十步以后到底会让任务变好还是变坏？

当然，如果 demonstration 足够好、数据足够丰富、模型上下文足够长，policy 完全可能从 demonstration 中**隐式地学到**长期策略。

但它的 objective 本身仍然是 imitation，而不是 long-horizon credit assignment。

所以：

$$
\boxed{
\text{distribution shift}
}
$$

和：

$$
\boxed{
\text{缺少显式长期反馈}
}
$$

是两个不同的问题。

---

## 3. “Action 怎么输出”其实很早就不是只有一个答案

如果同一个 observation 对应唯一动作，那么：

$$
o\rightarrow a
$$

当然很自然。

但机器人行为经常不是 deterministic mapping。

例如机械臂面前有一个障碍物，可以：

$$
A_L=\text{从左边绕}
$$

也可以：

$$
A_R=\text{从右边绕}
$$

两个行为都可能成功。

如果直接使用 MSE regression，最优 point estimate 可能会接近：

$$
\hat A
\approx
\frac{A_L+A_R}{2}
$$

结果反而可能直直撞向障碍物。

因此在 [ACT](https://arxiv.org/abs/2304.13705)、[Diffusion Policy](https://arxiv.org/abs/2303.04137) 之前，机器人 policy 对输出分布的讨论其实已经很多：

- Gaussian / Gaussian Mixture Model
- Mixture Density Network
- [CVAE](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
- Autoregressive density model
- Energy-based policy
- Normalizing Flow
- 其他 structured policy representation

所以后面 ACT 和 Diffusion Policy 不应该理解为“之前所有人都只会做单点回归”。

更准确地说，它们是这条长期问题线中两个非常有代表性的现代方案：

$$
\boxed{
\text{如何表达一个复杂、连续、具有时间结构的机器人动作？}
}
$$

---

## 4. ACT：把 policy 输出的基本单位从一个动作变成一段动作

[ACT（Action Chunking with Transformers）](https://arxiv.org/abs/2304.13705)的重要点，不只是“用了 Transformer”。

它改变了 policy 输出的基本对象。

原来是：

$$
o_t\rightarrow a_t
$$

ACT 更接近：

$$
o_t
\rightarrow
A_t =
[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$

也就是说，一次预测一个 action chunk。

---

### 4.1 为什么 Action Chunk 有意义？

很多机器人动作本身就具有明显的时间结构。

例如：

> 伸手 → 对准杯子 → 闭合夹爪 → 抬起

如果每一个低层 control step 都重新独立预测：

$$
o_t\rightarrow a_t
$$

$$
o_{t+1}\rightarrow a_{t+1}
$$

模型容易受到微小 observation noise 和单步预测误差影响，动作也可能出现 jitter。

而如果一次预测：

$$
[a_t,\ldots,a_{t+H-1}]
$$

那么模型可以把一段运动整体表示出来。

它更像一个局部 motion primitive。

因此 Action Chunking 至少带来两个直觉上的变化：

$$
\boxed{
\text{减少连续单步决策带来的误差积累}
}
$$

以及：

$$
\boxed{
\text{让一段动作具有更强的 temporal coherence}
}
$$

ACT 原论文也正是针对 imitation learning 中 policy error compounding 和 demonstration non-stationarity 等问题提出 action chunking，并将整段 action sequence 作为 generative modeling 对象。

---

### 4.2 但 chunk 并不是越长越好

这里会自然产生一个 trade-off。

如果 chunk 很短：

$$
H\text{ 小}
$$

机器人会频繁重新观察和修正：

$$
\boxed{\text{closed-loop correction 强}}
$$

但是动作的整体一致性会弱一些。

如果 chunk 很长：

$$
H\text{ 大}
$$

则更容易生成完整、连贯的 movement：

$$
\boxed{\text{temporal coherence 强}}
$$

但如果中途环境发生变化，模型又一直执行旧计划：

$$
\boxed{\text{feedback correction 变弱}}
$$

所以 Action Chunking 并不是简单地：

> 一次预测很多步，然后闭眼执行到底。

ACT 使用 temporal ensembling，把不同时刻预测出来的 overlapping action chunks 进行融合，从而在动作连贯性和在线反馈之间做折中。

---

### 4.3 CVAE 在这里解决什么？

ACT 还用了 [CVAE](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)。

它背后的问题还是：

$$
o\rightarrow A
$$

不是一个严格 deterministic mapping。

即使同一个任务，人类 demonstration 也可能有不同操作风格：

$$
A^{(1)},A^{(2)},\ldots
$$

CVAE 给模型一个 latent variable：

$$
z
$$

于是可以表示：

$$
p(A|o) =
\int p(A|o,z)p(z|o)\,dz
$$

也就是说，ACT 已经不只是单纯的 deterministic action regression。

---

## 5. Diffusion Policy：从“一段动作”进一步走向“动作轨迹的联合分布”

ACT 之后，一个更自然的问题是：

> 如果 action trajectory 本身是复杂、多模态、高维的连续分布，能不能直接对整个分布做 generative modeling？

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 的答案是：

$$
\boxed{
p(A|o)
}
$$

其中：

$$
A=
[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$

不是单个 action，而是完整的 action trajectory / action chunk。

---

### 5.1

$$
o\rightarrow A
$$

和：

$$
o\rightarrow p(A|o)
$$

到底有什么区别？

第一个更像 point prediction：

$$
A=f_\theta(o)
$$

给定 observation，输出一条确定 trajectory。

第二个表示：

> 给定这个 observation，哪些 trajectory 是合理的？它们的概率结构是什么？

例如：

$$
p(A_L|o)>0
$$

$$
p(A_R|o)>0
$$

左绕和右绕可以同时成为两个 mode。

因此“加一个 $p$”不是写法上的修饰，而是重新定义 policy 的输出：

$$
\boxed{
\text{从一个答案}
\rightarrow
\text{一整个行为分布}
}
$$

---

### 5.2 Diffusion 不是沿着机器人时间做 Autoregressive Generation

这里一开始很容易混淆两个时间轴。

机器人真实时间：

$$
t,t+1,\ldots,t+H-1
$$

描述的是：

$$
[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$

而 diffusion time 是：

$$
K,K-1,\ldots,0
$$

描述整条 trajectory 从 noise 逐渐变干净：

$$
A^K
\rightarrow
A^{K-1}
\rightarrow
\cdots
\rightarrow
A^0
$$

最终：

$$
A^0 =
[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$

所以 Diffusion Policy 并不是：

$$
a_t\rightarrow a_{t+1}\rightarrow a_{t+2}
$$

这样沿 robot time 一个动作接一个动作生成。

它是在每个 diffusion step 上更新**整条** action trajectory。

---

### 5.3 并行生成为什么仍然可以学习前后动作依赖？

因为：

$$
\text{parallel}
\neq
\text{independent}
$$

denoising network 输入的就是：

$$
A^k =
[a_t^k,\ldots,a_{t+H-1}^k]
$$

整段 noisy trajectory。

它在更新某个：

$$
a_{t+i}
$$

时，可以利用整段 trajectory 的信息。

所以它学习的不是：

$$
\prod_i p(a_{t+i}|o)
$$

而是：

$$
\boxed{
p(a_t,a_{t+1},\ldots,a_{t+H-1}|o)
}
$$

这个 joint distribution。

Diffusion Policy 原论文也特别强调它对 multimodal action distribution、高维 action space 和训练稳定性的优势，并结合 receding-horizon control 来实际执行生成出来的 action sequence。

---

## 6. 第一条路线到这里可以先收束

从我们现在关注的角度，可以把机器人 action modeling 的变化压成：

$$
\boxed{
o_t\rightarrow a_t
}
$$

↓

$$
\boxed{
o_t\rightarrow a_{t:t+H}
}
$$

↓

$$
\boxed{
o_t\rightarrow p(a_{t:t+H}|o_t)
}
$$

也就是：

$$
\text{single action}
$$

↓

$$
\text{temporally structured action chunk}
$$

↓

$$
\text{multimodal joint trajectory distribution}
$$

这条路线真正一直在回答的是：

$$
\boxed{
\text{一个机器人 policy 的输出空间到底应该如何参数化？}
}
$$

---

## Part II：大 Transformer / VLM 带来了另一套非常成功的输出范式

## 7. Transformer 成功以后，一个很自然的想法：Everything as Tokens

语言模型最成功的训练形式之一是：

$$
p(x_i|x_{\lt i})
$$

即 autoregressive next-token prediction。

文本天然就是：

$$
[x_1,x_2,\ldots,x_n]
$$

Transformer 可以统一学习：

$$
p(x_1,\ldots,x_n) =
\prod_i p(x_i|x_{\lt i})
$$

后来 image、video 等 modality 也大量采用 token 化或者 sequence modeling。

于是到了机器人这里，一个极其自然的问题就是：

> vision 和 language 都能进入 Transformer，action 能不能也变成 token？

如果可以，那么：

$$
\text{Vision}
+
\text{Language}
+
\text{Action}
$$

就可以全部进入一个统一的 sequence model。

---

## 8. RT-1：先把机器人 policy 放进 scaling 范式

[RT-1（Robotics Transformer 1）](https://arxiv.org/abs/2212.06817)真正重要的问题不是：

> 如何更精确地建模一个 action distribution？

而是：

$$
\boxed{
\text{一个 Transformer 能不能吸收大量不同机器人任务，形成一个多任务 policy？}
}
$$

它输入短时间图像序列和自然语言任务描述：

$$
(o_{t-k:t},l)
$$

然后输出 robot action。

为了让 action 适配 Transformer 的 categorical token prediction，RT-1 对连续 action dimension 进行离散化。

也就是：

$$
a\in\mathbb R
$$

经过 quantization：

$$
a
\rightarrow
k\in\{1,\ldots,K\}
$$

最后输出：

$$
\text{action token}
$$

RT-1 使用大量真实机器人数据训练一个统一 policy，并展示一个模型覆盖大量真实机器人任务的可行性。

所以它的历史位置更适合概括为：

$$
\boxed{
\text{task-specific robot policy}
\rightarrow
\text{large-scale multi-task robot policy}
}
$$

它解决的主要是：

$$
\boxed{
\text{robot policy scaling}
}
$$

---

## 9. RT-2：Robot Data 不够，能不能把 VLM 的 Web Knowledge 直接带进来？

RT-1 的知识来源很大程度仍然是 robot data。

但 robot data 本身非常昂贵。

与此同时，VLM 已经通过海量互联网 image-text data 学到了大量：

- object semantics
- visual concepts
- language concepts
- commonsense relations

所以 [RT-2](https://arxiv.org/abs/2307.15818) 的问题变成：

$$
\boxed{
\text{VLM 已经学到的知识，能不能直接变成 robot behavior？}
}
$$

---

### 9.1 把 Action 当成另一种 Token

RT-2 的做法非常直接。

robot action 原本是连续向量：

$$
a_t\in\mathbb R^d
$$

把每个 dimension 离散化之后，可以得到：

$$
[k_1,k_2,\ldots,k_d]
$$

这些整数再映射成模型 vocabulary 中的 token。

于是 VLM 原本做：

$$
(image,text)
\rightarrow
text\ tokens
$$

现在可以做：

$$
(image,instruction)
\rightarrow
action\ tokens
$$

从网络形式来看，两者都成为：

$$
\boxed{
\text{next-token prediction}
}
$$

这是一个非常强的统一。

---

### 9.2 RT-2 真正带来的变化

它不只是扩大 robot dataset。

更重要的是把：

$$
\boxed{
\text{Web-scale vision-language knowledge}
}
$$

和：

$$
\boxed{
\text{robot demonstrations}
}
$$

放进同一个模型。

因此已经在 VLM 里学到的语义关系，有机会直接影响 robot action。

VLA 在这里开始成为一个比较明确的范式：

$$
\boxed{
\text{Vision}
+
\text{Language}
+
\text{Action}
}
$$

统一进一个 pretrained multimodal model。

所以从 RT-1 到 RT-2，更重要的变化是：

$$
\boxed{
\text{multi-task robot policy}
\rightarrow
\text{VLM knowledge transfer to robot control}
}
$$

---

## 10. OpenVLA：把这套 VLA 范式变成一个开放的研究基础

[OpenVLA](https://arxiv.org/abs/2406.09246) 的重要性和 RT-2 不完全一样。

它没有重新发明 Action 应该如何生成。

在 action output 上，它仍然基本延续：

$$
\text{continuous action}
\rightarrow
\text{discretization}
\rightarrow
\text{action tokens}
\rightarrow
\text{autoregressive prediction}
$$

它更重要的目标是：

$$
\boxed{
\text{让 large VLA 变得开放、可训练、可适配}
}
$$

OpenVLA 是一个 7B VLA，在大规模 [Open X-Embodiment](https://arxiv.org/abs/2310.08864) 机器人数据上训练，并强调：

- open-source
- multi-robot / cross-embodiment
- parameter-efficient adaptation
- 对新 robot / task 的 fine-tuning

因此它更像：

$$
\boxed{
\text{把 RT-2-style VLA 做成一个可研究的 foundation policy}
}
$$

---

## 11. 走到这里，会重新遇到一个很有意思的问题：Action 真的是 Language 吗？

RT-1 / RT-2 / OpenVLA 的设计逻辑其实非常合理。

因为手里已经有：

$$
\boxed{
\text{一个极其强大的 autoregressive Transformer / VLM}
}
$$

最简单的办法就是把新的 modality 也变成 token。

于是：

$$
\text{continuous robot action}
$$

被改造成：

$$
\text{discrete action tokens}
$$

整个系统就可以继续复用：

$$
p(x_i|x_{\lt i})
$$

这个成熟的训练范式。

但从机器人角度重新看，action 本身具有一些和 language 很不一样的性质：

$$
\boxed{
\text{continuous}
}
$$

$$
\boxed{
\text{high-frequency}
}
$$

$$
\boxed{
\text{strong temporal correlation}
}
$$

$$
\boxed{
\text{potentially multimodal}
}
$$

而自然语言更天然是离散 symbolic sequence。

所以这里出现了一个值得保留的问题：

> VLA 的早期设计，在一定程度上是不是在让机器人动作适应一个已经非常成功的大模型范式，而不是重新问“机器人动作本身最自然的输出形式是什么”？

这不意味着 autoregressive action prediction 是错的。

它完全可以表达：

$$
p(A|o) =
\prod_i p(a_i|a_{\lt i},o)
$$

并且它有非常大的工程优势：

- 可以直接复用大模型 architecture；
- vision / language / action 可以统一表示；
- next-token training infrastructure 已经非常成熟；
- sequence likelihood 的定义清晰。

但它带来的 inductive bias 是：

$$
\boxed{
\text{trajectory = sequence of discrete symbols}
}
$$

而 Diffusion / Flow 一类方法则更倾向：

$$
\boxed{
\text{trajectory = one continuous high-dimensional object}
}
$$

这两个视角并不等价。

---

## Part III：$\pi_0$ —— 两条路线重新汇合

## 12. $\pi_0$ 接受 VLM，但不再要求 Action 服从语言输出接口

[$\pi_0$](https://arxiv.org/abs/2410.24164) 最有意思的地方，不只是“使用了 [Flow Matching](https://arxiv.org/abs/2210.02747)”。

更重要的是，它重新划分了系统里不同部分的职责。

一边仍然保留：

$$
\boxed{
\text{pretrained VLM}
}
$$

负责：

- visual representation
- language understanding
- semantic knowledge
- task conditioning

另一边加入：

$$
\boxed{
\text{Action Expert}
}
$$

专门负责机器人 continuous action。

---

### 12.1 Action 又重新变回连续 trajectory

$\pi_0$ 不再要求：

$$
\text{continuous action}
\rightarrow
\text{256 bins}
\rightarrow
\text{text-like tokens}
$$

而是直接定义：

$$
A_t=
[a_t,a_{t+1},\ldots,a_{t+H-1}]
$$

并建模：

$$
\boxed{
p(A_t|o_t)
}
$$

其中 observation 包括：

$$
o_t=
[\text{images},\text{language},\text{proprioception}]
$$

论文中使用的 action chunk horizon 为：

$$
H=50
$$

然后使用 conditional flow matching 来训练 action generation。

这在抽象层面与 Diffusion Policy 很接近：

$$
\boxed{
\text{conditional generative modeling of continuous action chunks}
}
$$

区别在于具体 generative formulation 使用的是 flow matching，而不是 Diffusion Policy 原论文中的 diffusion process。

---

## 13. 为什么这可以看成两条路线重新接上？

在机器人 policy 这边，我们已经走到了：

$$
\text{BC}
\rightarrow
\text{ACT}
\rightarrow
\text{Diffusion Policy}
$$

也就是：

$$
\text{single action}
\rightarrow
\text{action chunk}
\rightarrow
\text{continuous multimodal trajectory distribution}
$$

另一边，大模型路线走到了：

$$
\text{Transformer}
\rightarrow
\text{VLM}
\rightarrow
\text{RT-1}
\rightarrow
\text{RT-2}
\rightarrow
\text{OpenVLA}
$$

也就是：

$$
\text{vision}
+
\text{language}
+
\text{action}
\rightarrow
\text{unified token sequence}
$$

$\pi_0$ 做的事情可以理解成：

$$
\boxed{
\text{VLM semantic backbone}
+
\text{continuous generative action policy}
}
$$

它保留大模型路线真正有价值的部分：

$$
\text{large-scale visual-language knowledge}
$$

但对 motor output 又重新采用机器人 policy 领域更自然的连续生成方式。

所以一个比较准确的说法是：

> $\pi_0$ 并不是证明 autoregressive action generation 是错误的，而是把“VLM 最擅长的表示方式”和“连续机器人控制最自然的表示方式”重新拆开。

---

## 14. 这里其实出现了一个很清楚的职责分工

可以粗略理解成：

### VLM

回答：

> 我看到了什么？
>
> 用户让我做什么？
>
> 这些物体之间是什么语义关系？

### Action Expert

回答：

> 具体未来几十个低层连续 motor commands 应该怎么运动？

因此整体变成：

$$
(\text{vision},\text{language},\text{robot state})
$$

↓

$$
\text{VLM representation}
$$

↓

$$
\text{Action Expert}
$$

↓

$$
A_t=[a_t,\ldots,a_{t+H-1}]
$$

这比强行让一个 language decoder 承担所有低层连续运动生成，在表示层面更有针对性。

---

## Part IV：Action Generation 之后，VLA 暴露出来的问题开始改变

到 $\pi_0$ 以后，问题已经不再只是：

$$
\text{“Action 应该用 AR 还是 Flow？”}
$$

真正困难的问题逐渐变成：

$$
\boxed{
\text{如何让一个 generalist robot 在复杂环境中长期工作？}
}
$$

Physical Intelligence 后续的 $\pi$ 系列正好可以看成几个问题的逐步展开。

---

## 15. $\pi_{0.5}$：从“会很多动作”到 Open-World Generalization

$\pi_0$ 已经能够跨多种 robot embodiment 输出 continuous action。

但一个 generalist policy 真正困难的是：

> 换一个训练时没见过的房子、物体、布局，能不能继续完成任务？

[$\pi_{0.5}$](https://arxiv.org/abs/2504.16054) 把重点放在：

$$
\boxed{
\text{open-world generalization}
}
$$

这里开始使用更加 heterogeneous 的训练数据和任务形式，不只是低层 robot action，还结合：

- high-level semantic prediction
- subtask-like supervision
- web multimodal data
- 多种 robot data

模型开始不只是学习：

> 当前手应该移动多少？

还要理解：

> 一个较长任务现在大概处于哪个阶段？

因此从 $\pi_0$ 到 $\pi_{0.5}$，研究重点开始从：

$$
\text{motor generation}
$$

转向：

$$
\boxed{
\text{generalization across environment and task context}
}
$$

---

## 16. $\pi^*_{0.6}$：Behavior Cloning 最开始的问题又回来了

如果一直依赖 expert demonstrations，模型看到的仍然主要是：

$$
d_{\text{expert}}
$$

但真正部署之后，机器人会产生自己的：

- success
- near-success
- failure
- recovery situation

所以 BC 最早的那个问题重新出现：

$$
\boxed{
\text{expert data}
\neq
\text{deployment experience}
}
$$

[$\pi^*_{0.6}$](https://arxiv.org/abs/2511.14759) 的重点转向：

$$
\boxed{
\text{Learning from Experience}
}
$$

也就是把：

$$
\text{demonstrations}
$$

和：

$$
\text{autonomous robot rollouts}
$$

以及：

$$
\text{human corrections / interventions}
$$

结合起来，再通过 RL 类方法继续提升 policy。

这里的范式变化很明显：

$$
\boxed{
\text{Imitation}
\rightarrow
\text{Experience + RL}
}
$$

它开始直接回应 BC 中另一个一直存在的问题：

> 仅仅模仿 expert action，不等于根据实际任务结果持续改进 policy。

---

## 17. MEM：Skill 够多以后，瓶颈开始变成“记得住吗？”

当局部 manipulation skill 已经越来越强时，一个很朴素的问题会变得越来越明显：

> 任务做了十分钟以后，机器人还知道自己刚才干了什么吗？

如果任务是：

> 整理厨房

机器人需要知道：

- 哪些东西已经收过；
- 哪一步失败了；
- 哪些目标还没有完成；
- 某个物体刚才在哪里；
- 当前任务进行到了哪个阶段。

你不可能无限制地把过去所有 raw RGB frame 都塞给 Transformer。

[Multi-Scale Embodied Memory（MEM）](https://arxiv.org/abs/2603.03596) 于是明确引入：

$$
\boxed{
\text{short-term visual memory}
+
\text{long-term semantic memory}
}
$$

短期保留高分辨率、细粒度近期状态；

长期把过去压缩成更抽象的 semantic / episodic information。

到这里，VLA 的研究问题已经明显从：

$$
(o_t,l)\rightarrow A_t
$$

开始走向：

$$
\boxed{
\text{partial observability}
+
\text{progress tracking}
+
\text{long-horizon memory}
}
$$

---

## 18. $\pi_{0.7}$：Subgoal、Steerability 和 World Model 开始进入 VLA

[$\pi_{0.7}$](https://arxiv.org/abs/2604.15483) 又进一步把问题变成：

> 一个 generalist model 能不能组合已有 skill，在新环境中按照不同策略完成新任务？

它开始使用更加丰富的 prompt / context，包括：

- task instruction
- subtask information
- episode metadata
- behavior quality / style information
- visual subgoal

也就是说，模型条件不只是：

$$
\text{“做什么”}
$$

还包括：

$$
\boxed{
\text{“应该以什么方式做”}
}
$$

这就是 steerability 的一部分。

---

### 18.1 World Model 又重新出现

这里有一个对我们后续阅读很重要的点。

$\pi_{0.7}$ 可以使用 world model 生成 visual subgoal：

$$
(o_t,l_{\text{subtask}})
\rightarrow
\hat o_{\text{subgoal}}
$$

然后 VLA 根据：

$$
(o_t,\hat o_{\text{subgoal}},l)
$$

生成动作。

这和 [PlaNet](https://proceedings.mlr.press/v97/hafner19a.html) / [Dreamer](https://arxiv.org/abs/1912.01603) 那种：

$$
(s_t,a_t)\rightarrow s_{t+1}
$$

再利用想象出的未来轨迹进行在线规划（PlaNet）或学习策略（Dreamer）的 forward-dynamics world model 不完全一样。

这里的 world model 更接近：

$$
\boxed{
\text{预测“下一阶段世界应该长什么样”}
}
$$

然后 VLA 解决：

$$
(o_{\text{current}},o_{\text{subgoal}})
\rightarrow
A
$$

这已经有一点 inverse-dynamics 的味道。

因此 world model 在现代 VLA 体系中，并不一定非要作为一个：

$$
\text{candidate action simulator}
$$

它也可以作为：

$$
\boxed{
\text{subgoal generator / future-state proposer}
}
$$

出现。

---

## 19. 到这里，VLA 的问题已经发生了明显变化

最开始关心的是：

$$
\boxed{
\text{机器人动作怎么预测？}
}
$$

于是有：

$$
\text{BC}
\rightarrow
\text{ACT}
\rightarrow
\text{Diffusion Policy}
$$

后来大模型路线关心：

$$
\boxed{
\text{一个 policy 怎么获得更多任务和知识？}
}
$$

于是有：

$$
\text{RT-1}
\rightarrow
\text{RT-2}
\rightarrow
\text{OpenVLA}
$$

然后两者在：

$$
\boxed{
\pi_0
}
$$

附近重新汇合：

$$
\text{VLM semantics}
+
\text{continuous generative motor policy}
$$

再往后，问题开始变成：

$$
\text{open-world generalization}
$$

$$
\text{learning from experience}
$$

$$
\text{memory}
$$

$$
\text{subtask}
$$

$$
\text{world model}
$$

$$
\text{skill composition}
$$

也就是说：

$$
\boxed{
\text{VLA 正在从“动作生成模型”逐渐变成更完整的 embodied agent}
}
$$

---

## 20. 回头看，两条路线最有意思的交叉点是什么？

目前我觉得最值得记录的是：

### 第一条线：机器人 Action Modeling

$$
\boxed{
o\rightarrow a
}
$$

↓

$$
\boxed{
o\rightarrow A
}
$$

↓

$$
\boxed{
o\rightarrow p(A|o)
}
$$

这条线不断问：

> 什么样的 output representation 才适合真实机器人运动？

---

### 第二条线：Foundation Model / VLA

$$
\boxed{
\text{multi-task robot Transformer}
}
$$

↓

$$
\boxed{
\text{VLM knowledge transfer}
}
$$

↓

$$
\boxed{
\text{Everything-as-Token}
}
$$

它不断问：

> 能不能把已有的大模型 scaling 能力迁移到机器人？

---

### 最后两者重新汇合

$$
\boxed{
\text{VLM}
+
\text{continuous generative action model}
}
$$

从这个角度看，$\pi_0$ 的意义并不只是：

> flow matching 比 autoregressive 更先进。

真正有意思的是：

$$
\boxed{
\text{系统开始重新尊重不同 modality 本身的结构差异}
}
$$

Vision / language 可以继续从 VLM 获得强大的 pretrained representation。

Motor control 则不必为了“统一”而强行变成 text-like token sequence。

---

## 21. 这里还没有解决的问题

到这里暂时不需要下结论说哪一种 VLA architecture 最终会赢。

反而有几个问题值得留着，后面进入 long-horizon 再回来判断。

### Action Chunk 到底解决了多少 Long-Horizon？

Action chunk 可以把：

$$
1000\text{ 个低层 control steps}
$$

压成：

$$
20\text{ 个局部 chunks}
$$

但这仍然不等于：

$$
\boxed{
\text{task-level long-horizon reasoning}
}
$$

它主要解决的是局部 temporal coherence。

---

### VLA 是否真的具有 Planning？

大规模 demonstration 当然可能让模型参数里吸收很多 planning pattern。

所以 VLA 并不一定表现得“完全没有规划能力”。

但：

$$
(o,l)\rightarrow A
$$

和：

$$
\text{candidate actions}
\rightarrow
\text{future rollout}
\rightarrow
\text{evaluation}
\rightarrow
\text{selection}
$$

仍然是两种不同机制。

一个是 amortized behavior。

另一个是 explicit look-ahead。

---

### Generative Action Policy 解决的不是 Task-Level Uncertainty

Diffusion / Flow 可以很好地处理：

$$
p(A|o)
$$

中的 motor multimodality。

但如果真正的不确定性是：

> 我现在到底应该先收盘子，还是先关冰箱？

那么这已经不是简单的 low-level action distribution 问题。

它需要：

- task state
- memory
- progress
- subgoal
- planning

等更高层表示。

---

### Memory / Planning / World Model 最终会不会都被一个大模型吸收？

现在还不好下结论。

一种路线是模块化：

$$
\text{VLA}
+
\text{Memory}
+
\text{Planner}
+
\text{World Model}
$$

另一种路线可能希望：

$$
\boxed{
\text{一个更大的 sequence / multimodal model}
}
$$

把这些功能逐渐隐式吸收。

目前 $\pi_{0.7}$ 这类工作反而说明，至少在当前阶段，显式提供：

- metadata
- memory
- visual subgoal
- world model context

依然有实际价值。

---

## 22. 暂时的理解

现在回头看，VLA 的发展不能只概括成：

$$
\text{模型变大}
$$

或者：

$$
\text{数据变多}
$$

至少有两条非常清楚的问题线。

第一条是：

$$
\boxed{
\text{机器人动作应该怎么表示和生成？}
}
$$

从单步 regression，到 action chunk，再到 multimodal continuous trajectory generation。

第二条是：

$$
\boxed{
\text{如何把 Foundation Model 的 scaling 和世界知识带进机器人？}
}
$$

从 multi-task Transformer，到 VLM knowledge transfer，再到 cross-embodiment VLA。

这两条路线一度通过：

$$
\boxed{
\text{action tokenization + autoregressive prediction}
}
$$

被统一到同一个大模型范式。

然后到了 $\pi_0$ 一类模型，又开始重新拆分：

$$
\boxed{
\text{语义继续使用 VLM 擅长的表示，
连续控制使用机器人 policy 更自然的生成方式。}
}
$$

而在动作输出逐渐成熟以后，真正困难的问题又继续向外扩：

$$
\text{generalization}
\rightarrow
\text{experience learning}
\rightarrow
\text{memory}
\rightarrow
\text{subgoal}
\rightarrow
\text{world model}
\rightarrow
\text{long-horizon}
$$

所以目前比较自然的下一问已经不是：

> 下一个更大的 VLA 是什么？

而是：

$$
\boxed{
\text{一个通用 VLA 要真正变成长期工作的 embodied agent，
还缺哪些机制？}
}
$$

这也是后面继续看 Long-Horizon、Memory、Planning 和 World Model × VLA 时，需要重新回来的问题。

---

## 参考论文

1. Ross, S., Gordon, G., & Bagnell, D. **[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)](https://proceedings.mlr.press/v15/ross11a.html)**. AISTATS, 2011.
2. Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. **[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT / ALOHA)](https://arxiv.org/abs/2304.13705)**. RSS, 2023.
3. Chi, C., et al. **[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)**. RSS, 2023.
4. Brohan, A., et al. **[RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)**. 2022.
5. Brohan, A., et al. **[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)**. 2023.
6. Kim, M. J., et al. **[OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)**. 2024.
7. Black, K., et al. **[$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)**. Physical Intelligence, 2024.
8. Physical Intelligence. **[$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)**. 2025.
9. Physical Intelligence. **[$\pi^*_{0.6}$: A VLA that Learns from Experience](https://arxiv.org/abs/2511.14759)**. 2025.
10. Physical Intelligence. **[MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/abs/2603.03596)**. 2026.
11. Physical Intelligence. **[$\pi_{0.7}$: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483)**. 2026.
