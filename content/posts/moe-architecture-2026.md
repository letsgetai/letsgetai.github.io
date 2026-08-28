---
title: "主流大模型为什么都是 MoE：从 DeepSeek-V3 到 Qwen3.8 的架构演进"
date: 2026-08-25
draft: false
categories: ["AI Infrastructure"]
---


> 结论先行：2026 年的主流开源模型（DeepSeek-V3/V4、Qwen3.8）不约而同选择了 MoE——把 FFN 拆成数百个专家、每个 token 只激活其中几个。这不是跟风，而是三个结构性问题同时被 MoE 解决的必然结果：容量与算力解耦、注意力与解码的双重优化、加速机制内建于训练。这篇文章从 DeepSeek-V3 的细节讲起，把 MoE 的每个零件、注意力的两条改进路线、MTP 的来龙去脉，以及“小模型为什么不用 MoE”，按我自己的逻辑线一次讲清楚。

## 引子：一个值得注意的现象

2024 年之前，主流大模型是两种形态：dense 的（GPT-3、Llama）和稀疏的（GShard 式的 MoE）。到 2026 年，事情变了：DeepSeek 全线 MoE（V3 的 671B、V4 的 284B/1.6T），Qwen 的主力也变成 MoE（Qwen3.8-2.4T-A95B），连“80B 总参、3B 激活”的中间档（Qwen3-Next）都是 MoE。dense 只留在小模型区（Qwen 的 0.6B-32B）。

为什么？答案是 MoE 同时解决了两类问题：训练侧，它把“模型容量”和“每 token 算力”解耦，让你可以养一个 600B 的模型而每次推理只花 37B 的钱；推理侧，它和注意力压缩、MTP（多 token 预测）叠加，把单步成本和步数同时压下来。下面从零件讲起。

## 一、MoE 是什么：把 Transformer 的 FFN 拆开

以 DeepSeek-V3（2024 年 12 月）为例——它是第一个把 MoE 细节大规模定型的模型，也是理解后续一切的地基：671B 总参数、37B 每 token 激活。

### 1.1 专家的结构与数量

Transformer 的 FFN 层被替换成 MoE：1 个**共享专家** + 256 个**路由专家**，每个 token 激活 Top-8 个路由专家，专家中间维 2048。共享专家永远参与，负责通用能力；路由专家做细粒度分工。

和更早的 GShard 式 MoE（Top-2 门控、专家粒度粗）相比，DeepSeekMoE 的两点区别是：专家更细（数量多、每个专家维度小），且混入共享专家。直觉是：细粒度专家让分工更灵活，共享专家保证每个 token 至少有一条稳定的计算路径，不会因为路由失误而完全失去某类能力。

每个专家的网络本身是一个 **SwiGLU FFN**（激活函数：$\text{Swish}(xW) \odot xV$ 的门控结构），这和 dense 模型的 FFN 没有本质区别——MoE 换的是“网络的数量和路由方式”，不是“网络内部的结构”。

### 1.2 路由：token 怎么选专家

路由分三步：

1. **算 affinity**：token 表示与每个专家的质心向量做内积，再过激活函数。V3 用 Sigmoid，V4 换成 $\mathrm{Sqrt}(\mathrm{Softplus}(\cdot))$（数值更稳）。
2. **Top-K 选择 + 归一化**：取 affinity 最高的 8 个专家，把选中的 affinity 归一化作为门控权重。
3. **负载均衡**：这是 MoE 最容易翻车的地方——如果所有 token 都涌向少数专家，计算会严重倾斜。V3 的方案叫**无辅助损失负载均衡**：给每个专家维护一个 bias，路由时把 bias 加到 affinity 上参与排序，但门控权重仍用原始 affinity；每步训练结束后，过载专家的 bias 减一点、欠载专家加一点（更新速度 $\gamma=0.001$）。这样均衡来自动态 bias 而非损失项，避免了传统 aux loss 的副作用——论文的消融显示，aux-loss-free 模型在所有评测上一致优于纯 aux loss 模型，且专家的专业化程度更强（Figure 9：aux-loss-free 的专家负载分布更分散）。

此外还有两个工程细节：**节点受限路由**（每个 token 最多发往 4 个节点，控制通信量）和**不丢 token**（均衡有效所以全程不丢，训练推理都不丢）。

![DeepSeek-V3 基本架构](/images/moe-architecture-2026/deepseek-v3-basic-arch.png)

### 1.3 每个设计解决什么问题

| 设计 | 解决的问题 |
| --- | --- |
| 细粒度专家 + 共享专家 | 分工灵活性 + 稳定计算路径 |
| Top-K 门控 + affinity 归一化 | 连续可导的路由权重 |
| 无 aux-loss 负载均衡 | 均衡与性能的 tradeoff（aux loss 损伤能力） |
| 节点受限路由 | 训练通信成本 |
| 不丢 token | 避免能力波动与推理不可控 |

## 二、注意力机制的两条改进路线

MoE 解决的是 FFN 侧的容量问题，注意力侧是另一场仗：上下文越来越长，注意力的时间和内存成本随长度增长。两家走了两条不同的路。

### 2.1 DeepSeek 路线：压缩得越来越狠

- **V3：MLA（维度压缩）**。把每个 token 的 K/V 低秩压缩成一个潜在向量 $c_t^{KV}$，推理只缓存它加一条解耦的 RoPE key，每 token 从 128×128 维降到 512+64=576 维。压缩的是**单个 token 的表示维度**。
- **V3.2：DSA（稀疏）**。在 MLA 基础上引入稀疏注意力，每个 query 只 attend 少量 KV。
- **V4：CSA/HCA（长度压缩 + 稀疏 + 滑窗）**。这是把注意力重构了：CSA 先把每 4 个 token 的 KV 加权压缩成 1 个条目，再用轻量索引器做 top-512 稀疏选择，最后用共享 KV 的 MQA 做核心注意力，外加 128 token 的滑窗分支保局部细节；HCA 更狠，每 128 个 token 合并成 1 个条目、不做稀疏。效果：1M 上下文下 KV cache 约为 BF16 GQA8 基线的 **2%**（BF16/FP8 混合存储再省一半）。

配套的稳定性设计：mHC 残差连接（把残差映射矩阵约束到双随机矩阵流形，防止深层堆叠数值不稳）、Muon 优化器（混合 Newton-Schulz 正交化）、核心注意力前对 query 和 KV 做 RMSNorm（免去 QK-Clip）。V4 配置：43 层、1 共享 + 256 路由 Top-6、前 3 层用 Hash 路由（按 token ID 哈希定专家，不用学；报告未说明动机，推断为浅层 token 语义简单、省去路由学习成本），并去掉了节点受限路由（并行策略重设计来维持训练效率）。

![DeepSeek-V4 整体架构](/images/moe-architecture-2026/deepseek-v4-arch.svg)

![CSA 机制](/images/moe-architecture-2026/deepseek-v4-csa.svg)

![HCA 机制](/images/moe-architecture-2026/deepseek-v4-hca.svg)

### 2.2 Qwen 路线：线性注意力 + 全注意力交错

Qwen 的判断相同、做法不同。Qwen3-Next（80B/3B 激活）和放大版 Qwen3.8（2.4T/95B 激活）用 **Gated DeltaNet 线性注意力（75% 层）+ Gated Attention 全注意力（25% 层）** 交错排列（3:1）。线性注意力把状态压缩进一个固定大小的隐状态，推理复杂度 O(1)，但召回弱；全注意力召回强但贵。Qwen 的实验结论是 3:1 混合优于任何单一架构。

这里有个容易混淆的点需要澄清：**交错注意力不是新东西**。ModernBERT（[arXiv 2407.20270](https://arxiv.org/abs/2407.20270)，2024）就已经用 local/global 交替注意力了——但 ModernBERT 是 **dense 编码器**，激活率 100%，没有“3.7% 激活率”这回事。**3.7% 激活率是 Qwen3-Next 的 MoE 数字**（80B 总参 / 3B 激活，512 专家 Top-10+1）。ModernBERT 贡献的是“交错注意力”这个先例，Qwen 把它搬进生成式 MoE，并配上了超稀疏的专家结构。

![Qwen3-Next 架构](/images/moe-architecture-2026/qwen3-next-architecture.png)

### 2.3 深究：KV 压缩的两条路

- **维度压缩（MLA）**：把每个 token 的 K/V 投影到低维潜在空间。压缩率固定（与长度无关），实现简单，KV 复用性好（前缀缓存友好）。
- **长度压缩（CSA/HCA）**：把连续 m 个 token 的 KV 加权合并成一个条目。压缩率随长度线性增长，信息有损（合并即丢弃细粒度），且对位置编码、缓存管理提出新要求（V4 专门设计了经典 KV cache + state cache 的双层布局）。

DeepSeek 从 V3 到 V4 的切换说明一个判断：当上下文从 128K 推到 1M，瓶颈从“每个 token 占多大”变成“token 数量本身”，所以必须动长度。Qwen 的线性注意力本质也是长度压缩（状态化），两条路殊途同归。

## 三、MTP 与推测解码：加速从推理侧走进训练侧

### 3.1 推测解码：解码为什么慢

解码是带宽问题：每生成一个 token 都要把全部参数从显存读一遍。推测解码的思路是：用一个小模型先猜 K 个 token，大模型一次前向并行验证，配合修正拒绝采样保证输出分布与原模型完全一致。猜得准就有 2-3x 加速。关键瓶颈是“草稿从哪来”——独立小模型要额外训练和部署，且和目标模型匹配度决定加速上限。

### 3.2 相关工作：MTP 是谁先做的

MTP（Multi-Token Prediction）不是 DeepSeek 发明的，但顺序式实现是它的原创。脉络分四步：

- **ProphetNet（Qi et al. 2020，[arXiv 2001.04063](https://arxiv.org/abs/2001.04063)）**：种子想法。在 seq2seq 预训练里预测“未来 n-gram”（同时预测接下来 n 个 token），发现能提升下游表现。这是最早的多 token 预测预训练。
- **Gloeckle et al. 2024（Meta，[arXiv 2404.19737](https://arxiv.org/abs/2404.19737)）**：正式把多 token 预测变成解码器 LLM 的预训练目标，用并行独立头同时预测后续 K 个 token。效果：13B 模型在 HumanEval 上多解约 12%、MBPP 多解约 17%；推理侧配合自投机解码（用训练好的额外头当 drafter），7B 4-token 模型代码生成约 3.0x 加速、自然语言约 2.7x。这是 DeepSeek-V3 的直接灵感。
- **EAGLE（[arXiv 2401.15077](https://arxiv.org/abs/2401.15077)）**：特征空间自回归。不在 token 空间硬猜，而是用“backbone 最终隐藏态 + 下一 token embedding”预测下一隐藏态，再做投机解码的 drafter（EAGLE-3 最高 6.5x）。DeepSeek 的 MTP 模块结构借鉴了它。
- **DeepSeek-V3（2024 年 12 月）**：把两者组合并改造——用 EAGLE 的结构（拼接、过层）做 Gloeckle 式的事（预训练目标），并强调“每个预测深度保持完整因果链”（顺序式，区别于 Gloeckle 的并行头）。“顺序式 MTP 训练目标”这个组合是 V3 的原创设计。

### 3.3 DeepSeek-V3 的实现：一层怎么做到 85-90%

V3 的 MTP 用 D 个顺序模块（D=1），第 k 个模块的输入是：

$$h_i'^k = M_k[\mathrm{RMSNorm}(h_i^{k-1}); \mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))]$$

即“上一层表示（k=1 时为主模型最终隐藏态）+ 已知下一个 token 的 embedding”拼接后线性投影，过一个**完整的 transformer block（MLA 注意力 + MoE FFN，与主模型同宽）**，再用共享输出头预测第 k+1 个未来 token。embedding 与输出头与主模型共享（省显存），损失权重前 10T token 用 0.3、后 4.8T 降到 0.1。

![MTP 实现](/images/moe-architecture-2026/deepseek-v3-mtp.svg)

**实测数据（V3 报告 §6）**：第二 token 的接受率 85%~90%，解码速度约 1.8x TPS——这就是“一层”的效果。

为什么一层就这么准？两个原因：

1. **任务被输入变简单了**：输入里已经给了“下一个 token 是谁”，猜“再下一个”大多只需要局部共现信息，大部分难度被这个额外输入消掉了。
2. **这一层不是小层**：它和主模型同宽（完整 MLA + MoE 层），浅（深度 1）但不少（宽度全）。

剩下的 10-15% 缺口，是 t_{i+2} 真依赖长程上下文、或上一步 t_{i+1} 本身猜错的情况。

训练收益与推理收益分离：消融显示 15.7B/228.7B 两档加 MTP 后评测一致提升，且推理时丢弃模块成本不变——训练侧收益“白拿”；推理侧 MTP 模块成为免费的高匹配 drafter（85-90% 接受率）。

### 3.4 Qwen 的改进：多步训练、解耦、全注意力

Qwen3-Next/Qwen3.8 在 MTP 上做了三处改进，补 V3 的两个短板：

1. **多步训练（修 exposure bias）**：训练时让 MTP 链用自己的预测 rollout 几步再反传，而不是只喂真值——对齐推理时的多步自回归分布。V3 训练只喂真值（teacher forcing），推理却连续复用多步，存在“训练-推理不一致”，这是 Qwen 最大的一处改进。
2. **sidecar 解耦**：MTP 是独立权重文件（单独分发、可单独微调/量化，HF 不内置、需 SGLang/vLLM 的 NEXTN 算法），部署灵活。
3. **full-attention drafter**：MTP 模块内部特意用全注意力而非线性注意力（draft 需要强召回），和 DeepSeek 的 MTP block 复用 MLA 是同一个逻辑。

### 3.5 深究一：一层就够吗？——正确采样是关键

**一层 vs 多层**：层数越多对“下一位置隐藏态”的近似越准，但第二层 token 的接受率要乘上第一层（约 85%x85%≈72%），边际收益递减而成本线性上升，D=1 大概率是性价比甜点。公开报告没有“一层 vs 多层”的严格消融（V3 只验证了 D=1），这是推断。

**暴露偏差（exposure bias）**：训练时 MTP 模块看到的是真实 token，推理时看到的是自己上一步的预测——连续迭代时误差累积、分布会“飘”。但**正确采样兜底**：投机解码的验证是修正拒绝采样，对照 backbone 的真实分布，漂移的 draft token 会被拒绝，输出分布严格等于主模型——漂移只降接受率（降速），不破坏分布（不降质）。vLLM 的实现正是同一个 MTP block 循环复用做多步 draft。

**反面教训**：[CLP 论文](https://arxiv.org/abs/2606.10935)（arXiv 2606.10935）指出，此前 gate 式并行 MTP 失败（加速不了或 repetition >0.5%），根因是接受时用置信度门控、没做分布匹配的采样——“飘”的 token 直接进输出，上下文被污染后重复自强化。修法是 Backbone-as-Architect（主模型恒产第一个 token）+ 正确采样，漂移就只降速不降质。

### 3.6 深究二：激活率如何影响推理速度

Qwen3-Next 的 3.7% 激活率常被拿来当卖点，但要小心一个误区：**激活参数少不等于硬件成本低**。MoE 推理时所有专家权重都要加载到显存，每 token 只算其中几个——省的是计算（FLOPs），不省带宽和显存占用。激活率越低，理论上 FLOPs 越低，但路由开销占比、专家负载不均衡的风险都会上升；实验也显示“专家越多、激活越少”不是单调更好（[arXiv 2508.18672](https://arxiv.org/abs/2508.18672) 的最优稀疏度研究）。所以 3.7% 是 Qwen 在容量、速度、稳定性之间试出来的平衡点，不是越低越好。

### 3.7 MTP 与注意力机制的关系

之前有个说法：“MoE 用了 MLA 就会跟 MTP 冲突”。核验后的结论分三层：

1. **无结构性冲突**。MLA 压缩“过去”（KV），MTP 优化“未来”（预测目标），作用维度正交。最硬的证据：DeepSeek-V3 的 MTP block 内部用的就是 MLA + MoE（vLLM 源码可查），V4 换了注意力（CSA/HCA）但 MTP 配置不变。
2. **真实竞争在预测头**。CLP 论文（arXiv 2606.10935）证明：并行式 MTP（比如 Medusa 式 gate 网络）里，预测第一个 token 的 MTP 头会跟主模型自己的 LM head 竞争，被接受时导致重复、不连贯输出。修法是 Backbone-as-Architect：主模型恒产第一个 token，MTP 头只负责后续。**DeepSeek 的顺序式 MTP 天生就是这个设计**（实现里位置 0 的输入被直接 mask 掉），所以没有这个问题——这也是它投机解码实测不降质的原因。
3. **工程耦合**。MTP drafter 是完整 transformer 层，推理时它的 KV、显存和主模型的缓存管理需要一起设计；V4 的异构 KV cache 加 MTP 怎么共管，报告没有讨论，属于开放问题。

## 四、小模型为什么不用 MoE

10B 以下几乎没有 MoE 开源模型，这不是保守，是成本结构决定的：

- **端侧实测**（[OLMoE-1B-7B](https://arxiv.org/abs/2606.21428)，2026）：激活仅 1.3B 的 MoE 在 M2 Pro 上比同激活规模 dense 慢约 10%，在 Jetson 上慢 31%，能耗 2.1×，8GB 显存打顶。原因就是上面说的：所有专家权重都得加载。
- **等资源实验**（[arXiv 2506.12119](https://arxiv.org/abs/2506.12119)）：约 250 个 2B/7B 级模型对比显示，设计得当的 MoE 在等算力/等数据下可以赢 dense，但需要更多数据——门槛在数据预算，小模型通常没有。
- **路由塌缩**：小 MoE（Qwen3-30B-A3B 等）在低资源语言上出现深层路由失效（[arXiv 2605.17598](https://arxiv.org/abs/2605.17598)），容量被浪费。
- **产业实践**：Qwen 家族 0.6B/4B/14B/32B 全是 dense，MoE 从 30B-A3B 起步；DeepSeek 最小的开源 MoE 也是 236B 级。80B-A3B 这类“大总参小激活”是服务端吞吐场景的特例，不是端侧答案。

一句话：**小模型的容量瓶颈该用数据质量解决，不该用参数分裂解决**。

## 五、结论：MoE 为什么成为主流

三个结构性原因：

1. **容量与算力解耦**：总参决定容量，激活参决定成本。这让“更大”和“更便宜”不再互斥，是唯一能在有限 GPU 预算下把模型做到 600B+ 的架构。
2. **注意力与解码的双重优化**：MoE 负责 FFN 侧，压缩/线性注意力负责长度侧，MTP 负责步数侧——三个正交的杠杆可以叠加，总延迟 ≈ 步数 × 单步成本。
3. **加速内建于训练**：MTP 让“草稿模型”免费长在主模型里，推理加速不再需要额外训练和部署第二个模型。

选型判断：短/中上下文用 dense 或 GQA/MLA 就够；百万 token 级上 CSA/HCA 或线性注意力二选一；MTP 深度 1 + 小损失权重，无脑加、优先顺序式；小模型默认 dense，只有服务端大 batch + 长上下文 + 总参预算充足才考虑 80B-A3B 形态。

## 材料来源

- DeepSeek-V3 技术报告：[arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-V4 技术报告：[arXiv 2606.19348](https://arxiv.org/abs/2606.19348)
- Qwen3-Next 官方博客：[qwen.ai](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd) 与[模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Qwen3.8-2.4T-A95B 模型卡：[Hugging Face](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- CLP：[arXiv 2606.10935](https://arxiv.org/abs/2606.10935)
- ModernBERT：[arXiv 2407.20270](https://arxiv.org/abs/2407.20270)
- OLMoE 端侧实测：[arXiv 2606.21428](https://arxiv.org/abs/2606.21428)
- 等资源 MoE 实验：[arXiv 2506.12119](https://arxiv.org/abs/2506.12119)
- 最优稀疏度：[arXiv 2508.18672](https://arxiv.org/abs/2508.18672)
- 路由塌缩：[arXiv 2605.17598](https://arxiv.org/abs/2605.17598)
- vLLM 源码：deepseek_mtp.py / qwen3_next_mtp.py

- MTP 相关工作：ProphetNet（[arXiv 2001.04063](https://arxiv.org/abs/2001.04063)）、Gloeckle et al.（[arXiv 2404.19737](https://arxiv.org/abs/2404.19737)）、EAGLE（[arXiv 2401.15077](https://arxiv.org/abs/2401.15077)）
全部数字为官方/论文报告口径，加速倍数跨场景外推需自行复测。
