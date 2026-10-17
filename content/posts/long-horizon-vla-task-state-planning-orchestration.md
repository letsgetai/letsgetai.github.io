---
title: "Long-Horizon VLA：任务状态、规划、执行与编排"
date: 2026-10-18
publishDate: 2026-10-18T00:00:00+08:00
draft: false
categories: ["Robotics"]
---

> 机器人需要在持续交互中维护一个可靠的任务状态，并据此不断规划、执行、验证和修正。
>
> 因此，一个比较自然的结构是：
>

$$
\boxed{
\text{Task State}
\rightarrow
\text{Planner}
\rightarrow
\text{Executor}
\rightarrow
\text{Orchestrator / Harness}
}
$$

>
> 然后环境反馈重新进入 Task State，形成闭环。
>
> 这里很多论文都会同时跨越多个模块，所以不强行把“一篇论文”归到唯一一类。更合适的做法是：**正文按问题和模块组织，论文作为跨模块案例穿插进去。**

---

## 1. Long-Horizon VLA 到底在解决什么问题？

普通 VLA 可以粗略写成：

$$
(o_t,l)\rightarrow A_t
$$

其中：

- $o_t$：当前视觉、本体感知等 observation；
- $l$：语言任务；
- $A_t$：当前要执行的一段 action chunk。

这对于局部 manipulation 已经是一个很强的范式：

$$
\text{当前看到什么}
+
\text{当前要做什么}
\rightarrow
\text{接下来怎么动}
$$

但任务一旦变长，问题就发生变化。

例如：

> 整理厨房。

机器人不只需要知道“现在怎么抓这个杯子”，还必须持续回答：

- 之前已经做过什么？
- 哪些任务已经完成？
- 当前 subtask 做到什么程度？
- 刚才那一步到底成功了吗？
- 当前状态和预期是否一致？
- 原来的计划还有效吗？
- 现在应该继续、重试、恢复，还是重新规划？
- 如果有多个未来选择，哪个更好？

因此 Long-Horizon 的核心问题可以写成：

$$
\boxed{
\text{如何在持续环境交互中维护一个可靠的 Task State，
并据此持续决策？}
}
$$

机器人其实一直都有环境反馈：

$$
a_t
\rightarrow
\text{environment}
\rightarrow
o_{t+1}
$$

问题并不是“有没有反馈”。

真正的问题是：

$$
\boxed{
\text{raw observation 能不能被持续转化成对长期决策有用的任务状态？}
}
$$

这就是后面 Memory、Progress、Verification、Recovery、Replanning 等模块逐渐出现的原因。

---

## 2. 总体框架：四个模块

目前看下来，可以把 Long-Horizon VLA 压成四个模块：

$$
\boxed{
\text{Task State}
\rightarrow
\text{Planner}
\rightarrow
\text{Executor}
\rightarrow
\text{Orchestrator}
}
$$

它们不是严格独立的系统组件，更像四类功能。

### Task State

回答：

> **我现在到底是什么情况？**

包括：

- 当前世界状态；
- 过去发生过什么；
- 哪些任务已经完成；
- 当前做到哪里；
- 当前是否失败；
- 当前 belief 是否需要修正。

### Planner

回答：

> **基于当前任务状态，下一步应该做什么？**

包括：

- goal decomposition；
- subgoal generation；
- remaining-plan prediction；
- replanning；
- world-model look-ahead；
- value / search。

### Executor

回答：

> **已经决定做什么以后，具体应该怎么动？**

这基本就是[前面 VLA / imitation policy 讨论的部分](https://letsgetai.github.io/posts/vla-action-modeling-generalist-policy/)：

$$
z_t\rightarrow A_t
$$

例如：

- action chunk；
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)；
- [flow matching](https://arxiv.org/abs/2210.02747)；
- [VLA Action Expert](https://arxiv.org/abs/2410.24164)。

### Orchestrator / Harness

回答：

> **Planner 和 Executor 如何稳定地闭环工作？**

包括：

- termination；
- switching；
- verification；
- retry；
- rollback；
- forward recovery；
- return-to-planner。

这里面很多东西确实更像是当前 VLA 能力不足时的 execution harness，而不完全等于传统意义上的 planning。

---

## Part I：Task State

## 3. 为什么当前 Observation 不够？

Long-Horizon manipulation 很多时候并不是仅凭当前 observation 就能决策的。

如果 policy 只有：

$$
\pi(a_t|o_t,l)
$$

就默认当前 observation 已经包含所有决策需要的信息。

但很多任务显然不满足这一点。

例如：

> Blue → Pink → Green，依次按按钮。

当前图像里三个按钮都还存在。

只看：

$$
o_t
$$

可能无法知道：

- Blue 还没按；
- Blue 已经按过，现在应该按 Pink；
- Blue 和 Pink 都按过，现在应该按 Green。

所以真正需要的是：

$$
o_{1:t}
$$

中与当前任务有关的信息。

这也是为什么 Long-Horizon 中的第一类工作通常从 Memory 开始。

---

## 4. Memory：过去的信息怎么留下来？

### 4.1 MemoryVLA：从当前 observation 到带历史的内部状态

[MemoryVLA](https://arxiv.org/abs/2508.19236) 的核心问题是：

$$
\boxed{
\text{当前 observation 不足以决定当前 action}
}
$$

因此它把当前 representation 分成两部分：

$$
\boxed{
\text{Perceptual Memory}
+
\text{Cognitive Memory}
}
$$

其中：

- Perceptual Memory 保留低层视觉细节；
- Cognitive Memory 保留更高层的语义信息。

它没有简单地把所有历史 frame 全部塞进 context，而是做：

$$
\boxed{
\text{Retrieve}
\rightarrow
\text{Fuse}
\rightarrow
\text{Consolidate}
}
$$

也就是：

1. 当前状态去历史 memory 中检索相关内容；
2. 当前信息和历史信息做 gated fusion；
3. 相似、冗余的 memory 进行压缩。

因此内部状态可以粗略写成：

$$
b_t=f(o_t,M_{t-1})
$$

然后：

$$
a_t\sim\pi(a_t|b_t,l)
$$

这里最重要的不是某个具体 attention 结构，而是一个认识：

$$
\boxed{
\text{Memory 不是单纯把 Context Window 拉长}
}
$$

真正的问题是：

$$
\boxed{
\text{存什么、什么时候取、怎么融合、怎么压缩}
}
$$

### 4.2 HELM：Memory 也可以是显式 Episodic Memory

[HELM](https://arxiv.org/abs/2604.18791) 的做法和 MemoryVLA 不一样。

它更接近显式 episodic memory：

$$
\mathcal M=\{(k_i,v_i)\}
$$

memory 中记录：

- 关键 observation；
- 当时执行的 subgoal；
- success / failure；
- 时间；
- compact state delta。

而且不是每一帧都存，而是优先记录：

$$
\boxed{
\text{subgoal success}
,\quad
\text{failure}
,\quad
\text{checkpoint}
}
$$

这说明 Memory 也可以从：

$$
\text{latent feature memory}
$$

走向：

$$
\boxed{
\text{event-oriented episodic memory}
}
$$

这两个方向现在都还存在，没有统一。

---

## 5. Memory 还不够：还需要知道当前做到哪里

即使模型记住了过去，仍然存在一个问题：

$$
\boxed{
\text{“我记得发生过什么”}
\neq
\text{“我知道当前做到哪里”}
}
$$

例如：

> 把杯子放进抽屉。

执行过程中可能经历：

$$
\text{还没抓}
\rightarrow
\text{已经抓住}
\rightarrow
\text{移动中}
\rightarrow
\text{已经放进去}
\rightarrow
\text{可以松手}
$$

当前 task phase 本身就是一个重要状态变量。

---

## 6. Progress：显式表示当前执行进程

### 6.1 PALM：连续的 progress state

[PALM](https://arxiv.org/abs/2601.07060) 显式加入：

$$
p_t\in[0,1]
$$

表示当前 subtask 的进度。

于是 task state 不再只是：

$$
(o_t,M_t)
$$

而开始包含：

$$
\boxed{
(o_t,M_t,p_t)
}
$$

PALM 同时还预测：

- 当前应该关注什么 object；
- 什么区域适合 contact；
- 什么区域适合 placement；
- 接下来应该发生什么 motion。

也就是：

$$
\boxed{
\text{Affordance State}
+
\text{Progress State}
}
$$

这里一个重要变化是：

> Progress 不只是为了让模型“知道时间过去了多久”，而是让它区分当前处于任务的哪个 phase。

例如两个 observation 很像：

$$
o_i\approx o_j
$$

但：

$$
p_i=0.2,\qquad p_j=0.9
$$

对应的动作可能完全不同。

### 6.2 Progress 还可以用于 Termination

PALM 进一步把 progress 用于：

$$
\boxed{
\text{什么时候结束当前 skill}
}
$$

例如：

$$
p_t>\phi
$$

就触发 subtask transition。

这说明一个变量可以跨模块发挥作用：

- 在 Task State 里，$p_t$ 表示进度；
- 在 Orchestrator 里，$p_t$ 又可以决定何时切换。

这也说明：**论文和模块不是一一对应关系。**

---

## 7. 从 Latent State 到 Structured Task State

MemoryVLA 和 PALM 更偏隐式 latent representation。

但进入更复杂的 Planner 后，很多工作开始显式维护 task state。

### 7.1 Goal2Skill：Precondition / Postcondition

[Goal2Skill](https://arxiv.org/abs/2604.13942) 不再把一个 subtask 只表示成：

$$
z_k=\text{“pick up cup”}
$$

而是包含：

$$
\tau_k=
(
\ell_k,
\operatorname{pre}_k,
\operatorname{post}_k,
...)
$$

这意味着一个 subtask 同时定义：

$$
\boxed{
\text{什么时候可以执行}
+
\text{什么叫执行成功}
}
$$

于是执行后可以比较：

$$
\boxed{
\text{Expected State}
\overset{?}{=}
\text{Observed State}
}
$$

Task State 开始变得：

$$
\boxed{
\text{structured}
+
\text{verifiable}
}
$$

而不是只有一个神经网络 latent。

### 7.2 LoHo-Manip：Done + Remaining

[LoHo-Manip](https://arxiv.org/abs/2604.21924) 又给出一个更简单的 task-state 表示：

$$
\boxed{
b_t=[\text{Done},\text{Remaining},\text{Current Visual Trace}]}
$$

它不追求显式维护完整 symbolic world state。

而是只保留：

> 哪些已经完成、哪些还剩。

这是一种非常 task-oriented 的 state abstraction。

### 7.3 $\tau_0$-VLA：Memory 本身也可能错

更进一步，[$\tau_0$-VLA](https://arxiv.org/abs/2608.16885) 开始考虑：

$$
\boxed{
\text{Memory / Task State 本身可能不正确}
}
$$

例如：

- memory 落后于真实状态；
- memory 比真实状态提前；
- memory 和当前 observation 发生冲突。

这时候需要：

$$
(o_t,M_{t-1})
\rightarrow
M_t^{corrected}
$$

也就是：

$$
\boxed{
\text{Task-State Revision}
}
$$

这已经比“把历史留下来”更接近一个真正的 belief update 问题。

---

## 8. Task State 这一条线可以怎么理解？

从目前这些工作看，可以粗略写成：

$$
\boxed{o_t}
$$

↓

$$
\boxed{(o_t,M_t)}
$$

Memory

↓

$$
\boxed{(o_t,M_t,p_t)}
$$

Progress

↓

$$
\boxed{
b_t=[\text{history},\text{progress},\text{done},\text{remaining},\text{failure},...]}
$$

Structured Task State

↓

$$
\boxed{
b_t\xrightarrow{\text{new evidence}}b_{t+1}}
$$

State Revision

真正稳定的问题不是“到底应该用 Memory Bank 还是自然语言 summary”。

而是：

$$
\boxed{
\text{一个长期 agent 到底需要维护什么 Task State？
这个状态怎么根据新 observation 持续修正？}
}
$$

---

## Part II：Planner

## 9. 有了 Task State，下一步才是 Planning

假设已经有：

$$
b_t
$$

以及最终目标：

$$
g
$$

Planner 要解决：

$$
\boxed{
P(b_t,g)\rightarrow z_t
}
$$

其中：

$$
z_t
$$

是下一个 semantic subgoal / skill。

这部分可以从简单到复杂分成几个层次。

---

## 10. 第一层：Goal Decomposition

最简单的 Planner：

$$
g\rightarrow[z_1,z_2,\ldots,z_K]
$$

例如：

> 整理桌子。

拆成：

$$
\text{pick cup}
\rightarrow
\text{place cup}
\rightarrow
\text{pick plate}
\rightarrow
\text{place plate}
$$

这已经能显著降低低层 VLA 的有效 horizon。

VLA 不再需要直接解决：

$$
\text{十分钟任务}
$$

而只需要不断解决：

$$
\boxed{
\text{几秒钟级别的局部 skill}
}
$$

这也是 Hierarchical VLA 的基本逻辑。

---

## 11. 第二层：Closed-Loop Replanning

一次生成完整计划并执行到底并不可靠。

因为执行结果会改变环境：

$$
b_t\xrightarrow{z_t}b_{t+1}
$$

而真实：

$$
b_{t+1}
$$

可能和计划中的预期不同。

### 11.1 Goal2Skill：反馈真正修改 Future Plan

Goal2Skill 的闭环是：

$$
\text{Plan}
\rightarrow
\text{Execute}
\rightarrow
\text{Observe}
\rightarrow
\text{Verify}
\rightarrow
\text{Update State}
\rightarrow
\text{Revise Plan}
$$

如果当前 subtask 失败，reflection 可以得到：

$$
\boxed{
\text{retry}
\quad/\quad
\text{adjust}
\quad/\quad
\text{replan}
}
$$

这里第一次比较明确地出现：

$$
\boxed{
\text{feedback 不只是触发 retry，
而是可以修改后续计划本身}
}
$$

---

## 12. 第三层：Receding-Horizon Task Planning

### 12.1 LoHo-Manip：Done + Remaining

Goal2Skill 更像：

$$
\text{维护一份计划，然后修它}
$$

LoHo-Manip 则换了一种思路：

> 每次看到新的 observation，重新判断“已经做完什么、剩下什么”。

即：

$$
(o_t,g)
\rightarrow
[
\text{Done},
\text{Remaining}
]
$$

执行一点以后：

$$
o_{t+1}
$$

再次：

$$
(o_{t+1},g)
\rightarrow
[
\text{Done}',
\text{Remaining}'
]
$$

从控制思想上看，这和 MPC 很相似：

$$
\boxed{
\text{Plan}
\rightarrow
\text{Execute Locally}
\rightarrow
\text{Observe}
\rightarrow
\text{Replan}
}
$$

区别是这里的 planning variable 不再是低层连续 action：

$$
a_t,a_{t+1},\ldots
$$

而是：

$$
\boxed{
\text{semantic task sequence}
}
$$

因此可以把它理解成：

$$
\boxed{
\text{Task-Level Receding Horizon}
}
$$

---

## 13. 第四层：Counterfactual Planning

前面的 Planner 虽然会不断 replanning，但大多数情况下仍然是：

$$
(b_t,g)
\xrightarrow{\text{one forward}}
z_t
$$

也就是：

> 看见当前状态，直接猜下一步。

但这和经典 planning 仍然有差别。

真正的 look-ahead 应该是：

$$
z_A\rightarrow\hat b_A
$$

$$
z_B\rightarrow\hat b_B
$$

然后比较：

$$
V(\hat b_A),V(\hat b_B)
$$

再决定。

---

## 14. $\tau_0$-VLA：World Model 被重新拿回 Planner

$\tau_0$-VLA 把高层 decision 显式写成 search。

先生成候选：

$$
z_1,\ldots,z_N
$$

World Model：

$$
(o_t,z_i)
\rightarrow
\hat o_i
$$

预测：

> 如果执行这个 semantic subgoal，完成以后世界大概是什么样。

然后 Value Model：

$$
V(g,z_i,\hat o_i)
$$

评价：

> 这个未来对最终任务好不好。

于是出现：

$$
\boxed{
\text{Propose}
\rightarrow
\text{Imagine}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Choose}
}
$$

再配合 beam search：

$$
z_1
\rightarrow
\hat o_1
\rightarrow
z_2
\rightarrow
\hat o_2
\rightarrow
\cdots
$$

这才真正回到我们之前学过的：

$$
\boxed{
\text{Prediction}
+
\text{Evaluation}
+
\text{Search}
}
$$

只不过 planning variable 从低层 action 变成了：

$$
\boxed{
\text{semantic subgoal}
}
$$

---

## 15. World Model 在这里应该怎么定位？

世界模型的使用位置由预测用途决定。在 $\tau_0$-VLA 这类基于候选子目标搜索的系统中，它提供 Planner 内部的前瞻机制：预测候选子目标对应的未来，再结合 value 评估与 beam search 选择后续子目标。

$$
\boxed{\text{World Model 为 Planner 提供前瞻评估}}
$$

普通 Planner：

$$
b_t\rightarrow z_t
$$

加入 World Model：

$$
b_t
\rightarrow
\{z_i\}
\rightarrow
\{\hat b_i\}
\rightarrow
\{V_i\}
\rightarrow
z^*
$$

也就是：

$$
\boxed{
\text{Proposal}
+
\text{World Model}
+
\text{Value}
+
\text{Search}
}
$$

世界模型还可以参与失败监测、恢复目标选择及未来表示与训练迁移，文末专题将展开这些用途。

---

## Part III：Executor

## 16. Executor 在 Long-Horizon 中的位置反而更明确了

Planner 决定：

$$
z_t=\text{“pick up the cup”}
$$

Executor 负责：

$$
(o_t,z_t)\rightarrow A_t
$$

其中：

$$
A_t=[a_t,\ldots,a_{t+H-1}]
$$

这一部分就是[前面 VLA 笔记已经展开过的](https://letsgetai.github.io/posts/vla-action-modeling-generalist-policy/)：

- [Behavior Cloning](https://publications.ri.cmu.edu/alvinn-an-autonomous-land-vehicle-in-a-neural-network)；
- [Action Chunking](https://arxiv.org/abs/2304.13705)；
- Diffusion Policy；
- Flow Matching；
- VLA Action Expert。

所以在这篇笔记里不需要重新详细展开。

Long-Horizon 只是把 VLA 的位置变得更清楚：

$$
\boxed{
\text{VLA = Local Skill Executor}
}
$$

---

## 17. Planner → Executor 的 Interface 本身也是一个问题

虽然 Executor 不需要重讲，但 Planner 如何向 Executor 表达 intention 很重要。

### 17.1 Language Subgoal

最常见：

$$
z_t=\text{“pick up the cup”}
$$

优点是语义清楚。

缺点是 spatial grounding 可能不够。

### 17.2 Visual Trace

LoHo-Manip 加入：

$$
\gamma_t=[(x_1,y_1),\ldots,(x_k,y_k)]
$$

告诉 Executor：

> 大概沿图像哪个方向运动。

于是 interface 变成：

$$
\boxed{
\text{Language}
+
\text{Visual Trace}
}
$$

也就是：

> 做什么 + 大概往哪里做。

### 17.3 Visual Subgoal

[$\pi_{0.7}$](https://arxiv.org/abs/2604.15483) 等工作进一步使用：

$$
\hat o_{\text{subgoal}}
$$

表示：

> 当前 subtask 完成以后，世界应该长什么样。

于是 Executor 可以解决：

$$
(o_{\text{current}},o_{\text{subgoal}})
\rightarrow
A
$$

这有一点 inverse dynamics 的味道。

所以 Planner → Executor interface 目前也没有统一：

$$
\boxed{
\text{Language}
\quad vs\quad
\text{Visual Trace}
\quad vs\quad
\text{Visual Subgoal}
}
$$

---

## Part IV：Orchestration / Harness

## 18. 为什么单纯 Planner + VLA 还不够？

一个 naive hierarchy：

$$
\text{Planner}
\rightarrow
z_t
\rightarrow
\text{VLA}
$$

仍然有一堆 execution-level 问题：

- 当前 skill 应该执行多久？
- 什么时候把控制权还给 Planner？
- 当前动作真的合理吗？
- 当前 subgoal 真的完成了吗？
- 失败以后是 retry 还是 replan？

这些问题本身不完全属于 Planning。

它们更像：

$$
\boxed{
\text{Orchestration / Harness}
}
$$

---

## 19. Termination：什么时候结束当前 Skill？

如果 Planner 给：

$$
z_t=\text{pick cup}
$$

VLA 到底执行多久？

可能有几种方案。

### Fixed Duration

$$
T=8s
$$

时间到了就切换。

### Predicted Duration

Planner 事先预测：

> 这个 skill 大概执行 6 秒。

### Success-Based Termination

真正检测：

$$
\boxed{
\text{subgoal 是否完成}
}
$$

[Hi-VLA systematic study](https://arxiv.org/abs/2606.10267) 在长时程与推理任务上的评估支持：

$$
\boxed{
\text{success-based switching}
}
$$

比纯 time-driven 更可靠；这组实验中的成功检测使用了仿真器特权状态。

PALM 则用：

$$
p_t>\phi
$$

把 progress 和 termination 连起来。

---

## 20. Verification：模型提出的动作到底该不该执行？

普通 VLA：

$$
o_t\rightarrow a_t\rightarrow\text{execute}
$$

HELM 在中间加入：

$$
\boxed{
P(\text{failure}|o_t,a_t,g_t,M_t)
}
$$

也就是说：

> 当前这个 proposed action，在当前任务上下文里会不会出错？

这里很重要的一点是：

$$
\boxed{
\text{Action 是否正确，取决于 Task History}
}
$$

例如：

> put mug in cabinet

如果 mug 已经放进去过，这个动作即使几何上可行，在 task-level 也是错误的。

所以 Verifier 实际是在利用：

$$
\text{Current State}
+
\text{Memory}
+
\text{Task}
$$

判断：

$$
\text{Action Validity}
$$

---

## 21. Recovery：发现错了以后怎么办？

知道：

$$
\text{failure=True}
$$

还不够。

系统还要选择：

$$
\boxed{
\text{retry}
,\quad
\text{rollback}
,\quad
\text{forward recovery}
,\quad
\text{return to planner}
}
$$

### 21.1 Retry

偶然 grasp 失败：

$$
\text{retry same skill}
$$

### 21.2 Rollback

HELM 的仿真实验中可以回到最近成功 checkpoint：

$$
s'_t\rightarrow s_{\text{checkpoint}}
$$

然后重新执行当前 subgoal。

### 21.3 Forward Recovery

物理环境很多时候无法真的 rollback。

例如杯子已经掉到了新的位置。

这时：

$$
s'_t
\rightarrow
\text{new recovery behavior}
$$

也就是从当前 corrupted state 往前恢复。HELM-Fwd 是 HELM 针对无法回滚的情形设计的前向恢复变体。

### 21.4 Return to Planner

如果局部 recovery 已经解决不了：

$$
\boxed{
\text{Harness}
\rightarrow
\text{Planner}
}
$$

让高层重新规划未来任务。

所以 Orchestrator 最核心的问题其实是：

$$
\boxed{
\text{什么时候局部处理，
什么时候必须重新高层决策？}
}
$$

---

## 22. 四个模块之间真正困难的是 Interface

这些论文之所以高度耦合，根本原因是：

$$
\boxed{
\text{Long-Horizon 的难点不只在模块本身，
更在模块之间的信息接口。}
}
$$

### Task State → Planner

$$
b_t\rightarrow z_t
$$

问题：

> 什么样的 task state 才足够支持高层决策？

MemoryVLA、PALM、Goal2Skill、LoHo-Manip、$\tau_0$ 都给了不同答案。

### Planner → Executor

$$
z_t\rightarrow A_t
$$

问题：

> 高层 intention 应该用什么 representation 传给低层？

可能是：

$$
\text{language}
$$

也可能是：

$$
\text{visual trace}
$$

或者：

$$
\text{visual subgoal}
$$

### Executor → Task State

执行以后：

$$
A_t\rightarrow o_{t+1}
$$

然后需要得到：

$$
\boxed{
\text{success / progress / failure / new task state}
}
$$

PALM、HELM、Goal2Skill 都在研究这个接口。

### Task State → World Model → Planner

如果 Planner 需要 look-ahead：

$$
(b_t,z_i)
\rightarrow
\hat b_{t+1}^{(i)}
$$

然后：

$$
V(\hat b_{t+1}^{(i)})
$$

这是 $\tau_0$-VLA 等工作开始研究的接口。

---

## 23. 两种闭环

到目前为止，Long-Horizon 系统里其实出现了两类不同的闭环。

### 23.1 Reality Loop

机器人真的执行：

$$
\boxed{
\text{Plan}
\rightarrow
\text{Act}
\rightarrow
\text{Observe}
\rightarrow
\text{Update State}
\rightarrow
\text{Recover / Replan}
}
$$

MemoryVLA、PALM、HELM、Goal2Skill、LoHo-Manip 主要都在不同程度上补这个 loop。

它的特点是：

$$
\boxed{
\text{先进入真实世界，再根据真实反馈修正}
}
$$

### 23.2 Imagination Loop

World Model 加入以后：

$$
\boxed{
\text{Propose}
\rightarrow
\text{Imagine}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Choose}
}
$$

这发生在真正执行之前。

也就是：

$$
\boxed{
\text{Counterfactual Planning}
}
$$

这时候 Agent 不只是：

> 做错以后改。

而开始：

> 做之前先比较几个未来。

这才比较接近我们传统意义上的 planning。

---

## 24. Paper × Module 对应关系

这些论文大多跨多个模块，因此更适合用“主贡献 / 次贡献”的方式理解。

| Paper | Task State | Planner | Executor | Orchestrator / Harness |
| --- | --- | --- | --- | --- |
| **[Hi-VLA Systematic Study](https://arxiv.org/abs/2606.10267)** | 次：observation / memory | 次：VLM planner | 次：VLA controller | **主：termination / orchestration** |
| **[MemoryVLA](https://arxiv.org/abs/2508.19236)** | **主：perceptual + cognitive memory** | | 次：memory-conditioned action | |
| **[PALM](https://arxiv.org/abs/2601.07060)** | **主：progress + affordance** | | 次：diffusion action | **次：progress-based termination** |
| **[HELM](https://arxiv.org/abs/2604.18791)** | 次：episodic memory | | 次：base VLA | **主：verification + recovery** |
| **[Goal2Skill](https://arxiv.org/abs/2604.13942)** | **主：structured memory + pre/post state** | **主：reflection / replanning** | 次：skill executor | **主：verification / retry / adjust** |
| **[LoHo-Manip](https://arxiv.org/abs/2604.21924)** | 次：Done + Remaining | **主：receding-horizon replanning** | 次：visual-trace conditioned VLA | 次：implicit recovery |
| **[$\tau_0$-VLA](https://arxiv.org/abs/2608.16885)** | 次：memory correction | **主：world model + value + search** | 次：flow-based VLA | 次 |

这里最重要的不是表格本身，而是承认：

$$
\boxed{
\text{Paper}\neq\text{Module}
}
$$

一篇系统论文完全可能同时研究：

$$
\text{Task State}
+
\text{Planner}
+
\text{Verification}
$$

正文应该围绕模块写，论文只是不同模块上的案例。

---

## 25. Paper Cards

下面只保留每篇论文最值得记住的角色，避免正文变成论文流水账。

### [Hi-VLA Systematic Study](https://arxiv.org/abs/2606.10267)

#### 核心问题

$$
\boxed{
\text{Planner + VLA 到底应该怎么 orchestration？}
}
$$

#### 主要涉及

- Planner capability；
- low-level VLA steerability；
- observation representation；
- memory；
- termination / switching。

#### 最重要的认识

$$
\boxed{
\text{仅仅加一个 VLM Planner 不够，
Planner 和 Executor 之间怎么闭环本身非常重要。}
}
$$

---

### [MemoryVLA](https://arxiv.org/abs/2508.19236)

#### 核心问题

$$
\boxed{
\text{当前 observation 不足时，历史怎么进入 policy？}
}
$$

#### 主要涉及

- perceptual memory；
- cognitive memory；
- retrieval；
- gated fusion；
- consolidation。

#### 最重要的认识

$$
\boxed{
\text{Memory 不是简单增加 context length，
而是构造 task-relevant history representation。}
}
$$

---

### [PALM](https://arxiv.org/abs/2601.07060)

#### 核心问题

$$
\boxed{
\text{当前 subtask 做到哪里了？}
}
$$

#### 主要涉及

- affordance foresight；
- continuous progress；
- progress-conditioned action；
- progress-based switching。

#### 最重要的认识

$$
\boxed{
\text{Task State 不只需要 history，
还需要 phase / progress。}
}
$$

---

### [HELM](https://arxiv.org/abs/2604.18791)

#### 核心问题

$$
\boxed{
\text{VLA 提出的动作可能错，怎么提前发现并恢复？}
}
$$

#### 主要涉及

- episodic memory；
- state/action verifier；
- rollback；
- forward recovery。

#### 最重要的认识

$$
\boxed{
\text{Action 是否合理依赖 task history；
发现错误以后还必须有 recovery mechanism。}
}
$$

---

### [Goal2Skill](https://arxiv.org/abs/2604.13942)

#### 核心问题

$$
\boxed{
\text{执行反馈怎么真正修改 future plan？}
}
$$

#### 主要涉及

- structured memory；
- precondition；
- postcondition；
- verification；
- reflection；
- retry / adjust / replan。

#### 最重要的认识

$$
\boxed{
\text{反馈不只是用来 retry，
而是可以改变整个后续计划。}
}
$$

---

### [LoHo-Manip](https://arxiv.org/abs/2604.21924)

#### 核心问题

$$
\boxed{
\text{为什么要死守第一次生成的 Todo List？}
}
$$

#### 主要涉及

- Done / Remaining；
- receding-horizon task planning；
- visual trace；
- implicit recovery。

#### 最重要的认识

$$
\boxed{
\text{每次根据最新状态重新预测剩余任务，
可以把 task planning 做成闭环。}
}
$$

---

### [$\tau_0$-VLA](https://arxiv.org/abs/2608.16885)

#### 核心问题

$$
\boxed{
\text{为什么一定要执行以后才知道 subgoal 选错？
能不能先预测不同未来再选择？}
}
$$

#### 主要涉及

- memory correction；
- candidate subgoal generation；
- world model；
- value model；
- beam search；
- test-time computation。

#### 最重要的认识

$$
\boxed{
\text{Planner 开始从直接猜下一步，
转向“预测未来 + 评价 + 搜索”。}
}
$$

这时 World Model 真正重新进入 planning loop。

---

## 26. 哪些更像真正的长期问题，哪些更像当前 Harness？

当前这些模块里，有一部分问题比较基础：

$$
\boxed{
\text{Partial Observability / Memory}
}
$$

$$
\boxed{
\text{Task State}
}
$$

$$
\boxed{
\text{Progress}
}
$$

$$
\boxed{
\text{Temporal Abstraction}
}
$$

$$
\boxed{
\text{Uncertainty}
}
$$

$$
\boxed{
\text{Credit / Value}
}
$$

$$
\boxed{
\text{Counterfactual Future}
}
$$

这些即使 base VLA 继续变强，也不会简单消失。

它们可能被更大的模型隐式吸收，但问题本身仍然存在。

另一方面：

- 某个特定 failure classifier；
- 某套 rollback rule；
- 某种 prompt state machine；
- 某个 handcrafted success detector；

更可能是：

$$
\boxed{
\text{当前 Base VLA 不够可靠时的阶段性 Harness}
}
$$

因此当前很多系统都有双重性质：

$$
\boxed{
\text{Harness} =
\text{真正必要的 Agent Mechanism}
+
\text{对当前 Base Model 的能力补丁}
}
$$

现在这两者还没有完全分开。

---

## 27. 暂时的理解

如果把这一轮 Long-Horizon 阅读压成一句话：

普通 VLA 主要解决：

$$
\boxed{
\text{“当前该怎么动？”}
}
$$

而 Long-Horizon 系统开始补：

$$
\boxed{
\text{“我现在到底是什么任务状态？”}
}
$$

$$
\boxed{
\text{“接下来应该做什么？”}
}
$$

$$
\boxed{
\text{“当前执行是不是仍然正确？”}
}
$$

$$
\boxed{
\text{“如果不正确，怎么恢复或者改计划？”}
}
$$

以及更进一步：

$$
\boxed{
\text{“在真正行动前，能不能先比较不同未来？”}
}
$$

所以整个体系逐渐从：

$$
(o_t,l)\rightarrow A_t
$$

扩展成：

$$
\boxed{
\text{Observation / History}
\rightarrow
\text{Task State}
\rightarrow
\text{Planner}
\rightarrow
\text{VLA Executor}
\rightarrow
\text{Verification / Recovery}
\rightarrow
\text{New Task State}
}
$$

如果再加 World Model：

$$
\boxed{
\text{Task State}
\rightarrow
\text{Candidate Futures}
\rightarrow
\text{Value / Search}
\rightarrow
\text{Decision}
}
$$

这时候：

$$
\boxed{
\text{VLA}
+
\text{Memory}
+
\text{Planning}
+
\text{World Model}
}
$$

才真正开始构成一个完整的 Long-Horizon embodied agent。

---

## 补充专题：World Model 在 Long-Horizon 系统中的几种位置

在 Task State、Planner、Executor 与 Harness 的四模块框架中，世界模型的位置由预测用途决定。预测可以用于搜索动作、选择子目标、监测失败或选择恢复目标；未来表示的粒度与预测能力向控制的迁移，则是贯穿这些用途的设计选择。理解其作用，需要看清输入是什么、如何处理预测，以及输出改变了哪项决策。

### 动作级搜索：PILOT

[PILOT](https://arxiv.org/abs/2610.14335) 将视觉子目标与动作搜索结合起来：视频模型生成视觉子目标，策略提供动作序列初始化搜索，动作条件世界模型对候选序列进行 rollout，预测执行后的变化，用于评价候选。策略先验不以视觉子目标为输入，两者分别提供动作搜索的起点与视觉目标。

搜索采用 CEM，即反复采样动作序列、评分、保留较好的候选，再更新采样分布。执行采用 MPC：每轮只执行选中序列的一小段，随后重新观察、重新规划。世界模型参与候选动作的比较，实际观测持续修正后续选择。

论文在 4 个 UR5e 抓取放置任务上，每种方法每任务测试 20 次，共 80 次：PILOT 成功 55 次，成功率为 68.8%；同一微调后的 $\pi_0$ 直接执行成功 11 次，为 13.8%。在 Push-T 上，策略先验还减少了达到相同规划 cost 所需的 CEM 迭代数。

### 预测式失败监测：Foresight

[Foresight](https://arxiv.org/abs/2606.23085) 将当前观察和策略拟执行的下一段动作输入世界模型，再将预测的内部特征（latent）送入因果 Transformer，产生逐步失败分数，并校准判定阈值。输出可作为系统介入的信号。

训练采用整条任务最终成败的弱监督，无须逐帧标注失败起点。论文在 LIBERO-Long、ManiSkill-Long 四个任务、BEHAVIOR-1K 所选四个任务，以及 ReactorX200 三个任务和 Franka 一个任务上评测失败检测。它承担 Verifier 用途，评测指标衡量的是失败检测能力。

### 恢复目标选择：B2FF

[B2FF](https://arxiv.org/abs/2606.09258) 在执行前，从干净的初始观察生成未来图像里程碑库。执行偏离后，系统结合当前状态与历史，按恢复潜力选择图像目标，引导冻结的 VLA 从当前真实状态继续行动。

目标可以对应较早、当前或较后的任务进度：恢复可能需要回到先前阶段，也可能接向后续阶段。预测因此改变恢复时朝哪个目标行动，候选目标的可达性没有形式保证。

论文在注入故障的 LIBERO 四套件上报告平均成功率：UD-VLA 为 56.3%，已知扰动注入时机触发恢复的 B2FF 为 74.0%，在线触发版为 64.5%。该设置包含三类可恢复扰动，排除了不可逆故障；74.0% 衡量已知触发时机下的恢复效果，64.5% 对应在线触发与恢复结合后的表现。

### 未来表示与训练迁移：StructVLA

[StructVLA](https://arxiv.org/abs/2603.12553) 从轨迹中抽取稀疏、具有运动学意义的视觉帧，包括夹爪开合转换，以及运动阶段转换处的低速稳定状态，再由离线 VLM 去除冗余。这样的未来表示保留操作中的关键结构，使预测目标集中于有意义的阶段变化。

训练分为两个阶段：第一阶段学习结构帧预测；第二阶段将同一模型转为动作 token 预测，把视觉规划能力迁移到控制。世界模型在这里影响的是模型学习什么未来，以及这种能力如何进入动作预测。

### 这些预测改变了什么？

| 系统 | 预测对象 | 参与位置 | 改变的选择或学习 |
| --- | --- | --- | --- |
| PILOT | 候选动作条件下的未来 | Planner 的动作搜索／视觉子目标接口 | 选择当前执行的动作短前缀 |
| [$\tau_0$-VLA](https://arxiv.org/abs/2608.16885) | 候选语义子目标对应的未来 | Planner 的子目标搜索 | 结合 value 与 beam search 选择子目标 |
| Foresight | 拟执行动作对应的内部特征 | Harness 中的失败监测／Verifier | 输出可作为介入信号的失败分数 |
| B2FF | 未来图像里程碑 | 恢复目标选择 | 从当前真实状态选择恢复锚点 |
| StructVLA | 稀疏结构帧 | 未来表示与两阶段训练路线 | 将结构帧预测能力迁移到动作预测 |

这些用途可以跨越模块边界。Foresight 评测失败检测，输出可作为介入信号；其他系统则通过动作搜索、子目标选择、恢复或训练迁移影响控制。判断世界模型的作用，应先明确预测改变哪项决策或学习过程，再检验是否改善任务表现。

---

## 参考论文与入口

1. **[What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents](https://arxiv.org/abs/2606.10267)**

2. **[MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](https://arxiv.org/abs/2508.19236)**

3. **[PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2601.07060)**

4. **[HELM: Harness-Enhanced Long-horizon Memory for Vision-Language-Action Manipulation](https://arxiv.org/abs/2604.18791)**

5. **[Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection](https://arxiv.org/abs/2604.13942)**

6. **[Long-Horizon Manipulation via Trace-Conditioned VLA Planning](https://arxiv.org/abs/2604.21924)**（LoHo-Manip）

7. **[$\tau_0$-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation](https://arxiv.org/abs/2608.16885)**

8. **[PILOT: Guiding Long-Horizon World-Model Planning for Robot Manipulation with Visual Subgoals and Policy Priors](https://arxiv.org/abs/2610.14335)**

9. **[Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents](https://arxiv.org/abs/2606.23085)**

10. **[Back to the Familiar Future: Failure Recovery for VLA Policies via Pre-Imagined Milestone Selection](https://arxiv.org/abs/2606.09258)**

11. **[Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation](https://arxiv.org/abs/2603.12553)**
