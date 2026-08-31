---
title: "OPD 本质分析：从在线蒸馏公式、TRL 最小复现到 GSM8K 验证"
date: 2026-08-31T00:30:00+08:00
draft: false
categories: ["RL"]
---

> 一句话版本：OPD（on-policy distillation，在线策略蒸馏）用「学生自己采样的数据」加「教师逐 token 的密集信号」，同时绕开离线蒸馏的分布错位和纯 RL 的信号稀疏。我们基于 TRL 做了一次最小复现，得到三个可量化的结论：数据筛选把 GSM8K 准确率从 11.52% 抬到 43.06%；换更强的教师做离线蒸馏几乎不动（43.06% → 43.21%）；只有「强教师 + 完整生成预算」的组合下，on-policy 的价值才兑现（54% > 42% > 39%）。

本文面向已经接触过 RL 和蒸馏、想理解 OPD 来龙去脉并动手复现的读者。主线：OPD 本质是什么 → TRL 怎么实现 → 我们的实验证明了什么 → 结论能否被外部证据验证。

## 开篇：OPD 的本质

### 为什么现在都在谈 OPD

OPD 最近在大模型后训练里升温：Qwen3、GLM-5、DeepSeek 的后训练实践都涉及「在线蒸馏」思路，社区里也有不错的入门视频：[从 RL 到 OPD：Qwen3、GLM-5、DeepSeek 都在关注什么？](https://www.bilibili.com/video/BV1bNGz6xEQf/)。核心主张一句话：保留 RL 里「学生自己探索」的 on-policy 采样，同时引入教师对每个 token 的密集反馈，缓解传统蒸馏的分布错位，也降低纯 RL 里信号稀疏、成本高、训练不稳定的压力。

### 三范式对比

| 维度 | 离线蒸馏 offline KD | OPD 在线策略蒸馏 | RL（PPO/GRPO） |
| --- | --- | --- | --- |
| 采样分布 | 教师或固定数据（off-policy） | 学生当前策略（on-policy） | 学生当前策略（on-policy） |
| 学习信号 | 教师逐 token 分布 | 教师逐 token 分布 | 奖励（通常稀疏） |
| 是否要奖励模型 | 否 | 否 | 是 |
| 分布错位 | 有 | 无 | 无 |
| 信号密度 | 高 | 高 | 低 |
| 不稳定风险 | 低 | 中（依赖采样质量） | 高 |

本质是两个维度的交集：「谁的数据」决定分布是否错位——离线蒸馏用教师生成的数据，学生学的是「自己不太会产生的输入」上的分布，且固定数据不会随学生进步自适应；「谁的信号」决定密度——教师逐 token 的 KL 目标密集，RL 的奖励通常只出现在序列末尾。OPD 取交集：学生数据 × 教师信号。

### 公式：广义 JSD

OPD 的损失函数来自 GKD（[Generalized Knowledge Distillation for Auto-regressive Language Models](https://arxiv.org/abs/2306.13649)，Agarwal et al., ICLR 2024），用 β 把两种 KL 插值：

$$L = \beta \cdot KL(P \| M_\beta) + (1-\beta) \cdot KL(Q_\theta \| M_\beta), \quad M_\beta = \beta P + (1-\beta) Q_\theta$$

其中 $P$ 是教师分布、$Q_\theta$ 是学生分布、$M_\beta$ 是两者的几何混合。β 的两个端点对应两种 KL：

- β → 0：forward KL $KL(P\|Q_\theta)$，mass-covering，学生尽量覆盖教师的分布，鼓励多样性；
- β → 1：reverse KL $KL(Q_\theta\|P)$，mode-seeking，学生收紧到教师的高概率区域，分布更锐利。

TRL 的实现和论文公式一致：对两个分布取 logsumexp 混合，再按 β 加权求和（β=0/1 走 forward/reverse KL 特判分支，[DistillationTrainer 文档](https://huggingface.co/docs/trl/en/distillation_trainer)）。注意方向：蒸馏的 KL 是「学生向教师靠拢」，RLHF 里的 KL 正则方向相反——「策略不偏离参考模型」，这两个方向很容易混。

### 需要优化的点：四个旋钮

1. **β 调度**：固定 0.5 还是退火（0 → 1，先覆盖后收紧）。TRL 没有内置退火，一个 callback 就能加（见第一节）。
2. **采样完整性**（max_completion_length）：学生生成被截断时，损失只算在截断前缀上。对 GSM8K 这种长推理任务，这是生死参数，第四节有量化证据。
3. **教师质量**：教师弱时逐 token 目标本身就是错的，学生会收敛到高置信的错误模式。
4. **数据筛选**：蒸馏数据里混入「答案不可提取/错误」的样本，会直接拖垮效果（第四节实验 1）。

### 需要新增的资源

相对 SFT，OPD 每步要：学生采样一个完整 rollout（生成整条推理链）、学生和教师各一次前向（教师冻结但显存双份）。教师越大推理成本越高——我们强教师跑 150 步花了 105 分钟，弱教师 250 步不到 30 分钟。纯 RL 还需要奖励模型/评测器，OPD 不需要。

### 训练时大概是什么样子

一轮循环四步：取 prompt 批 → 学生按当前策略采样完整回答（temperature/top_p）→ 学生与教师分别前向，取 completion 位置逐 token logits → 算逐 token JSD 损失并只更新学生（LoRA）梯度，教师冻结。

训练曲线看起来和普通训练没有区别：loss 平滑下降、JSD 缓慢收敛。真正有信息量的是行为指标——学生熵、completion 触顶率、终止率。我们会在第四节展示：弱教师那组熵从 0.276 塌缩到 0.248，loss 却还在降，光看 loss 完全发现不了问题。

## 第一节：代码实践——TRL 的 DistillationTrainer

### (a) 怎么用

TRL 从 0.x 时代的 GKDTrainer 演进到现在的 DistillationTrainer。我们环境是 trl 1.10.0 + transformers 5.15.1，核心用法：

```python
from trl import DistillationConfig, DistillationTrainer

cfg = DistillationConfig(
    output_dir="outputs/opd",
    max_steps=250,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    bf16=True,
    beta=0.5,                    # JSD 插值：0=forward KL，1=reverse KL
    max_completion_length=384,   # 学生采样上限（关键旋钮）
    temperature=0.7,
    top_p=0.9,
    disable_dropout=True,
)
trainer = DistillationTrainer(
    model="Qwen/Qwen3-0.6B",              # 学生（可挂 peft LoRA）
    teacher_model="Qwen/Qwen3-4B-Instruct-2507",
    args=cfg,
    train_dataset=ds,
    peft_config=lora,
)
trainer.train()
```

关键参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| beta | 1.0（合法域 [0,1]） | JSD 插值：0=forward KL，1=reverse KL，中间为广义 JSD |
| max_completion_length | 512 | 学生采样上限，决定损失覆盖多长 |
| temperature / top_p | 1.0 / 1.0 | 学生采样分布，越接近 1 探索越多 |
| disable_dropout | False | 训练时关 dropout，稳定采样 |

### (b) 核心部分：它和之前的算法差在哪

三种方法的数据流对比：

```text
SFT:   固定数据 -> 学生前向 -> CE 损失
PPO:   学生采样 -> 奖励打分 -> 稀疏 advantage -> 策略梯度
OPD:   学生采样 -> 教师逐 token 打分 -> 密集 JSD -> 学生梯度
```

三个主要区别：

1. **数据谁产生**：SFT 用固定数据；OPD 的学生每步自己采样（on-policy），输入分布随学生当前能力自适应，这是解决分布错位的机制。
2. **信号是什么**：OPD 是教师对每个 token 的分布目标，密度和 SFT 一样高；PPO/GRPO 的奖励通常只在序列末尾，难任务上容易出现 advantage 全 0 或全 1——我们主线里叫「二元陷阱」。
3. **谁的梯度**：只有学生更新，教师冻结但占显存。TRL 用 chunked 前向（核心函数 `_chunked_divergence_loss` 和 `_chunk`）逐块算 JSD，把峰值显存从 2×batch×seq×vocab 降到 chunk 级别，否则双模型前向根本放不下。

源码里还有一道硬校验：学生和教师词表必须一致（报错是 The student model has vocab_size X but the teacher model has vocab_size Y）。这直接决定模型选型——Qwen2.5 系列里 0.5B（151936）和 7B（152064）词表不一致，蒸馏直接报错；我们因此换成同词表族的 Qwen3-0.6B + Qwen3-4B/8B。

一个 TRL 1.10 的 API 断层值得记录：旧 GKDTrainer 的 lmbda（on-policy 数据比例）参数在 DistillationTrainer 里没有了，只剩 beta 一个插值旋钮；GKDConfig 也不存在（[旧版 GKD Trainer 文档](https://huggingface.co/docs/trl/en/gkd_trainer)是历史 API）。如果要把这套东西移植到别的框架（比如 veRL），这两个旋钮要自己接。

### β 退火：6 行 callback

TRL 没有内置退火，但 beta 只是 trainer 上的一个属性，写个 callback 即可：

```python
class BetaAnnealCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.total = max(1, state.max_steps)
    def on_step_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is not None:
            frac = min(1.0, state.global_step / self.total)
            trainer.beta = 0.0 + (1.0 - 0.0) * frac  # 0 -> 1
```

我们在实验里用固定 β=0.5 和这个退火各跑了一组。

## 第二节：最小实现

### (a) TRL 的最小实现

最小可跑版本（实验脚本 train_opd.py 的骨架）：

```python
import json, random, re
from datasets import load_dataset
from peft import LoraConfig
from trl import DistillationConfig, DistillationTrainer

# 1) 数据：只需要 prompt，学生自己采 completion，不需要教师预生成答案
ds = load_dataset("json", data_files="train_prompts.jsonl", split="train")

# 2) LoRA 学生
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  bias="none", task_type="CAUSAL_LM")

# 3) 训练
cfg = DistillationConfig(
    output_dir="outputs/opd", max_steps=250,
    per_device_train_batch_size=8, learning_rate=1e-4,
    bf16=True, beta=0.5, max_completion_length=384,
    temperature=0.7, top_p=0.9, disable_dropout=True,
)
trainer = DistillationTrainer(
    model="Qwen/Qwen3-0.6B", teacher_model="Qwen/Qwen3-4B-Instruct-2507",
    args=cfg, train_dataset=ds, peft_config=lora,
)
trainer.train()
trainer.save_model("outputs/opd/final")
```

加上第一节的 β 退火 callback、再挂一个 SwanLab callback 记曲线，就是我们实验的实际代码。OPD 和离线蒸馏在数据准备上最大的差别：离线蒸馏要先拿教师把整份训练集生成一遍（我们弱教师、强教师各生成 1000 条），OPD 完全不需要预生成，数据侧只有 prompt。

### (b) 外部库

| 库 | 作用 |
| --- | --- |
| trl 1.10.0 | DistillationTrainer 本体 |
| transformers 5.15.1 | 模型、分词器与训练基础设施 |
| peft | LoRA 学生 |
| datasets | 数据加载（GSM8K、JSONL） |
| torch | 后端 |
| swanlab（可选） | 训练曲线记录（[SwanLab](https://github.com/SwanHubX/SwanLab) offline 模式，之后 sync 回传） |

### (c) 整体逻辑与三个细节

```text
train_prompts.jsonl（1000 条 GSM8K question）
  -> 每步：学生采样 completion（max_completion_length=384）
  -> 学生 + 教师前向，completion 位置逐 token JSD
  -> 更新 LoRA 学生
  -> 每 10 步记录 loss / 熵 / completion 统计
```

三个容易踩的细节：

1. **left padding**：生成端必须 left padding，否则右 padding 会污染采样起点（很多 RL 框架踩过同一个坑）。
2. **max_completion_length 是采样完整性参数**：GSM8K 推理链长，第一版用 384，completion 100% 触顶、损失只算在截断前缀上——这是弱教师组崩塌的直接诱因之一。
3. **筛选标准 = 评测标准**：筛选蒸馏数据用的 extract_answer 和评测是同一份逻辑（同时处理 #### 和 \boxed{}），保证「数据契约」和「验收标准」一致，否则筛选实验的结论不可信。

## 第三节：实验与效果

### (a) 实验设计

| 配置 | 值 |
| --- | --- |
| 学生 | Qwen3-0.6B（LoRA r16/alpha32/dropout0.05，lr 1e-4，batch 8，bf16，seed 42） |
| 弱教师 | Qwen3-4B-Instruct-2507（greedy 0-shot 43%，200 子集测得） |
| 强教师 | Qwen3-8B（greedy 0-shot 66.5%，200 子集、2048 上限 + 双格式提取测得） |
| 数据 | GSM8K train 1000（seed 42 打乱），test 全量 1319 |
| 评测 | greedy、0-shot、双格式提取（#### / \boxed{}），max_new 2048；强教师对比用 4096 |

九个臂，一次只改一个变量：

1. sft-baseline：SFT 金标答案（250 步）
2. offline-kd：SFT 在弱教师生成的原始 1000 条上（不筛选）
3. kd-hint：offline-kd 模型 + 评测时加「请以 #### 结尾」的格式提示
4. kd-random：弱教师数据随机抽 256 条
5. kd-filtered：弱教师数据筛出 256 条「答案正确且可提取」
6. opd：弱教师 on-policy，β=0.5，384 上限
7. opd-anneal：弱教师 on-policy，β 退火 0→1
8. kd8b-v2：强教师筛 699 条（69.9% 正确率）离线蒸馏
9. opd8b：强教师 on-policy，β=0.5，1024 上限，150 步

### (b) 实验效果

弱教师全量 1319 评测：

| 臂 | 准确率 | 正确/总数 |
| --- | --- | --- |
| SFT 金标 | 39.12% | 516/1319 |
| offline-KD 原始 1000 条 | 11.52% | 152/1319 |
| offline-KD + 格式提示 | 24.94% | 329/1319 |
| offline-KD 随机 256 条 | 3.33% | 10/300（300 子集） |
| offline-KD 筛选 256 条 | 43.06% | 568/1319 |
| OPD β=0.5 | 10.08% | 133/1319 |
| OPD β 退火 0→1 | 9.55% | 126/1319 |

强教师（100 子集、4096 上限）：

| 臂 | 准确率 | 口径 |
| --- | --- | --- |
| SFT 金标 | 39.12% | 全量参考 |
| offline-KD 筛选 699 条 | 42% | 100 子集 |
| OPD 150 步 | 54% | 100 子集 |

![GSM8K 各臂准确率对比](/images/opd/gsm8k-acc.png)

三层结论：

1. **数据质量是决定性变量**：同样 256 条数据，随机抽 3.33%，筛出正确样本后 43.06%——打平并略超 SFT 金标（39.12%），还超过了教师自己（43%）。学生学的不是「教师风格」，而是「正确且可提取」的答案格式。
2. **离线蒸馏有硬天花板**：换 66.5% 的强教师、筛 699 条，43.21%——几乎不动。offline 数据再干净也补不上「教师路径 ≠ 学生路径」的分布错位。
3. **on-policy 的价值要教师够强 + 生成够完整才兑现**：强教师 + 150 步 OPD 54% > offline-KD 42% > SFT 39%；弱教师下 OPD 反而崩到 10%，根因见下一节。

## 第四节：实验分析与验证

### (a) 溯源：提升到底来自哪里

**实验 1：数据筛选（+31.5pp）**。offline-KD 在原始 1000 条上只有 11.52%——弱教师 43% 的准确率意味着过半样本的答案是错的，模型在学错误答案。筛出 256 条正确样本后直接到 43.06%，反超教师 43% 的上限。再补一个对照：随机 256 条只有 3.33%，说明提升来自「正确样本」这个属性，不是样本量或随机性。kd-hint（24.94%）说明格式提示只能救一半——答案错了就是错了。

**实验 2：弱教师 OPD 为什么崩**。看训练曲线：

![OPD 训练曲线对比](/images/opd/opd-train-curves.png)

弱教师组（灰线）250 步：熵从 0.276 一路塌缩到 0.248，completion 100% 触顶 384（损失只算在截断前缀上），JSD loss 从 0.078 降到 0.053 还在降——loss 全程正常，行为指标先坏了。叠加教师本身只有 43% 准确率，逐 token 目标大量是错的，学生收敛到一个高置信的错误模式。我们做了 sanity check：让这个模型翻译句子、写诗，它输出大段中文思考后被截断，回答质量明显退化——不是完全坍塌，但行为已经坏了。

**实验 3：强教师为什么能行**。强教师组（黑线）150 步：熵从 0.232 升到 0.273（保持），1024 上限虽然也 100% 触顶（terminated 率为 0），但教师目标质量高，学生分布没有塌缩。代价是训练成本：150 步 105 分钟，弱教师 250 步不到 30 分钟。

**实验 4：评测上限也是口径陷阱**。强教师 OPD 用默认 2048 上限全量评测只有 0.08%（1/1319）——学生学到了更长的推理链，2048 截断后提取不到答案；同一模型在 4096 上限的 100 子集上是 54%。具体案例：强教师 8B 有一条生成在 1529 token 处被截断，答案提取为 None（金标 3）。测「变长」的模型，评测预算必须跟着涨。

### (b) 边界与口径

边界集中说明：子集口径（100/200/300）、单一数据集 GSM8K、LoRA 非全参、β 只试了 0.5 与退火、GPU 为共享环境。这些是后续要补的验证面，不改变上面三个结论的相对方向。

## 结尾：结论

回到开篇的问题——OPD 的 on-policy 机制到底有没有用：

1. **成立有两个前提**：教师足够强、学生采样足够完整。两个前提都不满足时（弱教师 + 384 截断），它比离线蒸馏更容易崩——错误目标和高置信塌缩是叠加的。
2. **数据筛选是所有方法里性价比最高的杠杆**：+31.5pp、零额外算力，本质是把「评测标准」前移到数据契约。
