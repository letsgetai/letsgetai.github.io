---
title: "Codex Goal 模式：什么时候用、怎么用"
date: 2026-08-26
draft: false
categories: ["Agents"]
---


## 一个把 p95 压到 200ms 的任务

假设你让 agent 把服务的 p95 延迟压到 200 毫秒以下（p95 指 95% 的请求延迟低于这个值，是性能优化常用的目标——意思是只允许最慢的 5% 请求超过它）。这件事没法一条命令做完：要先跑 profile 定位热点，改实现，再跑基准对比，确认没有回退；没达标就换一个方向重来。十几轮工具调用，每一轮都改一点东西。开头 agent 清楚方向，二十轮之后它可能已经忘了最初的目标，或者在一个局部优化里越钻越深，或者自己宣布「完成了」——但基准根本没达标。

这类任务有个名字：长程任务（long-horizon task），指需要很多轮连续动作、最后还要验证结果才算完成的任务。你的问题很具体：这类任务里 agent 为什么容易跑偏、停不下来？Codex 的 goal 模式能不能解决？什么时候用、怎么写？

暂定回答：goal 模式解决的是「完成判定与续跑机制」的缺失，而不是模型能力本身。它适合有明确终点但路径不确定的任务；使用时要把目标写成可验证的证据面，并配合 token 预算；官方推荐的完整流程是先 /plan 访谈澄清，再 /goal 建立目标。

## 长程任务为什么难：断崖不在单步，在「串起来」

先排除一个常见误解：单步能力已经不是瓶颈。OpenAI 官方跑过 25 小时、1300 万 token 的连续任务实验，模型可以长时间稳定工作；METR 的度量进一步指出，「50% 可靠完成」的任务时长大约每 7 个月翻一番，2025 年初已经到了小时级——瓶颈是把长动作序列串起来，而不是单步技能。

benchmark 把这种断崖画得更直白。在 SWE-bench Verified（500 个单 issue 任务）上 gpt-5.2 能做到约 72.8%；换成 SWE-EVO（48 个 release 级任务，平均涉及 21 个文件），同样的模型掉到 22.9%；SWE-bench Pro（1865 个企业级长任务）上 Top 模型约 23%；LongCLI-Bench 的 20 个长程 CLI 任务，SOTA 不到 20%，多数 agent 卡在 30% 进度之前。

| Benchmark | 形态 | 代表性数字 |
| --- | --- | --- |
| SWE-bench Verified | 500 个单 issue 任务，人工筛选 | mini-SWE-agent 65%；gpt-5.2 约 72.8% |
| SWE-bench Pro | 1865 个企业级长任务 | Top 模型约 23% |
| SWE-EVO | 48 个 release 级任务，平均涉及 21 个文件 | 最佳 25%；gpt-5.2 从 Verified 的 72.8% 掉到 22.9% |
| LongCLI-Bench | 20 个长程 CLI 任务 | SOTA 不到 20%，多数卡在 30% 进度之前 |

不同基准的口径和难度不一致，数字只说明数量级；同一模型从单 issue 到长程任务的落差（72.8% → 22.9%）是多个独立基准的一致信号——问题不在单步能力，而在能不能把长任务走完。

任务变长后，agent 会撞上三个具体困难。第一个是目标漂移：目标最初只存在于系统提示里，几十轮之后上下文里全是中间产物，最初的约束被稀释，agent 顺着最近的局部进展走，而不是最初的方向。第二个是完成判定缺失：没有外部判据时，「做完」是模型自己说的，它可能看到一条测试通过就宣布胜利，即使那只是它刚改的那条路径。第三个是上下文膨胀：每轮对话都追加进上下文，任务后期上下文被中间过程占满，模型既看不清目标，也看不清证据。

所以，解决长程任务的关键不是更强的单步模型，而是回答两个问题：目标放在哪里，谁来判断完成。看现有方案的谱系，能看出一条演进线：第一代把目标写进系统提示或记忆文件（AGENTS.md、auto memory），启动时加载进上下文，目标在不在、记得记不住、判定标不标准，全看模型自觉；第二代用外部状态文件（PLANS.md 一类的 living document），agent 自己维护进度、决策和证据，比提示词强，但维护本身仍是自觉行为；第三代是工具层的 goal——目标持久化在工具层，每轮自动注入，完成与否按预设证据面判定。

| 方案 | 目标放在哪 | 谁判断完成 | 局限 |
| --- | --- | --- | --- |
| 提示词 / 记忆文件（AGENTS.md、auto memory） | 启动时加载进上下文 | 模型自觉 | 会遗忘、会漂移，判定标准不稳 |
| 外部状态文件（PLANS.md） | agent 自维护的文档 | 用户 + 模型对照验收 | 维护靠自觉，仍需人工推进 |
| 工具层 goal | 工具层持久化、每轮注入 | 证据面判定（Codex）/ 独立裁判（Claude） | 需要用户写清证据面与预算 |

演进的是「目标放在哪里、谁判断完成」这两件事：goal 把判定权从模型自觉移到了工具层。

![长程任务方案谱系](/images/codex-goal-mode/long-horizon-approaches.png)

谱系图的演进方向是「目标放哪里、谁判断完成」：从模型自觉逐步移到机制层。

## goal 机制：把目标变成每轮注入的强制约束

goal 解决三个困难的方式，是把「目标」从上下文里的一次性文本，变成工具层持久化、每轮强制注入的状态。它由四个机制组成。

**持久目标状态。** 目标存在线程级状态里（thread-scoped persisted state）：存在服务端、跟随这个线程，而不是塞进每轮对话——它不是全局记忆，不会污染其他线程；也不是项目指令，不会每次启动自动加载。

**事件驱动续跑。** 一轮结束、线程空闲、没有排队输入时，工具层自动让 agent 继续下一轮，不用你每次手动说「继续」。为了防自旋，计划类的输出不触发续跑；某一轮没有任何工具调用时也抑制续跑——避免模型空转刷轮次、假装在干活。

**证据判定完成。** goal 只能由模型在证据支撑下标记完成，证据是文件、测试、日志、产物这类可审计的东西；系统随目标注入的审计要求（completion audit / blocked audit）会要求模型在标记完成前先核对证据，并在无路可走时停下来报告而不是硬撑。**预算兜底。** 创建时设定 token 预算，触顶后状态进入 budget-limited 并汇报；注意预算耗尽不等于完成——agent 会告诉你「预算用完、还差什么」，而不是假装做完了。

这四个机制合起来，回答的正是上一章那两个问题：目标放在工具层，完成由证据判定。这也说清了它和「把要求写进上下文」的本质区别：上下文里写「目标是 X」，模型每一轮都可能遗忘或漂移，而且没有任何机制检查它是否还朝着 X；goal 则把目标变成每轮请求里携带的约束块，连同预算、续跑策略、完成审计一起注入。agent 不是「记得」目标，而是「被注入」目标。

goal 的状态机很简单：active（进行中）→ paused（暂停）、complete（完成）、budget-limited（预算触顶）、cleared（被清除）。关键在权威边界：模型可以创建 goal，并且只在证据支撑时标记 complete；暂停、恢复、清除和预算转换都由用户或系统控制——goal 不是无人监管的自主性。

![goal 状态机与权威边界](/images/codex-goal-mode/goal-state-machine.png)

箭头标注谁触发转移：模型只负责创建与基于证据标记完成，其余转移都在用户/系统手里。

需要提醒：Codex 的 goal 和 Claude 的 goal 同名不同构——前者由主模型按证据面自审，后者由每轮独立的轻量裁判模型判定。差异和取舍放在附录 A。

## goal 的效果：研究怎么说

产品级 goal 没有公开的标准化 A/B 评测，但它的机制不是孤例。goal 的四个机制——持久目标状态、自动续跑、证据判定、预算——在学术上等价于「显式任务状态外置 + 独立验证」这一类机制，而这类机制在 2026 年的长程基准上有量化消融证据。

| 研究 | 机制 | 量化提升 |
| --- | --- | --- |
| [LongHorizon-Harness](https://arxiv.org/abs/2608.01964) | 任务状态外置 + 只读审计循环（Manage-Execute-Audit） | WeaveBench 51.8%→80.7%；Terminal-Bench 2.1 69.7%→77.2%；OSWorld 2.0 2.8%→8.3%；Claude Opus 4.7 子集 20.0%→34.3% |
| [PushBench](https://arxiv.org/abs/2605.23574) | 定量目标持久性（QGP）基准：状态跟踪检索 / backlog-tracking / 标准完成门控三类控制器对比 | 状态跟踪检索控制器 69–78% 成功并消除重复提交；backlog-tracking 工作单元控制器 25–50%；同一设置下标准/完成门控控制器 0 个实例完成 |
| [StructAgent](https://arxiv.org/abs/2607.11388) | verifier 背书的状态转移 | OSWorld-Verified：Qwen3.5-9B 27.0%→46.9%；Qwen3.5-27B 31.6%→62.2% |

三项研究做的其实是同一件事：把目标与进度从模型脑子里搬到机制层，让验证独立于生成——LongHorizon-Harness 外置任务状态并加只读审计循环，PushBench 证明目标持久本身决定成败，StructAgent 用独立验证器背书状态转移。goal 就是这件事的产品化：机制层持有目标与预算、每轮注入、按证据判定，而不是指望模型自觉。

生产侧的观察指向同一个痛点。Reza Rezvani 的 Claude /goal 生产实测里有一个 17 个服务文件的认证调用迁移任务：之前每轮手动「keep going」11 次，goal 之后自动跑到完成。这种 babysitting tax（看护税）——每轮都要人推一把——正是 goal 要消除的那部分成本。

最后补一个相邻发现。LongCLI-Bench 发现自我纠错提升有限，而 plan injection 加人工交互指导显著提升，支持「机制与人类协作比纯模型自驱更可靠」。goal 的定位正是把机制（状态、判定、预算）和人的介入（写证据面、审批）组合起来，而不是把任务完全扔给模型。

## 什么时候用、怎么写

先用三档判断。目标模糊，连「什么算完成」都要先调研——先 /plan：先做只读调研和访谈式澄清，把任务边界、验收标准问清楚，产出计划。计划产出后有两条路：直接确认执行，把已确定的方案走完——路径已定，通常一轮完成、比较快；或者关掉确认界面不执行，把计划固化成 goal——自动续跑，适合路径还没完全确定、需要探索的任务。目标明确、路径不确定——直接 /goal：压 p95、超参调优、多轮重构这类终点可验证、中间步骤未知的任务，正是 goal 的适用区。单行修改、单轮问答——两者都不用，goal 有创建和审计成本，杀鸡不用牛刀。

把计划固化成 goal，官方给出三种衔接方式，任选其一：a) 让 Codex 把计划转成带可度量成功标准的 goal 文本，再 /goal 启动（官方 long-running-work 推荐流程）；b) 把计划存成 PLAN.md，然后 /goal Implement PLAN.md，goal 直接指向计划文件（官方 follow-goals 迁移例子）；c) 跳过 plan，直接让 Codex 起草 goal 草稿（「Help me turn this into a strong /goal: 我想做……」），收紧后激活。为什么 plan 执行和 goal 执行不同：plan 解决方向不清——方案确定后，执行就是照做一遍；goal 解决执行不持续——路径未知时把目标绑在线程上自动续跑、按证据判定完成；plan 的确认执行是可选项，不与 goal 冲突。

怎么写，官方 cookbook 给出六要素：最终状态（Outcome）、证据面（Verification surface）、不回归项（Constraints）、可用边界（Boundaries）、每轮选动作策略（Iteration policy）、阻塞停下条件（Blocked stop condition）。套到 p95 的例子：

```text
/goal 把服务的 p95 延迟压到 200ms 以下，
verified by 基准脚本 bench.sh 输出中 p95 < 200ms，且测试套件全部通过；
while preserving 现有 API 语义和正确性；
Use 工作区、基准脚本和 git；
Between iterations 先跑基准确认当前基线，再选一个最大热点优化，跑完复测，没达标就换下一个热点；
If blocked 连续两轮 p95 无改善时停下，报告瓶颈分析、已尝试方案和剩余候选。
```

两个写法要点：证据面必须是可审计的东西（命令输出、测试结果、日志、产物），而不是「做得更好」这类主观词；迭代策略必须明确「每轮之后怎么选下一步」——这正是防止漂移的关键。配套方面，goal 通常配合 token 预算（防失控）、审批（关键动作前确认）、worktree（并行实验互不污染）；其中预算建议一定给，goal 解决「停不下来」靠的就是预算和阻塞条件这两把闸。

## 结论

回到开头的问题：goal 值不值得用？如果你经常跑「终点明确、路径未知」的多轮任务，值得——它把「目标放在哪、谁判断完成」从模型自觉变成工具约束，配合预算和证据面，能缓解跑偏和停不下来的问题；但它不提升单步能力，benchmark 断崖的另一半不是 goal 的职责。什么时候用：模糊任务先 /plan 再 /goal，明确任务直接 /goal，单轮任务都不用。怎么写：六要素，核心是证据面可审计、迭代策略明确。

最后集中说明一次本结论的边界：产品级 goal 没有公开的标准化 A/B 评测，但机制等价的研究给出了量化支持（见「goal 的效果」章）；具体机制以所用版本文档为准。

## 附录 A：Codex 与 Claude 的 goal，同名不同构

两者都把「每轮检查是否完成」做成了机制，但检查者不同：Codex 由主模型按证据面自审；Claude 把每轮结束后的判定外包给独立的轻量裁判模型（默认 Haiku），它读「条件+对话」返回 未达成/达成/不可能，并把理由注入下一轮。对照如下。

| 维度 | Codex goal | Claude /goal |
| --- | --- | --- |
| 本质 | 线程级持久化状态 | session 级「停止钩子（Stop hook）」包装，每轮结束后触发检查 |
| 判定者 | 主模型自审，按证据面审计 | 独立轻量裁判模型（默认 Haiku），读条件+对话 |
| 裁判能否取证 | 主模型审计时可调用工具取证 | 裁判不调用工具，条件必须写成「Claude 输出能证明」的形式 |
| 预算 | 显式 token 预算，触顶进 budget-limited | 无显式 token 预算，界面显示 token spend；可在条件里写轮次子句限长（例如「or stop after 20 turns」） |
| 错误处理 | 由线程状态管理 | 认证失败/余额耗尽/上下文溢出自动清除 goal |
| 非交互 | 支持 | claude -p "/goal <条件>" 支持 |
| 字符上限 | 4000 | 4000 |

plan 模式对照：

| 维度 | Codex /plan | Claude plan mode |
| --- | --- | --- |
| 本质 | 访谈式澄清流程 | 权限模式：只读调研+写计划，编辑被阻止直到批准 |
| 计划编辑 | 计划可编辑 | Ctrl+G 编辑计划 |
| 批准选项 | 确认后进入执行 | Yes auto / Yes manual / No keep planning |
| 批准后行为 | 按计划执行（路径已定，通常一轮完成） | 批准后进入编辑模式执行；goal 是另一条独立机制 |
| 强制范围 | 官方推荐流程 | bypass 模式下不强制 |

工程上的取舍：自审省一次推理、能看工具结果，但依赖主模型的诚实；独立裁判判定更「客观」、每轮成本固定，但看不见工具输出，条件必须能被纯文本证明。没有公开数据说谁更好。

![Codex 自审 vs Claude 独立裁判](/images/codex-goal-mode/codex-vs-claude-goal-arch.png)

两条路线的差别在完成判定：Codex 由主模型取证审计，Claude 由每轮独立的轻量裁判模型裁决并注入理由。

## 附录 B：技术追溯

时间线：

- 2025-03，METR 发布长程任务度量，给出「可靠完成时长约 7 个月翻倍」的量化基线，并把「串起长动作序列」定义为瓶颈。
- 2025 年上半年，τ-bench（ICLR 2025）显示 gpt-4o 在工具使用任务上 pass@1 仅 42%——长任务评测从一开始就不是高分场。
- 2025 年中，OpenAI 公开 25 小时/1300 万 token 连续运行实验，核心方法是外部状态文件（spec/plan/implement/documentation），官方称 durable project memory——「目标外置」路线的实证起点。
- 2026-05 前后，Codex 与 Claude 两侧的 goal 先后转正/成型，Codex 0.128.0+ 提供 /goal 命令族。
- 2026 年，机制研究（PushBench / LongHorizon-Harness）验证「显式状态外置 + 独立验证」机制在长程基准上的量化收益，为产品级 goal 提供机制级证据。

实现位置：goal 是「线程状态 + 上下文注入」的组合——目标持久化在服务端线程状态，每次续跑把最新状态（updatedAt、tokensUsed 随轮次变化）重新注入请求。对比之下，AGENTS.md/PLANS.md/CLAUDE.md 是启动时或按需加载、由模型自己读文件（拉取式）；goal 由 harness 每轮注入（推送式），目标不是可选项，而是请求的一部分。与记忆系统的关系：PLANS.md 是 agent 自维护的 living document，CLAUDE.md 是启动加载的上下文，MemGPT 是虚拟上下文分层记忆——它们解决「记得住」，goal 额外解决「每轮都必须对着它检查」。

## 附录 C：最小可复现示例（实测记录）

实测环境为 Codex 桌面端/CLI，创建 goal 后可观察到：

1. 注入格式：模型侧每轮请求携带 `<codex_internal_context source="goal">` 块，包含 objective、budget、续跑策略、完成审计、阻塞审计，且随轮次更新。
2. get_goal 返回：threadId、objective、status、tokensUsed、timeUsedSeconds、createdAt、updatedAt、remainingTokens、completionBudgetReport。
3. 线程单例：已有未完成 goal 时 create_goal 报错「cannot create a new goal because this thread has an unfinished goal; complete the existing goal first」。
4. 终态：update_goal 只能标记 complete 或 blocked。
5. 复现步骤：新开线程 → /goal <六要素文本> → 观察自动续跑与注入块 → get_goal 查看状态 → 完成后 update_goal complete。
6. 注意：goal 工具是 exec 级内建（无 mcp__ 前缀，非 MCP）；预算耗尽≠完成；每线程一个 goal。

## 附录 D：参考来源

- [METR: Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)
- [OpenAI: Run long-horizon tasks with Codex](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex)
- [OpenAI Cookbook: Using goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)
- [OpenAI Docs: Long-running work](https://developers.openai.com/codex/long-running-work)
- [Claude Docs: Goal](https://code.claude.com/docs/en/goal)
- [Claude Docs: Permission modes](https://code.claude.com/docs/en/permission-modes)
- [OpenAI Cookbook: Codex Exec Plans](https://developers.openai.com/cookbook/articles/codex_exec_plans)
- [Claude Docs: Memory](https://code.claude.com/docs/en/memory)
- [MemGPT（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560)
- [SWE-bench 排行榜](https://www.swebench.com/)
- [SWE-bench Pro（arXiv:2509.16941）](https://arxiv.org/abs/2509.16941)
- [SWE-EVO（arXiv:2512.18470）](https://arxiv.org/abs/2512.18470)
- [τ-bench](https://taubench.com/)
- [CodeClash](https://codeclash.ai/)
- [LongCLI-Bench](https://aclanthology.org/2026.findings-acl.1497/)
- [LongHorizon-Harness（arXiv:2608.01964）](https://arxiv.org/abs/2608.01964)
- [PushBench（arXiv:2605.23574）](https://arxiv.org/abs/2605.23574)
- [StructAgent（arXiv:2607.11388）](https://arxiv.org/abs/2607.11388)
- [Reza Rezvani: Claude Code Goal in Production](https://alirezarezvani.medium.com/claude-code-goal-in-production-3-tested-use-cases-that-work-5ab5f449a3c7)
- [Codex manual](https://developers.openai.com/codex/codex-manual.md)
