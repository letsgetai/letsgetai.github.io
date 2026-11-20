---
title: "工作型数字 Agent：环境分类、两根趋势与可展示案例"
date: 2026-11-20
draft: false
categories: ["Agent"]
---

工作型数字 Agent 该按闭环环境来分，而不是按产品名或模式名来分。本文先立坐标系，再讲两根趋势，并给每类环境配上能看清工作方式的案例。案例优先来自开源仓库里的演示；官方闭源若展示清楚，只作对照。核对截止于 2026-11-20（Asia/Shanghai）。星数与仓库时间以 GitHub API 为准；可视化只写本轮真正打开过的文件或视频帧。

调研截止日期：2026-11-20（Asia/Shanghai）。本报告只收录本轮打开过原文、仓库或可视化文件的材料。星数与仓库时间以 GitHub API 为准；未能用 API 复核的，不写星数。日期一律换算为 Asia/Shanghai，并标明时区。没有时刻的官方页面，只写日期并注明来源字段。

## 一、框架

先立坐标系，再讲趋势和案例。产品名、模式名、厂商名都不另开分类。

### 1. 第一刀：按闭环环境分

先看这件事在什么环境里做成。数字一侧是五类，物理/具身是同一刀切出来的另一侧，只作对照。

- Deep Research：在公开或受控来源里搜索、阅读、综合。做成，是一份有依据的报告。
- Coding Agent：在仓库和终端里改、跑、验。做成，是能过检查的改动。
- GUI / 办事型：在现成软件界面里看、点、填。做成，是把这件事在现成系统里点完。
- Personal Agent：在授权过的账号、日程、记忆和权限里，跨应用、跨时间代劳，并知道何时问人。做成，是在授权范围内持续办事。
- Scientific Agent：假设、实验、验证、成文。做成，是可检验的科学结果。它不是检索综述，也不是一次可合并的代码改动。
- 物理/具身：跟真实世界交互，动作难撤销。只作对照，不与数字五类混排成一条进化阶梯。

Muse 是 Personal 的旗舰，不是新类。Autoresearch 挂在 Scientific，不是新类，更不是 Coding Agent 的下一站。ChatGPT、WorkBuddy 一类产品里的模式切换，只是同一产品切入不同环境。

### 2. 第二刀：短程还是长程

短程和长程两边都有，不是新类。长程也不是“只是跑得更久”，更不是自主或自进化的同义词。

Towards Long-Horizon Agents 的项目页把长程写成结构性质，而不是时钟时间，并把它和长时间运行、自主、自进化区分开。任务层级是嵌套的：H1 Intra-Context，是同一窗口内的多步耦合；H2 Cross-Context，是超出单窗口，必须把状态外置并检查点、交接、恢复；H3 Cross-Task，是目标会切换的开放任务流，还要积累可复用技能。本轮打开了 https://long-horizon-agents.github.io/ ；Preprints.org 原稿页本轮返回 403，具体发布日未能在源站复核。项目页 BibTeX 记为 2026 年 7 月，DOI 10.20944/preprints202607.1328.v1。

### 3. Agent 模块

Skills、自进化是能力手段，用来支撑更长、更靠近真实世界的任务。它们不单列成类，也不升成第三根趋势。后面只用一个开源例子说明自进化长什么样，再用官方 Skills 说明它是可装载的程序性知识，而不是新的闭环环境。

## 二、两根趋势

坐标系立住以后，只用两根趋势把图讲活。

第一根：越来越接触真实世界。硬度在增加，但数字五类加物理侧是并行环境，不是一条必须拾级而上的进化阶梯。检索合成、仓库终端、屏幕点击、物理世界交互，是不同闭环，不是前后取代。

第二根：任务越来越长。资料侧从一次性检索增强，走到多步搜索、阅读和综合成报告。软件侧从补全，走到在仓库里连续改、跑、验，并进入需要人验收的自动流程。个人侧是跨会话不断线，任务关掉应用后还在继续，必要时回来问人。科学侧是实验闭环，按可检验指标决定保留或丢弃，而不是把 Coding Agent 再跑一夜。

判断、权限和核验收到对应例子里，不升级成第三根趋势。

## 三、趋势下的案例：从旧到新

每类环境至少放一个本轮真正打开过可视化的主案例。精选，不堆清单。官方闭源如果展示清楚，可以作对照，但必须标明不是开源。

### 趋势一：并行环境变硬

顺序是检索合成 → 仓库终端 → 屏幕点击 → 物理世界交互。请把它读成环境变硬，不要读成一代淘汰一代。

**检索合成。** 开源主案例是 GPT Researcher（https://github.com/assafelovic/gpt-researcher）。它把问题拆成研究问题，并行检索和阅读，再汇总成带引用的报告。本轮打开了仓库 README 里的界面演示 GIF：https://github.com/user-attachments/assets/3a56a008-fcbd-4373-9e40-8d6027120cfb 。画面从“What should we research?”的提问框，走到正在撰写的《Why Nvidia Stock Is Going Up in 2026》，右侧列出 16 个来源、11 个站点，报告写到 2,173 词并可下载。这是 Deep Research 的做成形态：有依据的报告。它不是实验闭环，也不是在仓库里提交可合并改动。仓库创建于 2023-05-12 18:33 Asia/Shanghai；默认分支 main 最近一次提交是 2026-11-09 23:17 Asia/Shanghai；未归档，仍在更新；API 星数 30,785。官方对照是 OpenAI 的 Introducing deep research（https://openai.com/index/introducing-deep-research/），页面标注 2025-02-02，并写明 2026-02-10 更新，可接 MCP、可把检索限制在受信任站点、可实时跟踪并中断。这篇是官方产品说明，不是开源。本轮打开了文章和头图，头图只是望远镜图标，不是研究报告界面的操作录屏，因此不以头图当作工作方式展示。

![GPT Researcher 从提问走到带 16 个来源的研究报告](https://github.com/user-attachments/assets/3a56a008-fcbd-4373-9e40-8d6027120cfb)

*GPT Researcher 仓库 README 中的界面演示 GIF。本轮已打开。*

**仓库终端。** 开源主案例是 OpenHands（https://github.com/OpenHands/OpenHands）。它是面向仓库和工程流程的 Coding Agent 控制台，当前界面叫 Agent Canvas，可以在本地或远程后端上跑 agent，并配置自动化。本轮打开了官方截图 https://assets.openhands.dev/screenshot/automation-preview.png ：Automate 页上有活跃的 “PR Triage Digest”“Nightly Security Pass”“Docs Sync on Push”，以及 “GitHub PR review copilot”“GitHub repository monitor” 一类模板；左侧会话列表是具体工程任务。另打开了 README 里的架构图 https://github.com/user-attachments/assets/cb6de6f5-ac30-4d04-a76a-b5c259f0c163 ，画的是笔记本上的 Agent Canvas 连接到本地、虚拟机或 OpenHands Cloud 的 Agent Server 与 Automations。这张图说明它落在仓库和流程侧，而不是个人生活代劳。需要说清楚：本轮打开的是控制台和架构，不是一次“从 issue 自动开出 PR”的完整录屏。人还要在这类流程里做验收和授权。仓库创建于 2024-03-13 11:33 Asia/Shanghai；默认分支 main 最近一次 API 核到的提交是 2026-11-20 07:16 Asia/Shanghai（SHA 428b3dc2a0e8d3d0f689a909e2607e33d973da76）；同一天 API 的 pushed_at 为 2026-11-20 10:09 Asia/Shanghai，可能还有更晚推送；未归档，仍在更新；API 星数 93,859。同一侧的从旧到新，可以用 SWE-agent 家族作更早、更“issue→补丁”的开源对照：SWE-agent 创建于 2024-04-02 12:09 Asia/Shanghai，默认分支最近提交停在 2026-07-16 23:21 Asia/Shanghai，README 已说明当前开发重心转到 mini-swe-agent；mini-swe-agent 默认分支最近提交是 2026-09-03 13:05 Asia/Shanghai，本轮打开了 https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini.gif?raw=true ，看到终端里把“写一个数独求解器”拆成建目录、写代码、跑测试，并在执行前询问 “Execute? Enter to confirm”。这是仓库终端里的短-中程闭环，加上人确认，还不是跨会话的个人代劳。

![OpenHands Automate 控制台，含 PR 分诊和夜间巡检等自动化](https://assets.openhands.dev/screenshot/automation-preview.png)

*OpenHands 官方 Automate 截图。本轮已打开。看到的是控制台，不是一次自动开 PR 的完整录屏。*

![mini-swe-agent 在终端里写数独求解器，执行前需要确认](https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini.gif?raw=true)

*mini-swe-agent 官方演示 GIF。本轮已打开。*

**屏幕点击。** 开源主案例是 browser-use（https://github.com/browser-use/browser-use）。它让模型在真实浏览器里操作现成网页。本轮打开了 README 标注为 “Browser Use V4 booking a driving test” 的演示 GIF：https://github.com/user-attachments/assets/135885e8-1141-4e10-b719-bf690ae7d260 。画面从 cloud.browser-use.com 的任务框，进入加州 DMV 预约页，选择预约类型，遇到“Select all squares with buses”的验证码，最后到达 “Appointment Scheduled”。这是 GUI / 办事型的做成形态：在现成系统里把事点完。它不是仓库里的代码改动，演示里也出现了验证码和不可轻易撤销的预约确认，人仍然要盯权限和误操作。仓库创建于 2024-11-01 00:00 Asia/Shanghai；默认分支最近提交是 2026-11-18 15:06 Asia/Shanghai；未归档，仍在更新；API 星数 121,529。官方对照是 Google 把 computer use 做进 Gemini 3.5 Flash 的博客（https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-computer-use-gemini-3-5-flash/ ，页面标注 2026-06-24），文中写明可要求对敏感或不可逆动作做显式确认，并在发现间接提示注入时停止任务。Anthropic 的 Computer Use 与 Skills、Files API 一起在 2026-08-20 宣布一般可用（JSON-LD 的 datePublished；https://claude.com/resources/articles/computer-use-skills-api-files-api）。这些是官方能力说明，不是开源仓库。

![browser-use 在加州 DMV 网站完成预约，过程中出现验证码](https://github.com/user-attachments/assets/135885e8-1141-4e10-b719-bf690ae7d260)

*browser-use 仓库 README 中的预约演示 GIF。本轮已打开。*

**物理世界交互。** 对照案例只选一个：NVIDIA Isaac GR00T（https://github.com/NVIDIA/Isaac-GR00T）。它是面向通用机器人的基础模型仓库，不是数字侧的新类。本轮从 Git LFS 打开了 README 中的三段真实交互 GIF：https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/main/media/unitree_g1.gif 里，Unitree G1 人形机器人把小推车上的黑色包装盒放到白色架子上；https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/main/media/agibot_g1.gif 里，AgiBot G1 的双臂在桌面整理碗碟和餐具；https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/main/media/yam.gif 里，YAM 双臂操作台上的红色衣物。这是跟真实世界交互、难以撤销的一侧。它不是 GUI Agent，也不是 Scientific Agent。仓库创建于 2025-03-12 02:34 Asia/Shanghai；默认分支最近提交是 2026-11-18 06:53 Asia/Shanghai；未归档，仍在更新；API 星数 8,401。

![Unitree G1 把包装盒放到架子上](https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/main/media/unitree_g1.gif)

*NVIDIA Isaac GR00T README 中的 Unitree G1 演示。本轮已从 media.githubusercontent.com 打开。*

### 趋势二：任务越来越长

这一根按资料、软件、个人、科学四条线从旧到新，不另开场景章。Autoresearch 只放在科学侧，不要写成 Coding 的下一站。

**资料侧：从 RAG 到 Deep Research。** 旧的一端是单次检索增强生成，问一句、取几段、答一句。新的一端是多步搜索、阅读、综合，并给出引用。GPT Researcher 的界面演示已经显示：一次提问会经历计划、写稿、列来源，最终得到可下载的长报告。OpenAI Deep Research 的官方说明把它明确写成可独立做数十分钟、引用可核对的研究报告；2026-02-10 的更新还加上实时跟踪和中断。资料侧变长的是研究过程，不是实验训练循环。

**软件侧：从补全到自动流程、人做验收。** 旧的一端是行内补全。新的一端是 agent 在仓库和终端里连续行动，并进入围绕 PR、仓库事件和定时任务的自动化。OpenHands 的 Automate 截图给出的是这一端的控制台：PR 分诊、PR 审查、仓库监控、夜间巡检，人都还在模板、权限和结果验收里。mini-swe-agent 的终端演示更靠近“接到任务后改代码并跑测试”，每一步执行仍可要求人确认。SWE-agent 主仓库把“拿 GitHub issue 自动修复”写进定位，但它的默认分支已停在 2026-07-16 23:21 Asia/Shanghai，不宜再当成正在推进的主阅读；它说明这条线从哪里来，不说明这条线今天还在哪里更新。Cursor 等厂商博客可以作官方对照，本报告不以厂商博客作为软件侧唯一证据。

**个人侧：跨会话不断线。** 主案例是 Muse。它不是开源，也不是新类，而是 Personal Agent 的旗舰。官方介绍页 https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/ 标注 2026-09-08，页上也出现 2026-11-10；澳洲发布页 https://about.fb.com/news/2026/11/muse-is-now-available-in-australia/ 标注 2026-11-17。文中写明：Muse 跑在独立的 Secure VM 里，可在人关掉应用后继续办事，发送邮件或下单前要征得同意，并提供审计轨迹。本轮完整下载并抽帧查看了两个官方视频。https://about.fb.com/wp-content/uploads/2026/09/Introducing-Muse_Sizzle-Video.mp4 （约 82.6 秒）里，Muse 主动提到训练进展，并问“我找到了 11 月的本地 10 公里赛，要我帮你报名吗？”，用户回复 “Yes, sign me up!”。https://about.fb.com/wp-content/uploads/2026/09/Muse_Shopping.mp4 （约 25.7 秒）里，Muse 列出旅行婴儿车选项，状态变为 “Setting up checkout...”，随后说明将用尾号 1234 的 Visa 下单，并出现 “Browser Completed · Purchased Glide Pro”。这是跨应用、跨时间代劳，并且把确认放在敏感动作前。开源替补没有找到同样干净的“生活代劳”可视化。Letta 的当前源码已转到 https://github.com/letta-ai/letta-code ，仓库仍在更新，默认分支最近提交是 2026-11-20 09:51 Asia/Shanghai，API 星数 3,876。本轮打开了 https://raw.githubusercontent.com/letta-ai/letta-code/main/assets/letta-code-demo.gif ，看到的是 “Resumed agent” 之后在仓库里审计测试覆盖率、读文件、跑命令。这能说明有可恢复的持久 agent，但展示的是编码任务，不能当作 Personal 主案例。Open Interpreter 的当前 README 已把自己写成面向低成本模型的 coding agent，并说明是 Codex 的分支，不再适合作为 Personal 替补。

Muse 的两个官方视频如下。第一段约 82.6 秒，Muse 主动询问是否报名本地路跑；第二段约 25.7 秒，完成比价并在声明使用卡片后显示下单完成。

<video controls preload="metadata" src="https://about.fb.com/wp-content/uploads/2026/09/Introducing-Muse_Sizzle-Video.mp4">Muse 介绍视频：主动询问是否报名 10 公里赛。</video>

<video controls preload="metadata" src="https://about.fb.com/wp-content/uploads/2026/09/Muse_Shopping.mp4">Muse 购物演示：比价、确认后下单。</video>

**科学侧：实验闭环，而不是写代码或检索综述。** 主案例是 karpathy/autoresearch（https://github.com/karpathy/autoresearch）。agent 只改 `train.py`；每次训练预算是 5 分钟墙钟；用 val_bpb 决定 keep 或 discard；`program.md` 由人改；改进的是小模型，不是 agent 自己的研究技能。本轮打开了 README 顶部的 https://raw.githubusercontent.com/karpathy/autoresearch/master/progress.png ：标题为 “Autoresearch Progress: 83 Experiments, 15 Kept Improvements”，绿线是保留下来的验证损失下降，灰点是被丢弃的实验。这正是科学侧的做成形态：可检验的结果，而不是一篇检索综述，也不是一次给业务仓库提的 PR。仓库创建于 2026-03-07 06:00 Asia/Shanghai；默认分支 master 最近一次提交是 2026-03-26 08:07 Asia/Shanghai；未归档，但主线代码到调研截止日已约 8 个月没有新提交。它是 2026 年的标杆仓库，不是一份还在滚动的活清单。API 星数 100,290。可补的开源对照是 The AI Scientist（https://github.com/SakanaAI/AI-Scientist）。本轮打开了 https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif ：动画从 Idea Generation（想法、新颖性检查、打分归档），到 Experiment Iteration（模板、用 LLM 和 aider 改代码、执行脚本、根据数值结果更新计划），再到 Paper Write-Up 和 LLM Paper Reviewing，并虚线回到想法生成。这是实验闭环加写稿，不是 Deep Research，也不是业务代码合并。该仓库以及 AI-Scientist-v2 的默认分支最近提交都在 2025-12-19 15:46 Asia/Shanghai，未归档，但主线已停约 11 个月，不能当活清单。

![Autoresearch 的 83 次实验中 15 次被保留](https://raw.githubusercontent.com/karpathy/autoresearch/master/progress.png)

*karpathy/autoresearch 仓库中的 progress.png。本轮已打开。*

![The AI Scientist 从想法、实验迭代到写稿和审稿的闭环](https://raw.githubusercontent.com/SakanaAI/AI-Scientist/main/docs/anim-ai-scientist.gif)

*The AI Scientist 仓库中的实验闭环动画。本轮已打开。主线已停。*

## 四、其他洞察

没有发现必须单列成第三根趋势、又不能折回上述两根的新洞察。

判断、权限和核验都很重要，但它们已经落在例子里。browser-use 的预约和验证码、Google Computer Use 对不可逆动作的确认、Muse 在发送和下单前的询问、OpenHands 与 mini-swe-agent 里的人验收，说的都是权限和误操作。Autoresearch 用 val_bpb 决定去留，The AI Scientist 在写稿后还经过自动审稿，相关综述也把“结果能不能被外人核验”写成科学 agent 的缺口。这些都是两根趋势走下去以后，人还要盯的东西，不必再立第三根。

Skills 和自进化同样不单列。Anthropic 在 2025-10-16 的文章里把 Agent Skills 写成可发现、按需加载的说明书和脚本目录；Darwin Gödel Machine 则给出一个开源模块例子：系统改的是编码 agent 自身，再用 SWE-bench / Polyglot 做经验验证，并在沙箱和人监督下进行。本轮打开了 https://raw.githubusercontent.com/jennyzzt/dgm/main/misc/overview.gif ，看到 Darwinian Agent Archive 从初始 agent 长出子代树，右侧是父代代码、自修改、子代代码，以及 SWE-Bench/Polyglot 评测。仓库 https://github.com/jennyzzt/dgm 创建于 2025-05-23 22:25 Asia/Shanghai，默认分支最近提交是 2025-08-13 18:40 Asia/Shanghai，未归档，但主线已停；API 星数 2,531。官方博客 https://sakana.ai/dgm/ 标注 2025-05-30；论文 arXiv:2505.22954 的 abs 写明 2025-05-29 提交，2026-03-12 修订到 v3。数字是 SWE-bench 20.0%→50.0%，Polyglot 14.2%→30.7%。它说明自进化是手段，而且改进对象是编码 agent，不是科学发现本身。OpenAI 的 Tax AI 案例（https://openai.com/index/building-self-improving-tax-agents-with-codex/ ，标注 2026-05-27）也只作模块对照：上线时约 25% 申报达到 75% 字段正确，六周后到 86%，仍要业务人员审。本轮未把这两项写成新趋势。

![Darwin Gödel Machine 用评测验证对编码 agent 自身的修改](https://raw.githubusercontent.com/jennyzzt/dgm/main/misc/overview.gif)

*jennyzzt/dgm 仓库中的 overview.gif。本轮已打开。这是模块例子，不是新趋势。*

## 五、一页判断

外面做到哪。数字五类都已经有能看清工作方式的实例。Deep Research 能从问题走到带引用的长报告，开源和官方产品都在这条线上。Coding Agent 已经从补全走到在仓库里连续改代码、跑测试，并进入围绕 PR 和仓库事件的自动化控制台；最活的开源主线是 OpenHands，SWE-agent 家族说明了“issue→补丁”这条研究路线，但主仓库已经不再是当前更新中心。GUI Agent 能在真实网站上把预约这类事点完，官方 computer use 也已经把确认和停止条件写进安全说明。Personal Agent 的最完整展示目前来自 Muse 的官方视频和文档：跨会话继续办事，敏感动作先问人；开源侧还缺少同等清晰的生活代劳录屏。Scientific Agent 已经能按固定预算和验证指标自动做实验取舍，Autoresearch 是清晰标杆，但它的主线已经约 8 个月没有新提交；The AI Scientist 把实验到写稿的闭环画得很清楚，主线同样已经停更。物理侧的开源仓库可以展示真实机器人搬盒子、整理桌面，但那是对照，不是数字 agent 的下一站。

人还要盯什么。第一是权限和不可逆动作：下单、发信、提交预约、在真实世界里抓取物体，都需要确认、审计和可撤销策略。第二是验收：自动提改动或自动巡检，不等于可以无人看管，PR 和运行结果仍要人收。第三是核验：报告要有引用，实验要有指标和可复现线索，论文式产出尤其不能只看“跑完了”。第四是边界：模式切换不会改变环境分类，自进化和 Skills 也不会把个人代劳、科学发现或屏幕操作变成同一类事情。

对企业和运维型场景意味着什么。先按闭环环境对任务，而不是按厂商或产品模式对任务。已有业务系统往往没有干净 API，GUI Agent 和 computer use 说的是在现成界面里把流程点完，同时把不可逆步骤留给人。仓库和终端侧更适合把检查、补丁和例行巡检做成可验收的自动流程，而不是指望一次补全解决运维。需要跨系统、跨班次盯着办的事务，接近 Personal 这一侧，关键是授权范围、审计和何时升级问人。若目标是可检验的指标改善或实验结论，应走科学闭环，而不是把 Coding Agent 通宵改业务代码。Skills 可以沉淀组织里的操作手册，自进化如果要做，也应关在可评测、可回滚、有人监督的范围里。环境会继续变硬，任务会继续变长，但判断、权限和核验不会因此消失。


---

# 附录：本轮调研资料全表

调研截止日期：2026-11-20（Asia/Shanghai）。本表收录本轮打开、尝试打开或用 API 核对过的相关资料，不限于正文主案例。星数只在 GitHub API 成功返回时填写。日期一律换算为 Asia/Shanghai；没有时刻的，只写日期并注明字段来源。打不开或核对失败的，如实记录。

| 资料 | 类型 | 日期与更新状态（Asia/Shanghai） | 本轮如何核 | 选或不选 | 说明 |
| --- | --- | --- | --- | --- | --- |
| arXiv:2601.12560 v1，Agentic AI 架构与分类 | 论文 | abs 标注 Submitted on 18 Jan 2026。本轮 arXiv API 查询在后一次请求中返回 500，提交时刻未能在本轮 API 复核。 | 打开 abs / HTML | 框架对照，不作主案例 | 环境分类更粗，用作边界对照，不改本报告已定的数字五类加物理对照。 |
| arXiv:2608.05179，Autonomous Research Agents | 论文 | abs 标注 Submitted on 29 Jun 2026。标识月份 2608 与 6 月提交不一致，两处都不抹平。本轮 arXiv API 时刻未复核成功。 | 打开 abs / HTML | 科学侧核验问题的阅读，不作主案例 | 把“结果能不能被外人核验”作为组织问题。不升成第三根趋势。 |
| arXiv:2607.13104，Self-Improvements in Modern Agentic Systems | 论文 | abs 标注 Submitted on 14 Jul 2026。本轮 API 时刻未复核成功。 | 打开 abs / HTML | 模块阅读 | 把自改进写成手段。配套仓库见下一行。 |
| https://github.com/selfimproving-agent/Awesome-Self-Improving-Agents | awesome / 仓库 | 创建 2025-12-28 13:29；默认分支最近提交 2026-11-13 23:32；`pushed_at` 同；未归档。API 星数 640。 | API + atom | 模块清单，不作主案例 | 仍在更新。 |
| arXiv:2609.04894，Digital/Social/Virtual/Physical | 论文 | abs 标注 Submitted on 4 Sep 2026。文中文献截止 2026-08-31。API 时刻未复核成功。 | 打开 HTML | 框架对照 | 分类与本报告不完全相同，只作边界对照。强调授权和核验，但不单列新趋势。 |
| Towards Long-Horizon Agents，DOI 10.20944/preprints202607.1328.v1 | 论文 / 项目页 | 项目页 BibTeX：year 2026，month July，publisher Preprints。Preprints.org 原稿页本轮 403，具体日未能复核。 | 打开 https://long-horizon-agents.github.io/ ；preprints.org 失败 | 第二刀的对照 | 项目页写明 H1 Intra-Context、H2 Cross-Context、H3 Cross-Task，并与长时间运行、自主、自进化区分。 |
| https://github.com/RUC-NLPIR/Awesome-Long-Horizon-Agents | awesome / 仓库 | 创建 2026-07-09 10:13；默认分支最近提交 2026-11-02 18:16；未归档。API 星数 873。 | API + atom | 长程阅读清单 | 仍在更新。 |
| https://github.com/luo-junyu/Awesome-Agent-Papers | awesome / 仓库 | 创建 2025-03-21 11:40；默认分支最近提交 2025-11-07 17:27；`pushed_at` 2026-11-12 23:22，与 main 最近提交不一致。未归档。API 星数 2,878。 | API + 提交页 | 已停用为主阅读 | main 在 2025-11-07 后再无提交，不能当 2026 年活清单。配套论文 arXiv:2503.21460 按任务要求已停作主阅读。 |
| https://github.com/yenanjing/awesome-ai-for-science | awesome / 仓库 | 创建 2026-03-30 15:03；默认分支最近提交 2026-11-19 19:31，说明为自动发现更新。未归档。API 星数 14。 | API | 科学侧扫描源 | 仍在自动更新，星数很少，只作发现渠道。 |
| https://openai.com/index/introducing-deep-research/ | 官方博客 | 标注 2025-02-02，文中 2026-02-10 更新。无时刻。 | 打开正文和头图 | 官方对照 | 非开源。头图不是工作界面，见 cases.md。 |
| https://github.com/assafelovic/gpt-researcher | 仓库 + 演示 GIF | 创建 2023-05-12 18:33；main 最近提交 2026-11-09 23:17；未归档。API 星数 30,785。 | API + 打开 GIF | 主案例 | 见 cases.md。YouTube 演示未观看。 |
| https://github.com/stanford-oval/storm | 仓库 | 创建 2024-03-25 00:23；main 最近提交 2025-10-01 02:07；未归档。API 星数 31,911。 | API + atom + 打开 README 图 | 不选为主案例 | 能生成带引用的长文，属于资料侧；主线已停约 13 个月。本轮打开了 https://raw.githubusercontent.com/stanford-oval/storm/main/assets/two_stages.jpg ，是 Prewriting / Writing 流程图，不是产品操作录屏。在线预览页未作为主证据。 |
| https://github.com/langchain-ai/open_deep_research | 仓库 | 创建 2024-11-21 01:37；最近提交 2026-08-11 02:13；**已归档**。API 星数 12,674。 | API | 不选 | 归档仓库，不作为主阅读或主案例。 |
| https://github.com/OpenHands/OpenHands | 仓库 + 截图 | 创建 2024-03-13 11:33；main 最近提交 2026-11-20 07:16；`pushed_at` 2026-11-20 10:09；未归档。API 星数 93,859。 | API + 打开截图和架构图 | 主案例 | 见 cases.md。 |
| https://www.openhands.dev/product/canvas | 产品页 | 本轮打开 HTML | 打开 | 配套 | 说明 Agent Canvas 是本地可视工作区。 |
| https://github.com/SWE-agent/SWE-agent | 仓库 | 创建 2024-04-02 12:09；main 最近提交 2026-07-16 23:21；`pushed_at` 2026-11-17 06:30；未归档。API 星数 20,835。 | API + atom | 对照，主线已放缓 | README 已把开发重心指向 mini-swe-agent。 |
| https://github.com/SWE-agent/mini-swe-agent | 仓库 + GIF | 创建 2025-06-29 04:18；main 最近提交 2026-09-03 13:05；`pushed_at` 2026-11-17 06:30；未归档。API 星数 9,862。 | API + 打开 GIF | 软件侧对照 | 终端演示已打开。main 已约两个半月无提交。 |
| https://github.com/SWE-agent/swe-agent-media | 媒体仓库 | 创建 2025-07-22 01:34；`pushed_at` 2025-11-15 09:29；未归档。API 星数 2。 | API + 下载 GIF | 演示托管 | 本轮打开 mini.gif、swebench.gif、inspector.gif 中的前两项并抽帧；inspector.gif 已下载，未作为正文证据展开。 |
| https://swe-agent.com | 文档站 | 本轮根路径只返回跳转到 latest/ 的小页面 | 打开根路径 | 配套 | 未把跳转页当作演示。YouTube https://www.youtube.com/watch?v=fcr8WzeEXyk 未观看。 |
| https://cursor.com/blog/cursor-3 | 官方博客 | 任务书记录 2026-04-02 | 本轮未再作为主证据打开 | 仅可作厂商对照 | 按任务要求不以厂商博客为软件侧唯一证据，故不写入正文主案例。 |
| https://cursor.com/blog/cursor-sdk 或 https://cursor.com/blog/agent-sdk | 官方博客 | 任务书记录 2026-10-20 | 本轮未作为主证据打开 | 不选为主案例 | 同上。 |
| https://github.com/browser-use/browser-use | 仓库 + 演示 GIF | 创建 2024-11-01 00:00；main 最近提交 2026-11-18 15:06；未归档。API 星数 121,529。 | API + 打开 GIF | 主案例 | 见 cases.md。 |
| https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-computer-use-gemini-3-5-flash/ | 官方博客 | 标注 Jun 24, 2026，无时刻 | 打开正文 | GUI 官方对照 | 写明不可逆动作可要求人确认。非开源。 |
| https://claude.com/resources/articles/computer-use-skills-api-files-api | 官方博客 | JSON-LD datePublished 2026-08-20，无时刻 | 打开正文并抽取 JSON-LD | GUI / 模块对照 | Computer use、Skills API、Files API 一般可用。 |
| https://www.anthropic.com/research/equipping-agents-for-the-real-world-with-agent-skills | 官方博客 | 标注 Oct 16, 2025，无时刻 | 打开正文 | 模块说明 | Skills 是可装载的程序性知识，不单列成类。 |
| https://github.com/xlang-ai/OSWorld | 仓库 | 创建 2023-10-16 09:49；main 最近提交 2026-11-13 00:40；未归档。API 星数 3,269。 | API + atom + 尝试官网 | 不选为主案例 | 这是计算机环境评测基准，不是一个可展示的办事 agent 产品。官网 https://os-world.github.io 本轮返回 HTTP 500，页面未打开。 |
| https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/ | 官方博客 + 视频 | 标注 2026-09-08，页上同时出现 2026-11-10 | 打开正文，完整下载并抽帧两个 mp4 | Personal 主案例 | 非开源。见 cases.md。 |
| https://about.fb.com/news/2026/11/muse-is-now-available-in-australia/ | 官方博客 | 标注 2026-11-17 | 打开正文 | 配套 | 澳洲可用与本地连接器说明。页内另有 https://about.fb.com/wp-content/uploads/2026/11/Muse-Sizzle-16x9-1.mp4 ，本轮未下载观看。 |
| https://github.com/letta-ai/letta | 仓库 | 创建 2023-10-11 15:38；main 最近提交 2026-10-19 06:28，内容是把活跃源码指向 letta-code；未归档。API 星数 25,496。 | API + 打开 README | 历史入口 | 不再是当前实现主仓。 |
| https://github.com/letta-ai/letta-code | 仓库 + GIF | 创建 2025-10-25 12:17；main 最近提交 2026-11-20 09:51；未归档。API 星数 3,876。 | API + 打开 GIF | 开源替补，不升主案例 | 演示是恢复后的编码任务，对不上生活代劳。 |
| https://github.com/openinterpreter/openinterpreter | 仓库 | 创建 2023-07-14 15:10；main 最近提交 2026-11-15 20:12；未归档。API 星数 68,918。旧地址 OpenInterpreter/open-interpreter 会 301 到此。 | API + 打开 README | 不选为 Personal 替补 | 当前定位是面向低成本模型的 coding agent，README 写明基于 Codex。原 Python 项目指向社区分叉 endolith/open-interpreter，本轮未再核那个分叉。 |
| https://github.com/karpathy/autoresearch | 仓库 + 图 | 创建 2026-03-07 06:00；master 最近提交 2026-03-26 08:07；未归档。API 星数 100,290。 | API + 打开 progress.png 与 README | 科学主案例 / 标杆 | 主线已停约 8 个月，仍未归档。 |
| https://github.com/SakanaAI/AI-Scientist | 仓库 + GIF | 创建 2024-08-12 17:32；main 最近提交 2025-12-19 15:46；未归档。API 星数 15,026。 | API + atom + 打开 GIF | 科学对照 | 可视化确认实验闭环。主线已停约 11 个月。论文 arXiv:2408.06292 本轮未把 API 时刻写入成功结果。 |
| https://github.com/SakanaAI/AI-Scientist-v2 | 仓库 | 创建 2025-04-08 14:37；main 最近提交 2025-12-19 15:46；未归档。API 星数 7,533。 | API + atom | 对照，不单列 | 与 v1 同日停在主线。 |
| https://github.com/NVIDIA/Isaac-GR00T | 仓库 + GIF | 创建 2025-03-12 02:34；main 最近提交 2026-11-18 06:53；未归档。API 星数 8,401。 | API + 打开三份 LFS GIF | 具身对照主案例 | raw.githubusercontent.com 只返回 LFS 指针；完整 GIF 从 media.githubusercontent.com 打开。 |
| https://github.com/Physical-Intelligence/openpi | 仓库 | 创建 2024-10-21 23:23；main 最近提交 2026-08-25 00:27；未归档。API 星数 14,708。 | API + atom | 不选 | 具身只保留一个仍在更新且可视化已打开的对照。openpi 主线已约 3 个月无提交。 |
| https://github.com/openvla/openvla | 仓库 | 默认分支最近提交 2025-03-24 07:40（atom）。本轮 API 一度 403，星数未核。 | atom | 不选 | 主线停更已久。 |
| https://github.com/jennyzzt/dgm | 仓库 + GIF | 创建 2025-05-23 22:25；main 最近提交 2025-08-13 18:40；未归档。API 星数 2,531。 | API + 打开 overview.gif | 模块例子 | SakanaAI/Darwin-Godel-Machine 本轮 API 404，正确仓库是 jennyzzt/dgm。 |
| https://sakana.ai/dgm/ | 官方博客 | 任务书与本轮检索一致为 2025-05-30 | 检索与论文交叉核，博客正文未作为唯一可视化来源 | 模块配套 | 可视化以仓库 GIF 为准。 |
| arXiv:2505.22954，Darwin Gödel Machine | 论文 | abs：Submitted on 29 May 2025，last revised 12 Mar 2026（v3） | 打开 abs | 模块配套 | SWE-bench 20.0%→50.0%，Polyglot 14.2%→30.7%。 |
| https://openai.com/index/building-self-improving-tax-agents-with-codex/ | 官方博客 | 任务书记录 2026-05-27 | 本轮未把该页作为新的可视化源重新打开 | 模块对照 | 按已定口径：上线时约 25% 申报达 75% 字段正确，六周后 86%，仍要业务人员审。非开源，不写进趋势。 |
| FutureHouse/paperqa、FutureHouse/crow | 仓库尝试 | 本轮 API 返回 404 | API | 打不开 / 仓库名不成立 | 未继续猜测其他组织名。 |
| https://os-world.github.io | 项目页 | 本轮 HTTP 500，空页 | curl | 打不开 | 不能声称看过 OSWorld 官网演示。 |

## 已停更或不再当主阅读的清单

- luo-junyu/Awesome-Agent-Papers：main 停在 2025-11-07 17:27 Asia/Shanghai，不能当 2026 年活清单。
- langchain-ai/open_deep_research：已归档，最近提交 2026-08-11 02:13 Asia/Shanghai。
- SWE-agent/SWE-agent：未归档，但 main 停在 2026-07-16 23:21 Asia/Shanghai，项目自己把重心转到 mini-swe-agent。
- SWE-agent/mini-swe-agent：未归档，main 停在 2026-09-03 13:05 Asia/Shanghai，演示可用，但不是当前最活的软件主线。
- karpathy/autoresearch：未归档，main 停在 2026-03-26 08:07 Asia/Shanghai，是标杆不是活清单。
- SakanaAI/AI-Scientist 与 AI-Scientist-v2：未归档，main 都停在 2025-12-19 15:46 Asia/Shanghai。
- jennyzzt/dgm：未归档，main 停在 2025-08-13 18:40 Asia/Shanghai。
- stanford-oval/storm：未归档，main 停在 2025-10-01 02:07 Asia/Shanghai。
- Physical-Intelligence/openpi：未归档，main 停在 2026-08-25 00:27 Asia/Shanghai，未选为具身主案例。
- openvla/openvla：main 停在 2025-03-24 07:40 Asia/Shanghai。

## 本轮打不开或日期未能完全核对的来源

- Preprints.org 上 Towards Long-Horizon Agents 原稿页：HTTP 403。项目页已打开，BibTeX 只有 2026 年 7 月，具体日未在源站复核。
- arXiv API（export.arxiv.org）本轮后一次查询返回 500，因此 2601.12560、2608.05179、2607.13104、2609.04894 的精确提交时刻未能在本轮 API 复核；abs 上的日期行已打开。
- https://os-world.github.io ：HTTP 500。
- FutureHouse/paperqa、FutureHouse/crow：GitHub API 404。
- GPT Researcher 与 SWE-agent 的 YouTube 演示：未观看，不算已打开展示。
- 澳洲 Muse 页中的 Muse-Sizzle-16x9-1.mp4：未下载。
- Cursor 两篇博客、OpenAI Tax AI 页：本轮没有作为新的可视化源重新打开；日期沿用任务书已核口径，并在表中标明。
- 部分 GitHub API 请求曾触发未认证速率限制或 301/403，最终以跟重定向后的仓库名为准；All-Hands-AI/OpenHands、OpenInterpreter/open-interpreter 均已解析到新路径。

