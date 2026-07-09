![](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/socialverse_logo.png)

<div align="center">
  <a href="README_CN.md">中文</a> | <a href="../README.md">EN</a>
 </div>

**社会科学研究的 AI 时代入口。**

从政策评估、量表开发、复杂抽样，到空间分析、网络分析与质性编码，socialverse 试图把社会科学研究中最常用、也最容易被 AI Agent 错用的方法，组织进**同一个可复现的研究工作台**：数据、研究设计、模型、诊断、证据链、图表与论文交付物，都可以在一个对象里被记录、调用和审计。

它面向经济学、政治学、社会学、心理学、传播学、人口学、公共卫生、传播学与数字人文等领域的研究者。你不需要计算机背景，也能看懂它在帮你做什么；如果你在使用 AI Agent，它也能为模型提供一套明确、可查询、可执行的分析接口，而不是让模型凭记忆“编”命令。

---

## 我们在做的事：为 AI 时代的社会科学搭基础设施

socialverse 是 **AI 时代社会科学研究的一套方法基础设施**。

我们的判断很直接：当前前沿大模型已经足够有用，能够阅读文献、生成代码、解释统计结果，也能辅助质性编码和论文复现；但让它直接做社会科学分析时，真正容易出问题的地方往往不是“会不会说”，而是**该用什么方法、假设是否成立、结果如何复现、结论能否被证据支撑**。换句话说，社会科学 AI Agent 缺的不只是更强的模型，还缺一套可靠、可查询、可执行、可编排、可审计的分析底座。

socialverse 就是这套底座：研究者可以直接使用它完成规范化工作流，AI Agent 也可以基于明确的工具契约与方法约束稳定调用它，而不是凭记忆“编”命令、临时拼流程，或生成不可追溯的结论。

它是 **AI4S（AI for Science）** 中面向社会科学的一块 —— **AI4Social**。与 AI4Bio 中已经成熟的基础设施 [**omicverse**](https://github.com/omicverse/omicverse) 类似，socialverse 的目标不是做一个单点工具，而是为一个学科群体提供可复用的研究对象、方法命名、执行接口和审计轨迹。

## 它能帮你做什么

| 你的场景                                              | 以往的麻烦                                                   | 用 socialverse                                               |
| ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **做政策评估**,想说清"某项改革有没有因果影响、有多大" | 在 Stata 里拼双重差分、担心平行趋势,还得打听眼下流行的"异质稳健"新做法(怕被说方法过时),代码散在一堆文件里 | 说清处理组与政策时点,**平行趋势检验 → 经典双重差分 → 最新反事实估计**一条龙,直接出事件研究图和论文级回归表,每步留痕 |
| **开发问卷 / 量表**,想知道信度效度、有几个维度        | SPSS 做因子分析,换个地方算信度,换份数据又从头来              | **因子分析(探索/验证)、信度、结构方程、IRT** 都在同一处,命名一致,结果直接写进"测量"部分 |
| **用带权重的大型调查**(CHARLS / NHANES / CGSS)        | 忘了加权就出错,加权又得记一堆抽样语法                        | **声明一次抽样设计**(权重 / 分层 / 抽样单元),之后所有估计自动按设计加权,不会跑成"简单随机样本" |
| **有访谈 / 文本**,要主题编码、把观点追溯回原文        | 质性软件和定量软件是两个世界,混合方法来回倒腾                | **主题编码、引语溯源、反身备忘**与定量分析同包,混合方法不用换工具 |

![](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/studystate_logo.png)

## 数据支撑：瓶颈在基础设施，而不只是模型能力

当前前沿大模型已经具备较强的语言理解、代码生成与研究辅助能力，能够参与社会科学文本标注、质性编码、数据分析与论文复现等任务 [1, 2]。但在社会科学分析中，真正决定可靠性的往往不是“能不能给出一个答案”，而是能否**在具体研究问题下正确选择方法、识别统计与因果假设、调用合适的数据与代码工具、执行可复现分析，并把结论限定在证据能够支持的范围内**。

已有 benchmark 显示，这一瓶颈非常具体：

- **StatQA**：包含 **11,623** 个统计分析样本；GPT-4o 的最好表现为 **64.83%**，主要错误集中在 *statistical method applicability errors*，也就是“知道方法名，但不知道什么时候该用” [3]。
- **REPRO-Bench**：包含 **112** 个社会科学论文可复现性评估任务；最好的 agent 准确率仅 **21.4%** [4]。
- **CORE-Bench**：包含 **90** 篇论文、**270** 个任务，覆盖计算机科学、社会科学和医学；最佳 agent 在最难任务上的准确率也只有 **21%** [5]。

这些结果说明：社会科学 AI4S 的关键瓶颈，并不只是模型“聪不聪明”，而是模型能否被一套**可靠的方法、数据、执行与验证机制**约束起来。

反过来，工具增强与执行环境已经被证明能够显著放大 agent 的表现：

- **InfiAgent-DABench** 将数据分析 agent 放入真实执行环境中评估；ICML 2024 正式版本包含 **603** 个数据分析问题和 **124** 个 CSV 文件 [6]。
- **Data Interpreter** 通过任务拆解、代码执行与逐步验证，将 InfiAgent-DABench 上的准确率从 **75.9%** 提升到 **94.9%** [7]。

这与工具使用、推理-行动框架的更早结论一致 [8, 9]：在复杂分析任务中，**可靠的基础设施 + 工具契约 + 执行反馈 + 审计机制**，能够显著放大模型的能力。

**socialverse 的定位正是这样一个社会科学分析底座**：它把研究中的数据结构、统计方法、质性分析流程、因果推断工具、复现规范与结果审计机制，组织成 AI Agent 可以稳定调用和编排的能力层。研究者可以直接使用这些规范化工作流；AI Agent 也能基于明确的工具契约与方法约束完成分析，而不是凭记忆编造命令、临时拼接流程，或在缺乏验证的情况下生成看似合理却不可追溯的结论。

---

> 简单说：**socialverse 希望成为社会科学研究者“从数据到论文”的 AI 时代入口**——方法可解释，结果可复现，图表可直接使用，结论可以追溯。

---

## 安装

```bash
pip install socialverse
```

核心只依赖 `numpy` + `pandas`；各方法的重型后端（statsmodels、scipy、networkx、scikit-learn、matplotlib 等）按需自动加载。没装某个后端时，相关方法会提示你安装，而不是让整个程序直接崩掉。

> 项目状态：socialverse 正在持续开发中。README 中列出的 API 代表当前设计目标与逐步开放的能力图谱；具体函数的可用性以当前版本文档、测试与 release note 为准。

---

## 研究对象:StudyState —— 一次研究 = 一个对象

做一项研究,你手上通常散落着一堆东西:原始数据、研究设计、跑出来的一串结果、各种稳健性检验、最后要放进论文的图表……常规做法是用一堆零散的变量和中间文件把它们接来接去——时间一长就乱,几个月后自己都说不清哪张图是用哪版数据、哪个设定跑出来的,想复现很难。

socialverse 把这些**统一收进一个精心设计的对象 `StudyState`**。你可以把它想成一个**项目档案袋 / 研究工作台**:数据放进去,之后每做一步分析,结果就自动归档进去,最后出图、出表也从里面取。

它的 12 个格子(槽位)**不是随意划分的,而是对应一项社科研究真实的生命周期**——从你手上的材料,到研究怎么设计、变量是什么,到你到底要估计什么、靠什么假设识别,再到跑出的结果、诊断、证据,最后到伦理合规与交付物。每个格子内部还有一套约定的字段(比如「研究设计」格里就有 `panel_id` / `time` / `treatment` / `weights` / `strata` / `psu`),既是提示也是规范:**写错格子名会当场报错**,保证一项研究从头到尾结构良好、好交接。

分析的三步都在这**同一个对象**上进行,你不用手动把数据和结果传来传去——`sv.pp` 往里写,`sv.tl` 读了再写回,`sv.pl` 从里读出:

```text
  sv.pp 准备 ──写入──▶  sv.tl 分析 ──读+写──▶  sv.pl 出图表 ──读出──▶  图 / 表

  StudyState 的 12 个格子 = 一项研究的生命周期(冒号后是每格常装的字段):

  ┌ 材料 ─────────────────────────────────────────────────────────
  │  sources         原始输入:datasets · corpora · bib · scans
  │  design          研究设计:panel_id · time · treatment · weights · strata · psu
  │  variables       变量表:outcome · exposure · controls · scales · constructs
  │  corpus · codes  文本 / 质性编码:documents · dfm · tei · themes · segments   〔质性〕
  ├ 提问 ─────────────────────────────────────────────────────────
  │  estimand        估计目标:target · population · effect
  │  identification  识别假设:strategy · dag · parallel_trends · iv_validity
  ├ 结果 ─────────────────────────────────────────────────────────
  │  models          拟合结果:did · event_study · cox · topic · network
  │  diagnostics     诊断 / 稳健:pretrend · balance · robustness · reliability · sensitivity
  │  evidence        证据链:citations · verified_bib · quote_index · claim_evidence
  ├ 收尾 ─────────────────────────────────────────────────────────
  │  governance      伦理合规:ethics · data_use · pii_status · ai_disclosure
  │  artifacts       交付物:figures · tables · docx · pdf · scripts
  └ 贯穿全程 ──────────────────────────────────────────────────────
     provenance      台账:每一步「哪个函数 · 什么参数 · 产出什么」自动记录,自带可复现审计轨迹
```

整项研究的来龙去脉都在一个地方,每一步自动记进台账 `provenance`,所以结果**天然可追溯、可复现**——这正是写论文、被审稿、和别人接手时最需要的。

> 熟悉生物信息学的话:`StudyState` 之于社科分析,约等于 **AnnData 之于单细胞分析**——都是那个"贯穿始终、装下整项研究"的标准对象。只不过社科数据(问卷 ≠ 语料 ≠ 网络)塞不进一个矩阵,所以它组织的是一项研究的**各个组成部分**,而不是一个数据矩阵。

日常你其实只跟它打两种交道,其余都由分析函数自动读写,不必记住内部结构:

```python
study = sv.StudyState()

study.write("variables", "outcome", "employment")   # ① 告诉它一件事(哪个是结果变量)

study.models["did"]          # ② 取结果:双重差分的点估计 / SE / CI / 稳健性
study.diagnostics["bacon"]   #    Goodman-Bacon 分解
study.artifacts["tables"]    #    生成的回归表
```

配套的三条命名轴:**`sv.pp`** 准备(读入 / 声明设计 / 构建语料 / 脱敏)· **`sv.tl`** 分析(因果 / 回归 / 测量 / 多层 / 空间 / 网络 / 质性)· **`sv.pl`** 出图出表(森林图 / 事件研究图 / 生存曲线 / 出版级回归表)。

---

## 快速上手

理解了那个 `study` 对象,上手就是"往里写、让函数跑、取结果"。以一个**双重差分(DiD)**为例——声明面板设计、检验平行趋势、估计效应、出事件研究图,只要几行:

```python
import socialverse as sv
import pandas as pd

df = pd.read_csv("policy_panel.csv")          # 你的面板数据(每行一个 单位×年份)

study = sv.StudyState()                        # 装下整项研究的对象
study.write("variables", "outcome", "employment")
sv.pp.ingest(study, data=df)
sv.pp.declare_design(study, panel_id="state", time="year",
                     treatment="treated", first_treated="reform_year")

sv.tl.parallel_trends(study)                   # 先检验平行趋势
sv.tl.did(study)                               # 双重差分 ATT(聚类稳健 SE + 稳健性)
sv.pl.event_study_plot(study)                  # 事件研究图

print(study.models["did"])                     # 点估计、置信区间、多口径 SE 一并给出
```

---

## 能做什么(按研究任务)

### 因果推断

| 方法                              | 一句话                                      | 函数                                                         |
| --------------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| 双重差分 DiD / 事件研究           | 面板政策评估的主力,自带聚类 SE 与稳健性     | `sv.tl.did` · `sv.tl.event_study`                            |
| 平行趋势检验                      | DiD 的前置诊断                              | `sv.tl.parallel_trends`                                      |
| 反事实插补 FEct/IFEct             | 现代异质稳健 DiD,修正交错采纳的负权重偏误   | `sv.tl.fect`                                                 |
| Sun-Abraham / 两步 DiD / 局部投影 | 交互加权事件研究、Gardner 两步、LP 脉冲响应 | `sv.tl.sun_abraham` · `sv.tl.did2s` · `sv.tl.local_projection` |
| Goodman-Bacon 分解                | 诊断 TWFE-DiD 的"禁忌比较"权重              | `sv.tl.bacon_decompose`                                      |
| 合成控制 / 合成 DiD               | 加权对照拟合反事实路径                      | `sv.tl.synthetic_control` · `sv.tl.synth_did`                |
| 断点回归 RDD                      | 断点处局部多项式跳跃                        | `sv.tl.rdd`                                                  |
| 工具变量 / 2SLS / Shift-share     | 两阶段最小二乘、Bartik 移份额工具           | `sv.tl.iv_regress` · `sv.tl.bartik_iv`                       |
| 倾向得分匹配                      | 近邻匹配 + 平衡诊断                         | `sv.tl.psm`                                                  |
| 中介分析                          | 直接/间接效应 bootstrap 分解                | `sv.tl.mediation`                                            |
| 因果图识别 + 反驳                 | DAG→后门/前门/IV 识别 + 安慰剂等敏感性反驳  | `sv.tl.dag_identify` · `sv.tl.dag_refute`                    |
| 异质处理效应(CATE)                | 双重机器学习、因果森林、S/T/X 元学习器      | `sv.tl.dml` · `sv.tl.causal_forest` · `sv.tl.metalearners`   |
| 分位处理效应                      | 效应在结果分布各分位上的差异                | `sv.tl.qte`                                                  |
| Honest-DiD 敏感性                 | 结论对平行趋势违背的稳健性                  | `sv.tl.honest_did`                                           |

### 回归建模

| 方法                                              | 函数                            |
| ------------------------------------------------- | ------------------------------- |
| 线性 / logit / probit / poisson(GLM,稳健/聚类 SE) | `sv.tl.glm`                     |
| 多项 / 有序 logit                                 | `sv.tl.mlogit` · `sv.tl.ologit` |
| 边际效应(AME)                                     | `sv.tl.margins`                 |

### 测量与量表

| 方法                                         | 函数                      |
| -------------------------------------------- | ------------------------- |
| 验证性 / 探索性因子分析                      | `sv.tl.cfa` · `sv.tl.efa` |
| 结构方程模型                                 | `sv.tl.sem`               |
| 项目反应理论 IRT                             | `sv.tl.irt`               |
| 信度(Cronbach α / McDonald ω / ICC)          | `sv.tl.reliability`       |
| 评分者间一致(Cohen/Fleiss κ、Krippendorff α) | `sv.tl.interrater`        |

### 复杂抽样

| 方法                        | 函数                                           |
| --------------------------- | ---------------------------------------------- |
| 声明抽样设计(权重/分层/PSU) | `sv.pp.declare_design` · `sv.tl.design_survey` |
| 设计基础加权估计            | `sv.tl.survey_estimate`                        |

### 多层 / 生存

| 方法                                                 | 函数               |
| ---------------------------------------------------- | ------------------ |
| 多层(混合效应)模型                                   | `sv.tl.multilevel` |
| 生存分析(Cox / KM / 时变协变量 / log-rank / PH 诊断) | `sv.tl.survival`   |

### 空间 / 网络 / 集合论

| 方法                             | 函数                                                  |
| -------------------------------- | ----------------------------------------------------- |
| 空间自相关(Moran)/ 空间回归(SAR) | `sv.tl.spatial_autocorr` · `sv.tl.spatial_regression` |
| 网络构建 / ERGM / 随机行动者模型 | `sv.tl.build_network` · `sv.tl.ergm` · `sv.tl.saom`   |
| 定性比较分析 QCA(fsQCA)          | `sv.tl.qca`                                           |

### 人口学 / 不平等

| 方法                               | 函数                                       |
| ---------------------------------- | ------------------------------------------ |
| 生命表 / 人口分解(Kitagawa)        | `sv.tl.life_table` · `sv.tl.decomposition` |
| Oaxaca-Blinder 分解(工资差距/歧视) | `sv.tl.oaxaca`                             |

### 质性 / 文本 / 人文

| 方法                                      | 函数                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| 主题编码 / 引语溯源 / 反身备忘            | `sv.tl.code_themes` · `sv.tl.trace_quotes` · `sv.tl.reflexive_memo` |
| 理论透镜(Foucault / Bourdieu / Weber)     | `sv.tl.foucault_discourse` · `sv.tl.bourdieu_field` · `sv.tl.weber_ideal_type` |
| 校勘 / TEI 编码 / 文体计量(Burrows Delta) | `sv.tl.philology_collate` · `sv.tl.tei_encode` · `sv.tl.stylometry` |

### 出图与制表

森林图、事件研究图、生存曲线、RDD 图、Moran 散点、合成控制路径、树状图……都在 `sv.pl.*`；还能一键出**出版级回归表**(booktabs LaTeX / Markdown / 纯文本):

```python
sv.pl.regtable(study, models=[("TWFE", study.models["did"]),
                              ("FEct", study.models["fect"])], format="latex")
```

### 治理与文献

- **治理**:伦理检查、数据使用合规、AI 使用披露 —— `sv.gov.ethics_check` · `sv.gov.data_use_check` · `sv.gov.ai_use_disclosure`
- **文献**:免费文献检索、引文真伪核验(防幻觉)、参考管理 —— `sv.lit.search_free` · `sv.lit.verify_citations` · `sv.lit.citation_manage`

---

## 熟悉 R / Stata / SPSS?

按你已经会的命令名就能找到对应方法——每个方法都带 `py-<命令>` 别名(`py-` 表示 Python 重构版):

```python
sv.tl.multilevel   # = R lme4::lmer / Stata mixed        （别名 py-lmer / py-mixed）
sv.tl.survival     # = Stata stcox / R survival::coxph    （别名 py-stcox / py-coxph）
sv.tl.cfa          # = R lavaan                           （别名 py-lavaan）
sv.tl.rdd          # = rdrobust                           （别名 py-rdrobust）
```

> 完整的「Stata / SPSS / R 命令 × socialverse 方法」对照表见 [docs/README-full.md](../docs/README-full.md)。

---

## 为什么用 socialverse

- **方法齐全**:一个包覆盖社科主要的定量 + 质性方法,不必在十几个库之间切换、拼接。
- **命名统一**:`pp`(准备)/ `tl`(分析)/ `pl`(出图)三条命名轴,一致好记。
- **结果可复现**:每步分析都记录在同一个研究对象上,估计与稳健性一并输出,直接可写进论文。
- **诚实降级**:装了哪个后端就用哪个；没装也给出方法学指引,而不是硬崩。

---

## 引用与许可

- 许可:GPL-3.0-or-later
- 项目主页:<https://github.com/omicverse/socialverse> · PyPI:<https://pypi.org/project/socialverse/>
- 方法出处:每个函数的文档字符串标注了对应的原始学术文献,请在论文中引用**原方法文献**。

---

## 参考文献

1. Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., & Yang, D. (2024). Can Large Language Models Transform Computational Social Science? *Computational Linguistics*, 50(1), 237–291.
2. Abdurahman, S., Ziabari, A. S., Moore, A. K., Bartels, D. M., & Dehghani, M. (2025). A Primer for Evaluating Large Language Models in Social-Science Research. *Advances in Methods and Practices in Psychological Science*.
3. Zhu, Y., et al. (2024). Are Large Language Models Good Statisticians? *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track*. (StatQA)
4. Hu, C., Zhang, L., Lim, Y., Wadhwani, A., Peters, A., & Kang, D. (2025). REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research? *Findings of the Association for Computational Linguistics: ACL 2025*.
5. Siegel, Z. S., Kapoor, S., Nadgir, N., Stroebl, B., & Narayanan, A. (2025). CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark. *arXiv preprint / CORE-Bench project*.
6. Hu, X., Zhao, Z., Wei, S., Chai, Z., Ma, Q., Wang, G., Wang, X., Su, J., Xu, J., Zhu, M., Cheng, Y., Yuan, J., Li, J., Kuang, K., Yang, Y., Yang, H., & Wu, F. (2024). InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*.
7. Hong, S., Lin, Y., Liu, B., et al. (2025). Data Interpreter: An LLM Agent for Data Science. *Findings of the Association for Computational Linguistics: ACL 2025*.
8. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *Advances in Neural Information Processing Systems 36*.
9. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations (ICLR 2023)*.
