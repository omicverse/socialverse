# socialverse

**AI时代社会科学研究的入口。**

从政策评估、量表开发和复杂调查，到空间分析、网络分析和定性编码，**socialverse** 将社会科学中最常用的方法——也是AI智能体最容易滥用的方法——组织成**一个可复现的研究工作台**：数据、研究设计、模型、诊断、证据链、图表和论文成果都被记录、调用和审计在一个单一对象中。

它是为经济学、政治学、社会学、心理学、传播学、人口统计学、公共卫生和数字人文等领域的研究者设计的。你不需要计算机科学背景就能理解它为你做了什么——如果你使用AI智能体，socialverse为模型提供了**清晰、可查询、可执行的分析接口**，而不是让它从记忆中"发明"命令。

socialverse是**AI4S（AI用于科学）**——**AI4Social**——的社会科学部分，是AI生物学基础设施[omicverse](https://github.com/omicverse/omicverse)的姊妹项目：相同的注册表驱动设计，不同的领域。

```{button-link} tutorials/index.html
:color: primary
:shadow:
浏览教程 →
```

## AI时代的社会科学基础设施

前沿语言模型已经足够有用，能够阅读文献、生成代码、解释统计结果、协助定性编码和复现论文。但当要求它们**直接执行**社会科学分析时，真正的失败点往往不是"模型能说话吗？"——而是**应该使用哪种方法、假设是否成立、结果是否可复现、以及结论是否得到证据支持**。

socialverse就是那个缺失的基础：**可靠、可查询、可执行、可组合和可审计**的分析层。研究者直接运行标准化工作流；AI智能体通过显式工具契约和方法论约束来调用它们，而不是临时拼凑工作流或产出似是而非但无法溯源的结论。

## 它能帮助你做什么

| 你的场景 | 原来的痛点 | 用socialverse |
| --- | --- | --- |
| **政策评估**——改革是否产生因果效应，效应有多大？ | 用Stata组装DiD、担忧平行趋势、追赶当下的"异质性稳健"方法、代码散落各处 | 声明处理组+政策时间，然后在一个链条中运行**平行趋势诊断→经典DID→现代反事实估计**——事件研究图表、论文级表格和每一步的溯源 |
| **量表/问卷开发**——可靠性、效度、维度性 | 用SPSS做因子分析、在其他地方计算可靠性、数据变化时重新开始 | **探索性因子分析/验证性因子分析、可靠性、结构方程模型和项反应理论**在一个地方，命名一致，结果为测量部分做好准备 |
| **大型加权调查**（CHARLS/NHANES/CGSS） | 忘记使用权重是错的；使用权重意味着要记住复杂的调查设计命令 | **声明一次抽样设计**（权重/分层/初级抽样单位）——之后的估计自动遵循该设计而非简单随机样本 |
| **访谈/文本**——主题编码+来源可溯的主张 | 定性和定量软件是分开的世界；混合方法=不断的导出/导入 | **主题编码、引用溯源和反思备忘录**与定量分析在同一个包中——混合方法而无需切换工具 |

## 瓶颈在于基础设施，而不仅是模型能力

现有基准测试显示差距是具体的：在**StatQA**（11,623个统计任务）上，GPT-4o的最佳成绩是**64.83%**，错误集中在*方法适用性*上——"知道方法名，但不知道何时使用它"；在**REPRO-Bench**（112个论文复现任务）上，最好的智能体仅达到**21.4%**；在**CORE-Bench**上，最难的级别接近**21%**。相反，执行环境+工具契约使智能体的能力大幅提升——**Data Interpreter**通过任务分解、代码执行和逐步验证，将InfiAgent-DABench的准确度从**75.9%→94.9%**。

教训是：可靠的**基础设施+工具契约+执行反馈+审计机制**将一个有能力的模型变成可信赖的分析工具。这正是socialverse提供的。

```{image} _static/studystate_logo.png
:alt: StudyState——socialverse核心的可复现研究对象
:width: 360px
:align: center
```

```{note}
**注册表是脊柱。** 每个分析都是一个有显式契约的注册函数——`requires`（必须先填充哪些研究槽位）、`produces`（它写回什么）和`tier`。`StudyState`将数据和证据链从一步携带到下一步，因此下游估计器**拒绝在未声明设计上运行**，而不是默默返回错误的数字。调用`sv.list_functions()`查看所有可用的。
```

## 里面有什么

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🎯 调查与因果推断
设计基础的调查估计（权重/分层/初级抽样单位）、固定效应回归、交错式DiD、断点回归/合成对照、匹配、工具变量、中介分析。
:::

:::{grid-item-card} 📊 元分析
随机/混合效应、网络、稳健方差、剂量反应和诊断准确度元分析，以及发表偏差诊断。
:::

:::{grid-item-card} 🧭 心理测量学
可靠性、级内相关系数、探索性/验证性因子分析、结构方程模型和项反应理论。
:::

:::{grid-item-card} ⏱ 生存与纵向数据
Kaplan-Meier、Cox比例风险、条件logit、参数加速失效时间模型和多水平模型。
:::

:::{grid-item-card} 🗺 空间与网络
空间自相关/空间自回归、时空重心轨迹、网络分析、指数随机图模型和定性比较分析。
:::

:::{grid-item-card} 📜 定性与文本
带引用溯源的主题编码+反思备忘录、文体学、文献对勘和OCR→锚定转录。
:::

::::

在底层，许多估计器是**奇偶性门控的端口**，来自社会科学家已经信任的R包（`metafor`、`survey`、`survival`、`lavaan`、`MatchIt`、`did`等），用纯`numpy`/`scipy`重新实现——无`rpy2`、无R运行时、无Stata许可证——每个都附带一个测试，将Python结果与R参考值固定在`1e-6`以内。参见[R包端口概览](tutorials/external/index.md)。

## 快速开始

```bash
pip install socialverse
```

```python
import socialverse as sv

sv.list_functions()     # 每个注册的分析，按类别分组

st = sv.StudyState()    # 在步骤之间携带数据+证据链
```

然后完成**引导式笔记本**（从原始数据到可溯源结果的真实分析）或侧栏中的**R包端口**参考。

```{toctree}
:hidden:
:maxdepth: 1

安装 <Installation.md>
教程总览 <tutorials/index.md>
R 包端口 <tutorials/external/index.md>
更新日志 <Release_notes.md>
```

```{toctree}
:hidden:
:caption: 快速入门
:maxdepth: 1

tutorials/notebooks/01_registry_and_studystate
tutorials/notebooks/10_full_study_evidence_chain
```

```{toctree}
:hidden:
:caption: 因果推断与准实验
:maxdepth: 1

tutorials/notebooks/02_causal_did
tutorials/notebooks/11_quasi_experiment
tutorials/notebooks/17_regression_iv_matching_mediation
tutorials/external/pydid
tutorials/external/pymatchit
tutorials/external/pyfixest
```

```{toctree}
:hidden:
:caption: 元分析与证据合成
:maxdepth: 1

tutorials/notebooks/22_meta_analysis_basics
tutorials/notebooks/23_multilevel_and_robust_meta
tutorials/notebooks/24_meta_regression_and_moderators
tutorials/notebooks/25_publication_bias_and_sensitivity
tutorials/notebooks/26_network_meta_analysis
tutorials/notebooks/27_specialized_designs
tutorials/external/pymetafor
tutorials/external/pynetmeta
tutorials/external/pyrobumeta
tutorials/external/pymada
```

```{toctree}
:hidden:
:caption: 复杂调查与流行病学
:maxdepth: 1

tutorials/notebooks/03_complex_survey
tutorials/external/pysurvey
```

```{toctree}
:hidden:
:caption: 计量经济学
:maxdepth: 1

tutorials/notebooks/04_econometrics_replication
```

```{toctree}
:hidden:
:caption: 心理测量学与测量
:maxdepth: 1

tutorials/notebooks/12_psychometrics
tutorials/external/pypsych
tutorials/external/pylavaan
```

```{toctree}
:hidden:
:caption: 生存与纵向数据
:maxdepth: 1

tutorials/notebooks/13_multilevel_survival
tutorials/external/pysurvival
```

```{toctree}
:hidden:
:caption: 空间、网络、定性比较分析与人口学
:maxdepth: 1

tutorials/notebooks/07_theory_lens_network
tutorials/notebooks/14_spatial_analysis
tutorials/notebooks/15_qca_demography
tutorials/notebooks/16_networks_stylometry
tutorials/external/pyqca
tutorials/external/pyergm
tutorials/external/pydemography
```

```{toctree}
:hidden:
:caption: 定性、文本与文献学
:maxdepth: 1

tutorials/notebooks/05_qualitative_coding
tutorials/notebooks/06_text_philology
```

```{toctree}
:hidden:
:caption: 研究治理与文献
:maxdepth: 1

tutorials/notebooks/08_governance_gates
tutorials/notebooks/09_literature_citation
tutorials/notebooks/28_systematic_review_governance
```

```{toctree}
:hidden:
:caption: 端到端复现
:maxdepth: 1

tutorials/notebooks/18_reproduction_rossi_cox
tutorials/notebooks/19_reproduction_jpsp2023_mediation
tutorials/notebooks/20_reproduction_hh2015_staggered_did
tutorials/notebooks/21_reproduction_401k_dml_cate
tutorials/notebooks/29_reproduction_ecr_multilevel_prevalence
```
