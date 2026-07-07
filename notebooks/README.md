# socialverse 教程

21 本可端到端运行、**带真实输出**的教学 notebook。风格参照 [omicverse_guide](https://github.com/Starlitnightly/omicverse):每本先把一种社会科学 / 人文的分析方法讲清楚——它解决什么问题、关键前提是什么、要走哪几步——再用 `socialverse` 顺手地跑通它,结尾留下一份可复现的证据链。多数用内置玩具数据([`socialverse.datasets`](../socialverse/datasets/)),已在真实环境执行,输出与图表齐全。

> **[21 · 复现因果 ML 教科书案例:401(k) 资格对家庭财富(DML + 异质效应 + DAG)](21_reproduction_401k_dml_cate.ipynb)** 用 SIPP 1991 **公开微数据**端到端复现 Chernozhukov 等 (2018) 的 DML 运行案例,走一整条**现代因果链**:因果图识别 `sv.tl.dag_identify` → 双重机器学习 `sv.tl.dml` → 因果森林异质效应 `sv.tl.causal_forest` → 反驳 `sv.tl.dag_refute`。socialverse 的 **DML ATE ≈ \$9,900 与报告的 ~\$9,000 吻合**(朴素差 \$19.5k 偏高一倍),森林揭示效应随收入从 ~\$0 到 ~\$22k 的强异质。这本展示 0.3.0 新补的**现代因果三大件**(对标 DoWhy/EconML,全原生实现)。
>
> **[20 · 复现方法复杂的顶刊论文:交错采纳 DiD(直接民主与移民归化)](20_reproduction_hh2015_staggered_did.ipynb)** 用 Harvard Dataverse 上**公开的 1211 市镇面板**端到端复现 Liu-Wang-Xu (2024, *AJPS*) 的 HH2015 案例:声明面板设计 → 平行趋势 → **TWFE DiD** → 事件研究 → **反事实插补估计量 `sv.tl.fect`** → 稳健性。socialverse 的 **TWFE ATT = +1.339 与论文逐位吻合**,`sv.tl.fect` 的反事实插补 ATT(bootstrap SE ≈ 论文 0.197、placebo 同样不显著)把 TWFE 往异质稳健方向修正;这本顺带把 DiD 家族升级成**吸收高维固定效应**(千级面板可跑)、并补进**反事实插补估计量**(FEct/IFEct)。
>
> **[19 · 复现近年顶刊(JPSP 2023)的中介效应](19_reproduction_jpsp2023_mediation.ipynb)** 用作者**公开的原始数据**复现一篇 2023 年 JPSP 论文的核心中介发现,`sv.tl.mediation` 估的间接效应 **ACME≈0.31 [0.08,0.53] 与发表值逐位吻合**——近年顶刊 + 公开数据版的"论文=函数链"。
>
> **[18 · 完整复现:Rossi 累犯实验的 Cox 模型](18_reproduction_rossi_cox.ipynb)** 用**真实公开数据**(Rossi 随机实验)端到端复现一项已发表研究的**整套分析**(含时变就业 Cox),socialverse 系数与 Allison (2014) 发表值**逐位吻合(最大偏差 < 0.002)**——"一篇论文 = 一条 socialverse 函数链"的字面证明。

每本都配套一个 jupytext `.py`(干净可 diff 的源)与一个已执行的 `.ipynb`(含输出与图)。

## 怎么运行

```bash
pip install -e ".[full]"     # numpy / pandas / statsmodels / scipy / networkx / matplotlib
jupyter lab notebooks/       # 逐格运行;或直接阅读已执行的 .ipynb
```

## 入门

| # | 教程 |
|---|---|
| 01 | [快速上手:用一个最小分析认识 StudyState 与函数注册表](01_registry_and_studystate.ipynb) |

## 回归底座与因果推断

| # | 教程 | 对标 |
|---|---|---|
| 17 | [回归底座与因果工具箱:GLM、工具变量、匹配与中介](17_regression_iv_matching_mediation.ipynb) | R `glm`/`ivreg`/`MatchIt`/`mediation` · Stata `logit`/`ivregress`/`psmatch2`/`mediate` |
| 02 | [用双重差分评估一项政策的因果效应](02_causal_did.ipynb) | pyfixest · R `fixest`/`did` |
| 20 | [复现顶刊交错采纳 DiD:直接民主与移民归化(HH2015)](20_reproduction_hh2015_staggered_did.ipynb) | Liu-Wang-Xu 2024 AJPS · R `fect`/`did` |
| 21 | [复现因果 ML 教科书案例:401(k)→财富(DML+因果森林+DAG)](21_reproduction_401k_dml_cate.ipynb) | Chernozhukov 2018 · DoWhy / EconML / DoubleML |
| 11 | [没有随机分配时,如何识别因果:断点回归与合成控制](11_quasi_experiment.ipynb) | `rdrobust` · `gsynth` |
| 04 | [把一篇实证论文打包成可复现的复现件](04_econometrics_replication.ipynb) | R `fixest` + 复现管线 |

## 调查与测量

| # | 教程 | 对标 |
|---|---|---|
| 03 | [从一份复杂抽样调查里,得到面向总体的诚实推断](03_complex_survey.ipynb) | samplics · R `survey` |
| 12 | [心理测量:用因子分析、IRT 和结构方程反推看不见的构念](12_psychometrics.ipynb) | semopy · R `lavaan`/`mirt` |
| 13 | [嵌套数据与时间到事件:多层模型与生存分析](13_multilevel_survival.ipynb) | R `lme4`/`survival` |

## 文本、网络与空间

| # | 教程 | 对标 |
|---|---|---|
| 07 | [用理论透镜读文本、用网络分析读关系](07_theory_lens_network.ipynb) | networkx/igraph + 学术传统 |
| 16 | [从一张社交网络里读出结构,再从文字里认出作者](16_networks_stylometry.ipynb) | R `ergm`/`RSiena` · `stylo` |
| 14 | [空间数据的自相关与空间回归](14_spatial_analysis.ipynb) | PySAL · R `spdep` |
| 06 | [从扫描件到校勘本:OCR、TEI 编码与抄本谱系重建](06_text_philology.ipynb) | Tesseract/Kraken · TEI · CollateX |

## 质性与组态

| # | 教程 | 对标 |
|---|---|---|
| 05 | [给一批访谈做质性编码:去标识、主题分析与引语溯源](05_qualitative_coding.ipynb) | CAQDAS(NVivo/QualCoder) |
| 15 | [组态与分解:定性比较分析(QCA)与形式人口学](15_qca_demography.ipynb) | R `QCA` · `demography` |

## 治理、文献与研究闭环

| # | 教程 | 对标 |
|---|---|---|
| 08 | [在跑分析之前,先过研究治理这三道闸门](08_governance_gates.ipynb) | sdcMicro · 期刊政策 |
| 09 | [核验一份参考文献,揪出稿件里的幻觉引用](09_literature_citation.ipynb) | Zotero + CrossRef/OpenAlex |
| 10 | [一次可复核的小型研究:从伦理审查到证据链](10_full_study_evidence_chain.ipynb) | socialverse 的差异化收束 |

---

这 16 本覆盖 registry 现有全部函数。每种方法的依赖契约(requires/produces/prerequisites/auto_fix)与对标的现实 Py/R 包,见 [../docs/CONTRACT_CARDS.md](../docs/CONTRACT_CARDS.md);整个人文社科计算生态的调研与设计依据,见 [../docs/LANDSCAPE.md](../docs/LANDSCAPE.md)。
