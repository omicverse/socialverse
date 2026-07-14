# 教程

socialverse 提供两类互补的教程。在**侧边栏中选择一个类别**查看相关内容，或从下面的卡片开始。

- **引导式笔记本** — 可执行、端到端的真实分析工作流演练，从原始数据到防守性的、溯源追踪的结果。
- **R 包移植** — 每个 `socialverse.external` 移植的参考教程：R 包的功能、Python API、R↔Python 字典，以及 `1e-6` 精度证据。
  查看[移植概览](external/index.md)了解完整的精度对照表。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 快速入门
:link: notebooks/01_registry_and_studystate.html
注册表 + `StudyState`，以及作为一个完整可追踪证据链的完整研究。
:::

:::{grid-item-card} 🎯 因果推断
:link: notebooks/02_causal_did.html
DiD、准实验（RDD / 合成对照）、IV、匹配、中介。
`pydid` · `pymatchit` · `pyfixest`。
:::

:::{grid-item-card} 📊 元分析
:link: notebooks/22_meta_analysis_basics.html
基础 → 多层级/稳健 → 元回归 → 出版偏差 → 网络元分析。
`pymetafor` · `pynetmeta` · `pyrobumeta` · `pymada`。
:::

:::{grid-item-card} 🧪 复杂调查与计量经济学
:link: notebooks/03_complex_survey.html
设计化调查估计与计量经济学复现。`pysurvey`。
:::

:::{grid-item-card} 🧭 心理计量学与生存分析
:link: notebooks/12_psychometrics.html
信度、因子分析、SEM；Kaplan–Meier、Cox、AFT。`pypsych` · `pylavaan` · `pysurvival`。
:::

:::{grid-item-card} 🗺 空间分析、网络、QCA、人口统计学
:link: notebooks/14_spatial_analysis.html
空间分析、网络 + 风格学、QCA + 人口统计学。`pyqca` · `pyergm` · `pydemography`。
:::

:::{grid-item-card} 📜 定性研究、文本与文献学
:link: notebooks/05_qualitative_coding.html
主题编码及引文追踪；文本缩放、风格学和文献学。
:::

:::{grid-item-card} ⚖️ 治理与文献
:link: notebooks/08_governance_gates.html
研究-治理门槛、引用/文献工作流、系统评价治理。
:::

:::{grid-item-card} 🔁 端到端复现
:link: notebooks/18_reproduction_rossi_cox.html
完整论文复现 — Rossi/Cox、JPSP 中介、错开 DiD、401(k) DML、ECR 患病率。
:::

::::