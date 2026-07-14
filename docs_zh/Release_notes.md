# 发布说明

## 0.7.2

后端对等性发布：四个剩余的接线缝隙已关闭，并添加和注册了一批之前缺失的R函数。

- **已接线到对等性门控端口的缝隙**：Kaplan–Meier现在使用R精确实现`pysurvival.km`（Greenwood标准误 + 置信带）；`survey_estimate`现在展示设计基础总计（`svytotal`）；`efa(method="pa")`选择R `psych::fa(fm="pa")`主轴因子分析（默认保持PCA）。
- **跨端口新增26个函数**，各自拥有自己的`1e-6`对等性门控——例如用于荟萃分析的单项研究BLUP、`survey_by` / `survey_ratio` / `survey_ciprop`、条件logit与参数AFT生存分析、ICC及相关性检验矩阵、直接模糊集校准、ERGM统计量 / 三元组普查、具有高维固定效应的Poisson-PMLE等。
- **9个新注册表条目**；`sv.tl`现在公开134个注册函数。
- 外部对等性门控：确定性核心上**115项测试，均在`1e-6`时通过**；完整套件**181项通过**，无回归。

## 0.7.1

**`socialverse.external`** 层首次发布——14个R包已按照[omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr)协议重构为对等性门控的纯`numpy`/`scipy`端口，并接线以替代之前的近似实现：

`pymetafor` · `pysurvey` · `pysurvival` · `pydemography` · `pyqca` · `pyfixest` · `pyrobumeta` · `pynetmeta` · `pypsych` · `pylavaan` · `pymatchit` · `pymada` · `pydid` · `pyergm`。

跨14个端口的70项对等性测试，在各包的确定性核心上均在`max_abs_err < 1e-6`时通过。随机分量（自助SE、MCMC-MLE）作为参考容差而非`1e-6`门控被记录。详见各[教程](tutorials/external/index.md)。

## 更早版本

- **0.6.x** — 原生荟萃分析模块（三层、约96个函数）、图形样式（`sv.style` / `sv.pl`）及OSF论文复现笔记本。
- **0.2.x / 0.1.x** — 核心引擎（注册表 + `StudyState` + 槽约定）及首批分析模块（`pp` / `tl` / `pl` / `gov` / `lit`）。
