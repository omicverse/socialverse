# socialverse 契约卡 (Contract Cards)

registry 全部 **54 个函数**的依赖契约,外加每个对标的现实 Python/R 冠军包。
自动从 `sv.registry.manifest()` 生成 —— 这既是文档,也是注册表的人类可读视图(对标 R 的 CRAN Task Views「策划目录」思路)。

> 读法:`requires → produces` 是机器可读的依赖契约(槽位见 [StudyState 词汇表](../README.md));`auto_fix` = 缺前置时的策略;「对标」列指出同类功能在现实生态里的权威实现。


## 准备 (sv.pp)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.pp.build_corpus** | community | `sources[corpora]` → `corpus[documents,units,manifest]` · `evidence[provenance]` | `escalate` | spaCy / textacy / quanteda::corpus |
| **sv.pp.ocr_tei** | plus | `sources[scans]` → `corpus[documents,tei]` · `artifacts[xml]` · `evidence[provenance]` | `auto` | pytesseract (Tesseract) / Kraken / eScriptorium |
| **sv.pp.redact_pii** | community | `corpus[documents]` → `corpus[documents]` · `governance[pii_status]` | `auto` | Presidio / spaCy NER / — |
| **sv.pp.declare_design** | plus | `sources[datasets]` → `design[panel_id,time,treatment,first_treated,weights,strata,psu,unit]` | `escalate` | samplics / survey::svydesign |
| **sv.pp.ingest** | community | ∅ → `sources[datasets]` | `none` | pandas / base/tibble |

## 因果计量 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.did** | plus | `design[panel_id,time,treatment]` · `variables[outcome]` · `identification[parallel_trends]` → `models[did,twfe]` · `diagnostics[robustness]` | 先跑 parallel_trends · `escalate` | pyfixest / linearmodels / fixest · did (Callaway–Sant Anna) |
| **sv.tl.event_study** | plus | `design[panel_id,time,treatment,first_treated]` · `variables[outcome]` → `models[event_study]` | `escalate` | pyfixest / fixest::sunab / did2s |
| **sv.tl.parallel_trends** | plus | `design[panel_id,time,treatment,first_treated]` · `variables[outcome]` · `estimand[target]` → `diagnostics[pretrend]` · `identification[parallel_trends]` | `escalate` | linearmodels / pyfixest / fixest / did |

## 准实验 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.rdd** | plus | `sources[datasets]` · `variables[outcome]` · `estimand[target]` → `models[rdd]` · `diagnostics[bandwidth]` | `escalate` | statsmodels local-linear / rdrobust |
| **sv.tl.synthetic_control** | pro | `sources[datasets]` · `design[treatment,time]` · `variables[outcome]` · `estimand[target]` → `models[synth]` · `diagnostics[pre_fit]` | `escalate` | scipy SLSQP / gsynth / augsynth / Synth |

## 复杂抽样 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.design_survey** | plus | `estimand[target]` → `design[sampling_frame]` · `variables[scales,constructs]` · `diagnostics[reliability,power]` | `none` | factor_analyzer / pingouin / psych / lavaan |
| **sv.tl.survey_estimate** | plus | `sources[datasets]` · `design[weights]` · `variables[outcome]` → `models[weighted_reg]` · `diagnostics[sensitivity]` · `artifacts[tables]` | `escalate` | samplics / survey (Lumley) / srvyr |

## 实证复现 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.replicate** | pro | `sources[datasets]` · `design[treatment]` · `estimand[target]` · `identification[strategy]` → `variables[controls]` · `models[twfe]` · `diagnostics[robustness,balance]` · `artifacts[scripts,tables]` | 先跑 did · `escalate` | pyfixest + statsmodels / fixest (+ targets/Quarto) |

## 心理测量 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.cfa** | plus | `sources[datasets]` → `models[cfa]` · `diagnostics[fit_indices]` | `escalate` | statsmodels.Factor / semopy / lavaan / psych |
| **sv.tl.irt** | plus | `sources[datasets]` → `models[irt]` · `diagnostics[item_info]` | `escalate` | scipy (girth) / mirt / TAM |
| **sv.tl.sem** | pro | `sources[datasets]` → `models[sem]` · `diagnostics[fit_indices]` | `escalate` | semopy / lavaan |

## 多层/生存 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.multilevel** | plus | `sources[datasets]` · `variables[outcome]` → `models[mixedlm]` · `diagnostics[variance_components]` | `escalate` | statsmodels.MixedLM / lme4 / brms |
| **sv.tl.survival** | plus | `sources[datasets]` · `variables[outcome]` → `models[cox,km]` · `diagnostics[ph_test]` | `escalate` | statsmodels.PHReg (Cox) / survival |

## 空间 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.spatial_autocorr** | plus | `sources[datasets]` → `diagnostics[moran]` · `models[lisa]` | `escalate` | numpy / PySAL esda / spdep |
| **sv.tl.spatial_regression** | pro | `sources[datasets]` · `variables[outcome]` → `models[sar]` · `diagnostics[spatial]` | `escalate` | numpy ML / spreg / spatialreg |

## 定性比较 QCA (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.qca** | plus | `sources[datasets]` · `variables[outcome]` → `models[qca]` · `diagnostics[consistency_coverage]` | `escalate` | (pure python) / QCA / SetMethods |

## 人口学 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.decomposition** | pro | `sources[datasets]` → `models[decomposition]` · `diagnostics[components]` | `escalate` | numpy / statsmodels / demography / oaxaca |
| **sv.tl.life_table** | plus | `sources[datasets]` → `models[life_table]` | `none` | numpy / demography / MortalityLaws |

## 质性方法 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.code_themes** | plus | `corpus[units]` → `codes[codebook,segments,themes,theme_map]` · `evidence[claim_evidence]` · `artifacts[tables]` | `escalate` | (none — CAQDAS gap) / RQDA / (NVivo, ATLAS.ti) |
| **sv.tl.reflexive_memo** | community | `codes[themes]` · `corpus[units]` → `evidence[provenance]` · `governance[ethics]` | `none` | (none — methodology) / (none) |
| **sv.tl.trace_quotes** | community | `corpus[units]` · `codes[segments]` → `evidence[quote_index]` | 先跑 code_themes · `escalate` | (none) / (none — standoff) |

## 文本/校勘 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.philology_collate** | plus | `corpus[documents]` → `models[stemma]` · `evidence[provenance]` · `artifacts[apparatus]` | `escalate` | collatex / (CollateX / Juxta) |
| **sv.tl.tei_encode** | plus | `corpus[documents]` → `corpus[tei]` · `artifacts[xml]` · `evidence[provenance]` | `escalate` | lxml / TEI-P5 / xml2 |

## 理论透镜 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.bourdieu_field** | plus | `codes[themes]` · `variables[constructs]` → `models[field_map]` · `evidence[claim_evidence]` | `escalate` | prince (MCA) / FactoMineR / ca |
| **sv.tl.foucault_discourse** | plus | `corpus[units]` → `evidence[claim_evidence]` | 先跑 code_themes · `auto` | (none — theory lens) / (none) |
| **sv.tl.weber_ideal_type** | plus | `sources[datasets]` → `models[ideal_type]` · `diagnostics[coverage]` · `governance[ethics]` · `evidence[claim_evidence]` | `escalate` | (none — theory lens) / (none) |

## 网络 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.build_network** | plus | `sources[datasets]` → `models[network]` · `diagnostics[coverage]` | `none` | networkx / igraph / igraph / statnet |
| **sv.tl.ergm** | pro | `sources[datasets]` → `models[ergm]` · `diagnostics[gof]` | `escalate` | statsmodels MPLE / ergm / statnet |
| **sv.tl.saom** | pro | `sources[datasets]` → `models[saom]` · `diagnostics[coevolution]` | `escalate` | numpy (descriptive) / RSiena |

## 文体计量 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.stylometry** | plus | `corpus[documents]` → `models[stylometry]` · `artifacts[figures]` | `escalate` | scipy hierarchical / stylo |

## 绘图 (sv.pl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.pl.event_study_plot** | community | `models[event_study]` → `artifacts[figures]` | `escalate` | matplotlib / ggplot2 / did |
| **sv.pl.forest** | community | `models[did]` → `artifacts[figures]` | `escalate` | matplotlib / seaborn / ggplot2 / dotwhisker |
| **sv.pl.manuscript_docx** | community | `sources[datasets]` → `artifacts[docx,pdf]` · `diagnostics[coverage]` | `auto` | python-docx + pandoc / officer / rmarkdown |
| **sv.pl.survey_dist** | community | `models[weighted_reg]` → `artifacts[figures]` | `escalate` | matplotlib / seaborn / ggplot2 / survey |
| **sv.pl.theme_map** | community | `codes[theme_map]` → `artifacts[figures]` | `escalate` | networkx + matplotlib / igraph / ggraph |
| **sv.pl.dendrogram** | community | `models[stylometry]` → `artifacts[figures]` | `escalate` | scipy + matplotlib / stylo / ggdendro |
| **sv.pl.km_curve** | community | `models[km]` → `artifacts[figures]` | `escalate` | matplotlib / survminer |
| **sv.pl.moran_scatter** | community | `diagnostics[moran]` → `artifacts[figures]` | `escalate` | matplotlib / spdep::moran.plot |
| **sv.pl.rdd_plot** | community | `models[rdd]` → `artifacts[figures]` | `escalate` | matplotlib / rdrobust::rdplot |
| **sv.pl.synth_path** | community | `models[synth]` → `artifacts[figures]` | `escalate` | matplotlib / gsynth::plot |

## 治理 (sv.gov)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.gov.ai_use_disclosure** | community | ∅ → `governance[ai_disclosure]` · `artifacts[tables]` · `evidence[provenance]` | `escalate` | (none — governance) / (none) |
| **sv.gov.data_use_check** | community | `sources[datasets]` → `governance[data_use]` | `escalate` | (none — governance) / (none) |
| **sv.gov.ethics_check** | community | `design[unit]` → `governance[ethics]` | `escalate` | (none — governance) / sdcMicro (k-anonymity) |

## 文献引证 (sv.lit)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.lit.citation_manage** | community | `sources[bib]` → `evidence[citations]` · `artifacts[tables]` | `none` | (none — methodology) / RefManageR |
| **sv.lit.literature_map** | community | `sources[bib]` → `evidence[landscape]` · `artifacts[figures]` | `auto` | (none — methodology) / (none) |
| **sv.lit.manuscript_review** | community | `sources[datasets]` · `evidence[verified_bib]` → `evidence[claim_evidence]` · `diagnostics[coverage]` · `artifacts[tables]` | 先跑 verify_citations · `auto` | (none — methodology) / (none) |
| **sv.lit.search_free** | community | ∅ → `sources[bib]` · `evidence[citations]` | `none` | requests (NCBI E-utils) / rentrez / europepmc |
| **sv.lit.zotero_bridge** | plus | `sources[bib]` → `sources[bib]` | `auto` | pyzotero (+ Zotero MCP) / (none) |
| **sv.lit.verify_citations** | community | `sources[bib]` → `evidence[verified_bib]` | `escalate` | (none — CrossRef/OpenAlex) / (none) |


---

**曾经的缺口现已全部实现**(真实计算 + 懒加载 champion 后端 fallback,见上表 quasi / psychometrics / longitudinal / spatial / setmethods / demography / stylometry / net 各类):RDD · 合成控制 · CFA/SEM · IRT · 多层 HLM · 事件史/Cox · 空间自相关/SAR · fsQCA · 生命表/Kitagawa 分解 · 文体计量 Delta · ERGM。诚实标注的近似:**ERGM = MPLE**(非 MCMC-MLE)· **SAOM = 描述性简化版**(非基于模拟的完整估计)· **SEM latent 不可用时退化为 path analysis**。全部经 `tests/test_gap_methods.py` 验证能复原已知 DGP 参数。
