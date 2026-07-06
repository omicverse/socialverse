# socialverse 契约卡 (Contract Cards)

registry 全部 **34 个函数**的依赖契约,外加每个对标的现实 Python/R 冠军包。
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

## 复杂抽样 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.design_survey** | plus | `estimand[target]` → `design[sampling_frame]` · `variables[scales,constructs]` · `diagnostics[reliability,power]` | `none` | factor_analyzer / pingouin / psych / lavaan |
| **sv.tl.survey_estimate** | plus | `sources[datasets]` · `design[weights]` · `variables[outcome]` → `models[weighted_reg]` · `diagnostics[sensitivity]` · `artifacts[tables]` | `escalate` | samplics / survey (Lumley) / srvyr |

## 实证复现 (sv.tl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.tl.replicate** | pro | `sources[datasets]` · `design[treatment]` · `estimand[target]` · `identification[strategy]` → `variables[controls]` · `models[twfe]` · `diagnostics[robustness,balance]` · `artifacts[scripts,tables]` | 先跑 did · `escalate` | pyfixest + statsmodels / fixest (+ targets/Quarto) |

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

## 绘图 (sv.pl)

| 函数 | tier | requires → produces | 前置/auto_fix | 对标 Py / R |
|---|---|---|---|---|
| **sv.pl.event_study_plot** | community | `models[event_study]` → `artifacts[figures]` | `escalate` | matplotlib / ggplot2 / did |
| **sv.pl.forest** | community | `models[did]` → `artifacts[figures]` | `escalate` | matplotlib / seaborn / ggplot2 / dotwhisker |
| **sv.pl.manuscript_docx** | community | `sources[datasets]` → `artifacts[docx,pdf]` · `diagnostics[coverage]` | `auto` | python-docx + pandoc / officer / rmarkdown |
| **sv.pl.survey_dist** | community | `models[weighted_reg]` → `artifacts[figures]` | `escalate` | matplotlib / seaborn / ggplot2 / survey |
| **sv.pl.theme_map** | community | `codes[theme_map]` → `artifacts[figures]` | `escalate` | networkx + matplotlib / igraph / ggraph |

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

缺口(现实有权威包、socialverse 待补,见 [LANDSCAPE.md](LANDSCAPE.md) 第 4 节):**SEM/CFA**(lavaan/semopy)· **IRT**(mirt)· **RDD**(rdrobust)· **合成控制**(gsynth)· **多层 HLM**(lme4)· **事件史**(survival)· **空间**(spdep/PySAL)· **ERGM/SAOM**(ergm/RSiena,Python 原生空白=高护城河)· **QCA**(fsQCA)· **人口学分解**· **文体计量 Delta**(stylo)。