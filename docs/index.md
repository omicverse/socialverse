# socialverse

**The AI-era entry point for social science research.**

From policy evaluation, scale development, and complex surveys to spatial analysis,
network analysis, and qualitative coding, **socialverse** organizes the methods most
often used in social science — and most easily misused by AI agents — into **one
reproducible research workbench**: data, research design, models, diagnostics,
evidence chains, figures, and paper deliverables are all recorded, invoked, and
audited in a single object.

It is built for researchers in economics, political science, sociology, psychology,
communication, demography, public health, and digital humanities. You do not need a
computer-science background to understand what it is doing for you — and if you use
an AI agent, socialverse gives the model a **clear, queryable, executable analysis
interface** instead of letting it "invent" commands from memory.

socialverse is the social-science part of **AI4S (AI for Science)** — **AI4Social** —
the sibling of the AI4Bio infrastructure
[omicverse](https://github.com/omicverse/omicverse): the same registry-driven design,
a different domain.

```{button-link} tutorials/index.html
:color: primary
:shadow:
Browse the tutorials →
```

## Infrastructure for social science in the AI era

Frontier language models are already useful enough to read literature, generate code,
explain statistical results, assist qualitative coding, and reproduce papers. But when
they are asked to *do* social science analysis directly, the real failure point is
rarely "can the model talk?" — it is **which method should be used, whether the
assumptions hold, whether the result can be reproduced, and whether the conclusion is
supported by evidence.**

socialverse is that missing foundation: a **reliable, queryable, executable,
composable, and auditable** analysis layer. Researchers run standardized workflows
directly; AI agents call them through explicit tool contracts and methodological
constraints, instead of improvising workflows or producing plausible-looking but
untraceable conclusions.

## What it helps you do

| Your scenario | The old pain point | With socialverse |
| --- | --- | --- |
| **Policy evaluation** — did a reform have a causal effect, and how large? | DiD assembled in Stata, parallel-trends worries, chasing the current "heterogeneity-robust" method, code scattered across files | State the treated group + policy timing, then run **parallel-trends diagnostics → classic DID → modern counterfactual estimation** in one chain — event-study figures, paper-ready tables, and a trace for every step |
| **Scale / questionnaire development** — reliability, validity, dimensionality | Factor analysis in SPSS, reliability computed elsewhere, restart when the data changes | **EFA / CFA, reliability, SEM, and IRT** in one place, consistent naming, results ready for the measurement section |
| **Large weighted surveys** (CHARLS / NHANES / CGSS) | Forgetting weights is wrong; using them means memorizing survey-design commands | **Declare the sampling design once** (weights / strata / PSU) — later estimates automatically follow the design instead of a simple random sample |
| **Interviews / text** — thematic coding + source-traceable claims | Qualitative and quantitative software are separate worlds; mixed methods = constant export/import | **Thematic coding, quote tracing, and reflexive memos** live in the same package as quantitative analysis — mixed methods without switching tools |

## The bottleneck is infrastructure, not just model capability

Existing benchmarks show the gap is concrete: on **StatQA** (11,623 statistical tasks)
GPT-4o's best is **64.83%**, with errors concentrated in *method-applicability* — "knowing
the method name, but not when to use it"; on **REPRO-Bench** (112 paper-reproduction
tasks) the best agent reaches only **21.4%**; on **CORE-Bench** the hardest tier tops out
near **21%**. Conversely, execution environments + tool contracts amplify agents sharply —
**Data Interpreter** lifts InfiAgent-DABench accuracy from **75.9% → 94.9%** through task
decomposition, code execution, and step-by-step verification.

The lesson: reliable **infrastructure + tool contracts + execution feedback + audit
mechanisms** turn a capable model into a trustworthy analyst. That is what socialverse
provides.

```{image} _static/studystate_logo.png
:alt: StudyState — the reproducible research object at the core of socialverse
:width: 360px
:align: center
```

```{note}
**The registry is the spine.** Every analysis is a registered function with an explicit
contract — `requires` (which study slots must be populated first), `produces` (what it
writes back), and a `tier`. A `StudyState` carries the data and the evidence chain from
step to step, so a downstream estimator **refuses to run on an undeclared design** rather
than silently returning a wrong number. Call `sv.list_functions()` to see everything
available.
```

## What's inside

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🎯 Survey & causal
Design-based survey estimation (weights / strata / PSU), fixed-effects regression,
staggered DiD, RDD / synthetic control, matching, IV, mediation.
:::

:::{grid-item-card} 📊 Meta-analysis
Random / mixed-effects, network, robust-variance, dose–response, and
diagnostic-accuracy meta-analysis, with publication-bias diagnostics.
:::

:::{grid-item-card} 🧭 Psychometrics
Reliability, ICC, exploratory / confirmatory factor analysis, SEM, and IRT.
:::

:::{grid-item-card} ⏱ Survival & longitudinal
Kaplan–Meier, Cox PH, conditional logit, parametric AFT, and multilevel models.
:::

:::{grid-item-card} 🗺 Spatial & network
Spatial autocorrelation / SAR, spatial-temporal center-of-gravity trajectories,
network analysis, ERGM, and QCA.
:::

:::{grid-item-card} 📜 Qualitative & text
Thematic coding with quote tracing + reflexive memos, stylometry, philological
collation, and OCR → anchored transcription.
:::

::::

Under the hood, many estimators are **parity-gated ports** of the R packages social
scientists already trust (`metafor`, `survey`, `survival`, `lavaan`, `MatchIt`, `did`, …),
re-implemented in pure `numpy` / `scipy` — no `rpy2`, no R runtime, no Stata licence — and
each ships a test pinning the Python result to the R reference within `1e-6`. See the
[R-package ports overview](tutorials/external/index.md).

## Quick start

```bash
pip install socialverse
```

```python
import socialverse as sv

sv.list_functions()     # every registered analysis, grouped by category

st = sv.StudyState()    # carries data + the evidence chain between steps
```

Then work through a **guided notebook** (a real analysis from raw data to a
provenance-tracked result) or an **R-package port** reference in the sidebar.

```{toctree}
:hidden:
:maxdepth: 1

Installation <Installation.md>
Tutorials overview <tutorials/index.md>
R-package ports <tutorials/external/index.md>
Release notes <Release_notes.md>
```

```{toctree}
:hidden:
:caption: Getting started
:maxdepth: 1

tutorials/notebooks/01_registry_and_studystate
tutorials/notebooks/10_full_study_evidence_chain
```

```{toctree}
:hidden:
:caption: Causal inference & quasi-experiments
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
:caption: Meta-analysis & evidence synthesis
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
:caption: Complex-survey & epidemiology
:maxdepth: 1

tutorials/notebooks/03_complex_survey
tutorials/external/pysurvey
```

```{toctree}
:hidden:
:caption: Econometrics
:maxdepth: 1

tutorials/notebooks/04_econometrics_replication
```

```{toctree}
:hidden:
:caption: Psychometrics & measurement
:maxdepth: 1

tutorials/notebooks/12_psychometrics
tutorials/external/pypsych
tutorials/external/pylavaan
```

```{toctree}
:hidden:
:caption: Survival & longitudinal
:maxdepth: 1

tutorials/notebooks/13_multilevel_survival
tutorials/external/pysurvival
```

```{toctree}
:hidden:
:caption: Spatial, networks, QCA & demography
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
:caption: Qualitative, text & philology
:maxdepth: 1

tutorials/notebooks/05_qualitative_coding
tutorials/notebooks/06_text_philology
```

```{toctree}
:hidden:
:caption: Research governance & literature
:maxdepth: 1

tutorials/notebooks/08_governance_gates
tutorials/notebooks/09_literature_citation
tutorials/notebooks/28_systematic_review_governance
```

```{toctree}
:hidden:
:caption: End-to-end reproductions
:maxdepth: 1

tutorials/notebooks/18_reproduction_rossi_cox
tutorials/notebooks/19_reproduction_jpsp2023_mediation
tutorials/notebooks/20_reproduction_hh2015_staggered_did
tutorials/notebooks/21_reproduction_401k_dml_cate
tutorials/notebooks/29_reproduction_ecr_multilevel_prevalence
```
