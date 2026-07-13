# Tutorials

socialverse ships two complementary kinds of tutorial. Pick a **category in the sidebar**
to see the relevant ones side by side, or start from the cards below.

- **Guided notebooks** — runnable, end-to-end walk-throughs of a real analysis workflow,
  from raw data to a defensible, provenance-tracked result.
- **R-package ports** — reference tutorials for each `socialverse.external` port: what the
  R package does, the Python API, the R↔Python dictionary, and the `1e-6` parity evidence.
  See the [ports overview](external/index.md) for the full parity table.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Getting started
:link: notebooks/01_registry_and_studystate.html
The registry + `StudyState`, and a full study as one traceable evidence chain.
:::

:::{grid-item-card} 🎯 Causal inference
:link: notebooks/02_causal_did.html
DiD, quasi-experiments (RDD / synthetic control), IV, matching, mediation.
`pydid` · `pymatchit` · `pyfixest`.
:::

:::{grid-item-card} 📊 Meta-analysis
:link: notebooks/22_meta_analysis_basics.html
Basic → multilevel/robust → meta-regression → publication bias → network meta.
`pymetafor` · `pynetmeta` · `pyrobumeta` · `pymada`.
:::

:::{grid-item-card} 🧪 Complex survey & econometrics
:link: notebooks/03_complex_survey.html
Design-based survey estimation and econometrics replications. `pysurvey`.
:::

:::{grid-item-card} 🧭 Psychometrics & survival
:link: notebooks/12_psychometrics.html
Reliability, factor analysis, SEM; Kaplan–Meier, Cox, AFT. `pypsych` · `pylavaan` · `pysurvival`.
:::

:::{grid-item-card} 🗺 Spatial, networks, QCA, demography
:link: notebooks/14_spatial_analysis.html
Spatial analysis, networks + stylometry, QCA + demography. `pyqca` · `pyergm` · `pydemography`.
:::

:::{grid-item-card} 📜 Qualitative, text & philology
:link: notebooks/05_qualitative_coding.html
Thematic coding with quote tracing; text scaling, stylometry, and philology.
:::

:::{grid-item-card} ⚖️ Governance & literature
:link: notebooks/08_governance_gates.html
Research-governance gates, citation/literature workflows, systematic-review governance.
:::

:::{grid-item-card} 🔁 End-to-end reproductions
:link: notebooks/18_reproduction_rossi_cox.html
Full paper reproductions — Rossi/Cox, JPSP mediation, staggered DiD, 401(k) DML, ECR prevalence.
:::

::::
