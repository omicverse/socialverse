# Tutorials

socialverse ships two complementary kinds of tutorial, organised here by topic:

- **Guided notebooks** (`▶`) — runnable, end-to-end walk-throughs of a real analysis
  workflow, from raw data to a defensible, provenance-tracked result.
- **R-package ports** (`{ }`) — reference tutorials for each `socialverse.external`
  port: what the R package does, the Python API, the R↔Python dictionary, and the
  `1e-6` parity evidence. See the [ports overview](external/index.md) for the full
  parity table.

Each category below lists the relevant notebooks and port references side by side.

```{toctree}
:hidden:

R-package ports — overview <external/index>
```

```{toctree}
:caption: Getting started
:maxdepth: 1

notebooks/01_registry_and_studystate
notebooks/10_full_study_evidence_chain
```

```{toctree}
:caption: Causal inference & quasi-experiments
:maxdepth: 1

notebooks/02_causal_did
notebooks/11_quasi_experiment
notebooks/17_regression_iv_matching_mediation
external/pydid
external/pymatchit
external/pyfixest
```

```{toctree}
:caption: Meta-analysis & evidence synthesis
:maxdepth: 1

notebooks/22_meta_analysis_basics
notebooks/23_multilevel_and_robust_meta
notebooks/24_meta_regression_and_moderators
notebooks/25_publication_bias_and_sensitivity
notebooks/26_network_meta_analysis
notebooks/27_specialized_designs
external/pymetafor
external/pynetmeta
external/pyrobumeta
external/pymada
```

```{toctree}
:caption: Complex-survey & epidemiology
:maxdepth: 1

notebooks/03_complex_survey
external/pysurvey
```

```{toctree}
:caption: Econometrics
:maxdepth: 1

notebooks/04_econometrics_replication
```

```{toctree}
:caption: Psychometrics & measurement
:maxdepth: 1

notebooks/12_psychometrics
external/pypsych
external/pylavaan
```

```{toctree}
:caption: Survival & longitudinal
:maxdepth: 1

notebooks/13_multilevel_survival
external/pysurvival
```

```{toctree}
:caption: Networks, QCA & demography
:maxdepth: 1

notebooks/07_theory_lens_network
notebooks/15_qca_demography
notebooks/16_networks_stylometry
external/pyqca
external/pyergm
external/pydemography
```

```{toctree}
:caption: Qualitative, text & spatial
:maxdepth: 1

notebooks/05_qualitative_coding
notebooks/06_text_philology
notebooks/14_spatial_analysis
```

```{toctree}
:caption: Research governance & literature
:maxdepth: 1

notebooks/08_governance_gates
notebooks/09_literature_citation
notebooks/28_systematic_review_governance
```

```{toctree}
:caption: End-to-end reproductions
:maxdepth: 1

notebooks/18_reproduction_rossi_cox
notebooks/19_reproduction_jpsp2023_mediation
notebooks/20_reproduction_hh2015_staggered_did
notebooks/21_reproduction_401k_dml_cate
notebooks/29_reproduction_ecr_multilevel_prevalence
```
