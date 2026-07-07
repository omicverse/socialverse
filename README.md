# socialverse

**A structured, dependency-annotated function registry for the social sciences and humanities.**

`socialverse` ports the mechanism that makes [omicverse](https://github.com/Starlitnightly/omicverse)'s
agent capability work — not its data model. In AI-for-biology, what lets an agent
plan a real analysis without hallucinating the API is **`ov.registry`**: every
function is registered with a machine-readable contract (`requires` / `produces` /
`prerequisites` / `auto_fix`), so an agent *queries the registry* instead of
guessing. AnnData is only the vocabulary that contract speaks in.

Social data is not commensurable (a survey ≠ a corpus ≠ a network), so there is no
"AnnData for social science" and there never will be. **So socialverse keeps the
registry and drops the container**: the 12-slot [`StudyState`](socialverse/_state.py)
is a light vocabulary — *not* a data matrix — that `requires`/`produces` speak in.

> The spine is the registry, not the container. Define the vocabulary first, register
> the federated tools against it, and an agent can plan, chain, verify, and auto-fix.

---

## Install

```bash
pip install -e .            # minimal (numpy + pandas)
pip install -e ".[full]"    # + statsmodels/scipy/networkx/matplotlib to run every chain
pip install -e ".[dev]"     # + pytest
```

Everything domain-specific (linearmodels, spaCy, lxml, pyfixest, python-docx, …) is
**federated and lazy-imported** — a chain degrades gracefully if its backend is absent.

## Query the registry (the whole point)

```python
import socialverse as sv

sv.registry.find("双重差分")          # fuzzy search (Chinese / English / abbrev / tool name)
sv.registry.get_prerequisites("did")  # what does DID require & produce? who satisfies each slot?
sv.registry.resolve_plan("sv.pl.forest")   # order the chain to reach a target
```

### Coming from R / Stata / SPSS?

Search by the command name you already know — every function carries `py-<command>`
aliases drawn from Stata, R, and SPSS (the `py-` marks the Python reimplementation):

```python
sv.registry.get("py-lmer")             # R lme4::lmer      -> sv.tl.multilevel
sv.registry.get("py-stcox")            # Stata stcox       -> sv.tl.survival
sv.registry.get("py-svyglm")           # R survey::svyglm  -> sv.tl.survey_estimate
sv.registry.find("mixed")              # bare command also fuzzy-matches
```

128 such aliases across the registry map `mixed`/`lmer`, `stcox`/`coxph`, `svyset`/
`svydesign`, `sem`/`lavaan`, `mirt`, `rdrobust`, `ergm`, `truthTable`, `lagsarlm`,
`oaxaca`, … onto their socialverse equivalents (see `socialverse/_compat_aliases.py`).

`get_prerequisites("did")` returns the same shape as omicverse's, so OmicOS's
`registry_lookup` tool can consume a `socialverse` registry unchanged:

```json
{
  "function": "socialverse.tl.did",
  "required_functions": ["parallel_trends"],
  "requires":  {"design": ["panel_id","time","treatment"], "identification": ["parallel_trends"]},
  "produces":  {"models": ["did","twfe"], "diagnostics": ["robustness"]},
  "auto_fix":  "escalate",
  "satisfied_by": {"identification.parallel_trends": ["parallel_trends"], "...": ["declare_design"]}
}
```

## Run a chain — grounded, not guessed

```python
import socialverse as sv
from socialverse import datasets

st = sv.StudyState()
st.write("estimand", "target", "ATT")           # the one user-supplied input
df = datasets.load_did_panel()

sv.pp.ingest(st, data=df)
sv.pp.declare_design(st, panel_id="firm_id", time="year",
                     treatment="treat_post", first_treated="first_treated")
sv.tl.parallel_trends(st)                        # must pass before DID is called causal
sv.tl.did(st)                                    # TWFE ATT + cluster-robust SE
sv.pl.forest(st)                                 # publication figure

print(st.summary())        # slots populated + a full provenance ledger
```

Call `sv.tl.did(st)` on an unprepared state and the registry **refuses**, telling you
exactly which slot is missing and which function produces it — the `leiden`-before-
`neighbors` guard, ported to social science:

```
socialverse.tl.did cannot run — unmet requires:
  - identification.parallel_trends (produced by: parallel_trends)
Query registry.get_prerequisites(...) or registry.resolve_plan(...) to plan the chain.
```

## The StudyState vocabulary (12 slots)

The social-science analog of AnnData's `obs / var / obsm / uns`. Every contract
speaks only in these slots (validated at registration):

| slot | holds |
|---|---|
| `sources` | raw inputs: datasets, corpora, manuscripts, .bib, scans |
| `design` | sampling frame, weights, strata, PSU, panel_id, time, treatment/timing |
| `variables` | codebook, outcome, exposure, controls, scales, constructs |
| `corpus` | documents, coding units, dfm, TEI |
| `codes` | qualitative codebook, coded segments, themes, theme map |
| `estimand` | ATT / prevalence / association + target population (**user-given**) |
| `identification` | DAG, parallel-trends, IV validity, exclusion, positivity |
| `models` | DID/TWFE, event-study, weighted regression, topic model, network, field map |
| `diagnostics` | pretrend, balance, robustness matrix, reliability α, sensitivity |
| `evidence` | claim→quote/citation links, quote-trace index, verified .bib, provenance |
| `governance` | IRB, consent, PII-redaction status, data-use licence, AI-use disclosure |
| `artifacts` | figures, tables, DOCX/PDF, TEI-XML, apparatus, reproducible scripts |

## Namespaces (two axes, like omicverse)

- **phase**: `sv.pp` (prepare) · `sv.tl` (analyze) · `sv.pl` (plot/render)
- **social-science axes**: `sv.gov` (governance gates) · `sv.lit` (literature & citation)

Governance is a first-class axis — in social science, ethics/licence/PII/AI-disclosure
gate almost every analysis, so they are registered functions with their own contracts,
not an afterthought.

## Method coverage (61 registered functions)

Each family is a real, tested implementation (pure numpy/scipy/statsmodels, with the
champion backend lazy-imported when present) — see [docs/CONTRACT_CARDS.md](docs/CONTRACT_CARDS.md).

- **regression base**: **GLM** (`glm` covers OLS / logit / probit / Poisson), **multinomial** (`mlogit`), **ordered** (`ologit`), **average marginal effects** (`margins`)
- **causal / quasi-experimental**: TWFE-DiD, event-study, **RDD** (local-linear), **synthetic control**, **IV / 2SLS** (`iv_regress`), **propensity-score matching / IPW** (`psm`), **causal mediation** (`mediation`)
- **econometrics**: 8-step replication pipeline (emits reproducible R/Stata scripts)
- **complex survey**: design-based weighted estimation (strata/PSU/weights)
- **psychometrics**: **CFA**, **SEM** (path fallback), **IRT** (2PL) — reliability, fit indices
- **longitudinal**: **multilevel/HLM** (MixedLM), **survival/event-history** (Cox PH, KM)
- **spatial**: **Moran's I / LISA**, **spatial-lag (SAR)** regression with impacts
- **networks**: descriptives, **ERGM** (MPLE), **SAOM** co-evolution (descriptive)
- **set-theoretic**: **fsQCA** (truth-table + Quine-McCluskey minimization)
- **demography**: **life tables**, **Kitagawa / Oaxaca decomposition**
- **text / DH**: corpus building, topic coding, OCR→TEI, philology collation, **stylometry (Burrows's Delta)**
- **qualitative**: reflexive thematic analysis, quote-traceability, theory lenses
- **governance / literature**: ethics/licence/AI-disclosure gates · search, citation-verify, review

### Built-in analysis chains (auto-derived from `requires ↔ produces`)

- **causal**: `ingest → declare_design → parallel_trends → did → event_study → forest`
- **quasi**: `ingest → rdd → rdd_plot` · `synthetic_control → synth_path`
- **survey**: `ingest → declare_design → design_survey → survey_estimate → survey_dist`
- **psychometrics**: `ingest → cfa → sem` · `irt`
- **longitudinal**: `ingest → multilevel` · `survival → km_curve`
- **spatial**: `ingest → spatial_autocorr → spatial_regression → moran_scatter`
- **qualitative**: `build_corpus → redact_pii → code_themes → trace_quotes → reflexive_memo → theme_map`
- **text / philology**: `ocr_tei → build_corpus → philology_collate → tei_encode` · `stylometry → dendrogram`
- **networks**: `build_network → ergm` · `saom`
- **QCA / demography**: `qca` · `life_table → decomposition`
- **literature / citation**: `search_free → zotero_bridge → citation_manage → verify_citations → manuscript_review`
- **governance (cross-cutting)**: `data_use_check · ethics_check · redact_pii · ai_use_disclosure`

## How it maps to OmicOS

This package is the concrete instantiation of the `humanities_social` domain's
registry table: its 54 registered functions cover all 26 `humanities_social` skills
plus the quantitative method families a social-science审稿 pipeline needs.
An OmicOS agent points its `registry_lookup` at `sv.registry` and gets the same
grounding it gets from `ov.registry` in the bio domain — query, plan, chain, auto-fix.

## Design notes

- **Registry first, tools second.** Contracts are the spine; implementations are
  federated wrappers over the field's best tools (statsmodels, linearmodels, pyfixest,
  networkx, spaCy, lxml …), never rewrites.
- **Provenance is built in.** Every registered call records params + slots touched into
  `state.provenance` — the reproducible/auditable "evidence spine".
- **Fail-soft.** A missing optional backend degrades one chain, never the import.

Licence: CC-BY-4.0.
