# pysurvey — R `survey` in Python

> Design-based (Taylor-linearization) estimation for complex probability samples — svydesign, svymean/svytotal, svyglm, svyby, svyratio, svyciprop — now callable from Python at 1e-6 parity with R `survey`, no R runtime required.

## What `survey` does

R's **survey** package (Thomas Lumley) is the standard tool for analyzing data collected under complex sampling designs — stratification, multistage cluster (PSU) sampling, unequal probability weights, and finite population correction (fpc). Naively running `lm`/`glm` on survey data ignores the design and produces wrong standard errors; `survey` instead computes **design-based** point estimates and Taylor-linearized ("ultimate cluster") variances that correctly propagate stratum and PSU structure into every statistic. Social scientists reach for it whenever they analyze national or cross-national survey data with published sampling weights (GSS, ANES, NHANES, DHS, World Values Survey, etc.) and need publication-correct SEs rather than the deceptively tighter SRS-style SEs.

## The port

`socialverse.external.pysurvey` exposes:

- `svydesign(data, weights, ids=None, strata=None, fpc=None)` — builds a `SurveyDesign` (weights + PSU ids + strata + fpc), returning a `SurveyDesign` dataclass with a `.degf` property (design degrees of freedom = #PSUs − #strata).
- `svymean(y, design, level=95.0)` — design-based mean of a column, with linearized SE, t-based CI, and df.
- `svytotal(y, design, level=95.0)` — design-based total, same machinery as `svymean`.
- `svyglm(y, X, design, level=95.0, add_intercept=True)` — design-based Gaussian GLM (weighted least squares) with the survey sandwich variance; returns coefficients, SEs, t-values, p-values, and CIs.
- `svyby(y, by, design, stat="svymean", level=95.0)` — per-level domain (subpopulation) `svymean`/`svytotal`, with variance taken over the **full** design (not a re-declared subset design) — matching R's `svyby` semantics.
- `svyratio(num, den, design, level=95.0)` — design-based ratio of two totals via the Taylor-linearized influence function.
- `svyciprop(y, design, level=95.0)` — logit-method CI for a survey proportion (matches R's `svyciprop(..., method="logit")`, the package default).
- `SurveyDesign` — the dataclass returned by `svydesign`.

The port is pure `numpy`/`scipy` (no rpy2, no R runtime) and implements the same ultimate-cluster Taylor-linearization estimator `survey` uses internally. It is wired into socialverse's registered survey-domain functions in `socialverse/tl/_survey.py`: `sv.tl.survey_estimate` (design-based weighted regression, delegating to `svydesign`/`svyglm`/`svymean`/`svytotal`), `sv.tl.survey_by` (delegating to `svyby`), and `sv.tl.survey_ratio` (delegating to `svyratio`), plus `sv.tl.survey_ciprop` (delegating to `svyciprop`). `sv.pp.design_survey` (or its alias) declares the design slots (`weights`/`strata`/`psu`/`fpc`) on the `StudyState` that these `tl.*` functions read.

:::{admonition} Parity gate
:class: note

This port is pinned to R `survey` 4.5 to `max_abs_err < 1e-6` on 8 deterministic parity tests (`socialverse/external/pysurvey/tests/test_parity.py`), covering both a stratified design (`apistrat`) and a one-stage cluster design (`apiclus1`) from the canonical `api` dataset shipped with `survey`.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pysurvey import svydesign, svymean, svyglm, svyby, svyratio, svyciprop

# A tiny stratified sample: 2 strata, unequal weights, with a finite
# population correction (fpc) — same shape as R's `apistrat` design.
data = {
    "api00":  [693, 762, 811, 528, 601, 742, 895, 615],
    "ell":    [42.0, 10.0, 5.0, 71.0, 55.0, 18.0, 3.0, 46.0],
    "meals":  [67.0, 24.0, 11.0, 85.0, 78.0, 33.0, 6.0, 61.0],
    "stype":  ["E", "E", "E", "E", "H", "H", "H", "H"],
}
weights = [33.7, 33.7, 33.7, 33.7, 22.1, 22.1, 22.1, 22.1]
fpc     = [140, 140, 140, 140, 90, 90, 90, 90]  # population PSUs per stratum

# Element sampling (ids=None -> one PSU per row), stratified, with fpc.
design = svydesign(data, weights=weights, ids=None, strata=data["stype"], fpc=fpc)
print("design df:", design.degf)  # (# PSUs) - (# strata)

# Design-based mean of api00, with linearized SE + t CI.
m = svymean("api00", design)
print("mean:", m["estimate"], "se:", m["se"], "ci:", (m["ci_lb"], m["ci_ub"]))

# Design-based GLM: api00 ~ ell + meals (survey sandwich variance).
X = np.column_stack([data["ell"], data["meals"]])
g = svyglm("api00", X, design)
print("coef:", g["coef"], "se:", g["se"], "pval:", g["pval"])

# Domain (subpopulation) means per stratum level (variance over the FULL design).
by = svyby("api00", "stype", design, stat="svymean")
print("levels:", by["levels"], "domain means:", by["estimate"])

# Design-based ratio (needs two more columns on the design).
data["api_stu"] = [450, 480, 520, 300, 350, 470, 560, 380]
data["enroll"]  = [520, 500, 540, 400, 420, 500, 580, 450]
design2 = svydesign(data, weights=weights, ids=None, strata=data["stype"], fpc=fpc)
r = svyratio("api_stu", "enroll", design2)
print("ratio:", r["estimate"], "se:", r["se"])

# Logit-method CI for a proportion: P(api00 > 700).
prop_indicator = (np.asarray(data["api00"], float) > 700).astype(float)
cp = svyciprop(prop_indicator, design)
print("proportion:", cp["estimate"], "95% CI:", (cp["ci_lb"], cp["ci_ub"]))
```

## R ↔ Python dictionary

| R (`survey`) | socialverse | notes |
|---|---|---|
| `svydesign(id=~1, strata=~stype, weights=~pw, fpc=~fpc, data=df)` | `svydesign(data, weights=pw, ids=None, strata=stype, fpc=fpc)` | `ids=None` (or `~1`) means element sampling — one PSU per row |
| `svydesign(id=~dnum, weights=~pw, fpc=~fpc, data=df)` | `svydesign(data, weights=pw, ids=dnum, fpc=fpc)` | one-stage cluster design; `strata=None` if unstratified |
| `svymean(~api00, ds)` / `coef()`/`SE()` | `svymean("api00", design)` → `{"estimate", "se", "df", "ci_lb", "ci_ub"}` | |
| `svytotal(~api00, ds)` | `svytotal("api00", design)` | same return shape as `svymean` |
| `svyglm(api00 ~ ell + meals, design=ds)` | `svyglm("api00", X, design)` where `X = np.column_stack([ell, meals])` | intercept added automatically unless `add_intercept=False` |
| `svyby(~api00, ~stype, ds, svymean)` | `svyby("api00", "stype", design, stat="svymean")` | domain variance from the full design, not a subset re-declaration |
| `svyratio(~api.stu, ~enroll, ds)` | `svyratio("api_stu", "enroll", design)` | Taylor-linearized ratio SE |
| `svyciprop(~I(api00>700), ds, method="logit")` | `svyciprop(indicator_array, design)` | logit method is the only (and R's default) method implemented |
| `degf(ds)` | `design.degf` | property on `SurveyDesign` |
| `df.residual(g)` | `svyglm(...)["df"]` | `= design.degf - (p - 1)` |
| workflow entry point | `sv.tl.survey_estimate` / `sv.tl.survey_by` / `sv.tl.survey_ratio` / `sv.tl.survey_ciprop` | registered `StudyState` functions in `socialverse/tl/_survey.py`, delegating to the port |

## Parity evidence

8 deterministic parity tests in `socialverse/external/pysurvey/tests/test_parity.py`, gated at `max_abs_err < 1e-6` (one test, the stratified `svytotal`, uses a slightly looser `1e-4` tolerance for the total's magnitude) against a reference JSON (`reference.json`) generated by the R driver `r_reference_driver.R`. The gated quantities are:

- stratified design: `svymean` estimate + SE, `svytotal` estimate + SE, `svyglm` coefficients + SEs (and residual df) for `api00 ~ ell + meals`, `svyby` per-stratum domain means + SEs, `svyratio` estimate + SE for `api.stu / enroll`, `svyciprop` estimate + variance + logit-CI bounds for `P(api00 > 700)`.
- one-stage cluster design (`apiclus1`): `svymean` estimate + SE, `svyglm` coefficients + SEs for `api00 ~ ell`.
- design degrees of freedom (`degf`) is asserted exactly (integer equality) in every test.

:::{admonition} No stochastic caveats
:class: warning

Unlike ports that wrap MCMC or bootstrap procedures, `pysurvey`'s estimators are closed-form (Taylor linearization), so all 8 tests are gated at the strict 1e-6 tolerance with no reference-tolerance relaxation, aside from the one `1e-4` total-magnitude exception noted above.
:::

Reproduce locally:

```bash
Rscript socialverse/external/pysurvey/tests/r_reference_driver.R
pytest socialverse/external/pysurvey/tests/
```

## In the socialverse workflow

Day-to-day, call the registered `sv.tl.survey_estimate` (design-based weighted regression), `sv.tl.survey_by` (domain estimates), `sv.tl.survey_ratio`, or `sv.tl.survey_ciprop` — these read the design slots (`weights`/`strata`/`psu`/`fpc`) that `sv.pp.design_survey`-style declaration writes onto `StudyState`, then delegate internally to this port. The registry enforces each function's `requires`/`produces` contract; use `registry_lookup` or `sv.list_functions()` to confirm the live signature and slot names before wiring a pipeline.
