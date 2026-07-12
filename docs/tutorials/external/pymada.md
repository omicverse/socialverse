# pymada — `mada` in Python

> Bivariate diagnostic test-accuracy meta-analysis (the Reitsma model, SROC/AUC, and HSROC coefficients) — callable from Python at 1e-6 parity with R `mada`, no R runtime required.

## What `mada` does

`mada` (Meta-Analysis of Diagnostic Accuracy) is the standard R package for pooling diagnostic test studies reported as 2×2 tables (TP/FN/FP/TN). Rather than pooling sensitivity and specificity separately — which ignores the negative correlation induced by threshold effects across studies — `mada::reitsma` fits the bivariate random-effects model of Reitsma et al. (2005): a joint linear mixed model on the logit-transformed sensitivity and false-positive rate, with an unstructured between-study covariance matrix, fit by (profiled) restricted maximum likelihood via `mvmeta`. From the fitted model `mada` also derives the Rutter–Gatsonis HSROC parametrisation and the area under the summary ROC (SROC) curve — the numbers social scientists and epidemiologists report for screening-instrument or diagnostic-test meta-analyses (e.g. pooling a depression-screening tool's sensitivity/specificity across validation studies).

## The port

`socialverse.external.pymada` exposes:

- **`reitsma(data=None, TP=None, FN=None, FP=None, TN=None, correction=0.5, correction_control="all")`** — fits the bivariate random-effects model on the logit scale (profiled REML, unstructured 2×2 `Psi`, R's own `vmmin`/BFGS optimizer with an analytic gradient). Returns a dict with `coefficients` (pooled logit-sens, logit-fpr), `vcov`, `Psi`, `se`, `sensitivity`, `false_pos_rate`, `logLik`, and `par`.
- **`calc_hsroc_coef(fit)`** — converts a `reitsma` fit into the Rutter–Gatsonis HSROC coefficients (`Theta`, `Lambda`, `beta`, `sigma2theta`, `sigma2alpha`) for the no-covariate case.
- **`AUC(fit, fpr=None, sroc_type="ruttergatsonis")`** — area under the Rutter–Gatsonis SROC curve (`AUC`, integrated over `fpr = 1:99/100` by default) and the partial AUC (`pAUC`, integrated over the observed FPR range from the fit's raw cell counts).

It is pure `numpy`/`scipy` — the REML profile likelihood, its analytic gradient, the IGLS starting values, and even R's `vmmin` BFGS line search are reimplemented directly in Python, with no R process spawned at runtime.

Inside socialverse it is wired into `sv.tl.dta_bivariate` (`socialverse/tl/_meta_dta.py`), which calls `external.pymada.reitsma` and `external.pymada.AUC` as its preferred backend, translating the port's (logit-sensitivity, logit-FPR) basis into socialverse's own (logit-sensitivity, logit-specificity) basis via the linear map `fpr = 1 - spec`. If the port's import or fit raises, `dta_bivariate` silently falls back to a native Nelder-Mead bivariate-normal approximation (`backend: "native"` in the output vs `backend: "pymada"`). `sv.tl.dta_glmm` in turn delegates to `dta_bivariate`.

:::{admonition} Parity gate
:class: note

The port is pinned to R `mada` to `max_abs_err < 1e-6` across 8 deterministic parity tests.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pymada import reitsma, AUC, calc_hsroc_coef

# A small diagnostic-accuracy dataset: 2x2 cell counts per study
# (TP/FN/FP/TN), analogous to mada's built-in AuditC fixture.
data = {
    "TP": [17, 18, 18, 6, 34, 19, 19, 34, 6, 32],
    "FN": [3, 3, 2, 1, 5, 3, 1, 4, 1, 4],
    "FP": [1, 4, 2, 1, 12, 6, 3, 12, 1, 9],
    "TN": [10, 12, 10, 5, 33, 27, 12, 33, 5, 30],
}

# Fit the Reitsma bivariate random-effects model (profiled REML).
fit = reitsma(data=data)

print("pooled sensitivity:", fit["sensitivity"])
print("pooled FPR:        ", fit["false_pos_rate"])
print("pooled specificity:", 1.0 - fit["false_pos_rate"])
print("coefficients (logit sens, logit fpr):", fit["coefficients"])
print("between-study covariance Psi:\n", fit["Psi"])
print("fixed-effects se:", fit["se"])

# Rutter-Gatsonis HSROC coefficients from the same fit.
hs = calc_hsroc_coef(fit)
print("HSROC Theta/Lambda/beta:", hs["Theta"], hs["Lambda"], hs["beta"])

# Area (and partial area) under the summary ROC curve.
auc = AUC(fit)
print("AUC / pAUC:", auc["AUC"], auc["pAUC"])

# --- Equivalent path through the registered socialverse function ---
import pandas as pd
import socialverse as sv

df = pd.DataFrame(data)
state = sv.StudyState()
state.write("sources", "datasets", df)                 # ingest the 2x2 table
state = sv.tl.dta_descriptives(state, tp="TP", fp="FP", fn="FN", tn="TN")
state = sv.tl.dta_bivariate(state, tp="TP", fp="FP", fn="FN", tn="TN")
summary = state.models["dta_bivariate"]
print(summary["backend"], summary["sensitivity"], summary["specificity"])
```

## R ↔ Python dictionary

| R (`mada`) | socialverse | notes |
|---|---|---|
| `reitsma(data)` | `socialverse.external.pymada.reitsma(data=...)` / `sv.tl.dta_bivariate(state, ...)` | port takes `TP/FN/FP/TN` directly or a `data=` mapping/DataFrame with those columns; `sv.tl.dta_bivariate` wraps it and remaps `fpr -> spec` |
| `fit$coefficients` | `fit["coefficients"]` | logit(sens), logit(fpr) in the port; `sv.tl.dta_bivariate` writes `mu_logit` as (logit sens, logit spec) |
| `fit$vcov`, `fit$Psi` | `fit["vcov"]`, `fit["Psi"]` | column-major-flattened in R's JSON dumps; the port returns them as 2×2 `numpy` arrays |
| `summary(fit)` (pooled sens/spec) | `fit["sensitivity"]`, `1 - fit["false_pos_rate"]` | `sv.tl.dta_bivariate` also writes `sens_ci`/`spec_ci` (Wald, 1.96·SE on the logit scale) |
| `mada:::calc_hsroc_coef(fit)` | `calc_hsroc_coef(fit)` | identical keys: `Theta`, `Lambda`, `beta`, `sigma2theta`, `sigma2alpha` |
| `AUC(fit)` | `AUC(fit)` | returns `{"AUC": ..., "pAUC": ...}`; `sv.tl.dta_bivariate` stores this under `summary["auc"]` when available |
| `data(AuditC); reitsma(AuditC)` | `sv.tl.dta_descriptives(state, ...)` then `sv.tl.dta_bivariate(state, ...)` | `dta_descriptives` computes per-study sens/spec/DOR/LR+/LR- with 0-cell continuity correction; `dta_bivariate` requires its `models["dta"]` output |

## Parity evidence

8 deterministic parity tests (`socialverse/external/pymada/tests/test_parity.py`) fit the port against R `mada::reitsma` on the canonical `AuditC` fixture (14 studies) and assert `max_abs_err < 1e-6` on:

- the fixed-effects `coefficients` (pooled logit-sensitivity, logit-FPR)
- their standard errors (`se`) and full `vcov` (column-major flattened)
- the between-study covariance `Psi` (column-major flattened)
- derived pooled `sensitivity` and `false_pos_rate`
- the Rutter–Gatsonis HSROC coefficients (`Theta`, `Lambda`, `beta`, `sigma2theta`, `sigma2alpha`)
- the SROC `AUC` and partial `pAUC`

Reference values are generated once by `tests/r_reference_driver.R` (via R `mada` + `jsonlite`) into `tests/reference.json`, which the Python tests load and compare against — no R process runs at test time.

:::{admonition} Deterministic, not stochastic
:class: warning

Everything gated here is deterministic REML/quadrature output — there is no bootstrap or MCMC step in `reitsma`/`AUC`/`calc_hsroc_coef`, so the 1e-6 gate is a strict numerical-agreement bound, not a looser stochastic-convergence tolerance. The port reimplements R's `vmmin` BFGS line search (not `scipy.optimize`) specifically so it lands on the same stationary point R does, rather than a sharper or different local optimum.
:::

To reproduce:

```bash
Rscript socialverse/external/pymada/tests/r_reference_driver.R
pytest socialverse/external/pymada/tests/
```

## In the socialverse workflow

Day to day, call `sv.tl.dta_descriptives` to get per-study sensitivity/specificity/DOR/LR± with continuity correction, then `sv.tl.dta_bivariate` (or its experimental alias `sv.tl.dta_glmm`) to get the Reitsma-pooled summary point and SROC/AUC — `dta_bivariate` prefers the `pymada` backend and only falls back to the native approximation if the port raises. The registry enforces the `requires={"models": ["dta"]}` / `produces={"models": ["dta_bivariate"]}` contract between these two steps; use `sv.list_functions()` or `registry_lookup("dta_bivariate")` to confirm the live signature and prerequisite chain before scripting a pipeline.
