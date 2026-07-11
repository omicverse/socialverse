# pylavaan — lavaan in Python

> Marker-variable maximum-likelihood confirmatory factor analysis with lavaan's full fit-index battery and score-test modification indices, gated to R `lavaan` at 1e-6 element-wise — no R runtime required.

## What `lavaan` does

`lavaan` is the standard R package for latent-variable modelling — confirmatory factor analysis (CFA) and structural equation models (SEM) fit by maximum likelihood, with the model specified in a compact formula-like syntax (`f =~ x1 + x2 + x3`). Social scientists reach for it to test whether a hypothesized measurement structure (e.g. a set of survey items loading onto latent constructs) fits the observed covariance structure, to obtain standardized loadings for construct validity, and to read off the full battery of fit indices (χ², CFI, TLI, RMSEA, SRMR, AIC/BIC) that reviewers expect in a psychometrics or SEM paper. `lavaan`'s `modindices()` further flags mis-specified paths — cross-loadings or residual covariances the model omits but that the data want — via univariate score (Lagrange-multiplier) tests with expected parameter changes (EPC).

## The port

`socialverse.external.pylavaan` exposes:

- **`parse_model(model)`** — parses lavaan `=~` measurement-model syntax into an ordered `{factor: [indicators]}` mapping (factor covariances are added automatically, matching `cfa()`'s default).
- **`cfa(model, data, meanstructure=False)`** — fits a marker-variable-identified ML CFA (covariance structure only) and returns a `CFAResult`.
- **`CFAResult`** — the fitted-model object; carries `.parameter_estimates()`, `.standard_errors()`, `.fit_measures()`, and `.modification_indices()` as methods.
- **`fit_measures(fitted_cfa)`** — the full `lavaan::fitMeasures()` battery as a dict (dots replaced by underscores: `rmsea_ci_lower`, `bic2`, `logl`, `gfi`, ...).
- **`modification_indices(fitted_cfa, sort=True, minimum_value=0.0)`** — univariate score-test modification indices + EPC for every currently-fixed parameter (cross-loadings and residual covariances), mirroring `lavaan::modindices()`.

It is pure `numpy`/`scipy` — normal-theory ML discrepancy minimized with L-BFGS-B, polished with a Fisher-scoring Newton refinement, expected-information standard errors, and an R-`uniroot`-faithful Brent root-finder for the RMSEA confidence interval — with **no R runtime dependency**.

It is wired into socialverse as the primary backend of the registered `sv.tl.cfa` function (`socialverse/tl/_psychometrics.py`): `cfa()` fits the raw data, `fit_measures()` populates `models['cfa']['fit_measures']` and the flat `diagnostics/fit_indices` keys (`CFI`, `RMSEA`, `SRMR`, `AIC`, `BIC`, ...), and `modification_indices()` populates `models['cfa']['modification_indices']`. If the port import or fit fails for any reason, `sv.tl.cfa` falls back to `semopy` or an internal statsmodels-based block-wise approximation.

:::{admonition} Parity gate
:class: note

The port is pinned to R `lavaan` to `max_abs_err < 1e-6` on 8 deterministic parity tests.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pylavaan import cfa, fit_measures, modification_indices

# Holzinger-Swineford-style 3-factor / 9-indicator measurement model, the
# canonical lavaan CFA example. Build a small synthetic dataset inline.
rng = np.random.default_rng(0)
n = 300
visual = rng.normal(size=n)
textual = rng.normal(size=n)
speed = rng.normal(size=n)

def item(latent, loading=0.8, noise=0.6):
    return loading * latent + noise * rng.normal(size=n)

data = {
    "x1": item(visual), "x2": item(visual), "x3": item(visual),
    "x4": item(textual), "x5": item(textual), "x6": item(textual),
    "x7": item(speed), "x8": item(speed), "x9": item(speed),
}

model = """visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9"""

# fit by ML (marker-variable identification, covariance-only, N divisor)
res = cfa(model, data)

# unstandardized + standardized (std.lv, std.all) loadings, ordered like
# lavaan's parameterEstimates()
for row in res.parameter_estimates():
    if row["op"] == "=~":
        print(row["lhs"], row["rhs"], row["est"], row["std_all"])

# full lavaan fitMeasures() battery: chisq, df, cfi, tli, rmsea (+ CI),
# srmr, aic, bic, bic2, logl, gfi, agfi, nfi, ...
fm = fit_measures(res)
print(fm["cfi"], fm["rmsea"], fm["rmsea_ci_lower"], fm["rmsea_ci_upper"], fm["srmr"])

# score-test modification indices + EPC for every fixed parameter
# (cross-loadings and residual covariances), sorted by MI descending
mi = modification_indices(res, sort=True)
print(mi[0])  # {'lhs': ..., 'op': '=~'/'~~', 'rhs': ..., 'mi': ..., 'epc': ...}
```

Or, calling through the registered socialverse function:

```python
import socialverse as sv

state = sv.tl.cfa(state, model_spec={
    "visual": ["x1", "x2", "x3"],
    "textual": ["x4", "x5", "x6"],
    "speed": ["x7", "x8", "x9"],
})
cfa_model = state.read("models", "cfa")
print(cfa_model["backend"])            # "pylavaan" when the port fit successfully
print(cfa_model["fit_measures"])       # full battery, lavaan-style keys
print(cfa_model["modification_indices"])
```

## R ↔ Python dictionary

| R (`lavaan`) | socialverse | notes |
|---|---|---|
| `cfa(model, data)` | `pylavaan.cfa(model, data)` | same `=~` syntax; marker-variable ID, ML, covariance-only (no mean structure), N (biased) divisor |
| `parameterEstimates(fit)` | `res.parameter_estimates()` | rows carry `est`, `se`, `std_lv`, `std_all`, ordered loadings → residual variances → factor variances → factor covariances |
| `fitMeasures(fit)` | `pylavaan.fit_measures(res)` / `res.fit_measures()` | dict keyed like lavaan (dots → underscores): `chisq`, `df`, `cfi`, `tli`, `rmsea`, `rmsea_ci_lower/upper`, `rmsea_pvalue`, `srmr`, `aic`, `bic`, `bic2`, `logl`, `gfi`, `agfi`, `nfi` |
| `modindices(fit)` | `pylavaan.modification_indices(res)` / `res.modification_indices()` | list of `{lhs, op, rhs, mi, epc}`, sortable, `minimum_value` filter |
| `lavParseModelString(model)` | `pylavaan.parse_model(model)` | `=~` (measured-by) operator only, sufficient for `cfa()` |
| — (called internally by `cfa()`) | `sv.tl.cfa(state, model_spec=...)` | registered socialverse function; wraps `pylavaan.cfa` + `fit_measures` + `modification_indices` and writes `models['cfa']` / `diagnostics/fit_indices` |

## Parity evidence

8 deterministic parity tests gate the port against R `lavaan` at `max_abs_err < 1e-6`, run on the classic 3-factor / 9-indicator (visual/textual/speed) measurement model:

- unstandardized loadings and variances (`est`)
- standardized loadings, both `std.lv` and `std.all`
- the compact fit-index set (`chisq`, `df`, `cfi`, `tli`, `rmsea`, `srmr`)
- the full `fitMeasures()` battery element-wise (`fmin`, `logl`, `unrestricted_logl`, `aic`, `bic`, `bic2`, `chisq`, `pvalue`, `baseline_chisq`, `cfi`, `tli`, `nfi`, `gfi`, `agfi`, `rmsea`, `rmsea_ci_lower`, `rmsea_ci_upper`, `rmsea_pvalue`, `srmr`), plus exact integer checks on `npar`, `df`, `baseline_df`
- the modification-index **formula**, verified deterministically by plugging lavaan's own fitted coefficients into the port's score-test machinery and comparing MI/EPC to lavaan directly (isolates the math from any optimizer difference)
- the modification-index **ranking**, checking the top-MI suggestion (`visual =~ x9`) matches lavaan's top-ranked entry

:::{admonition} End-to-end modification-index tolerance
:class: warning

R `lavaan`'s `nlminb` optimizer stops at a finite gradient tolerance (~5e-7), while the port's Newton-polished ML solution drives the gradient to ~1e-13. That optimizer slack, amplified by N, propagates into any quantity computed from lavaan's *own* end-to-end fit — including MI/EPC computed from the port's independently-optimized estimates. The end-to-end modification-index test therefore uses a documented `5e-4` tolerance rather than 1e-6; the 1e-6 gate on the formula itself (using lavaan's stored coefficients as input) is what actually pins the math.
:::

To reproduce:

```bash
Rscript socialverse/external/pylavaan/tests/r_reference_driver.R
pytest socialverse/external/pylavaan/tests/test_parity.py -v
```

## In the socialverse workflow

Call `sv.tl.cfa(state, model_spec={...})` for day-to-day CFA fitting — it drives the pylavaan port when available and writes `models['cfa']` (loadings, `fit_measures`, `modification_indices`) and `diagnostics/fit_indices` onto the `StudyState`. The registry enforces the `requires`/`produces` contract for `cfa`; use `sv.registry_lookup("cfa")` or `sv.list_functions()` to confirm the live signature and I/O contract before scripting against it.
