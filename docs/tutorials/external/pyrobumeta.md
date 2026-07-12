# pyrobumeta — robumeta in Python

> Robust variance estimation (RVE) meta-regression for dependent effect sizes — CORR/HIER working models, Tipton (2015) CR2 small-sample correction, and `impute_covariance_matrix` — callable from Python at 1e-6 parity with R, no R runtime.

## What `robumeta` does

`robumeta` (Hedges, Tipton & Johnson) fits meta-regressions when effect sizes are statistically dependent within studies — e.g. several outcomes, timepoints, or subgroups reported by the same study — without requiring the researcher to specify the exact within-study covariance structure. It offers two working models, correlated effects ("CORR", for effect sizes sharing a common measure/outcome within a study) and hierarchical effects ("HIER", for effect sizes nested at multiple levels), estimates variance components by method-of-moments, and — with `small=TRUE` (the default) — applies the Tipton (2015) CR2 bias-reduced sandwich estimator with Satterthwaite degrees of freedom so p-values and CIs stay well-calibrated with few studies. Social scientists reach for it constantly in education, psychology, and public-health meta-analyses where "one row per effect size, many rows per study" is the norm and clustering by study cannot be ignored. `clubSandwich::impute_covariance_matrix` and `coef_test(vcov="CR2")` are the companion utilities for building a working correlation-based V matrix and re-running the CR2 test off any fitted model.

## The port

- `robu(effect_size, var_eff_size, studynum, covariates, modelweights="CORR", rho=0.8, small=True)` — fits the RVE meta-regression: WLS on first-stage weights, method-of-moments τ² (CORR) or τ²+ω² (HIER), refit with variance-component-adjusted weights, then (if `small=True`) the CR2 sandwich covariance and Satterthwaite dfs. Returns a dict with `b`, `SE`, `t`, `dfs`, `prob`, `CI_L`, `CI_U`, `tau_sq` (+ `omega_sq` for HIER, `I2` for CORR), `N`, `M`, `p`.
- `impute_covariance_matrix(vi, cluster, r, return_list=True)` — builds the block-diagonal working covariance matrix `clubSandwich::impute_covariance_matrix` produces under a constant within-cluster correlation `r` (diagonal = `vi`, off-diagonal = `r * sqrt(vi_i * vi_j)`); returns a list of per-cluster blocks or the full `(M, M)` matrix.
- `coef_test(fit, vcov="CR2")` — packages a `robu()` fit's already-computed CR2 `SE`/`dfs` into the per-coefficient test table `clubSandwich::coef_test()` returns (`beta`, `SE`, `tstat`, `df`, `p_val`).

The port is pure numpy/scipy — no R runtime, no rpy2, no subprocess call-out. It is wired into socialverse primarily by `sv.tl.robu` (`socialverse/tl/_meta_rve.py`), which calls `pyrobumeta.robu` as its faithful backend (falling back to an internal sandwich estimator only if the port raises), and secondarily calls `pyrobumeta.coef_test` for the CR2 Satterthwaite test table and `pyrobumeta.impute_covariance_matrix` when the caller supplies `impute_r`/`within_corr`. A separate, independently-implemented sandwich estimator lives in `sv.tl.ma_robust` / `sv.tl.ma_che` for CR0/CR1/CR2 inference outside the robumeta working-model framework — it does not call this port.

:::{admonition} Parity gate
:class: note

The port is pinned to R `robumeta` 2.1 / `clubSandwich` 0.7.0 to `max_abs_err < 1e-6` across 4 deterministic parity tests.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pyrobumeta import robu, coef_test, impute_covariance_matrix

# --- a tiny dependent-effect-size dataset: 6 rows nested in 3 studies -------
effectsize = np.array([0.20, 0.35, 0.10, 0.50, 0.42, 0.28])
var_eff    = np.array([0.04, 0.05, 0.03, 0.06, 0.05, 0.04])
studyid    = np.array([1, 1, 2, 2, 3, 3])
males      = np.array([0.5, 0.5, 0.4, 0.4, 0.6, 0.6])   # study-level moderator
college    = np.array([0.3, 0.3, 0.6, 0.6, 0.2, 0.2])

# --- CORR working model, rho=0.8, CR2 small-sample correction (default) ----
fit = robu(
    effect_size=effectsize, var_eff_size=var_eff, studynum=studyid,
    covariates=[males, college], modelweights="CORR", rho=0.8, small=True,
)
print("coefficients (intercept, males, college):", fit["b"])
print("robust SE:                               ", fit["SE"])
print("Satterthwaite df:                         ", fit["dfs"])
print("tau^2, I^2:                               ", fit["tau_sq"], fit["I2"])

# --- Tipton (2015) CR2 coefficient test table -------------------------------
ct = coef_test(fit, vcov="CR2")
print("t-stats:", ct["tstat"], " p-values:", ct["p_val"])

# --- impute a block-diagonal working V under an assumed within-study r=0.7 -
blocks = impute_covariance_matrix(var_eff, studyid, r=0.7, return_list=True)
print("per-study V blocks:", [b.shape for b in blocks])
```

Same fit through the wired socialverse pipeline function (`sv.tl.robu`), which
drives the same `pyrobumeta.robu` backend from a `StudyState` holding a
`meta_effects` model:

```python
import socialverse as sv

state = sv.pp.meta_effects(state, ...)          # produces yi/vi/study columns
state = sv.tl.robu(state, model="CORR", rho=0.8)  # dispatches to pyrobumeta.robu
print(state.models["meta_rve"]["coefs"])
print(state.models["meta_rve"]["coef_test_cr2"])  # CR2 Satterthwaite test table
```

## R ↔ Python dictionary

| R (`robumeta` / `clubSandwich`) | socialverse | notes |
|---|---|---|
| `robu(effectsize ~ x1 + x2, data=d, studynum=, var.eff.size=, modelweights="CORR", rho=0.8, small=TRUE)` | `socialverse.external.pyrobumeta.robu(effect_size, var_eff_size, studynum, covariates, modelweights="CORR", rho=0.8, small=True)` / `sv.tl.robu(state, model="CORR", rho=0.8)` | covariates passed as a list of arrays (no intercept); intercept is prepended internally, matching the R formula's implicit intercept. |
| `robu(..., modelweights="HIER")` | `robu(..., modelweights="HIER")` / `sv.tl.robu(state, model="HIER")` | HIER also returns `omega_sq` (no `I2`). |
| `mc$reg_table$b.r`, `$SE`, `$dfs`, `$prob`, `$CI.L`/`$CI.U` | `fit["b"]`, `fit["SE"]`, `fit["dfs"]`, `fit["prob"]`, `fit["CI_L"]`/`fit["CI_U"]` | one entry per coefficient, intercept first. |
| `mc$mod_info$tau.sq`, `$I.2`, `$omega.sq` | `fit["tau_sq"]`, `fit["I2"]`, `fit["omega_sq"]` | I2 only for CORR, omega_sq only for HIER. |
| `clubSandwich::coef_test(mc, vcov="CR2")` | `socialverse.external.pyrobumeta.coef_test(fit, vcov="CR2")` | reuses the CR2 SE/df already computed by `robu(..., small=True)`. |
| `clubSandwich::impute_covariance_matrix(vi, cluster, r, return_list=TRUE)` | `socialverse.external.pyrobumeta.impute_covariance_matrix(vi, cluster, r, return_list=True)` / `sv.tl.robu(state, impute_r=0.7)` | scalar `r` only (constant within-cluster correlation path). |

## Parity evidence

4 deterministic parity tests in `socialverse/external/pyrobumeta/tests/test_parity.py`, gated at `max_abs_err < 1e-6` against `socialverse/external/pyrobumeta/tests/reference.json` (generated by `r_reference_driver.R` against R's `robumeta::corrdat` and `robumeta::hierdat` fixtures):

- `test_corr` — CORR working model on `corrdat` (N=39 studies, M=172 rows, p=3 covariates): coefficients `b`, robust `SE`, `t`, Satterthwaite `dfs`, `prob`, `CI_L`/`CI_U`, `tau_sq`, `I2`.
- `test_hier` — HIER working model on `hierdat` (5 covariates): same quantities plus `tau_sq` and `omega_sq` (no `I2` for HIER).
- `test_impute_covariance_matrix` — `impute_covariance_matrix` block-diagonal V (r=0.7) on `corrdat`: every per-cluster block's entries and the vector of block sizes.
- `test_coef_test_cr2` — `coef_test(fit, vcov="CR2")` on the CORR fit: `beta`, `SE`, Satterthwaite `df`, `tstat`, `p_val`.

All four quantities are fully deterministic (no bootstrap or MCMC step in this port), so the 1e-6 gate is a strict numerical-equality check, not a stochastic-tolerance one.

:::{admonition} Scope limit
:class: warning

Only `small=TRUE` (the default, and the statistically relevant path for RVE) is ported for the CORR/HIER working models; the `small=FALSE` (HC0-style, uncorrected) branch and user-supplied-weights working models are not parity-tested. `impute_covariance_matrix` supports only a scalar (constant) within-cluster `r`, not clubSandwich's per-cluster vector or `ar1` correlation structures.
:::

Reproduce:

```bash
Rscript socialverse/external/pyrobumeta/tests/r_reference_driver.R
pytest socialverse/external/pyrobumeta/tests/
```

## In the socialverse workflow

Day to day, call `sv.tl.robu(state, model="CORR"|"HIER", rho=0.8)` after `sv.pp.meta_effects` — it dispatches straight to this port and additionally writes a Tipton CR2 coefficient-test table (`coef_test_cr2`) into `state.models["meta_rve"]`. The registry enforces the contract (`requires={"models": ["meta_effects"]}`, `produces={"models": ["meta_rve"]}`); confirm the live signature and aliases at any time with `sv.list_functions()` or `registry_lookup("robu")` rather than trusting this page if the code has moved on.
