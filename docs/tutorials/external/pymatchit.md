# pymatchit — R `MatchIt` in Python

> Propensity-score preprocessing for causal inference — logistic-regression propensity scores, greedy nearest-neighbour matching, `WeightIt`-style balancing weights, Mahalanobis distances, and the full `summary.matchit` balance table — now callable from Python at 1e-6 parity with R `MatchIt`, no R runtime required.

## What `MatchIt` does

R's **MatchIt** (Ho, Imai, King, Stuart) is the standard preprocessing tool for observational causal inference: it selects (or reweights) comparison units so that the treated and control groups look alike on observed covariates *before* an outcome model is fit, following the "matching as nonparametric preprocessing" recipe. Its `nearest`/`glm` default estimates a propensity score by logistic regression and greedily pairs each treated unit to its nearest available control on that score; its `summary()` reports standardized mean differences, variance ratios, and eCDF statistics to check whether the match actually balanced the covariates. Social scientists reach for it (often together with `WeightIt` for IPW weights) whenever they need a defensible, auditable covariate-balancing step ahead of a difference-in-means or outcome regression.

## The port

`socialverse.external.pymatchit` exposes:

- `glm_logit_ps(X, y, max_iter=25, tol=1e-8)` — fits a binomial logit GLM by IRLS exactly as R's `glm.fit` does (intercept-first design matrix, R's `mu=(y+0.5)/2` start, deviance-based convergence); returns `(coef, fitted_ps)`.
- `nearest_match(distance, treat)` — greedy 1:1 nearest-neighbour matching without replacement, treated units processed in `m.order="largest"` (descending propensity-score) order; returns a `dict {treated_index: matched_control_index}`.
- `smd(x, treat, weights=None)` — standardized mean difference using the full-sample treated-group SD as denominator (MatchIt's convention), optionally weighted for the after-matching case.
- `matchit(X, treat, covariates=None)` — the full `nearest`/`glm` pipeline: fits the propensity score, matches, and reports before/after SMD; returns a `MatchItResult`.
- `MatchItResult` — dataclass-like container holding `ps_coef`, `distance` (fitted PS), `pairs`, `weights`, `smd_before`, `smd_after`, `smd_vars`.
- `get_w_from_ps(ps, treat, estimand="ATE", treated=1)` — `WeightIt::get_w_from_ps` port converting a binary propensity score to ATE/ATT/ATC balancing weights.
- `mahalanobis_dist(X, treat)` — `MatchIt:::mahalanobis_dist` port: scales covariates, computes the pooled within-group covariance (with MatchIt's small-sample correction), and returns the `n1 x n0` pairwise Mahalanobis distance matrix between treated and control rows.
- `balance_table(X, treat, weights=None, covariates=None)` — the `summary.matchit(standardize=TRUE)` balance columns (`Std. Mean Diff.`, `Var. Ratio`, `eCDF Mean`, `eCDF Max`) for each covariate, before or after adjustment.

The port is pure `numpy`/`scipy`, no R runtime. It is wired into socialverse's registered causal-inference function `sv.tl.psm` (`socialverse/tl/_matching.py`): `psm` prefers `pymatchit.glm_logit_ps` for the propensity-score step (falling back to `statsmodels.Logit` or a plain numpy IRLS only if the port raises), and — after doing its own with-replacement nearest-neighbour matching / IPW ATT-weighting — calls `pymatchit.balance_table` for the extra `Var. Ratio`/`eCDF` balance columns and `pymatchit.get_w_from_ps` (estimand `"ATT"`) to summarize the effective control sample size.

:::{admonition} Parity gate
:class: note

This port is pinned to R `MatchIt` 4.7.2 (`__matchit_reference_version__`) to `max_abs_err < 1e-6` on 12 deterministic parity tests (`socialverse/external/pymatchit/tests/test_parity.py`), run against the classic Lalonde NSW/CPS observational fixture.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pymatchit import (
    matchit, get_w_from_ps, mahalanobis_dist, balance_table,
)

# A tiny synthetic covariate set: age, education, and two lagged earnings
# columns, plus a 0/1 treatment indicator (same layout as the Lalonde fixture
# the parity tests use: age, educ, re74, re75).
rng = np.random.default_rng(0)
n = 20
age  = rng.integers(20, 50, n).astype(float)
educ = rng.integers(8, 16, n).astype(float)
re74 = rng.normal(5000, 2000, n).clip(min=0)
re75 = rng.normal(5000, 2000, n).clip(min=0)
treat = np.array([1] * 8 + [0] * 12)

X = np.column_stack([age, educ, re74, re75])
covariates = ["age", "educ", "re74", "re75"]

# Full nearest/glm pipeline: PS-logit fit + greedy 1:1 matching + before/after SMD.
res = matchit(X, treat, covariates=covariates)
print("PS coefficients (intercept, age, educ, re74, re75):", res.ps_coef)
print("fitted propensity scores:", res.distance)
print("matched pairs (treated_idx -> control_idx):", res.pairs)
print("SMD before:", dict(zip(res.smd_vars, res.smd_before)))
print("SMD after: ", dict(zip(res.smd_vars, res.smd_after)))

# WeightIt-style ATT balancing weights from the same fitted propensity score.
w_att = get_w_from_ps(res.distance, treat, estimand="ATT", treated=1)
print("ATT weights (treated=1, control=ps/(1-ps)):", w_att)

# Mahalanobis distance matrix (treated x control) on the raw covariates.
D = mahalanobis_dist(X, treat)
print("Mahalanobis distance shape:", D.shape)  # (n_treated, n_control)

# Full summary.matchit(standardize=TRUE) balance table, before vs. matched.
bt_before = balance_table(X, treat, weights=None, covariates=covariates)
bt_after = balance_table(X, treat, weights=res.weights, covariates=covariates)
print("Var. Ratio before:", dict(zip(bt_before["vars"], bt_before["var_ratio"])))
print("Var. Ratio after: ", dict(zip(bt_after["vars"], bt_after["var_ratio"])))
```

## R ↔ Python dictionary

| R (`MatchIt`) | socialverse | notes |
|---|---|---|
| `glm(treat ~ ., family=binomial(), data=df)` | `glm_logit_ps(X, y)` → `(coef, fitted_ps)` | intercept prepended automatically; matches R's IRLS element-wise |
| `matchit(treat ~ age + educ + re74 + re75, data=df, method="nearest", distance="glm")` | `matchit(X, treat, covariates=["age","educ","re74","re75"])` → `MatchItResult` | `m.order="largest"`, without replacement, matching MatchIt's defaults |
| `match.data(m)$weights` | `MatchItResult.weights` | 1 for matched treated/control, 0 otherwise |
| `summary(m)$sum.all` / `summary(m)$sum.matched` | `smd(x, treat, weights)` / `balance_table(X, treat, weights)` | `matchit()` reports SMD only; `balance_table` adds Var. Ratio + eCDF |
| `WeightIt::get_w_from_ps(ps, treat, estimand="ATT")` | `get_w_from_ps(ps, treat, estimand="ATT", treated=1)` | also supports `"ATE"` / `"ATC"` |
| `MatchIt:::mahalanobis_dist(formula, data)` | `mahalanobis_dist(X, treat)` | returns the `n1 x n0` treated-by-control distance matrix |
| workflow entry point | `sv.tl.psm(state, method="nn", ...)` | registered `StudyState` function in `socialverse/tl/_matching.py`; delegates the PS step to `glm_logit_ps` and the extended balance columns to `balance_table`/`get_w_from_ps` |

## Parity evidence

12 deterministic parity tests in `socialverse/external/pymatchit/tests/test_parity.py`, gated at `max_abs_err < 1e-6` against a reference JSON (`reference.json`) generated by the R driver `r_reference_driver.R` on the Lalonde fixture. The gated quantities are:

- propensity-score logistic-regression coefficients (`glm_logit_ps` via `matchit`);
- fitted propensity scores (the `distance` vector);
- standardized mean differences before **and** after matching (`smd_before`/`smd_after`);
- `WeightIt::get_w_from_ps` balancing weights for all three estimands (ATE, ATT, ATC);
- the full `n1 x n0` `mahalanobis_dist` pairwise distance matrix;
- the `summary.matchit(standardize=TRUE)` balance table (`Std. Mean Diff.`, `Var. Ratio`, `eCDF Mean`, `eCDF Max`), both before and after matching.

:::{admonition} Documented tie-break limitation
:class: warning

The exact *pairing* of controls whose propensity score is bit-identical to another candidate is not bit-reproduced — it follows R MatchIt's internal C++ scan order rather than the port's. On the Lalonde fixture, 177/185 matched pairs reproduce R exactly; the port asserts that every one of the 8 residual disagreements involves two controls with a `0.0` propensity-score gap (the *multiset* of matched-control propensity scores is identical to R), which is why the after-matching SMD and every other distance-based balance statistic still reproduce at 1e-6. This is a reference-tolerance limitation on individual pair identity, not on any numerical quantity the gate checks.
:::

Reproduce locally:

```bash
Rscript socialverse/external/pymatchit/tests/r_reference_driver.R
pytest socialverse/external/pymatchit/tests/
```

## In the socialverse workflow

Day-to-day, call the registered `sv.tl.psm` (propensity-score matching / IPW estimate of the ATT, with pre/post covariate balance) — it prefers this port for the propensity-score fit and for the extended `Var. Ratio`/`eCDF` balance columns, falling back gracefully if the port import fails. The registry enforces `psm`'s `requires`/`produces` contract (`sources['datasets']`, `design['treatment']`, `variables['outcome']` → `models['psm']`, `diagnostics['balance']`); use `registry_lookup` or `sv.list_functions()` to confirm the live signature before wiring it into a pipeline.
