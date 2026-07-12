# pyfixest — `fixest` in Python

> High-dimensional fixed-effects OLS (`feols`), Poisson PMLE with fixed effects (`fepois`), and Newey-West HAC OLS — the `fixest` estimators social scientists actually run for panel/gravity regressions — callable from Python with no R runtime, pinned to R `fixest` 0.14.2 at `max_abs_err < 1e-6`.

## What `fixest` does

`fixest` is the R package that made high-dimensional fixed-effects estimation fast: it absorbs one or more fixed-effect dimensions (unit, time, unit×time, ...) via iterated demeaning instead of building dummy-variable design matrices, and pairs that with `fixest`-specific cluster-robust and multiway-clustered standard errors. `feols()` is the OLS workhorse behind most modern panel/DiD/TWFE specifications in economics and political science; `fepois()` extends the same demeaning machinery to Poisson pseudo-maximum-likelihood (PPML), the standard estimator for gravity models, trade flows, and other non-negative count/multiplicative outcomes (Santos Silva & Tenreyro). Social scientists reach for it because closed-form dummy-variable OLS becomes intractable once you have thousands of fixed-effect levels, and because its small-sample cluster corrections have become the de facto standard people expect a "real" `fixest` regression table to reproduce.

## The port

- `feols(y, X, fe, cluster)` — one- or two-way fixed-effects OLS via the within (iterated group-mean demeaning) estimator, with `fixest`-compatible cluster-robust vcov (`ssc`: `adj=TRUE`, `cluster.adj=TRUE`, `cluster.df="conventional"`, `fixef.K="nested"`) and the within-R².
- `fepois(y, X, fe, cluster=None, tol=1e-10, maxit=1000)` — Poisson pseudo-maximum-likelihood (PPML) with high-dimensional fixed effects via IRLS on the Poisson log-link, fixed effects concentrated out by *weighted* alternating within-demeaning, and the same `fixest` nested cluster small-sample correction as `feols`.
- `newey_west(y, X, lag, add_intercept=True, order=None)` — OLS with a Newey-West (Bartlett-kernel) HAC vcov, matching `feols(y ~ X)` re-summarised with `vcov = NW(lag) ~ t` under `fixest`'s default `ssc(adj=TRUE)`.
- `demean(M, fe_codes, tol=1e-12, maxit=100000)` — the underlying partialling-out primitive (alternating projections across one or more fixed-effect dimensions); exported mainly for reuse/testing rather than direct end-user use.

All four are pure `numpy` (no R runtime, no `rpy2`, no compiled extensions). The port is wired into socialverse through two registered pipeline functions in `socialverse/tl/_econ.py`:

- `sv.tl.replicate` — an end-to-end AER-style replication pipeline (balance table → baseline TWFE → robustness matrix → publication table). Its baseline TWFE step tries the `pyfixest` port first (`_twfe_pyfixest_port`, calling `feols`), falls back to the real `pyfixest` package if installed, and finally to a `statsmodels` OLS path — writing the result to `models['twfe']`. When a time index is available it additionally attaches a Newey-West HAC companion (`_newey_west_hac`, calling `newey_west`) under `diagnostics['newey_west']`.
- `sv.tl.poisson_fe` — PPML with fixed effects, calling the port's `fepois` via `_ppml_pyfixest_port`, writing `models['fepois']`.

:::{admonition} Parity gate
:class: note

The port is pinned to R `fixest` 0.14.2 at `max_abs_err < 1e-6` on 6 deterministic parity tests (`socialverse/external/pyfixest/tests/test_parity.py`).
:::

## Quickstart

```python
import numpy as np

from socialverse.external.pyfixest import feols, fepois, newey_west

# --- a tiny synthetic panel: 5 units x 4 periods -------------------------
rng = np.random.default_rng(0)
n_id, n_t = 5, 4
idv = np.repeat(np.arange(1, n_id + 1), n_t)          # unit id (one-way FE)
timev = np.tile(np.arange(1, n_t + 1), n_id)           # time period
x = rng.normal(size=n_id * n_t)
y = 2.0 * x + idv * 0.3 + rng.normal(scale=0.5, size=n_id * n_t)

# --- one-way FE OLS, clustered by unit -------------------------------------
r1 = feols(y, x, fe=idv, cluster=idv)
print("coef:", r1["coef"], "se:", r1["se"], "within R2:", r1["within_r2"])

# --- two-way FE OLS (unit + time), clustered by unit ------------------------
r2 = feols(y, x, fe=[idv, timev], cluster=idv)
print("twfe coef:", r2["coef"], "n_clusters:", r2["n_clusters"])

# --- PPML with fixed effects (count outcome) --------------------------------
counts = rng.poisson(lam=np.exp(0.4 * x)).astype(float)
r3 = fepois(counts, x, fe=idv, cluster=idv)
print("ppml coef:", r3["coef"], "deviance:", r3["deviance"], "n_iter:", r3["n_iter"])

# --- OLS with Newey-West (Bartlett-kernel) HAC SEs, ordered by time ---------
X2 = np.column_stack([x, rng.normal(size=n_id * n_t)])
r4 = newey_west(y, X2, lag=2, order=timev)
print("nw coef:", r4["coef"], "nw se:", r4["se"])
```

Equivalently, through the registered socialverse pipeline (uses the same `feols`/`fepois` port under the hood, with automatic schema resolution and fallback):

```python
import socialverse as sv

state = sv.StudyState()
state = sv.tl.replicate(state, data=df, outcome="y", treatment="x",
                         unit="id", time="time", cluster="id")
twfe = state.models["twfe"]           # backend == "pyfixest" when the port path is used
```

## R ↔ Python dictionary

| R (`fixest`) | socialverse | notes |
|---|---|---|
| `feols(y ~ x \| id, cluster = ~id)` | `feols(y, x, fe=id, cluster=id)` | one-way FE, no intercept column (absorbed by FE) |
| `feols(y ~ x \| id + time, cluster = ~id)` | `feols(y, x, fe=[id, time], cluster=id)` | two-way FE, `fe=` as a list of grouping vectors |
| `fepois(y ~ x \| id, cluster = ~id)` | `fepois(y, x, fe=id, cluster=id)` | PPML; `cluster=None` defaults to the first FE dim, matching fixest's default |
| `feols(y ~ x1 + x2, vcov = NW(lag) ~ t)` | `newey_west(y, X, lag=lag, order=t)` | Newey-West HAC OLS; `add_intercept=True` prepends the constant fixest implies |
| `summary(fit)$coeftable` | `result["coef"]`, `result["se"]` | per-slope coefficient / SE arrays, in `X` column order |
| `fit$sigma2` / within-R² | `result["within_r2"]` | `feols` only |
| `fit$deviance` | `result["deviance"]` | `fepois` only |
| high-level driver | `sv.tl.replicate(...)`, `sv.tl.poisson_fe(...)` | schema-resolved, falls back gracefully if preconditions unmet |

## Parity evidence

6 parity tests in `socialverse/external/pyfixest/tests/test_parity.py`, each asserting `max_abs_err < 1e-6` against `socialverse/external/pyfixest/tests/reference.json` (generated by `r_reference_driver.R` against `fixest` 0.14.2):

- `test_oneway_id` — one-way FE `feols`: coefficient, cluster-robust SE, within-R², `nobs`, `nparams` (parameter count under the nested small-sample rule).
- `test_oneway_id_clustertime` — same design, clustered on the time dimension instead of the unit FE, to exercise the nested-vs-non-nested small-sample correction path.
- `test_twoway` — two-way (unit + time) FE `feols`: coefficient, SE, within-R², `nparams`.
- `test_fepois_oneway_id` — one-way FE `fepois` (PPML/IRLS): coefficient, SE, deviance at convergence, `nobs`.
- `test_newey_west_lag3` / `test_newey_west_lag2` — Newey-West HAC OLS at two truncation lags: coefficient vector and HAC SE vector (coefficients are lag-invariant since the lag only affects the vcov, which the two tests both confirm).

All six quantities are deterministic closed-form (or IRLS-converged) numbers — no bootstrap or MCMC step is involved anywhere in this port, so there are no stochastic-tolerance caveats to document here.

Reproduce:

```bash
# regenerate reference.json from the real R package (requires R + fixest installed)
Rscript socialverse/external/pyfixest/tests/r_reference_driver.R

# run the parity tests against the committed reference.json (no R required)
pytest socialverse/external/pyfixest/tests/test_parity.py -v
```

## In the socialverse workflow

Day to day, call `sv.tl.replicate(state, ...)` for the full TWFE replication pipeline (balance table, baseline estimate, robustness matrix, Newey-West companion, publication table) or `sv.tl.poisson_fe(state, ...)` directly for a PPML fixed-effects fit — both are registered in the `econ` category and route through the `feols`/`fepois` port automatically when its preconditions (at least one FE dimension plus a cluster variable) are met. The registry enforces each function's `requires`/`produces` contract; use `sv.list_functions()` or `registry_lookup("replicate")` / `registry_lookup("poisson_fe")` to confirm the live signature and I/O contract before wiring it into a larger pipeline.
