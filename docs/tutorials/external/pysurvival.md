# pysurvival — R's `survival` in Python

> Kaplan-Meier, Cox proportional hazards, conditional (exact) logistic regression, and parametric AFT models — the core of R's `survival` package, callable from Python at 1e-6 parity, no R runtime required.

## What `survival` does

`survival` (Therneau) is the reference implementation of event-history analysis in R and the estimator every methods paper cites when it says "we fit a Cox model in R." Social scientists reach for it for duration/hazard analysis of any time-to-event outcome — job separation, protest onset, marriage dissolution, policy adoption, recidivism — with right-censored data. Its `coxph` (Efron-tie partial likelihood) and `survfit` (Kaplan-Meier with Greenwood variance) are the de facto standard, and `clogit`/`survreg` extend the same machinery to matched case-control designs and fully parametric accelerated-failure-time models respectively.

## The port

- `km(time, event, conf_level=0.95)` — Kaplan-Meier estimator with Greenwood cumulative-hazard standard errors and log-transformed confidence bands; matches `survfit(Surv(time,status)~1)`.
- `coxph(time, event, X, ties="efron", maxiter=30, eps=1e-9)` — Cox proportional-hazards regression by Newton-Raphson on the partial likelihood, with both **Efron** (R default) and **Breslow** tie handling; also returns Harrell's concordance.
- `clogit(y, strata, X, maxiter=30, eps=1e-9)` — conditional (fixed-effects) logistic regression for stratum-matched data via the exact conditional likelihood (equivalent to a stratified Cox model with `method="exact"`).
- `survreg(time, status, X, dist="weibull", maxiter=50, eps=1e-10)` — parametric accelerated-failure-time MLE for Weibull, exponential, or lognormal error distributions, on the log-time scale.

Each function returns a small `dataclass` (`KMResult`, `CoxResult`, `ClogitResult`, `SurvregResult`) carrying coefficients, standard errors, z/p-values, variance-covariance matrix, and log-likelihoods. The implementation is pure `numpy`/`scipy` — no R runtime, no `rpy2`, no subprocess.

It is wired into socialverse as the numeric backend of three registered functions in `socialverse/tl/_longitudinal.py`:

- `sv.tl.survival` — delegates to `pysurvival.coxph` (Efron ties) for the non-time-varying Cox fit that underlies the `models["cox"]` output (time-varying/Andersen-Gill designs fall back to `statsmodels.PHReg` instead, since left-truncation isn't covered by this port).
- `sv.tl.conditional_logit` — delegates entirely to `pysurvival.clogit`.
- `sv.tl.aft_survreg` — delegates entirely to `pysurvival.survreg`.

:::{admonition} Parity gate
:class: note

This port is pinned to R `survival` 3.8.3 to `max_abs_err < 1e-6` on **9** deterministic parity tests (`socialverse/external/pysurvival/tests/test_parity.py`).
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pysurvival import km, coxph, clogit, survreg

# --- toy right-censored duration data (10 units) ---
rng = np.random.default_rng(0)
time = np.array([5, 6, 6, 8, 9, 10, 12, 14, 14, 20], dtype=float)
event = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1])       # 1 = event, 0 = censored
age = np.array([50, 62, 40, 58, 47, 65, 39, 71, 44, 55], dtype=float)
sex = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

# 1) Kaplan-Meier — matches survfit(Surv(time, status) ~ 1)
km_fit = km(time, event)
print(km_fit.time)      # unique event/censoring times
print(km_fit.surv)      # KM survival estimate S(t)
print(km_fit.median)    # median survival time

# 2) Cox PH — Efron ties (R default)
X = np.column_stack([age, sex])
cox_fit = coxph(time, event, X, ties="efron")
print(cox_fit.coef, cox_fit.se, cox_fit.pval)   # log-HR, SE, p-value per covariate
print(np.exp(cox_fit.coef))                      # hazard ratios
print(cox_fit.concordance)                        # Harrell's C

# 3) Conditional logistic regression — matched sets
y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
strata = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
Xc = np.column_stack([age, sex])
clogit_fit = clogit(y, strata, Xc)
print(clogit_fit.coef, np.exp(clogit_fit.coef))   # conditional log-OR, OR

# 4) Parametric AFT — Weibull, with an explicit intercept column
Xs = np.column_stack([np.ones_like(time), age, sex])
aft_fit = survreg(time, event, Xs, dist="weibull")
print(aft_fit.coef, aft_fit.scale, aft_fit.loglik)

# Equivalent day-to-day call inside a socialverse workflow:
# state = sv.tl.survival(state, time="time", event="event", covariates=["age", "sex"])
```

## R ↔ Python dictionary

| R (`survival`) | socialverse | notes |
|---|---|---|
| `survfit(Surv(time, status) ~ 1)` | `pysurvival.km(time, event)` | rows at every unique observed time; `std_err` is the Greenwood SE of the cumulative hazard, R's convention |
| `coxph(Surv(time,status) ~ x, ties="efron")` | `pysurvival.coxph(time, event, X, ties="efron")` / `sv.tl.survival(...)` | Newton-Raphson partial likelihood; `sv.tl.survival` uses this for non-time-varying designs |
| `coxph(..., ties="breslow")` | `pysurvival.coxph(time, event, X, ties="breslow")` | Breslow tie handling |
| `clogit(case ~ x + strata(id))` | `pysurvival.clogit(y, strata, X)` / `sv.tl.conditional_logit(...)` | exact conditional likelihood over matched sets |
| `survreg(Surv(time,status) ~ x, dist="weibull")` | `pysurvival.survreg(time, status, X, dist="weibull")` / `sv.tl.aft_survreg(...)` | `X` must include an intercept column, matching R's `~ x` design matrix |
| `survreg(..., dist="exponential")` | `pysurvival.survreg(..., dist="exponential")` | scale fixed at 1 |
| `survreg(..., dist="lognormal")` | `pysurvival.survreg(..., dist="lognormal")` | Gaussian error on log-time |
| `summary(km)$table["median"]` | `KMResult.median` | first time at which `surv <= 0.5` |
| `cox$concordance["concordance"]` | `CoxResult.concordance` | Harrell's C |

## Parity evidence

**9** parity tests in `socialverse/external/pysurvival/tests/test_parity.py`, each asserting `max_abs_err < 1e-6` against R `survival` 3.8.3 output captured by `tests/r_reference_driver.R` (`reference.json`), on the canonical `lung` and `infert` datasets:

- **KM**: `time`, `n.risk`, `n.event`, `surv`, `std.err` at 1e-6, plus an exact match on the reported median; log-transformed CI bounds (`lower`/`upper`) checked separately at 1e-6.
- **Cox (Efron)**: coefficients, standard errors, z-statistics, and the (null, fitted) log-likelihood pair, all at 1e-6.
- **Cox (Breslow)**: coefficients and log-likelihood at 1e-6.
- **Cox concordance**: checked at a looser 1e-3 tolerance (see warning below).
- **clogit**: coefficients, standard errors, and (null, fitted) conditional log-likelihood on the `infert` matched case-control data, at 1e-6.
- **survreg** (Weibull, exponential, lognormal): coefficients, scale, standard errors (in R's `vcov` ordering `[coef..., Log(scale)]`), and log-likelihood, each at 1e-6, on the `lung` dataset.

:::{admonition} One looser-tolerance quantity
:class: warning

`cox.concordance` (Harrell's C) is gated at **1e-3**, not 1e-6 — R's `coxph` concordance uses an internal tie-resolution/robust-variance algorithm this port approximates with a direct pairwise comparison; the point estimate agrees closely but not bit-for-bit. All coefficient/SE/log-likelihood quantities remain at the full 1e-6 gate.
:::

Reproduce locally:

```bash
Rscript socialverse/external/pysurvival/tests/r_reference_driver.R
pytest socialverse/external/pysurvival/tests/
```

## In the socialverse workflow

Call `sv.tl.survival` for Cox + KM, `sv.tl.conditional_logit` for matched-set fixed-effects logit, and `sv.tl.aft_survreg` for parametric AFT — day to day these are the entry points, not the `pysurvival` functions directly. Each is registered with an explicit `requires`/`produces` contract that the registry enforces at call time; run `sv.list_functions()` or `registry_lookup("survival")` (and likewise for `"conditional_logit"` / `"aft_survreg"`) to confirm the live signature and default kwargs before scripting against it.
