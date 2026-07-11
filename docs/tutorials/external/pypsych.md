# pypsych — psych in Python

> Classical psychometrics — Cronbach's α, McDonald's ω, ICC inter-rater reliability, `corr.test`, and principal-axis factor analysis — parity-gated to R's `psych` at 1e-6, no R runtime required.

## What `psych` does

`psych` (William Revelle) is the workhorse R package for classical test theory and exploratory factor analysis in the social and behavioral sciences. Researchers reach for it to score multi-item scales (`alpha`, `omega`) before trusting a composite, to quantify inter-rater agreement on coded/observational data (`ICC`), to get a full correlation matrix with pairwise significance in one call (`corr.test`), and to run principal-axis factor extraction (`fa(fm="pa")`) as the traditional psychometric alternative to principal components. It is the de facto standard cited in psychometrics methods sections across psychology, education, and survey research.

## The port

- `smc(R)` — squared multiple correlations of each variable against the rest, computed as `1 - 1/diag(solve(R))` on a correlation matrix (used internally to seed factor-analysis communalities).
- `cronbach_alpha(X)` — the `psych::alpha` total row from a raw item matrix: `raw_alpha`, `std_alpha`, `G6` (Guttman's lambda-6), and `average_r` (mean inter-item correlation). No automatic key reversal, matching `check.keys=FALSE`.
- `fa_pa(R, nfactors=1, min_err=1e-3, max_iter=50)` — principal-axis factor analysis (`fa(fm="pa")`): iterates the leading eigenvector loading with the diagonal reset to the model communalities until the communality-sum change drops below `min_err`. Returns `loadings`, `communality`, `uniqueness`.
- `omega_total(R, communality=None, nfactors=1)` — McDonald's ω_total computed from a one-factor PA solution: `1 - Σ(1 - h²)/sum(R)`.
- `ICC(ratings, alpha=0.05)` — the six Shrout & Fleiss intraclass correlation coefficients (`ICC1`, `ICC2`, `ICC3`, `ICC1k`, `ICC2k`, `ICC3k`) from a subjects×raters matrix via two-way ANOVA, each with its F ratio, df, p-value, and confidence bounds.
- `corr_test(x)` — Pearson correlation matrix plus pairwise n, t, raw two-sided p, and standard error, replicating `psych::corr.test(method="pearson", normal=TRUE)` (unadjusted p — Holm adjustment and Fisher-z CIs are not part of this port).

The port is pure numpy/scipy — no rpy2, no R installation, no subprocess call into Rscript at runtime. It is wired into socialverse as the numerical backend behind four registered functions:

- `sv.tl.reliability` — calls `cronbach_alpha` and `omega_total` internally to build the full internal-consistency report.
- `sv.tl.icc` — calls `pypsych.ICC` directly on the resolved subjects×raters matrix.
- `sv.tl.correlation_test` — calls `pypsych.corr_test` directly on the resolved numeric variables.
- `sv.tl.efa` with `method="pa"` (or `fm="pa"`) — calls `pypsych.fa_pa` for R-exact principal-axis extraction (the `efa` default without this kwarg is a principal-component extraction, not this port).

:::{admonition} Parity gate
:class: note

pypsych is pinned to R `psych` 2.6.5 at `max_abs_err < 1e-6` across 8 deterministic parity tests (`socialverse/external/pypsych/tests/test_parity.py`), on the `psych::bfi` first-5-items fixture (2709 complete-case subjects) plus a fixed 6-subject×4-rater ICC fixture.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pypsych import cronbach_alpha, fa_pa, omega_total, ICC, corr_test

# --- a tiny 5-item, 8-subject congeneric scale (toy data, not the bfi fixture) ---
rng = np.random.default_rng(0)
true_trait = rng.normal(size=8)
items = np.column_stack([
    true_trait * 0.9 + rng.normal(scale=0.4, size=8) for _ in range(5)
])

alpha = cronbach_alpha(items)
print("Cronbach's alpha:", alpha)
# {'raw_alpha': ..., 'std_alpha': ..., 'G6': ..., 'average_r': ...}

# fa_pa works on a correlation matrix, not the raw item matrix
R = np.corrcoef(items, rowvar=False)
fa = fa_pa(R, nfactors=1)
print("PA loadings:", fa["loadings"])
print("PA communality:", fa["communality"])

omega = omega_total(R, communality=fa["communality"])
print("McDonald's omega_total:", omega)

# --- inter-rater reliability: 6 subjects rated by 4 judges ---
ratings = rng.integers(1, 8, size=(6, 4)).astype(float)
icc = ICC(ratings)
print("ICC types:", icc["type"])
print("ICC point estimates:", icc["ICC"])

# --- correlation matrix with pairwise significance ---
ct = corr_test(items)
print("r matrix:\n", ct["r"])
print("raw two-sided p matrix:\n", ct["p"])

# --- equivalently, via the registered sv.tl functions on a StudyState ---
import socialverse as sv
import pandas as pd

df = pd.DataFrame(items, columns=[f"item{i}" for i in range(1, 6)])
state = sv.StudyState()
state.write("sources", "datasets", df)

state = sv.tl.reliability(state)
print(state.diagnostics["reliability"]["cronbach_alpha"])

state = sv.tl.efa(state, method="pa", n_factors=1)
print(state.models["efa"]["estimator"])  # "pypsych.fa_pa (R psych::fa fm='pa', principal-axis)"
```

## R ↔ Python dictionary

| R (`psych`) | socialverse | notes |
|---|---|---|
| `alpha(items)$total` | `pypsych.cronbach_alpha(items)` / `sv.tl.reliability(state)` | returns `raw_alpha`/`std_alpha`/`G6`/`average_r`; no key reversal (`check.keys=FALSE`) |
| `omega(items)$omega.tot` | `pypsych.omega_total(R, communality=...)` / `sv.tl.reliability(state)` | this port's ω_tot is the closed form on the **public** `fa(fm="pa")` solution, not psych's internal auto-key-reversal + GPArotation minres pipeline — see Parity evidence |
| `fa(items, nfactors=1, fm="pa")` | `pypsych.fa_pa(R, nfactors=1)` / `sv.tl.efa(state, method="pa")` | takes a correlation matrix `R`, not the raw item matrix; returns `loadings`, `communality`, `uniqueness` |
| `ICC(ratings, lmer=FALSE)` | `pypsych.ICC(ratings, alpha=0.05)` / `sv.tl.icc(state)` | `ratings` is subjects × raters; returns all six ICC1/2/3(+k) types with F/df/p/CI |
| `corr.test(x, method="pearson")` | `pypsych.corr_test(x)` / `sv.tl.correlation_test(state)` | returns raw (unadjusted) two-sided p; psych's Holm adjustment and Fisher-z CIs are not gated |
| `smc(R)` | `pypsych.smc(R)` | internal helper, also usable standalone |

## Parity evidence

The port is checked against R `psych` 2.6.5 by 8 deterministic parity tests in `socialverse/external/pypsych/tests/test_parity.py`, driven by `r_reference_driver.R` and compared against `reference.json`:

- **`cronbach_alpha`**: `raw_alpha`, `std_alpha`, `G6`, `average_r` on the `psych::bfi` first-5-items complete-case fixture (n = 2709) — gated at 1e-6.
- **`fa_pa` (`fa(fm="pa")`, nfactors=1)**: `loadings`, `communality`, `uniqueness` on the same fixture's correlation matrix — gated at 1e-6.
- **`omega_total`**: two variants both gated at 1e-6 against the R driver's matching closed-form computation — one on the raw fixture's PA communalities, one after the same key reversal `psych::omega()` applies internally.
- **`ICC`**: all six point estimates (`ICC1/2/3`, `ICC1k/2k/3k`) plus F, df1, df2, p, and `MSW` on a fixed 6-subject×4-rater matrix — gated at 1e-6; confidence bounds (`lower`/`upper`, closed-form qf/Satterthwaite) gated at 1e-6 in a separate test.
- **`corr_test`**: the Pearson r matrix, raw two-sided p, and se on the 5-item complete-case fixture — gated at 1e-6; the t matrix's finite entries gated at 1e-6, with an exact match on which cells are `+Inf` (the diagonal).

:::{admonition} Documented reference-tolerance limit
:class: warning

`psych::omega()`'s own `omega.tot` is **not** element-wise reproducible from the public `fa(fm='pa')` solution: psych's `omega()` runs a separate internal pipeline (automatic negative-item key reversal followed by GPArotation minres extraction), not the one-factor PA solution this port exposes. One test in the suite (`test_omega_close_to_psych_reference`) checks this port's ω_tot, after applying the *same* key reversal, lands within a looser **1e-3** tolerance of `psych::omega()`'s reported `omega.tot` — confirming it is the same quantity, not a 1e-6 parity claim. The raw (un-reversed) fixture's ω legitimately differs by roughly 0.13 from `psych::omega()` and is intentionally not gated.
:::

To reproduce:

```bash
Rscript socialverse/external/pypsych/tests/r_reference_driver.R
pytest socialverse/external/pypsych/tests/test_parity.py -v
```

## In the socialverse workflow

Day to day, call `sv.tl.reliability`, `sv.tl.icc`, `sv.tl.correlation_test`, or `sv.tl.efa(method="pa")` on a `StudyState` rather than importing pypsych directly — the registry enforces each function's `requires`/`produces` contract (e.g. `reliability` requires `sources['datasets']` and produces `diagnostics['reliability']`) so a resolver can't quote a reliability coefficient before it has actually run. Use `sv.list_functions()` or `registry_lookup` to confirm the live signature and category (`psychometrics`) before scripting against it.
