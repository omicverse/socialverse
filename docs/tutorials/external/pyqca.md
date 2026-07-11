# pyqca — R `QCA` in Python

> Qualitative Comparative Analysis — truth-table construction, Quine–McCluskey Boolean minimization, parameters of fit, direct calibration, and necessity superset search — now callable from Python at 1e-6 parity with R `QCA`, no R runtime required.

## What `QCA` does

R's **QCA** package (Adrian Duşa) is the reference implementation of Charles Ragin's set-theoretic method for mapping *configurations* of conditions onto an outcome — the standard tool for crisp-set, multi-value, and fuzzy-set QCA (csQCA/mvQCA/fsQCA) in comparative and case-oriented social science. Where regression asks "how much does each variable matter on average," QCA asks "which *combinations* of conditions are sufficient (or necessary) for the outcome," building a truth table over the 2^k configurational corners, minimizing it with Boolean (Quine–McCluskey) logic to a minimal sum-of-products solution, and reporting each path's consistency (inclusion) and coverage. Social scientists reach for it for small/medium-N comparative research (welfare-state typologies, democratization, policy configurations) where causal complexity, equifinality, and conjunctural causation are the point, not a nuisance to average away.

## The port

`socialverse.external.pyqca` exposes:

- `truth_table(data, outcome, conditions, incl_cut=0.8)` — fuzzy-set truth table (R `truthTable`): builds the observed (non-remainder) corners, per-row case count `n`, sufficiency inclusion `incl`, PRI, and the `OUT` column assigned by the `incl_cut` cut-off. Returns a `TruthTable`.
- `minimize(tt, include=None)` — conservative (complex) Quine–McCluskey Boolean minimization of a `TruthTable`'s `OUT=1` corners to prime implicants and an irredundant essential-PI cover, with per-term `inclS`/`PRI`/`covS`/`covU` and solution-level `overall` fit. Only the conservative solution (`include=None`, no remainder-row simplifying assumptions) is supported; parsimonious solutions raise `NotImplementedError`.
- `pof(terms, data, outcome, conditions)` — parameters of fit (`inclS`/`PRI`/`covS`/`covU` per term, plus solution-level `overall`) for an explicit list of terms, given either as implicant tuples or term strings like `"DEV*~URB*LIT"`.
- `calibrate(x, type="fuzzy", method="direct", thresholds=None, logistic=True, idm=0.95)` — direct calibration of raw numeric data into set-membership scores. `type="fuzzy"` is the 3-anchor logistic direct method (exclusion/crossover/inclusion thresholds, inclusion degree of membership `idm`); `type="crisp"` is `findInterval` on sorted cut-points. Only `method="direct"` and `logistic=True` are supported.
- `superSubset(data, outcome, conditions=None, incl_cut=1.0, cov_cut=0.0, ron_cut=0.0, depth=None)` — necessity superset search (R `superSubset`, `relation="necessity"`): enumerates conjunctions (fuzzy `min`) and minimal disjunctions (fuzzy `max`) of the conditions and reports those clearing `incl_cut`/`cov_cut`/`ron_cut`, with `inclN`/`RoN`/`covN`.
- `TruthTable` — the dataclass-like result of `truth_table`, carrying `conditions`, `rownames`, `rows`, `OUT`, `n`, `incl`, `PRI`.

The port is pure `numpy`/`scipy` (no rpy2, no R runtime). It is wired into socialverse's registered set-methods functions in `socialverse/tl/_setmethods.py`: `sv.tl.qca` builds its own truth table and `OUT` coding in-module, then delegates the Quine–McCluskey minimization and per-term/solution fit to `pyqca.minimize` (falling back to an in-module Quine–McCluskey implementation only if the `pyqca` call raises), and also delegates necessity superset search to `pyqca.superSubset` for the `models['qca']['necessity']` block. `sv.tl.calibrate` delegates directly to `pyqca.calibrate` for both the fuzzy-direct and crisp paths. `sv.tl.necessity_analysis` delegates directly to `pyqca.superSubset` for a standalone necessity report (`models['necessity']` / `diagnostics['necessity']`).

:::{admonition} Parity gate
:class: note

This port is pinned to R `QCA` 3.25 to `max_abs_err < 1e-6` on 10 deterministic parity tests (`socialverse/external/pyqca/tests/test_parity.py`), driven by an 18-case comparative dataset (Ragin-style development/urbanization/literacy/industrialization/stability conditions against a survival outcome).
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pyqca import truth_table, minimize, pof, calibrate, superSubset

# Tiny fuzzy-set dataset: 6 cases, two conditions (DEV, URB), one outcome (SURV).
data = {
    "DEV":  [0.81, 0.99, 0.58, 0.16, 0.07, 0.98],
    "URB":  [0.12, 0.89, 0.98, 0.07, 0.16, 0.99],
    "SURV": [0.05, 0.95, 0.89, 0.12, 0.42, 0.95],
}

# 1. Truth table: observed corners only, with n / incl (consistency) / PRI / OUT.
tt = truth_table(data, outcome="SURV", conditions=["DEV", "URB"], incl_cut=0.8)
print("rownames:", tt.rownames)   # 1-based R-style row ids
print("OUT:", tt.OUT, "incl:", tt.incl, "PRI:", tt.PRI)

# 2. Quine-McCluskey minimization of the OUT=1 corners (conservative solution).
sol = minimize(tt)
print("terms:", sol["terms"])                 # e.g. ["DEV*URB", ...]
print("term inclS/covS:", sol["inclS"], sol["covS"])
print("solution overall:", sol["overall"])     # {"inclS":..., "PRI":..., "covS":...}

# 3. Parameters of fit for an explicit set of terms (e.g. a hypothesised path).
fit = pof(["DEV*URB"], data, outcome="SURV", conditions=["DEV", "URB"])
print("pof inclS/PRI/covS:", fit["inclS"], fit["PRI"], fit["covS"])

# 4. Direct calibration: raw scale -> fuzzy membership (3-anchor logistic).
raw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
fs = calibrate(raw, type="fuzzy", method="direct", thresholds=[3, 5.5, 8], idm=0.95)
print("fuzzy scores:", fs)                     # 0.5 at the crossover (x=5.5)

# Crisp calibration: count of cut-points each value equals/exceeds.
crisp = calibrate(raw, type="crisp", thresholds=[3, 5.5, 8])
print("crisp set values:", crisp)               # integers in 0..3

# 5. Necessity superset search: which conditions/combinations are necessary for SURV?
ss = superSubset(data, outcome="SURV", conditions=["DEV", "URB"],
                  incl_cut=0.9, cov_cut=0.6)
print("necessary terms:", ss["terms"])
print("inclN/RoN/covN:", ss["incl_cov"]["inclN"], ss["incl_cov"]["RoN"], ss["incl_cov"]["covN"])
```

## R ↔ Python dictionary

| R (`QCA`) | socialverse | notes |
|---|---|---|
| `truthTable(data, outcome, conditions, incl.cut = 0.8)` | `pyqca.truth_table(data, outcome, conditions, incl_cut=0.8)` | observed rows only (remainder rows with `n==0` are dropped, matching R's default table). |
| `minimize(tt, include = "")` (conservative/complex, no `?` in `include`) | `pyqca.minimize(tt, include=None)` | only the conservative solution is ported; parsimonious/`include="?"` raises `NotImplementedError`. |
| `pof(expression, data)` | `pyqca.pof(terms, data, outcome, conditions)` | `terms` may be implicant tuples or `"A*~B"`-style strings. |
| `calibrate(x, type = "fuzzy", method = "direct", thresholds, logistic = TRUE, idm = 0.95)` | `pyqca.calibrate(x, type="fuzzy", method="direct", thresholds=..., logistic=True, idm=0.95)` | indirect/TFR calibration and the non-logistic direct path are out of scope. |
| `calibrate(x, type = "crisp", thresholds)` | `pyqca.calibrate(x, type="crisp", thresholds=...)` | `findInterval(x, sort(thresholds))`. |
| `superSubset(data, outcome, conditions, relation = "necessity", incl.cut, cov.cut)` | `pyqca.superSubset(data, outcome, conditions, incl_cut=1.0, cov_cut=0.0, ron_cut=0.0, depth=None)` | necessity relation only; conjunctions + minimal disjunctions, R's `sqrt(.Machine$double.eps)` cut-off tolerance is reproduced. |
| (whole fsQCA pipeline, script) | `sv.tl.qca(state, conditions=..., outcome=..., threshold=0.8, ...)` | in-module truth table/coding + `pyqca.minimize`/`pyqca.superSubset` delegation; falls back to an in-module Quine–McCluskey on any `pyqca` exception. |
| `calibrate(...)` on a data-frame column | `sv.tl.calibrate(state, column=..., thresholds=..., type=...)` | delegates directly to `pyqca.calibrate`; optionally writes the calibrated column back to `sources['datasets']`. |
| `superSubset(..., relation = "necessity")` | `sv.tl.necessity_analysis(state, outcome=..., conditions=..., incl_cut=0.9, cov_cut=0.6)` | delegates directly to `pyqca.superSubset`. |

## Parity evidence

`socialverse/external/pyqca/tests/test_parity.py` runs 10 tests against `tests/reference.json` (generated by `tests/r_reference_driver.R` against R `QCA` 3.25) at `max_abs_err < 1e-6` (exact integer equality, `tol=0`, for the discrete columns). The gated quantities are:

- **Truth table**: observed row ids (`rownames`), the condition bit pattern per row, the `OUT` coding, per-row case count `n` (all exact), and per-row sufficiency inclusion `incl` and PRI (float, 1e-6).
- **Minimization**: the prime-implicant term strings (exact string match) and, per term, `inclS`, `PRI`, `covS`, `covU` (float, 1e-6), plus solution-level `overall` `inclS`/`PRI`/`covS`.
- **Calibration**: the fuzzy-direct logistic scores (float, 1e-6) and the crisp `findInterval` set values (exact integers) for a fixed 10-point raw scale with 3 anchors.
- **superSubset**: the necessity term list (exact string match, conjunctions then minimal disjunctions in R's report order) and, per term, `inclN`, `RoN`, `covN` (float, 1e-6).

:::{admonition} Scope limits, honestly stated
:class: warning

The port only implements the deterministic combinatorial-exact core of `QCA`. `minimize(..., include=...)` with remainder simplifying assumptions (parsimonious/intermediate solutions) is explicitly unsupported and raises `NotImplementedError`. `calibrate` only implements the fuzzy-direct logistic path and crisp `findInterval`; the indirect/TFR calibration method is documented in the docstring as out-of-scope for this gate. There is no stochastic component anywhere in this port (unlike, say, a bootstrap or MCMC-based reconstruction) — all 10 parity tests compare exact deterministic outputs.
:::

To reproduce:

```bash
Rscript socialverse/external/pyqca/tests/r_reference_driver.R
pytest socialverse/external/pyqca/tests/
```

## In the socialverse workflow

Day to day, call `sv.tl.qca` for the full fsQCA pipeline (truth table through minimized solution and necessity), `sv.tl.calibrate` to turn a raw column into calibrated set-membership scores before running `qca`, and `sv.tl.necessity_analysis` for a standalone necessity superset report. The registry enforces each function's `requires`/`produces` contract (e.g. `sv.tl.necessity_analysis` requires `sources['datasets']` and `variables['outcome']` and produces `models['necessity']`/`diagnostics['necessity']`) — use `registry_lookup` or `sv.list_functions()` to confirm the live signature and contract before scripting against it.
