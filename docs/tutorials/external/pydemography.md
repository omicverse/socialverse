# pydemography — R `demography` life tables in Python

> Period life-table construction (`nMx → nqx → lx / Lx / Tx / ex`) plus Kitagawa
> and Oaxaca-Blinder rate-difference decomposition, callable from Python at
> 1e-6 parity with the R engine — no R runtime required.

## What `demography` does

The R `demography` package (and the closely related `DemoTools` ecosystem) is
the standard toolkit demographers use to turn a raw age-specific mortality
schedule into the actuarial machinery of a period life table: survivorship
(`lx`), person-years lived in each interval (`Lx`), the survivorship-weighted
person-years remaining (`Tx`), and life expectancy at each age (`ex`,
including `e0` at birth). Its numerical engine, `demography:::lt`, implements
the classic Chiang/Keyfitz separation-factor treatment of the infant and
child intervals (`a0`, `a1`) and the standard `qx` closure formula, which is
why demographers, actuaries, and social scientists studying mortality,
population aging, or cross-national life-expectancy gaps reach for it rather
than re-deriving the closure by hand. Kitagawa (1955) rate decomposition and
Oaxaca-Blinder (1973) mean-gap decomposition are not part of `demography`
itself, but they are the natural companions social scientists pair with a
life table — splitting a crude-rate or outcome gap between two groups into a
"composition/endowment" component and a "rate/coefficient" component.

## The port

- `life_table(mx, sex="total", startage=0, agegroup=1)` — faithful port of
  `demography:::lt`. Builds the full actuarial column set (`ax`, `mx`, `qx`,
  `lx`, `dx`, `Lx`, `Tx`, `ex`, `nx`, each a numpy array) plus the scalar
  `e0`, from an age-specific mortality schedule `mx`.
- `life_expectancy(mx, sex="total", startage=0, agegroup=1, age=0)` —
  convenience wrapper that runs `life_table` and returns `ex` at a single
  requested age (default `e0`).
- `kitagawa(c1, r1, c2, r2)` — Kitagawa (1955) decomposition of a crude-rate
  difference `R2 - R1` into a `rate_effect` and a `composition_effect` that
  sum exactly to `total`.
- `oaxaca(yA, xA, yB, xB)` — twofold Oaxaca-Blinder decomposition of a
  mean-outcome gap `meanYA - meanYB` into `explained` (endowments, using
  group B's OLS coefficients as the reference structure) and `unexplained`
  (coefficients).

The port is pure `numpy` (`_ols` uses `numpy.linalg.lstsq` for the Oaxaca
regressions) — no `scipy`, no R runtime, no subprocess call into R. It is
wired into socialverse's registry via `socialverse/tl/_demography.py`:

- `sv.tl.life_table(...)` delegates to `external.pydemography.life_table`
  whenever the supplied interval widths match a single-year (`agegroup=1`)
  or the standard `1, 4, 5, 5, …` five-year (`agegroup=5`) schedule and `mx`
  has no missing values; the returned `lx`/`dx`/`Lx`/`Tx` are rescaled from
  the port's radix-1 convention to the requested `radix`. It falls back to a
  built-in life-table builder otherwise, and reports which backend ran in
  `models.life_table["backend"]`.
- `sv.tl.decomposition(...)` delegates the scalar Kitagawa split to
  `external.pydemography.kitagawa`, called as
  `kitagawa(cA, mA, cB, mB)` so that `rate_effect + composition_effect`
  reproduces `crude_B - crude_A` exactly; it likewise reports the backend
  used (`"pydemography"` vs `"builtin"`) in
  `models.decomposition["backend"]`.

:::{admonition} Parity gate
:class: note

The port is pinned to the R `demography` engine (life table) and to the
published Kitagawa / Oaxaca-Blinder closed forms (computed independently
in the R driver) to `max_abs_err < 1e-6` across 6 deterministic parity
tests.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pydemography import (
    life_table, life_expectancy, kitagawa, oaxaca,
)

# --- 1. Period life table from a single-year mortality schedule --------
mx = [0.02, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.30]

lt = life_table(mx, sex="female", startage=0, agegroup=1)
print(lt["e0"])          # life expectancy at birth
print(lt["lx"][:3])      # survivorship at ages 0, 1, 2
print(lt["ex"][:3])      # remaining life expectancy at ages 0, 1, 2

# life_expectancy() is a one-line wrapper around life_table()["ex"][age]
e0_male = life_expectancy(mx, sex="male", startage=0, agegroup=1, age=0)
print(e0_male)

# --- 2. Kitagawa decomposition of a crude-rate difference --------------
# c1/c2: age-composition shares (each sums to 1); r1/r2: group rates
c1 = [0.40, 0.35, 0.15, 0.10]
c2 = [0.25, 0.30, 0.25, 0.20]
r1 = [0.005, 0.010, 0.030, 0.090]
r2 = [0.004, 0.009, 0.028, 0.085]

ki = kitagawa(c1, r1, c2, r2)
print(ki["total"], ki["rate_effect"], ki["composition_effect"])
assert np.isclose(ki["rate_effect"] + ki["composition_effect"], ki["total"])

# --- 3. Oaxaca-Blinder decomposition of a mean-outcome gap --------------
rng = np.random.default_rng(0)
nA, nB = 50, 50
xA = rng.normal(5, 1, size=(nA, 2))
xB = rng.normal(4, 1, size=(nB, 2))
yA = 2.0 + 1.5 * xA[:, 0] + 0.9 * xA[:, 1] + rng.normal(0, 0.5, nA)
yB = 1.3 + 1.2 * xB[:, 0] + 1.0 * xB[:, 1] + rng.normal(0, 0.5, nB)

ox = oaxaca(yA, xA, yB, xB)
print(ox["gap"], ox["explained"], ox["unexplained"])
assert np.isclose(ox["explained"] + ox["unexplained"], ox["gap"])
```

## R ↔ Python dictionary

| R (`demography`) | socialverse | notes |
|---|---|---|
| `demography:::lt(mx, sex, startage, agegroup)` | `sv.tl.life_table(...)` / `pydemography.life_table(mx, sex, startage, agegroup)` | `sv.tl.life_table` reads `mx`/age/width columns from `state.datasets`, rescales `lx`/`dx`/`Lx`/`Tx` by `radix`, and reports `backend` in the returned model |
| `lifetable(mx)$ex[age]` | `pydemography.life_expectancy(mx, sex, startage, agegroup, age)` | one-line `ex` lookup, no direct `sv.*` wrapper |
| custom Kitagawa (1955) closed form in the R driver | `sv.tl.decomposition(...)` / `pydemography.kitagawa(c1, r1, c2, r2)` | `sv.tl.decomposition` calls `kitagawa(cA, mA, cB, mB)`; `rate_effect + composition_effect == crude_B - crude_A` |
| custom twofold Oaxaca-Blinder closed form in the R driver | `pydemography.oaxaca(yA, xA, yB, xB)` | no direct `sv.tl.*` wrapper; `sv.tl.decomposition` runs its own age-index regression companion (`_oaxaca_blinder`) rather than calling this function |

## Parity evidence

6 deterministic pytest cases in
`socialverse/external/pydemography/tests/test_parity.py`, each asserting
`max_abs_err < 1e-6` against `reference.json` (produced by
`r_reference_driver.R`):

- `test_lifetable_female` / `test_lifetable_male` / `test_lifetable_total` —
  every life-table column (`ax`, `mx`, `qx`, `lx`, `dx`, `Lx`, `Tx`, `ex`,
  `nx`) and the scalar `e0`, for all three sex branches of the
  infant/child separation factors, over a 10-interval single-year schedule.
- `test_life_expectancy_e0` — the `life_expectancy(..., age=0)` wrapper
  reproduces `e0` from the full life table for all three sex branches.
- `test_kitagawa` — `R1`, `R2`, `total`, `rate_effect`, `composition_effect`,
  plus the exact adding-up identity `rate_effect + composition_effect == total`.
- `test_oaxaca` — `betaA`, `betaB` (OLS coefficient vectors), `meanYA`,
  `meanYB`, `gap`, `explained`, `unexplained`, plus the exact adding-up
  identity `explained + unexplained == gap`.

No stochastic or looser-tolerance quantities are involved — the life table
is a closed-form recursion and both decompositions are closed-form
algebraic splits, so all 6 tests are held to the same 1e-6 gate.

To reproduce:

```bash
Rscript socialverse/external/pydemography/tests/r_reference_driver.R
pytest socialverse/external/pydemography/tests/test_parity.py -v
```

## In the socialverse workflow

Day to day, call `sv.tl.life_table(...)` for a period life table and
`sv.tl.decomposition(...)` for a Kitagawa (+ companion Oaxaca-Blinder) rate
split — both transparently prefer the `pydemography` backend and report
which backend actually ran (`models.life_table["backend"]` /
`models.decomposition["backend"]`). The registry enforces each function's
declared `requires`/`produces` contract; use `registry_lookup("life_table")`
or `sv.list_functions()` to confirm the live signature and defaults before
scripting against it.
