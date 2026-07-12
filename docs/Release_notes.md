# Release notes

## 0.7.2

Backend-parity release: the four remaining wiring gaps were closed and a batch of
previously-missing R functions were added and registered.

- **Gaps wired to the parity-gated ports**: Kaplan–Meier now uses the R-exact
  `pysurvival.km` (Greenwood std.err + confidence bands); `survey_estimate` now
  surfaces a design-based total (`svytotal`); `efa(method="pa")` opts in to
  R `psych::fa(fm="pa")` principal-axis factoring (the default stays PCA).
- **26 newly-added functions** across the ports, each with its own `1e-6` parity
  gate — e.g. per-study BLUP for meta-analysis, `survey_by` / `survey_ratio` /
  `survey_ciprop`, conditional logit and parametric AFT survival, ICC and a
  correlation-test matrix, direct fuzzy-set calibration, an ERGM statistics /
  triad census, Poisson-PMLE with high-dimensional fixed effects, and more.
- **9 new registry entries**; `sv.tl` now exposes 134 registered functions.
- External parity gates: **115 tests, all green at `1e-6`** on the deterministic
  core; full suite **181 passed**, no regressions.

## 0.7.1

First release of the **`socialverse.external`** layer — 14 R packages
reconstructed as parity-gated pure-`numpy`/`scipy` ports, following the
[omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr) protocol,
and wired in to replace the previous approximate implementations:

`pymetafor` · `pysurvey` · `pysurvival` · `pydemography` · `pyqca` · `pyfixest`
· `pyrobumeta` · `pynetmeta` · `pypsych` · `pylavaan` · `pymatchit` · `pymada`
· `pydid` · `pyergm`.

70 parity tests across the 14 ports, all green at `max_abs_err < 1e-6` on each
package's deterministic core. Stochastic components (bootstrap SEs, MCMC-MLE) are
documented as reference-tolerance rather than gated to `1e-6`. See each
[tutorial](tutorials/external/index.md) for the specifics.

## Earlier

- **0.6.x** — native meta-analysis module (three tiers, ~96 functions), figure
  styling (`sv.style` / `sv.pl`), and the OSF paper-reproduction notebooks.
- **0.2.x / 0.1.x** — the core engine (registry + `StudyState` + slot contracts)
  and the first analysis modules (`pp` / `tl` / `pl` / `gov` / `lit`).
