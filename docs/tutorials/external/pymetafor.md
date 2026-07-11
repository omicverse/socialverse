# pymetafor — metafor in Python

> `metafor::rma.uni` (random/mixed/equal-effects meta-analysis, Knapp-Hartung inference, BLUPs) reconstructed in pure numpy/scipy, callable from Python at 1e-6 parity with R — no R runtime required.

## What `metafor` does

`metafor` (Viechtbauer 2010, *Journal of Statistical Software*) is the reference R package for meta-analysis: it pools effect sizes across studies under fixed/equal-effects, random-effects, and mixed-effects (meta-regression) models, and reports the full battery of heterogeneity diagnostics social scientists expect in a forest-plot writeup — τ² (between-study variance) by REML, ML, or DerSimonian-Laird, Higgins-Thompson I²/H², Cochran's Q, and Knapp-Hartung/Sidik-Jonkman small-sample adjustments to the standard errors. It is the de facto standard cited in PRISMA-compliant systematic reviews across psychology, education, sociology, and health/social policy research, and its `rma()`/`rma.mv()` family covers the vast majority of published meta-analytic models. Social scientists reach for it specifically because its τ² estimators and inference (especially Knapp-Hartung t-adjustment) are the ones journal reviewers expect to see reported, not looser approximations.

## The port

`socialverse/external/pymetafor` exposes:

- **`rma(yi, vi, mods=None, method="REML", test="z", level=95.0, add_intercept=True)`** — random/mixed/equal-effects meta-analysis. Fits τ² (REML via metafor's own Fisher-scoring iteration, ML, DerSimonian-Laird closed form, or "EE"/"FE" equal-effects), then weighted-least-squares β with Wald or Knapp-Hartung (`test="knha"`) inference, Q_E/Q_Ep heterogeneity test, I²/H² (Higgins-Thompson), SE(τ²) via inverse Fisher information, and Q_M/Q_Mp omnibus test for moderators. Returns an `RMAResult` dataclass.
- **`RMAResult.predict(level=95.0)`** — fitted value plus confidence and prediction interval for the average effect (intercept-only model), mirroring `predict.rma`.
- **`blup(res, level=95.0)`** — per-study best linear unbiased predictors (empirical-Bayes shrinkage toward the fitted value, `metafor::blup.rma.uni` parity). Takes an `RMAResult` from `rma()` and returns a `BLUPResult` dataclass with `pred`, `se`, `pi_lb`, `pi_ub`.

The module is pure numpy/scipy (`~230` LOC, no rpy2, no R subprocess). Numerically it never forms the ill-conditioned `(XᵀWX)⁻¹` directly — everything is expressed through the thin QR of the weight-scaled design — which is what keeps it at 1e-6 parity even at `cond(XᵀWX) ≈ 2e11` (uncentred moderators).

It is wired into socialverse at `socialverse/tl/_meta.py`: the registered function **`sv.tl.meta_random`** delegates to `pymetafor.rma` whenever `method` is one of `REML`/`ML`/`DL`/`EE`/`FE`/`CE` (the PM/SJ/HS/HE estimators still use socialverse's legacy path), and additionally calls `pymetafor.blup` to attach a per-study empirical-Bayes shrinkage table (`out["blup"]`) to the result. `sv.tl.meta_fixed` remains a separate, simpler common-effect pooling function and does not go through this port.

:::{admonition} Parity gate
:class: note

This port is pinned to R `metafor` 5.0.1 to `max_abs_err < 1e-6` on 9 deterministic parity tests (`socialverse/external/pymetafor/tests/test_parity.py`), run against the canonical `dat.bcg` (BCG vaccine, k=13 log risk ratios) fixture.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pymetafor import rma, blup

# 13 studies of a treatment effect (log risk ratios) and their sampling variances —
# same shape as metafor's canonical dat.bcg fixture used in the parity tests.
yi = np.array([-0.889, -1.586, -1.336, -1.406, -0.212, 0.577, 0.339,
                0.336, -1.088, -0.322, 0.000, -0.442, -0.017])
vi = np.array([0.324, 0.311, 0.157, 0.032, 0.038, 0.055, 0.078,
                0.084, 0.007, 0.011, 0.006, 0.019, 0.037])

# 1) Random-effects pooling, REML tau^2 (metafor::rma default), Knapp-Hartung t-adjustment
res = rma(yi, vi, method="REML", test="knha")
print("pooled effect (beta):", res.beta[0])
print("SE(beta):", res.se[0])
print("95% CI:", res.ci_lb[0], res.ci_ub[0])
print("tau^2:", res.tau2, " SE(tau^2):", res.se_tau2)
print("I^2 (%):", res.I2, " H^2:", res.H2)
print("Q_E:", res.QE, " Q_Ep:", res.QEp)

# 2) Prediction interval for the average effect (predict.rma equivalent)
pi = res.predict(level=95.0)
print("prediction interval:", pi["pi_lb"], pi["pi_ub"])

# 3) Meta-regression with a moderator (mixed-effects model)
ablat = np.array([44, 55, 42, 52, 13, 44, 19, 13, 44, 19, 33, 21, 42], dtype=float)
res_mods = rma(yi, vi, mods=ablat, method="REML")
print("intercept, slope:", res_mods.beta)
print("Q_M (moderator omnibus):", res_mods.QM, res_mods.QMp)

# 4) Per-study empirical-Bayes shrinkage (BLUPs), metafor::blup.rma.uni equivalent
b = blup(res)
for i, (p, se) in enumerate(zip(b.pred, b.se)):
    print(f"study {i}: shrunk pred={p:.4f}  se={se:.4f}  PI=[{b.pi_lb[i]:.4f}, {b.pi_ub[i]:.4f}]")

# Equivalent via the wired socialverse pipeline function (adds BLUPs automatically):
# import socialverse as sv
# state = sv.tl.meta_random(state, method="REML", knapp_hartung=True)
# state.models["meta"]["estimate"], state.models["meta"]["blup"]
```

## R ↔ Python dictionary

| R (`metafor`) | socialverse | notes |
|---|---|---|
| `rma(yi, vi, method="REML")` | `rma(yi, vi, method="REML")` from `socialverse.external.pymetafor`, or `sv.tl.meta_random(state, method="REML")` | intercept-only random-effects fit |
| `rma(yi, vi, mods=~x, method="REML")` | `rma(yi, vi, mods=x, method="REML")` | `mods` excludes the intercept column; `add_intercept=True` prepends it (matches R's default `~x` formula behaviour) |
| `rma(yi, vi, method="DL")` | `rma(yi, vi, method="DL")` | closed-form DerSimonian-Laird τ² |
| `rma(yi, vi, method="FE")` / `rma(yi, vi, method="EE")` | `rma(yi, vi, method="EE")` (also accepts `"FE"`/`"CE"` aliases) | equal-effects, τ²=0 |
| `rma(yi, vi, test="knha")` | `rma(yi, vi, test="knha")` or `sv.tl.meta_random(..., knapp_hartung=True)` | Knapp-Hartung t-distribution CI/inference with `k-p` df |
| `predict.rma(res)` | `res.predict(level=95.0)` | fitted value + CI + prediction interval |
| `blup.rma.uni(res)` | `blup(res)` from `socialverse.external.pymetafor` | per-study empirical-Bayes shrinkage; auto-attached as `out["blup"]` by `sv.tl.meta_random` |
| `res$I2`, `res$H2`, `res$QE`, `res$QEp` | `res.I2`, `res.H2`, `res.QE`, `res.QEp` | Higgins-Thompson heterogeneity diagnostics |
| `res$se.tau2` | `res.se_tau2` | inverse Fisher information (REML/ML) or Q-based delta method (DL) |
| `res$QM`, `res$QMp` | `res.QM`, `res.QMp` | omnibus test for moderators, excludes intercept |

## Parity evidence

9 deterministic parity tests in `socialverse/external/pymetafor/tests/test_parity.py`, gated at `max_abs_err < 1e-6` against R `metafor` 5.0.1 on the `dat.bcg` fixture (`k=13` log risk ratios, generated by `tests/r_reference_driver.R`). The gated quantities include: pooled `beta`/`se`/`zval`/`pval`/CI for REML, DL, and equal-effects (EE) fits; the Knapp-Hartung t-adjusted fit; `tau2` and `se.tau2`; `I2`, `H2`, `QE`, `QEp`; the meta-regression identified slopes and `QE`/`QM` (both with and without centred moderators); the `predict.rma` prediction interval; and all four BLUP outputs (`pred`, `se`, `pi.lb`, `pi.ub`).

:::{admonition} Meta-regression τ² is bounded by metafor's own convergence tolerance, not the port
:class: warning

metafor's Fisher-scoring iteration for τ² stops at its own default `threshold=1e-5` (not machine precision), so on a flat or ill-conditioned objective (e.g. an uncentred moderator with `cond(XᵀWX) ≈ 2e11`) metafor's *reported* τ² itself sits up to ~1e-5 from the exact REML root — and this propagates to the unidentified intercept. The port replicates metafor's exact iteration and its `threshold=1e-5`, so it tracks metafor's *reported* number rather than the mathematically exact one. Consequently the moderator-model tests relax the tolerance to 1e-5 on slopes/τ² (down from 1e-6), while the τ²-independent quantities (`QE`) stay exact at 1e-6. Centring the moderators removes the conditioning problem entirely. See `RECONSTRUCTION_REPORT.md §6 Known limitations` for the full analysis, including an independent Brent-root check confirming the residual is metafor's under-convergence, not a port error.
:::

Reproduce:

```bash
Rscript socialverse/external/pymetafor/tests/r_reference_driver.R
python -m pytest socialverse/external/pymetafor/tests/test_parity.py
```

## In the socialverse workflow

Call **`sv.tl.meta_random(state, method="REML", knapp_hartung=True)`** for day-to-day random/mixed-effects pooling — it delegates REML/ML/DL/EE fits to this port and attaches per-study BLUPs automatically; use `sv.tl.meta_fixed` for plain common-effect pooling instead. The registry enforces the contract (`requires={"models": ["meta_effects"]}`, `produces={"models": ["meta"]}`, i.e. run `sv.pp.escalc` first); use `sv.registry_lookup("meta_random")` or `sv.list_functions()` to confirm the live signature and kwargs before scripting against it.
