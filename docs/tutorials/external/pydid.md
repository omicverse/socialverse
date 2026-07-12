# pydid — `did` (Callaway & Sant'Anna) in Python

> Staggered difference-in-differences with group-time average treatment effects `att_gt` and their `aggte` aggregations, callable from Python at 1e-6 parity with R's `did` — no R runtime required.

## What `did` does

`did` (Brantly Callaway & Pedro H.C. Sant'Anna) is the reference R implementation of staggered-adoption difference-in-differences, the workhorse identification strategy whenever a treatment (a policy, a minimum-wage change, a platform rollout) turns on for different units at different calendar times. Instead of a single two-way fixed-effects coefficient — which mixes "good" 2x2 comparisons with "forbidden" comparisons between already-treated cohorts, and can be badly biased under treatment-effect heterogeneity — `did` builds the estimand up from disaggregated group-time average treatment effects `ATT(g,t)`: the effect for the cohort first treated in period `g`, measured at period `t`. Social scientists reach for it whenever they have panel data with a variable date of policy adoption and want a causal estimate that is robust to heterogeneous and dynamic treatment effects, plus the standard event-study / cohort / calendar-time summaries that referees expect.

## The port

- `att_gt(data, yname, tname, idname, gname, control_group="nevertreated", est_method="reg", anticipation=0, base_period="varying")` — computes the group-time ATT(g,t) for every treated cohort `g` and period `t`, using the outcome-regression 2x2 estimator (intercept-only, i.e. difference-in-means) against either never-treated or not-yet-treated controls. Returns an `ATTgtResult`.
- `aggte(res, type="simple", max_e=np.inf, min_e=-np.inf, na_rm=False)` — aggregates an `ATTgtResult` into `type='simple'` (pg-weighted overall ATT), `type='dynamic'` (event-study ATT(e) by relative event time plus overall), `type='group'` (one ATT per treatment cohort plus cohort-size-weighted overall), or `type='calendar'` (one ATT per calendar period plus plain-mean overall). Returns an `AGGTEResult`.
- `ATTgtResult` — container mirroring the relevant fields of R's `did` MP object (`group`, `t`, `att`, `glist`, `tlist`, `pg`, `gvar_unit`).
- `AGGTEResult` — container for an aggregation (`type`, `overall_att`, `egt`, `att_egt`).

The port is pure numpy — no rpy2, no R subprocess. It reproduces `did::att_gt` + `did:::compute.att_gt` + `DRDID::reg_did_panel` (intercept-only outcome regression, no covariates) and `did::aggte`, for panel data with `control_group` in `{'nevertreated', 'notyettreated'}`, `est_method='reg'`, `anticipation=0`, `base_period='varying'` (the R default). It is wired into socialverse through the causal module in `socialverse/tl/_causal.py`: the registered functions `sv.tl.did` and `sv.tl.event_study` both call the internal helper `_cs_estimate` (which itself calls `att_gt` / `aggte` from `socialverse.external.pydid`) to replace the ad-hoc TWFE point estimate with the parity-verified Callaway–Sant'Anna ATT — `did` swaps in the `simple`/`group`/`calendar` aggregations for the overall ATT, `event_study` swaps in the `dynamic` aggregation for the per-relative-period coefficients. Both functions fall back to the existing TWFE estimator (`backend="twfe"` in the result) if the design lacks the columns pydid needs or the port raises.

:::{admonition} Parity gate
:class: note

The port is pinned to R `did` (reference version 2.5.1) to `max_abs_err < 1e-6` on 7 deterministic parity tests.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pydid import att_gt, aggte

# Small synthetic staggered-adoption panel: 6 units observed over 4 periods.
# first.treat = 0 marks never-treated units (the did convention for "no G").
n_periods = [1, 2, 3, 4]
data = {
    "id":    np.repeat([1, 2, 3, 4, 5, 6], 4),
    "t":     np.tile(n_periods, 6),
    # cohort 2 (units 1-2) treated from t=2, cohort 3 (units 3-4) from t=3,
    # units 5-6 never treated (control group).
    "g":     np.repeat([2, 2, 3, 3, 0, 0], 4).astype(float),
    "y":     np.array([
        1.0, 1.1, 2.6, 3.4,   # unit 1 (g=2): jump at t=2
        1.2, 1.3, 2.9, 3.6,   # unit 2 (g=2)
        0.9, 1.0, 1.1, 2.8,   # unit 3 (g=3): jump at t=3
        1.1, 1.2, 1.3, 3.0,   # unit 4 (g=3)
        1.0, 1.15, 1.3, 1.45, # unit 5 (never treated): parallel trend
        0.8, 0.95, 1.1, 1.25, # unit 6 (never treated)
    ]),
}

res = att_gt(
    data, yname="y", tname="t", idname="id", gname="g",
    control_group="nevertreated", est_method="reg",
)
for g, t, att in zip(res.group, res.t, res.att):
    print(f"ATT(g={g:.0f}, t={t:.0f}) = {att:.4f}")

# Aggregate into a single overall ATT (pg-weighted average of post-treatment cells).
simple = aggte(res, type="simple", na_rm=True)
print("overall ATT (simple):", simple.overall_att)

# Event-study aggregation: one ATT per relative event time e = t - g.
dyn = aggte(res, type="dynamic", na_rm=True)
for e, att_e in zip(dyn.egt, dyn.att_egt):
    print(f"ATT(e={e:.0f}) = {att_e:.4f}")
print("overall dynamic ATT (average over e >= 0):", dyn.overall_att)
```

## R ↔ Python dictionary

| R (`did`) | socialverse | notes |
|---|---|---|
| `att_gt(yname=, tname=, idname=, gname=, control_group=, est_method=, data=)` | `socialverse.external.pydid.att_gt(data, yname, tname, idname, gname, control_group, est_method, anticipation, base_period)` | port takes a column-name -> array mapping instead of a `data.frame`; only `est_method='reg'`, `anticipation=0`, `base_period='varying'` are ported |
| `aggte(res, type="simple", na.rm=)` | `aggte(res, type="simple", na_rm=)` | pg-weighted mean over post-treatment ATT(g,t) |
| `aggte(res, type="dynamic", na.rm=)` | `aggte(res, type="dynamic", na_rm=)` | returns `egt` (event times) / `att_egt` / `overall_att` |
| `aggte(res, type="group", na.rm=)` | `aggte(res, type="group", na_rm=)` | one ATT per cohort, cohort-size-weighted overall |
| `aggte(res, type="calendar", na.rm=)` | `aggte(res, type="calendar", na_rm=)` | one ATT per calendar period, plain-mean overall |
| `res$att`, `res$se` (bootstrap SE) | `res.att` (point est. only — no bootstrap SE) | see [Parity evidence](#parity-evidence) for the SE caveat |
| — (called from analysis code) | `sv.tl.did(state, control_group="nevertreated")` | overall/group/calendar ATT via `_cs_estimate`, `models.did.backend == "pydid"` when active |
| — (called from analysis code) | `sv.tl.event_study(state)` | dynamic ATT(e) via `_cs_estimate`, `models.event_study.backend == "pydid"` when active |

## Parity evidence

7 parity tests gate the port against R `did` 2.5.1 on the canonical `mpdta` county minimum-wage panel, at `max_abs_err < 1e-6`, covering:

- `att_gt` point estimates (`group`, `t`, `att`) for every group-time cell, both with `control_group='nevertreated'` and `control_group='notyettreated'`;
- `aggte(type='simple')` overall ATT;
- `aggte(type='dynamic')` — every event-time `att.egt` value plus the overall dynamic ATT;
- `aggte(type='group')` — per-cohort `att.egt` plus the cohort-size-weighted overall ATT;
- `aggte(type='calendar')` — per-period `att.egt` plus the plain-mean overall ATT.

:::{admonition} Bootstrap standard errors are not parity-gated
:class: warning

R's `did` reports multiplier-bootstrap standard errors (`bstrap=TRUE`, Mammen/Rademacher weights redrawn every run), which are inherently stochastic and differ run-to-run even within R itself. The port does not reproduce this RNG and returns point estimates only. The test suite only sanity-checks that R's reported SEs are finite and positive (`test_se_documented`) — it does not, and cannot, assert element-wise SE parity. Downstream, `sv.tl.did` / `sv.tl.event_study` keep their existing TWFE cluster-robust SE and re-centre the confidence interval on the pydid point estimate.
:::

To reproduce:

```bash
Rscript socialverse/external/pydid/tests/r_reference_driver.R
pytest socialverse/external/pydid/tests/
```

## In the socialverse workflow

Call `sv.tl.did` for the overall/group/calendar Callaway–Sant'Anna ATT, or `sv.tl.event_study` for the dynamic event-time path — both fall back to a TWFE estimator (`backend="twfe"` in the result) when the design lacks a usable `first_treated`/panel structure, so always check `models.did.backend` / `models.event_study.backend` to confirm pydid actually ran. The registry enforces each function's `requires`/`produces` contract; use `registry_lookup` or `sv.list_functions()` to confirm the live signature before scripting against it.
