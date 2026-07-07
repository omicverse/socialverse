"""High-dimensional fixed effects in the DiD family.

``did`` / ``event_study`` / ``parallel_trends`` originally built explicit unit
dummies, which is infeasible once a panel has thousands of units (the design
matrix is huge and rank-deficient and OLS fails). These tests pin the
within-transformation path added for large panels: it recovers a known ATT at
scale, and it agrees with the explicit-dummy OLS on a small panel (Frisch-Waugh-
Lovell equivalence), so the fix changes feasibility, not the estimate.
"""
import numpy as np
import pandas as pd
import pytest

import socialverse as sv
from socialverse.tl._causal import _within_fit


def _staggered_panel(n_units=200, n_periods=12, att=2.0, seed=0):
    """Balanced staggered-adoption panel with unit + time FE and a known ATT."""
    rng = np.random.default_rng(seed)
    unit_fe = rng.normal(0, 3, n_units)
    time_fe = np.linspace(0, 2, n_periods)
    # half the units are ever-treated, adopting at staggered dates; half never.
    first = np.full(n_units, 0)  # 0 == never treated
    treated = rng.random(n_units) < 0.6
    first[treated] = rng.integers(3, n_periods - 1, treated.sum())
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            post = first[u] > 0 and t >= first[u]
            y = unit_fe[u] + time_fe[t] + att * post + rng.normal(0, 1)
            rows.append((u, t, int(post), int(first[u]), y))
    return pd.DataFrame(rows, columns=["unit", "period", "treat_post", "first_treated", "y"])


def _run_chain(df):
    st = sv.StudyState()
    st.write("estimand", "target", "ATT")
    st.write("variables", "outcome", "y")
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    sv.tl.event_study(st, base=-1)
    return st


def test_did_recovers_att_at_scale():
    """200 units × 12 periods — dummy OLS would choke; within recovers the ATT."""
    df = _staggered_panel(n_units=200, att=2.0)
    st = _run_chain(df)
    m = st.models["did"]
    assert m["estimator"] == "twfe_within_absorb_cluster"
    assert m["att"] == pytest.approx(2.0, abs=0.15)
    assert m["se"] is not None and m["se"] > 0
    # parallel trends should hold in this DGP (no pre-trend), and use the within path
    assert st.identification["parallel_trends"] in {"pass", "unknown"}
    assert st.models["event_study"]["estimator"] == "event_study_within_absorb_cluster"


def test_within_matches_dummy_ols_small_panel():
    """Frisch-Waugh-Lovell: absorbing FE == explicit dummies for point + robust SE."""
    sm = pytest.importorskip("statsmodels.api")
    df = _staggered_panel(n_units=30, n_periods=8, att=1.5, seed=3)
    valid = df["y"].notna()
    w = df.loc[valid]

    fit = _within_fit(w["y"], w["treat_post"].to_numpy(float),
                      w["unit"], w["period"], w["unit"])

    unit_d = pd.get_dummies(w["unit"].astype("category"), prefix="u", drop_first=True, dtype=float)
    time_d = pd.get_dummies(w["period"].astype("category"), prefix="t", drop_first=True, dtype=float)
    X = pd.concat([w[["treat_post"]].astype(float), unit_d, time_d], axis=1)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(w["y"].to_numpy(float), np.asarray(Xc, float)).fit(
        cov_type="cluster", cov_kwds={"groups": w["unit"].to_numpy()})

    # point estimate identical (FWL)
    assert float(fit["beta"][0]) == pytest.approx(float(res.params[1]), abs=1e-8)
    # cluster-robust SE matches statsmodels to a few percent (same small-sample adj.)
    se_within = float(np.sqrt(fit["V_cluster"][0, 0]))
    assert se_within == pytest.approx(float(res.bse[1]), rel=0.05)


def test_within_none_when_no_residual_variation():
    """A regressor collinear with the fixed effects has no within variation."""
    df = _staggered_panel(n_units=180, n_periods=6, seed=1)
    # treat_post constant within every unit -> absorbed by unit FE -> no variation
    df["treat_post"] = (df["unit"] % 2).astype(int)
    fit = _within_fit(df["y"], df["treat_post"].to_numpy(float),
                      df["unit"], df["period"], df["unit"])
    assert fit is None
