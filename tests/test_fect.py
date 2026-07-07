"""The counterfactual (imputation) estimator ``sv.tl.fect``.

FEct fits the untreated potential-outcome model on untreated cells, imputes the
counterfactual for treated cells, and averages the individual effects. These tests
pin: (1) additive FEct recovers a known ATT on a staggered panel with *dynamic*
effects where two-way-FE DID is biased; (2) IFEct with the right number of factors
recovers the ATT when a latent factor confounds the outcome; (3) always-treated
units (no pre-period) are dropped; (4) the block bootstrap yields a positive SE and
the placebo test does not flag a clean design.
"""
import numpy as np
import pandas as pd
import pytest

import socialverse as sv


def _panel(N=180, T=12, att=2.0, r_true=0, dynamic=False, seed=0):
    """Balanced staggered-adoption panel with unit + time FE, optional ``r_true``
    interactive factors, and a known (optionally dynamic) treatment effect."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 2, N)
    g = rng.normal(0, 1, T)
    lam = rng.normal(0, 1, (N, r_true)) if r_true else None
    f = rng.normal(0, 1, (T, r_true)) if r_true else None
    first = np.where(rng.random(N) < 0.55, rng.integers(4, T - 1, N), 0)  # 0 = never
    rows, eff = [], []
    for i in range(N):
        for t in range(T):
            treated = first[i] > 0 and t >= first[i]
            te = 0.0
            if treated:
                te = att * (t - first[i] + 1) / 3.0 if dynamic else att
            y = a[i] + g[t] + (lam[i] @ f[t] if r_true else 0.0) + te + rng.normal(0, 0.3)
            rows.append((i, t, int(treated), int(first[i]), y))
            if treated:
                eff.append(te)
    df = pd.DataFrame(rows, columns=["unit", "period", "treat_post", "first_treated", "y"])
    return df, float(np.mean(eff))


def _fit(df, **kw):
    st = sv.StudyState()
    st.write("variables", "outcome", "y")
    st.write("estimand", "target", "ATT")
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.fect(st, **kw)
    return st.models["fect"]


def test_additive_fect_recovers_dynamic_att():
    """Dynamic effects: FEct imputation recovers the true average, no factors."""
    df, true_att = _panel(att=2.0, dynamic=True, seed=1)
    m = _fit(df, r=0, nboots=80)
    assert m["estimator"] == "fect_additive_imputation"
    assert m["att"] == pytest.approx(true_att, abs=0.15)
    assert m["se"] is not None and m["se"] > 0
    assert m["ci"][0] < m["att"] < m["ci"][1]
    assert m["att_by_period"]  # dynamic path populated


def test_ifect_recovers_att_under_latent_factor():
    """A latent factor confounds trends; additive FEct is biased, IFEct(r=2) fixes it."""
    df, true_att = _panel(N=220, T=14, att=1.5, r_true=2, seed=2)
    add = _fit(df, r=0, nboots=1)
    ife = _fit(df, r=2, nboots=1)
    assert ife["estimator"] == "ifect_r2"
    # IFEct is at least as close to the truth as additive, and lands near it
    assert abs(ife["att"] - true_att) <= abs(add["att"] - true_att) + 1e-6
    assert ife["att"] == pytest.approx(true_att, abs=0.2)


def test_always_treated_units_dropped():
    """Units treated in every period have no counterfactual and must be dropped."""
    df, _ = _panel(att=2.0, seed=3)
    # force some units to be always-treated
    always = df["unit"].isin(df["unit"].unique()[:20])
    df.loc[always, "treat_post"] = 1
    df.loc[always, "first_treated"] = 0
    m = _fit(df, r=0, nboots=20)
    assert m["n_units_dropped"] >= 20
    assert m["att"] is not None


def test_placebo_clean_design_not_flagged():
    """A design with true parallel pre-trends should not trip the placebo test."""
    df, _ = _panel(att=2.0, seed=4)
    m = _fit(df, r=0, nboots=100, placebo=True, placebo_periods=3)
    pb = m["placebo"]
    assert pb is not None and pb.get("placebo_p") is not None
    assert pb["placebo_p"] > 0.05  # no spurious pre-treatment effect


def test_placebo_ifect_is_robust():
    """IFEct (r>=1) + placebo must not crash and must restrict to estimable units
    after holding out the pre-window (no degenerate-loading pollution)."""
    df, _ = _panel(N=200, T=14, att=1.5, r_true=1, seed=6)
    m = _fit(df, r=1, nboots=40, placebo=True, placebo_periods=3)
    pb = m["placebo"]
    assert pb is not None            # returned a result rather than raising
    assert "note" in pb              # always reports what it did
    if pb.get("placebo_p") is not None:
        assert pb["placebo_se"] is not None and pb["placebo_se"] > 0
