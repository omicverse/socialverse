"""Robust-DiD + heterogeneous/distributional treatment effects:
synth_did, honest_did, bartik_iv, metalearners, qte.

Each pins a known-truth synthetic DGP: the estimator recovers the effect where the
naive alternative is biased, honest_did reduces to the plain CI at M=0, and qte
recovers the distribution-varying effect.
"""
import numpy as np
import pandas as pd
import pytest

import socialverse as sv
from socialverse import datasets


def _st(df, outcome="y", treatment=None):
    st = sv.StudyState()
    st.write("variables", "outcome", outcome)
    if treatment:
        st.write("design", "treatment", treatment)
    sv.pp.ingest(st, data=df)
    return st


def test_bartik_iv_recovers_effect_where_ols_biased():
    rng = np.random.default_rng(0)
    n, K = 2000, 5
    sh = rng.dirichlet(np.ones(K), n)
    shocks = rng.normal(0, 1, K)
    z = sh @ shocks
    conf = rng.normal(0, 1, n)
    X = 0.8 * z + 0.6 * conf + rng.normal(0, 0.5, n)
    Y = 1.5 * X + 2.0 * conf + rng.normal(0, 0.5, n)   # true 1.5; conf confounds OLS
    df = pd.DataFrame({**{f"s{k}": sh[:, k] for k in range(K)}, "x": X, "y": Y})
    st = _st(df, treatment="x")
    sv.tl.bartik_iv(st, shares=[f"s{k}" for k in range(K)], shocks=list(shocks), endog="x")
    m = st.models["bartik_iv"]
    assert m["beta"] == pytest.approx(1.5, abs=0.2)
    assert abs(m["beta"] - 1.5) < abs(m["ols_beta"] - 1.5)   # beats OLS
    assert m["first_stage_F"] > 10 and m["weak_instrument"] is False


def test_metalearners_recover_ate_all_strategies():
    rng = np.random.default_rng(1)
    n = 2000
    X = rng.normal(0, 1, (n, 3))
    ps = 1 / (1 + np.exp(-X[:, 0]))
    T = (rng.random(n) < ps).astype(int)
    tau = 1.0 + 0.5 * X[:, 0]
    Y = tau * T + X[:, 0] + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2], "T": T, "y": Y})
    st = _st(df, treatment="T")
    sv.tl.metalearners(st, hetero=["x0", "x1", "x2"], learner="all")
    m = st.models["metalearners"]
    assert m["ate"] == pytest.approx(tau.mean(), abs=0.25)
    for lk in ("S", "T", "X"):
        assert m["ate_by_learner"][lk] == pytest.approx(tau.mean(), abs=0.3)
    assert m["cate_summary"]["p90"] > m["cate_summary"]["p10"]   # heterogeneity


def test_qte_captures_distributional_effect():
    rng = np.random.default_rng(2)
    n = 4000
    T = rng.integers(0, 2, n)
    base = rng.normal(0, 1, n)
    Y = base + T * (0.5 + 0.8 * (base > 0))   # effect larger in the upper tail
    df = pd.DataFrame({"T": T, "y": Y})
    st = _st(df, treatment="T")
    sv.tl.qte(st, quantiles=[0.1, 0.9], nboots=80)
    q = st.models["qte"]["qte"]
    assert q["0.9"]["qte"] > q["0.1"]["qte"]      # upper-tail effect larger
    assert q["0.1"]["se"] is not None


def test_synth_did_recovers_and_reports_jackknife():
    df = datasets.load_did_staggered(n_units=40, n_periods=16, att=2.0, seed=3)
    st = _st(df)
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.synth_did(st)
    m = st.models["synth_did"]
    assert m["att"] is not None and m["att"] > 1.0       # positive effect recovered
    assert m["se"] is not None and m["se"] > 0           # jackknife SE
    assert m["n_control"] >= 2


def test_honest_did_reduces_to_plain_ci_at_M0_and_reports_breakdown():
    df = datasets.load_did_staggered(n_units=40, n_periods=16, att=2.0, seed=3)
    st = _st(df)
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.sun_abraham(st)
    sv.tl.honest_did(st, target_period=3)
    h = st.diagnostics["honest_did"]
    ci0 = h["robust_ci"]["0.0"]
    # at M=0 the robust CI is exactly the sampling CI
    assert ci0["lo"] == pytest.approx(h["estimate"] - 1.96 * h["se"], abs=1e-6)
    assert ci0["hi"] == pytest.approx(h["estimate"] + 1.96 * h["se"], abs=1e-6)
    assert h["breakdown_M"] >= 0
    # the CI widens as M grows
    assert h["robust_ci"]["1.0"]["hi"] > h["robust_ci"]["0.0"]["hi"]


def test_honest_did_flat_pretrend_is_unbreakable_not_fragile():
    # a perfectly flat pre-trend with a significant effect is the MOST robust case:
    # breakdown must be +inf, not 0 (the earlier bug reported it as "fragile")
    st = sv.StudyState()
    sv.tl.honest_did(st, coefs={-3: (0.0, 0.1), -2: (0.0, 0.1), -1: (0.0, 0.1),
                                0: (2.0, 0.1), 1: (2.0, 0.1)}, target_period=1)
    h = st.diagnostics["honest_did"]
    assert h["breakdown_M"] == float("inf")
    assert "稳健" in h["verdict"]


def test_honest_did_bad_target_period_is_guarded():
    st = sv.StudyState()
    # target_period=9 is not among the post periods -> graceful, not a crash
    sv.tl.honest_did(st, coefs={-2: (0.0, 0.1), -1: (0.0, 0.1), 0: (1.0, 0.1)},
                     target_period=9)
    assert st.diagnostics["honest_did"].get("robust_ci") == {}  # returned _empty
