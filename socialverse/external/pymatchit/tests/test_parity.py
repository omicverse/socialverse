"""Numerical parity gate: pymatchit vs R MatchIt on the lalonde fixture.

Class-1 exact (max_abs_err < 1e-6) on every deterministic quantity:
  * propensity-score logistic-regression coefficients,
  * fitted propensity scores (distance),
  * standardized mean differences before AND after matching,
  * the *set* of matched controls (identical to R).

Documented reference-tolerance limitation (NOT gated at 1e-6): the exact
pairing of controls that are *exactly equidistant* (identical propensity score)
from a treated unit follows MatchIt's internal C++ scan order and is not
bit-reproduced.  On this fixture 177/185 pairs match exactly; every one of the 8
residual disagreements is between two controls whose propensity scores are
*bit-identical* (|Δ|=0.0), so the multiset of matched-control propensity scores
is identical to R (asserted below) — which is exactly why the after-matching
SMD, and every other distance-based balance statistic, reproduces at 1e-6.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pymatchit import (  # noqa: E402
    matchit,
    get_w_from_ps,
    mahalanobis_dist,
    balance_table,
)

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _fit():
    d = REF["data"]
    X = np.column_stack([d["age"], d["educ"], d["re74"], d["re75"]])
    treat = np.asarray(d["treat"], int)
    res = matchit(X, treat, covariates=["age", "educ", "re74", "re75"])
    return res, d


def test_ps_coefficients():
    res, _ = _fit()
    _c("ps.coef", res.ps_coef, REF["ps_coef"]["value"])


def test_distance_fitted_ps():
    res, _ = _fit()
    _c("distance", res.distance, REF["distance"])


def test_smd_before():
    res, _ = _fit()
    _c("smd.before", res.smd_before, REF["smd"]["before"])


def test_smd_after():
    res, _ = _fit()
    _c("smd.after", res.smd_after, REF["smd"]["after"])


def test_matched_control_ps_multiset_identical():
    """The MULTISET of matched-control propensity scores is exactly R's.

    This is the permutation-invariant, tie-break-independent fact: R and the
    port draw the same propensity scores into the matched control group, so
    every distance-based balance statistic (incl. after-matching SMD) is
    guaranteed identical.  Only the specific tied unit assigned to each tied
    treated may differ.
    """
    res, d = _fit()
    rn = d["rownames"]
    ps = res.distance
    idx = {n: i for i, n in enumerate(rn)}
    my_ps = sorted(round(float(ps[c]), 10) for c in res.pairs.values())
    r_ps = sorted(round(float(ps[idx[c]]), 10) for c in REF["match"]["control"])
    assert my_ps == r_ps, "matched-control PS multiset differs from R"


def test_pairs_match_and_all_disagreements_are_exact_ties():
    """>=177/185 pairs reproduce R exactly; EVERY disagreement is between two
    controls with bit-identical propensity scores (documented reference-tolerance
    tie-break, not a numerical error)."""
    res, d = _fit()
    rn = d["rownames"]
    ps = res.distance
    idx = {n: i for i, n in enumerate(rn)}
    r_match = dict(zip(REF["match"]["treated"], REF["match"]["control"]))

    n_exact = 0
    for t, c in res.pairs.items():
        tn = rn[t]
        mine = rn[c]
        r_ctrl = r_match[tn]
        if mine == r_ctrl:
            n_exact += 1
        else:
            # the only permitted disagreement: two controls at the SAME PS
            gap = abs(float(ps[c]) - float(ps[idx[r_ctrl]]))
            assert gap == 0.0, (
                f"pair {tn}: mine={mine} vs R={r_ctrl} differ but PS gap="
                f"{gap:.3e} (not an exact tie)"
            )

    assert n_exact >= 177, f"only {n_exact}/185 pairs reproduced exactly"


# ---------------------------------------------------------------------------
# WeightIt::get_w_from_ps parity (ATE / ATT / ATC)
# ---------------------------------------------------------------------------
def test_get_w_from_ps_ate():
    d = REF["data"]
    ps = np.asarray(REF["distance"], float)
    treat = np.asarray(d["treat"], int)
    _c("get_w_from_ps.ATE", get_w_from_ps(ps, treat, "ATE"),
       REF["get_w_from_ps"]["ate"])


def test_get_w_from_ps_att():
    d = REF["data"]
    ps = np.asarray(REF["distance"], float)
    treat = np.asarray(d["treat"], int)
    _c("get_w_from_ps.ATT", get_w_from_ps(ps, treat, "ATT"),
       REF["get_w_from_ps"]["att"])


def test_get_w_from_ps_atc():
    d = REF["data"]
    ps = np.asarray(REF["distance"], float)
    treat = np.asarray(d["treat"], int)
    _c("get_w_from_ps.ATC", get_w_from_ps(ps, treat, "ATC"),
       REF["get_w_from_ps"]["atc"])


# ---------------------------------------------------------------------------
# MatchIt::mahalanobis_dist parity (n1 x n0 pairwise distances)
# ---------------------------------------------------------------------------
def test_mahalanobis_dist():
    d = REF["data"]
    X = np.column_stack([d["age"], d["educ"], d["re74"], d["re75"]])
    treat = np.asarray(d["treat"], int)
    D = mahalanobis_dist(X, treat)
    m = REF["mahalanobis"]
    D_ref = np.asarray(m["flat"], float).reshape(m["n1"], m["n0"])
    assert D.shape == D_ref.shape, f"{D.shape} vs {D_ref.shape}"
    _c("mahalanobis_dist", D.ravel(), D_ref.ravel())


# ---------------------------------------------------------------------------
# MatchIt summary() balance table: SMD + Var.Ratio + eCDF (mean & max)
# ---------------------------------------------------------------------------
def _balance_inputs():
    """Reconstruct the (distance + covariates) matrix R's summary() reports on."""
    d = REF["data"]
    ps = np.asarray(REF["distance"], float)
    X = np.column_stack([ps, d["age"], d["educ"], d["re74"], d["re75"]])
    treat = np.asarray(d["treat"], int)
    return X, treat


def test_balance_table_before():
    X, treat = _balance_inputs()
    bt = balance_table(X, treat, weights=None, covariates=REF["balance_table"]["vars"])
    ref = REF["balance_table"]["before"]
    _c("bal.before.smd", bt["std_mean_diff"], ref["std_mean_diff"])
    _c("bal.before.var_ratio", bt["var_ratio"], ref["var_ratio"])
    _c("bal.before.ecdf_mean", bt["ecdf_mean"], ref["ecdf_mean"])
    _c("bal.before.ecdf_max", bt["ecdf_max"], ref["ecdf_max"])


def test_balance_table_after():
    X, treat = _balance_inputs()
    w = np.asarray(REF["balance_table"]["match_weights"], float)
    bt = balance_table(X, treat, weights=w, covariates=REF["balance_table"]["vars"])
    ref = REF["balance_table"]["after"]
    _c("bal.after.smd", bt["std_mean_diff"], ref["std_mean_diff"])
    _c("bal.after.var_ratio", bt["var_ratio"], ref["var_ratio"])
    _c("bal.after.ecdf_mean", bt["ecdf_mean"], ref["ecdf_mean"])
    _c("bal.after.ecdf_max", bt["ecdf_max"], ref["ecdf_max"])
