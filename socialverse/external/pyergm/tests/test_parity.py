"""Parity gate: pyergm MPLE vs R ergm on the canonical Padgett Florentine
marriage network (flomarriage ~ edges + nodecov('wealth'), estimate='MPLE').

Class-1 deterministic: MPLE is a convex logistic regression on dyads, so the
coefficients and model-based SEs must match ergm element-wise at 1e-6.

MCMC-MLE and RSiena SAOM are genuinely stochastic (class-2) and are out of the
deterministic 1e-6 scope — documented, not tested here.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pyergm import (  # noqa: E402
    TRIAD_CENSUS_LABELS,
    ergm_mple,
    summary_formula,
    triad_census,
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
    adj = np.asarray(d["adjacency"], float)
    wealth = np.asarray(d["wealth"], float)
    directed = bool(d["directed"])
    return ergm_mple(adj, ["edges", ("nodecov", wealth)], directed=directed)


def test_mple_coef():
    r = _fit()
    _c("mple.coef", r.coef, REF["mple"]["coef"])


def test_mple_se():
    r = _fit()
    _c("mple.se", r.se, REF["mple"]["se"])


def test_design_dimensions():
    # undirected 16-node net -> 120 dyads, 2 predictors (edges, nodecov)
    d = REF["data"]
    adj = np.asarray(d["adjacency"], float)
    from pyergm import build_design

    X, y, labels = build_design(adj, ["edges", ("nodecov", d["wealth"])])
    assert X.shape == (120, 2), X.shape
    assert y.sum() == np.triu(adj, 1).sum(), (y.sum(), np.triu(adj, 1).sum())
    assert labels == ["edges", "nodecov"]


# --------------------------------------------------------------------------- #
# summary(net ~ terms) — observed sufficient statistics (exact counts, 0 tol)
# --------------------------------------------------------------------------- #
def test_summary_undirected_stats():
    """flomarriage ~ edges+triangle+degree(0:6)+kstar(2)+nodecov+nodematch.

    Exact integer/real statistics — gated at 0 tolerance against ergm.
    """
    d = REF["data"]
    adj = np.asarray(d["adjacency"], float)
    terms = [
        "edges",
        "triangle",
        ("degree", list(range(0, 7))),
        ("kstar", 2),
        ("nodecov", d["wealth"]),
        ("nodematch", d["priorates"]),
    ]
    stats, labels = summary_formula(adj, terms, directed=False)
    exp = np.asarray(REF["summary_undirected"]["stats"], float)
    # counts + integer-valued covariate sums: exact match, 0 tolerance
    assert np.array_equal(stats, exp), f"got={stats}\n exp={exp}"


def test_summary_undirected_labels():
    d = REF["data"]
    adj = np.asarray(d["adjacency"], float)
    terms = [
        "edges",
        "triangle",
        ("degree", list(range(0, 7))),
        ("kstar", 2),
        ("nodecov", d["wealth"]),
        ("nodematch", d["priorates"]),
    ]
    _, labels = summary_formula(adj, terms, directed=False, attr_name=None)
    # bare-attr labels (no attr_name) still match the non-attr prefix
    assert labels[:10] == REF["summary_undirected"]["terms"][:10]
    assert labels[10] == "nodecov"
    assert labels[11] == "nodematch"


def test_summary_directed_stats():
    """directed 5-node fixture ~ edges+mutual+istar(2)+ostar(2)+i/odegree(1:2)."""
    d = REF["data"]
    Ad = np.asarray(d["dir_adjacency"], float)
    terms = [
        "edges",
        "mutual",
        ("istar", 2),
        ("ostar", 2),
        ("idegree", [1, 2]),
        ("odegree", [1, 2]),
    ]
    stats, labels = summary_formula(Ad, terms, directed=True)
    exp = np.asarray(REF["summary_directed"]["stats"], float)
    assert np.array_equal(stats, exp), f"got={stats}\n exp={exp}"
    assert labels == REF["summary_directed"]["terms"]


# --------------------------------------------------------------------------- #
# Holland-Leinhardt directed triad census (sna::triad.census) — 0 tol
# --------------------------------------------------------------------------- #
def test_triad_census_counts():
    d = REF["data"]
    Ad = np.asarray(d["dir_adjacency"], float)
    tc = triad_census(Ad)
    exp = np.asarray(REF["triad_census"]["counts"], float)
    assert np.array_equal(tc, exp), f"got={tc}\n exp={exp}"


def test_triad_census_labels_and_total():
    d = REF["data"]
    Ad = np.asarray(d["dir_adjacency"], float)
    n = Ad.shape[0]
    tc = triad_census(Ad)
    assert TRIAD_CENSUS_LABELS == REF["triad_census"]["labels"]
    # every triad classified exactly once: sum == C(n, 3)
    assert tc.sum() == n * (n - 1) * (n - 2) / 6
