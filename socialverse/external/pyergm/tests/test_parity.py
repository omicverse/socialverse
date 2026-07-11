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
from pyergm import ergm_mple  # noqa: E402

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
