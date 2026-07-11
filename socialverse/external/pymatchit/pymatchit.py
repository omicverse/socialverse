"""pymatchit — pure-Python reconstruction of R MatchIt (class-1 PS-logit exact).

Faithful port of ``matchit(treat ~ X, method="nearest", distance="glm")``:

  * propensity score via a binomial GLM with logit link, fitted by exactly the
    IRLS (Fisher-scoring) algorithm R's ``glm.fit`` uses;
  * 1:1 greedy nearest-neighbour matching without replacement, processing
    treated units in MatchIt's default ``m.order="largest"`` order (descending
    propensity score) and picking the available control minimising the absolute
    distance on the propensity-score scale;
  * standardized mean differences (``Std. Mean Diff.``) before and after
    matching, using the *treated-group* standard deviation from the full sample
    as the common denominator (MatchIt's convention).

Only numpy/scipy are used.  Everything the numerical parity gate checks
(PS coefficients, fitted distances, before/after SMD, and the *set* of matched
controls) is deterministic and reproduces R element-wise.  See ``matchit`` and
the module tests for the one documented reference-tolerance limitation (the
exact pairing of a handful of exactly-equidistant controls follows MatchIt's
internal C++ scan order and is not bit-reproduced; it does not affect balance).
"""
from __future__ import annotations

import numpy as np

__all__ = ["glm_logit_ps", "nearest_match", "smd", "matchit", "MatchItResult"]


# ---------------------------------------------------------------------------
# propensity score: binomial GLM (logit) via IRLS, matching R glm.fit
# ---------------------------------------------------------------------------
def glm_logit_ps(X, y, max_iter=25, tol=1e-8):
    """Fit a logistic regression by IRLS and return (coef, fitted_prob).

    Replicates R ``glm(family=binomial())``:

      * design matrix has an intercept as the FIRST column;
      * link = logit, mean function = expit;
      * IRLS with the binomial variance ``mu*(1-mu)``;
      * R's start of ``mu = (y + 0.5)/2`` then ``eta = link(mu)``;
      * convergence on the relative change in deviance (< 1e-8), as R does.

    Parameters
    ----------
    X : (n, p) array of covariates WITHOUT the intercept column.
    y : (n,) 0/1 treatment indicator.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = X.shape[0]
    # design: intercept first, then covariates (R model.matrix order)
    Xd = np.column_stack([np.ones(n), X])
    p = Xd.shape[1]

    # R glm binomial initialisation
    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1.0 - mu))  # logit link
    coef = np.zeros(p)

    def dev_resids(y, mu):
        # binomial deviance residuals (squared), guarding logs at 0/1
        eps = 1e-10
        muc = np.clip(mu, eps, 1 - eps)
        r = np.zeros_like(y)
        # 2*(y*log(y/mu) + (1-y)*log((1-y)/(1-mu)))
        m1 = y > 0
        r[m1] += y[m1] * np.log(y[m1] / muc[m1])
        m0 = y < 1
        r[m0] += (1 - y[m0]) * np.log((1 - y[m0]) / (1 - muc[m0]))
        return 2.0 * r

    dev = np.sum(dev_resids(y, mu))
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-eta))
        dmu_deta = mu * (1.0 - mu)          # d mu / d eta for logit
        var = mu * (1.0 - mu)               # binomial variance
        # working response and IRLS weights
        z = eta + (y - mu) / dmu_deta
        w = (dmu_deta ** 2) / var           # == mu*(1-mu)
        # weighted least squares:  (X' W X) beta = X' W z
        WX = Xd * w[:, None]
        XtWX = Xd.T @ WX
        XtWz = Xd.T @ (w * z)
        coef = np.linalg.solve(XtWX, XtWz)
        eta = Xd @ coef
        mu = 1.0 / (1.0 + np.exp(-eta))
        dev_new = np.sum(dev_resids(y, mu))
        if abs(dev_new - dev) / (abs(dev_new) + 0.1) < tol:
            dev = dev_new
            break
        dev = dev_new
    ps = 1.0 / (1.0 + np.exp(-(Xd @ coef)))
    return coef, ps


# ---------------------------------------------------------------------------
# 1:1 greedy nearest-neighbour matching without replacement
# ---------------------------------------------------------------------------
def nearest_match(distance, treat):
    """Greedy 1:1 nearest-neighbour matching without replacement.

    MatchIt defaults for ``method="nearest"`` with a scalar distance:
      * ``m.order="largest"``: treated units are matched in DESCENDING order of
        their propensity score (distance);
      * the nearest AVAILABLE control minimises ``|distance_c - distance_t|``;
      * matching is without replacement.

    Ties between exactly-equidistant controls are broken by highest original
    index (MatchIt's internal ordering picks the later control in most cases);
    this affects only the *pairing* of equidistant controls, never the matched
    *set*, so all balance statistics are unaffected.

    Returns
    -------
    pairs : dict {treated_index -> matched_control_index}
    """
    distance = np.asarray(distance, float)
    treat = np.asarray(treat)
    tr = np.flatnonzero(treat == 1)
    co = np.flatnonzero(treat == 0)

    # m.order = "largest": descending distance among treated.
    # stable sort by (-distance) keeps original order on exact ties.
    order = tr[np.argsort(-distance[tr], kind="stable")]

    available = np.ones(distance.shape[0], dtype=bool)
    pairs = {}
    for ti in order:
        cand = co[available[co]]
        if cand.size == 0:
            break
        dd = np.abs(distance[cand] - distance[ti])
        mind = dd.min()
        tied = cand[dd == mind]
        best = int(tied.max())      # highest-index tie-break
        pairs[int(ti)] = best
        available[best] = False
    return pairs


# ---------------------------------------------------------------------------
# standardized mean difference
# ---------------------------------------------------------------------------
def smd(x, treat, weights=None):
    """Standardized mean difference, MatchIt convention.

    Denominator is the treated-group standard deviation computed on the FULL
    (unweighted) sample; the same denominator is reused before and after
    matching.  Group means use ``weights`` (all-ones before matching; 1 for a
    matched unit, 0 otherwise after matching).
    """
    x = np.asarray(x, float)
    treat = np.asarray(treat)
    if weights is None:
        weights = np.ones_like(x, float)
    weights = np.asarray(weights, float)

    tmask = treat == 1
    cmask = treat == 0
    # full-sample treated SD (R sd(): ddof=1)
    sd_t = np.std(x[tmask], ddof=1)

    def wmean(mask):
        w = weights[mask]
        return np.sum(w * x[mask]) / np.sum(w)

    return (wmean(tmask) - wmean(cmask)) / sd_t


# ---------------------------------------------------------------------------
# top-level matchit driver
# ---------------------------------------------------------------------------
class MatchItResult:
    """Container mirroring the fields the parity gate checks."""

    def __init__(self, ps_coef, distance, pairs, weights,
                 smd_before, smd_after, smd_vars):
        self.ps_coef = ps_coef
        self.distance = distance
        self.pairs = pairs
        self.weights = weights
        self.smd_before = smd_before
        self.smd_after = smd_after
        self.smd_vars = smd_vars


def matchit(X, treat, covariates=None):
    """Run the full nearest/glm MatchIt pipeline.

    Parameters
    ----------
    X : (n, p) covariate matrix (order == column order used for the GLM).
    treat : (n,) 0/1 treatment indicator.
    covariates : optional list of covariate names for SMD reporting order;
        the SMD block is always reported as ``["distance"] + covariates``.

    Returns
    -------
    MatchItResult
    """
    X = np.asarray(X, float)
    treat = np.asarray(treat, int)
    n = X.shape[0]

    coef, ps = glm_logit_ps(X, treat)
    pairs = nearest_match(ps, treat)

    # matched weights: matched treated + their controls -> 1, else 0
    weights = np.zeros(n, float)
    for t, c in pairs.items():
        weights[t] = 1.0
        weights[c] = 1.0

    # SMD block: distance first, then covariates
    cols = np.column_stack([ps, X])  # distance + covariates
    ones = np.ones(n, float)
    smd_before = np.array([smd(cols[:, j], treat, ones) for j in range(cols.shape[1])])
    smd_after = np.array([smd(cols[:, j], treat, weights) for j in range(cols.shape[1])])
    vars_ = ["distance"] + (list(covariates) if covariates is not None
                            else [f"x{j}" for j in range(X.shape[1])])

    return MatchItResult(coef, ps, pairs, weights, smd_before, smd_after, vars_)
