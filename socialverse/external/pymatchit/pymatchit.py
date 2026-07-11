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

__all__ = [
    "glm_logit_ps",
    "nearest_match",
    "smd",
    "matchit",
    "MatchItResult",
    "get_w_from_ps",
    "mahalanobis_dist",
    "balance_table",
]


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


# ---------------------------------------------------------------------------
# WeightIt::get_w_from_ps — propensity-score -> balancing weights
# ---------------------------------------------------------------------------
def get_w_from_ps(ps, treat, estimand="ATE", treated=1):
    """Convert a binary propensity score to balancing weights (WeightIt).

    Faithful port of ``WeightIt::get_w_from_ps(ps, treat, estimand)`` for a
    **binary** treatment supplied as a length-n vector of P(treated).  R builds
    the 2-column matrix ``ps_mat = [1-ps, ps]`` whose second column is the
    *treated* level, then:

      * ``ATE`` : ``w = 1 / ps_mat[i, treat_i]``  → treated ``1/ps``, control
        ``1/(1-ps)``;
      * ``ATT`` : focal = treated level; treated get weight 1, controls get
        ``ps_mat[i, focal] / ps_mat[i, treat_i] = ps/(1-ps)``;
      * ``ATC`` : focal = control level; controls get weight 1, treated get
        ``(1-ps)/ps``.

    Parameters
    ----------
    ps : (n,) propensity scores, P(treat == ``treated``).
    treat : (n,) treatment indicator (values equal to ``treated`` mark treated).
    estimand : one of ``"ATE"``, ``"ATT"``, ``"ATC"`` (case-insensitive).
    treated : the value in ``treat`` denoting the treated level (default 1).

    Returns
    -------
    w : (n,) balancing weights, matching WeightIt element-wise.
    """
    ps = np.asarray(ps, float)
    treat = np.asarray(treat)
    n = ps.shape[0]
    est = str(estimand).upper()

    # ps_mat columns: [P(control level), P(treated level)] = [1-ps, ps]
    is_treated = treat == treated
    w = np.ones(n, float)

    if est == "ATE":
        w[is_treated] = 1.0 / ps[is_treated]
        w[~is_treated] = 1.0 / (1.0 - ps[~is_treated])
    elif est == "ATT":
        # focal = treated: treated weight 1, control ps/(1-ps)
        w[is_treated] = 1.0
        w[~is_treated] = ps[~is_treated] / (1.0 - ps[~is_treated])
    elif est == "ATC":
        # focal = control: control weight 1, treated (1-ps)/ps
        w[~is_treated] = 1.0
        w[is_treated] = (1.0 - ps[is_treated]) / ps[is_treated]
    else:
        raise ValueError(f"unsupported estimand {estimand!r} (ATE/ATT/ATC)")
    return w


# ---------------------------------------------------------------------------
# MatchIt::mahalanobis_dist — pairwise (scaled) Mahalanobis distances
# ---------------------------------------------------------------------------
def mahalanobis_dist(X, treat):
    """Pairwise Mahalanobis distances between treated and control units.

    Faithful port of ``MatchIt:::mahalanobis_dist(formula, data)`` on the
    ``treat``-supplied path used by ``matchit(distance="mahalanobis")``:

      1. ``X <- scale(X)`` — standardize each column (center by mean, divide by
         the sample SD, ``ddof=1``);
      2. ``var <- pooled_cov(X, treat)`` — the within-group-centered covariance
         with the binary small-sample correction ``* (n-1)/(n-2)``;
      3. whiten ``X`` by ``inv(var)`` via a Cholesky factor
         (``mahalanobize``: ``X @ chol(inv_var)`` reordered by the pivot), so
         Euclidean distance in the whitened space equals Mahalanobis distance;
      4. return the ``n1 x n0`` matrix of Euclidean distances between each
         treated row and each control column (``eucdist_internal``).

    Because the whitening only needs to satisfy
    ``W W' = inv(var)`` for the squared distance
    ``(x_t - x_c)' inv(var) (x_t - x_c)`` to be reproduced, any valid factor
    gives the *same* pairwise distances; we use the symmetric matrix square
    root of ``inv(var)`` for numerical stability.  Distances reproduce R
    element-wise.

    Parameters
    ----------
    X : (n, p) covariate matrix (no intercept).
    treat : (n,) 0/1 treatment indicator.

    Returns
    -------
    D : (n1, n0) array; ``D[i, j]`` is the Mahalanobis distance between the
        i-th treated unit and the j-th control unit (in original row order).
    """
    X = np.asarray(X, float)
    treat = np.asarray(treat, int)
    n, p = X.shape

    # scale(): center by column mean, divide by sample SD (ddof=1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Xs = (X - mu) / sd

    # pooled within-group covariance (binary treat -> * (n-1)/(n-2))
    Xc = Xs.copy()
    ut = np.unique(treat)
    for g in ut:
        m = treat == g
        Xc[m] -= Xc[m].mean(axis=0)
    cov = np.cov(Xc, rowvar=False, ddof=1)  # cov() on already-centered X
    cov = np.atleast_2d(cov) * (n - 1) / (n - len(ut))

    inv_var = np.linalg.solve(cov, np.eye(p))
    # symmetric square root of inv_var: whitening W with W W' = inv_var
    evals, evecs = np.linalg.eigh(inv_var)
    W = evecs @ np.diag(np.sqrt(np.clip(evals, 0, None))) @ evecs.T
    Xw = Xs @ W

    tr = np.flatnonzero(treat == 1)
    co = np.flatnonzero(treat == 0)
    diff = Xw[tr][:, None, :] - Xw[co][None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


# ---------------------------------------------------------------------------
# MatchIt summary() balance table: SMD + Var.Ratio + eCDF (mean & max)
# ---------------------------------------------------------------------------
def _wvar(x, w, bin_var=False):
    """MatchIt wvar(): reliability-weighted variance (w renormalized to 1)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    mx = np.sum(w * x)
    if bin_var:
        return mx * (1.0 - mx)
    return np.sum(w * (x - mx) ** 2) / (1.0 - np.sum(w ** 2))


def _ecdf_std(x, treat, w):
    """MatchIt qqsum(standardize=TRUE): standardized eCDF (mean, max) diff.

    Weights are renormalized to sum to 1 *within each treatment group*, the
    treated group's contributions are negated, and the running cumulative sum
    over sorted ``x`` is evaluated at each distinct-value boundary; ``meandiff``
    is the mean of the absolute boundary heights, ``maxdiff`` the max.
    """
    x = np.asarray(x, float)
    treat = np.asarray(treat)
    w = np.asarray(w, float).copy()

    # binary covariate: eCDF diff == |mean diff|
    ux = np.unique(x)
    if ux.size <= 2 and np.all((x == 0) | (x == 1)):
        t1 = treat == treat[0]
        d = abs(np.sum(w[t1] * x[t1]) / np.sum(w[t1])
                - np.sum(w[~t1] * x[~t1]) / np.sum(w[~t1]))
        return d, d

    # renormalize weights to sum to 1 within each group
    for g in np.unique(treat):
        m = treat == g
        w[m] = w[m] / w[m].sum()

    order = np.argsort(x, kind="stable")
    x_ord = x[order]
    w_ord = w[order]
    t_ord = treat[order]
    # negate the first-appearing group's weights (t == t[0] in R)
    first = t_ord == t_ord[0]
    w_signed = w_ord.copy()
    w_signed[first] = -w_signed[first]

    cs = np.abs(np.cumsum(w_signed))
    # keep positions at distinct-x boundaries: c(diff1(x_ord) != 0, TRUE)
    keep = np.empty(x_ord.shape[0], dtype=bool)
    keep[:-1] = np.diff(x_ord) != 0
    keep[-1] = True
    ediff = cs[keep]
    return float(np.mean(ediff)), float(np.max(ediff))


def balance_table(X, treat, weights=None, covariates=None):
    """MatchIt ``summary()`` balance columns, before and/or after weighting.

    For each covariate column of ``X`` returns the four balance statistics R's
    ``summary.matchit`` reports with ``standardize=TRUE``:

      * ``Std. Mean Diff.`` — ``(mean_t - mean_c) / sd_denom`` with
        ``sd_denom = sqrt(wvar(treated, s.weights))`` (the ``s.d.denom="treated"``
        default);
      * ``Var. Ratio`` — ``wvar(treated, w) / wvar(control, w)``;
      * ``eCDF Mean`` and ``eCDF Max`` — the standardized empirical-CDF
        distance summaries (see ``_ecdf_std``).

    Group means use ``weights`` (``w * s.weights``; here ``s.weights == 1``).
    The SMD denominator always uses the *unweighted* treated-group ``wvar`` so
    the same denominator applies before and after (MatchIt convention).

    Parameters
    ----------
    X : (n, p) covariate matrix.
    treat : (n,) 0/1 treatment indicator.
    weights : optional (n,) weights (matching or balancing).  ``None`` -> the
        unadjusted (all-ones) sample.
    covariates : optional column names for the returned dict order.

    Returns
    -------
    dict with keys ``"vars"``, ``"std_mean_diff"``, ``"var_ratio"``,
    ``"ecdf_mean"``, ``"ecdf_max"`` (each a list / array over columns).
    """
    X = np.asarray(X, float)
    treat = np.asarray(treat)
    n, p = X.shape
    ww = np.ones(n, float) if weights is None else np.asarray(weights, float)

    i1 = treat == 1
    i0 = treat == 0

    smd = np.empty(p)
    vr = np.empty(p)
    em = np.empty(p)
    ex = np.empty(p)

    for j in range(p):
        xx = X[:, j]
        bin_var = np.all((xx == 0) | (xx == 1))

        m_t = np.sum(ww[i1] * xx[i1]) / np.sum(ww[i1])
        m_c = np.sum(ww[i0] * xx[i0]) / np.sum(ww[i0])
        mdiff = m_t - m_c

        # s.d.denom = "treated": UNWEIGHTED (s.weights all 1) treated wvar
        std = np.sqrt(_wvar(xx[i1], np.ones(np.sum(i1)), bin_var))
        smd[j] = mdiff / std if abs(mdiff) > np.sqrt(np.finfo(float).eps) else 0.0

        if bin_var:
            vr[j] = np.nan
            em[j] = ex[j] = abs(mdiff)
        else:
            vr[j] = _wvar(xx[i1], ww[i1], bin_var) / _wvar(xx[i0], ww[i0], bin_var)
            em[j], ex[j] = _ecdf_std(xx, treat, ww)

    return {
        "vars": (list(covariates) if covariates is not None
                 else [f"x{j}" for j in range(p)]),
        "std_mean_diff": smd,
        "var_ratio": vr,
        "ecdf_mean": em,
        "ecdf_max": ex,
    }
