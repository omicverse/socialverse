"""pyfixest — pure-numpy reconstruction of R fixest::feols.

Implements one- and two-way fixed-effects OLS via the within (demeaning)
estimator, with fixest-compatible cluster-robust standard errors and the
within-R2 fit statistic.

Algorithm (matches fixest 0.14.2 defaults, class-1 deterministic):

* Fixed effects are partialled out by alternating projections (iterated
  group-mean demeaning), which is exact for one-way FE in one pass and
  converges for two-way FE.
* Slopes solve the demeaned normal equations  (X~'X~) b = X~'y~.
* Residuals are the within residuals  y~ - X~ b  (equal to the full-model
  residuals since the FE are orthogonal to the demeaned regressors).
* Clustered vcov (fixest default ssc: adj=TRUE, cluster.adj=TRUE,
  cluster.df="conventional", fixef.K="nested"):

      V = (X~'X~)^-1 [ sum_g s_g s_g' ] (X~'X~)^-1 * c

  with s_g = X~_g' res_g the per-cluster score sum, and small-sample
  correction

      c = G/(G-1) * (N-1)/(N-K)

  where K counts the slope coefficients plus the parameters of any fixed
  effect NOT nested in the cluster variable, plus 1 whenever at least one
  FE dimension IS nested in the cluster (the implicit intercept revealed by
  the nesting).  This is fixest's fixef.K="nested" rule.

* Within-R2 = 1 - SSR / SS(y~), the demeaned total sum of squares.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# demeaning
# --------------------------------------------------------------------------
def _group_demean(col, codes, n_levels):
    """Subtract group means of a single column given integer group codes."""
    sums = np.bincount(codes, weights=col, minlength=n_levels)
    cnts = np.bincount(codes, minlength=n_levels)
    means = sums / cnts
    return col - means[codes]


def _factorize(values):
    """Return (integer codes, n_levels) for a 1-D array of labels."""
    values = np.asarray(values)
    uniq, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.intp), len(uniq)


def demean(M, fe_codes, tol=1e-12, maxit=100000):
    """Partial out one or more fixed effects from every column of M.

    Parameters
    ----------
    M : (N, p) array
    fe_codes : list of (codes, n_levels) tuples, one per FE dimension
    """
    M = np.asarray(M, float).copy()
    if M.ndim == 1:
        M = M[:, None]
    if len(fe_codes) == 1:
        codes, nl = fe_codes[0]
        for j in range(M.shape[1]):
            M[:, j] = _group_demean(M[:, j], codes, nl)
        return M
    # multiple FE: alternating projections
    for _ in range(maxit):
        M_old = M.copy()
        for codes, nl in fe_codes:
            for j in range(M.shape[1]):
                M[:, j] = _group_demean(M[:, j], codes, nl)
        if np.max(np.abs(M - M_old)) < tol:
            break
    return M


# --------------------------------------------------------------------------
# fixed-effect parameter counting (for the (N-1)/(N-K) adjustment)
# --------------------------------------------------------------------------
def _fe_param_counts(fe_codes):
    """Degrees of freedom used by each FE dim, accounting for collinearity.

    The first FE dim contributes its number of levels.  Each subsequent FE
    dim contributes (levels - 1) because one level is collinear with the
    span of the earlier FE (the shared intercept).  This reproduces
    fixest's `nparams` for the demeaned model.
    """
    counts = []
    for i, (_, nl) in enumerate(fe_codes):
        counts.append(nl if i == 0 else nl - 1)
    return counts


def _is_nested(fe_codes_i, cluster_codes):
    """True if FE dim (codes) is nested within the cluster grouping."""
    fe_c = fe_codes_i[0]
    # nested iff every FE level maps to exactly one cluster level
    order = np.argsort(fe_c, kind="stable")
    fe_sorted = fe_c[order]
    cl_sorted = np.asarray(cluster_codes)[order]
    # within each block of equal fe, cluster must be constant
    boundaries = np.where(np.diff(fe_sorted) != 0)[0] + 1
    for block in np.split(cl_sorted, boundaries):
        if block.size and np.any(block != block[0]):
            return False
    return True


# --------------------------------------------------------------------------
# public estimator
# --------------------------------------------------------------------------
def feols(y, X, fe, cluster):
    """One/two-way fixed-effects OLS with clustered SE, fixest-compatible.

    Parameters
    ----------
    y : (N,) array-like — dependent variable
    X : (N,) or (N, p) array-like — regressors (no intercept; FE absorb it)
    fe : array-like or list of array-likes — one or two FE grouping vectors
    cluster : array-like — cluster grouping vector

    Returns
    -------
    dict with keys: coef (p,), se (p,), vcov (p,p), within_r2 (float),
    nobs, nparams, n_clusters, resid (N,).
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N, p = X.shape

    # normalise FE spec to a list
    if isinstance(fe, (list, tuple)) and not np.isscalar(fe[0]) and np.ndim(fe[0]) == 1:
        fe_list = list(fe)
    elif isinstance(fe, (list, tuple)) and np.ndim(fe) == 2:
        fe_list = [np.asarray(f) for f in fe]
    else:
        fe_list = [np.asarray(fe)]
    # guard: a single 1-D vector passed as list-of-scalars vs list-of-vectors
    if len(fe_list) >= 1 and np.ndim(fe_list[0]) == 0:
        fe_list = [np.asarray(fe)]

    fe_codes = [_factorize(f) for f in fe_list]
    cluster_codes, G = _factorize(cluster)

    # within transform
    Xw = demean(X, fe_codes)
    yw = demean(y, fe_codes).ravel()

    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta

    # clustered meat
    meat = np.zeros((p, p))
    for g in range(G):
        idx = cluster_codes == g
        s_g = Xw[idx].T @ resid[idx]          # (p,)
        meat += np.outer(s_g, s_g)
    V_raw = XtX_inv @ meat @ XtX_inv

    # fixest small-sample correction, fixef.K="nested"
    fe_counts = _fe_param_counts(fe_codes)
    any_nested = False
    K = p
    for i in range(len(fe_codes)):
        if _is_nested(fe_codes[i], cluster_codes):
            any_nested = True                 # nested FE params dropped from K
        else:
            K += fe_counts[i]
    if any_nested:
        K += 1
    correction = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
    V = V_raw * correction
    se = np.sqrt(np.diag(V))

    ssr = float(resid @ resid)
    ss_within = float(yw @ yw)
    within_r2 = 1.0 - ssr / ss_within

    nparams = p + sum(fe_counts)
    return {
        "coef": beta,
        "se": se,
        "vcov": V,
        "within_r2": within_r2,
        "nobs": N,
        "nparams": nparams,
        "n_clusters": G,
        "resid": resid,
    }


# --------------------------------------------------------------------------
# weighted demeaning (for the Poisson IRLS working response)
# --------------------------------------------------------------------------
def _wgroup_demean(col, w, codes, n_levels):
    """Subtract weighted group means of one column (weights ``w``)."""
    wsum = np.bincount(codes, weights=w, minlength=n_levels)
    wxsum = np.bincount(codes, weights=w * col, minlength=n_levels)
    means = wxsum / wsum
    return col - means[codes]


def _wdemean(M, w, fe_codes, tol=1e-13, maxit=100000):
    """Weighted partialling-out of the FE from every column of ``M``.

    Uses weighted alternating projections with weights ``w`` (the IRLS
    working weights).  Exact for one-way FE in a single sweep; iterates to
    convergence for multi-way FE.
    """
    M = np.asarray(M, float).copy()
    if M.ndim == 1:
        M = M[:, None]
    if len(fe_codes) == 1:
        codes, nl = fe_codes[0]
        for j in range(M.shape[1]):
            M[:, j] = _wgroup_demean(M[:, j], w, codes, nl)
        return M
    for _ in range(maxit):
        M_old = M.copy()
        for codes, nl in fe_codes:
            for j in range(M.shape[1]):
                M[:, j] = _wgroup_demean(M[:, j], w, codes, nl)
        if np.max(np.abs(M - M_old)) < tol:
            break
    return M


# --------------------------------------------------------------------------
# Poisson pseudo-ML with high-dimensional fixed effects (PPML)
# --------------------------------------------------------------------------
def fepois(y, X, fe, cluster=None, tol=1e-10, maxit=1000):
    """PPML (Poisson pseudo-maximum-likelihood) with HD fixed effects.

    Matches ``fixest::fepois(y ~ X | fe, cluster = ~cluster)`` (fixest
    0.14.2, class-1 deterministic) for panels where no FE group is dropped
    for all-zero outcomes / singletons.

    Algorithm — IRLS on the Poisson log-link.  At each iteration the working
    weights are ``w = mu`` and the working response is
    ``z = eta + (y - mu) / mu``.  The FE are partialled out of ``X`` and of
    ``z`` by *weighted* alternating within-demeaning (weights ``w``), so the
    fixed effects are concentrated out rather than carried as explicit
    dummies.  The slope solves the weighted demeaned normal equations
    ``(X~' W X~) b = X~' W z~`` and ``eta`` is updated from the implied fit;
    iterating to convergence of the deviance.

    Standard errors
    ---------------
    The observation score is ``s_i = X~_i (y_i - mu_i)`` (the demeaned
    regressor times the raw Poisson residual — this is the gradient of the
    concentrated log-likelihood).  The bread is ``(X~' W X~)^-1``.  Clustered
    vcov:

        V = bread [ sum_g (sum_{i in g} s_i)(...)' ] bread * c

    with the *same* fixest nested small-sample correction as :func:`feols`,
    ``c = G/(G-1) * (N-1)/(N-K)`` where ``K`` counts the slopes plus one for
    the intercept revealed by a cluster-nested FE.

    Parameters
    ----------
    y : (N,) count outcome
    X : (N,) or (N, p) regressors (no intercept; FE absorb it)
    fe : array-like or list of array-likes — FE grouping vector(s)
    cluster : array-like or None — cluster grouping (defaults to first FE)

    Returns
    -------
    dict with keys: coef (p,), se (p,), vcov (p,p), mu (N,), eta (N,),
    deviance (float), nobs, n_clusters, n_iter.
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N, p = X.shape

    if isinstance(fe, (list, tuple)) and np.ndim(fe[0]) == 1:
        fe_list = [np.asarray(f) for f in fe]
    else:
        fe_list = [np.asarray(fe)]
    fe_codes = [_factorize(f) for f in fe_list]

    if cluster is None:
        cluster_codes, G = fe_codes[0][0], fe_codes[0][1]
    else:
        cluster_codes, G = _factorize(cluster)

    # IRLS.  Initialise mu à la fixest/glm: mu0 = (y + mean(y)) / 2.
    mu = (y + y.mean()) / 2.0
    eta = np.log(mu)
    beta = np.zeros(p)

    def _deviance(y_, mu_):
        # Poisson deviance; y*log(y/mu) -> 0 as y -> 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y_ > 0, y_ * np.log(y_ / mu_), 0.0)
        return 2.0 * np.sum(t - (y_ - mu_))

    dev_old = _deviance(y, mu)
    n_iter = 0
    for n_iter in range(1, maxit + 1):
        w = mu                                   # IRLS working weights
        z = eta + (y - mu) / mu                  # working response
        Xw = _wdemean(X, w, fe_codes)
        zw = _wdemean(z, w, fe_codes).ravel()

        WX = w[:, None] * Xw
        XtWX = Xw.T @ WX
        XtWz = Xw.T @ (w * zw)
        beta = np.linalg.solve(XtWX, XtWz)

        # eta update: recover FE contribution from the fit.  With eta = z - r
        # where r is the working residual orthogonal (in W-metric) to the FE,
        #   eta_new = z - (zw - Xw beta)
        resid_w = zw - Xw @ beta
        eta = z - resid_w
        mu = np.exp(eta)

        dev = _deviance(y, mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            dev_old = dev
            break
        dev_old = dev

    # final bread & scores at convergence
    w = mu
    Xw = _wdemean(X, w, fe_codes)
    XtWX = Xw.T @ (w[:, None] * Xw)
    bread = np.linalg.inv(XtWX)
    scores = Xw * (y - mu)[:, None]              # (N, p) gradient contributions

    meat = np.zeros((p, p))
    for g in range(G):
        s_g = scores[cluster_codes == g].sum(axis=0)
        meat += np.outer(s_g, s_g)
    V_raw = bread @ meat @ bread

    # fixest nested small-sample correction (same rule as feols)
    K = p
    any_nested = False
    for i in range(len(fe_codes)):
        if _is_nested(fe_codes[i], cluster_codes):
            any_nested = True
    if any_nested:
        K += 1
    correction = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
    V = V_raw * correction
    se = np.sqrt(np.diag(V))

    return {
        "coef": beta,
        "se": se,
        "vcov": V,
        "mu": mu,
        "eta": eta,
        "deviance": dev_old,
        "nobs": N,
        "n_clusters": G,
        "n_iter": n_iter,
    }


# --------------------------------------------------------------------------
# Newey-West HAC (heteroskedasticity + autocorrelation consistent) vcov
# --------------------------------------------------------------------------
def newey_west(y, X, lag, add_intercept=True, order=None):
    """OLS with a Newey-West (Bartlett-kernel) HAC vcov, fixest-compatible.

    Matches ``feols(y ~ X)`` re-summarised with ``vcov = NW(lag) ~ t``
    (fixest 0.14.2 default ssc, ``adj=TRUE``).

    The meat is the Bartlett-weighted sum of score autocovariances

        S = Gamma_0 + sum_{l=1}^{L} (1 - l/(L+1)) (Gamma_l + Gamma_l')

    with score ``s_t = X_t * e_t``, ``Gamma_l = sum_t s_t s_{t-l}'``, sandwiched
    by ``(X'X)^-1`` and scaled by the fixest degrees-of-freedom adjustment
    ``N/(N-K)`` (``K`` = number of estimated coefficients incl. intercept).

    Parameters
    ----------
    y : (N,) outcome
    X : (N,) or (N, p) regressors (no intercept column)
    lag : int — the HAC truncation lag L
    add_intercept : bool — prepend an intercept column (fixest default)
    order : array-like or None — time ordering of the rows.  If given, rows
        are sorted by it before forming autocovariances (NW is order-aware).

    Returns
    -------
    dict with keys: coef (p,), se (p,), vcov (p,p), resid (N,), nobs, nparams.
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]

    if order is not None:
        perm = np.argsort(np.asarray(order), kind="stable")
        y = y[perm]
        X = X[perm]

    N = X.shape[0]
    if add_intercept:
        X = np.column_stack([np.ones(N), X])
    K = X.shape[1]

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    scores = X * resid[:, None]                  # (N, K)
    S = scores.T @ scores                        # Gamma_0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)                # Bartlett weight
        Gl = scores[l:].T @ scores[:-l]          # Gamma_l = sum_t s_t s_{t-l}'
        S += w * (Gl + Gl.T)

    V = XtX_inv @ S @ XtX_inv
    V *= N / (N - K)                             # fixest adj=TRUE for HAC
    se = np.sqrt(np.diag(V))

    return {
        "coef": beta,
        "se": se,
        "vcov": V,
        "resid": resid,
        "nobs": N,
        "nparams": K,
    }
