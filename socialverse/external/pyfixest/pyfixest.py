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
