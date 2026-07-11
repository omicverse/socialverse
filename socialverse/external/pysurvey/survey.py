"""Pure-Python reconstruction of R **survey** (Lumley) design-based estimation.

Reference-driven port under the Omicverse-RebuildR protocol; parity-gated against
survey 4.5 (``tests/test_parity.py``) on the canonical ``apistrat`` (stratified)
and ``apiclus1`` (one-stage cluster) designs at 1e-6.

Implements the **ultimate-cluster (Taylor-linearization)** variance estimator
used by ``svymean`` / ``svytotal`` / ``svyglm``:

    V(Σ zᵢ) = Σ_h (1 − f_h) · n_h/(n_h−1) · Σ_c (s_{hc} − s̄_h)(s_{hc} − s̄_h)ᵀ

where s_{hc} is the weighted-contribution total of PSU c in stratum h, n_h the
number of PSUs in stratum h, and f_h = n_h/N_h the sampling fraction from ``fpc``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg, stats

__all__ = ["svydesign", "svymean", "svytotal", "svyglm", "SurveyDesign"]


@dataclass
class SurveyDesign:
    y_cols: dict          # name -> array (carried columns)
    weights: np.ndarray
    strata: np.ndarray    # per-element stratum id (or all-0 if none)
    psu: np.ndarray       # per-element PSU id (element index if ids=~1)
    fpc: np.ndarray | None  # per-element population count of PSUs in its stratum
    n: int

    @property
    def degf(self) -> int:
        """degrees of freedom = (# PSUs) − (# strata)."""
        npsu = len({(s, p) for s, p in zip(self.strata, self.psu)})
        nstr = len(set(self.strata))
        return npsu - nstr


def svydesign(data, weights, ids=None, strata=None, fpc=None):
    """Build a survey design.

    data    : dict of column-name -> array-like.
    weights : array-like sampling weights.
    ids     : PSU id array, or None/1 for element sampling (ids=~1).
    strata  : stratum id array, or None for no stratification.
    fpc     : per-element population count (finite population correction), or None.
    """
    w = np.asarray(weights, float).ravel()
    n = w.size
    st = np.zeros(n, int) if strata is None else np.asarray(strata)
    if ids is None:
        psu = np.arange(n)
    else:
        psu = np.asarray(ids)
    fp = None if fpc is None else np.asarray(fpc, float).ravel()
    cols = {k: np.asarray(v, float).ravel() if np.asarray(v).dtype.kind in "fiu"
            else np.asarray(v) for k, v in data.items()}
    return SurveyDesign(y_cols=cols, weights=w, strata=st, psu=psu, fpc=fp, n=n)


def _vcov_total(Z, design: SurveyDesign):
    """Ultimate-cluster design variance of the total Σ Zᵢ.

    Z : (n,) or (n, p) per-element weighted contributions. Returns scalar or (p,p).
    """
    Z = np.asarray(Z, float)
    scalar = Z.ndim == 1
    if scalar:
        Z = Z[:, None]
    p = Z.shape[1]
    st, psu, fpc = design.strata, design.psu, design.fpc
    V = np.zeros((p, p))
    for h in np.unique(st):
        mh = st == h
        psu_h = psu[mh]
        Zh = Z[mh]
        uniq = np.unique(psu_h)
        n_h = uniq.size
        if n_h < 2:
            continue  # single PSU per stratum contributes 0 (survey's default)
        # PSU-level totals s_{hc}
        S = np.array([Zh[psu_h == c].sum(axis=0) for c in uniq])   # (n_h, p)
        sbar = S.mean(axis=0)
        D = S - sbar
        ssq = D.T @ D                                              # (p,p)
        f_h = 0.0
        if fpc is not None:
            Nh = float(fpc[mh][0])                                 # pop PSU count in h
            if Nh > 0:
                f_h = n_h / Nh
        V += (1.0 - f_h) * (n_h / (n_h - 1.0)) * ssq
    return float(V[0, 0]) if scalar else V


def _crit(level, df):
    a = (1 - level / 100.0) / 2.0
    return stats.t.ppf(1 - a, df)


def svymean(y, design: SurveyDesign, level=95.0):
    """Design-based mean of column ``y`` with linearized SE + t CI + df."""
    yv = design.y_cols[y] if isinstance(y, str) else np.asarray(y, float)
    w = design.weights
    sw = w.sum()
    ybar = float((w * yv).sum() / sw)
    z = w * (yv - ybar) / sw                    # linearized total variable
    var = _vcov_total(z, design)
    se = float(np.sqrt(var))
    df = design.degf
    crit = _crit(level, df)
    return {"estimate": ybar, "se": se, "df": df,
            "ci_lb": ybar - crit * se, "ci_ub": ybar + crit * se}


def svytotal(y, design: SurveyDesign, level=95.0):
    """Design-based total of column ``y`` with linearized SE."""
    yv = design.y_cols[y] if isinstance(y, str) else np.asarray(y, float)
    w = design.weights
    z = w * yv
    est = float(z.sum())
    se = float(np.sqrt(_vcov_total(z, design)))
    df = design.degf
    crit = _crit(level, df)
    return {"estimate": est, "se": se, "df": df,
            "ci_lb": est - crit * se, "ci_ub": est + crit * se}


def svyglm(y, X, design: SurveyDesign, level=95.0, add_intercept=True):
    """Design-based Gaussian GLM (weighted LS) with the survey sandwich
    variance and ``df.residual = degf − (p − 1)``.

    y : response column name or array. X : moderator matrix (no intercept col).
    """
    yv = design.y_cols[y] if isinstance(y, str) else np.asarray(y, float)
    M = np.asarray(X, float)
    if M.ndim == 1:
        M = M[:, None]
    Xd = np.hstack([np.ones((design.n, 1)), M]) if add_intercept else M
    w = design.weights
    p = Xd.shape[1]
    # weighted LS (survey uses the QR of the weight-scaled design)
    A = Xd.T @ (w[:, None] * Xd)
    beta = linalg.solve(A, Xd.T @ (w * yv), assume_a="sym")
    resid = yv - Xd @ beta
    # estimating-function contributions gᵢ = wᵢ xᵢ eᵢ  → sandwich meat
    G = (w * resid)[:, None] * Xd                  # (n, p)
    B = _vcov_total(G, design)                     # (p, p)
    Ainv = linalg.inv(A)
    V = Ainv @ B @ Ainv
    se = np.sqrt(np.diag(V))
    df = design.degf - (p - 1)
    crit = _crit(level, df)
    tval = beta / se
    return {"coef": beta, "se": se, "df": df,
            "tval": tval, "pval": 2 * stats.t.sf(np.abs(tval), df),
            "ci_lb": beta - crit * se, "ci_ub": beta + crit * se}
