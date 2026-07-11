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

__all__ = ["svydesign", "svymean", "svytotal", "svyglm", "SurveyDesign",
           "svyby", "svyratio", "svyciprop"]


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


def svyby(y, by, design: SurveyDesign, stat="svymean", level=95.0):
    """Domain (subpopulation) ``svymean``/``svytotal`` per level of ``by``.

    Matches R ``survey::svyby``: each domain statistic is computed as a
    **whole-sample linearized statistic** (the domain indicator enters the
    linearized variable, and the variance is taken over the *full* design —
    all strata/PSUs — NOT by re-declaring a design on the subset). This is
    what makes the domain SE differ from a naive ``svymean`` on a subset.

    y     : response column name (or array).
    by    : grouping column name (or array) — one statistic per distinct level.
    stat  : ``"svymean"`` or ``"svytotal"``.
    Returns dict: ``levels`` (sorted), ``estimate`` (per level), ``se``, ``df``.
    """
    yv = design.y_cols[y] if isinstance(y, str) else np.asarray(y, float)
    gv = design.y_cols[by] if isinstance(by, str) else np.asarray(by)
    w = design.weights
    levels = sorted(np.unique(gv).tolist())
    ests, ses = [], []
    for lev in levels:
        dom = (gv == lev)                      # domain indicator, whole sample
        wd = w * dom                           # weight zeroed outside domain
        if stat == "svymean":
            swd = wd.sum()
            est = float((wd * yv).sum() / swd)
            # domain-mean linearized total variable, defined on ALL elements
            z = wd * (yv - est) / swd
        elif stat == "svytotal":
            z = wd * yv
            est = float(z.sum())
        else:  # pragma: no cover
            raise ValueError(f"unsupported stat: {stat!r}")
        ests.append(est)
        ses.append(float(np.sqrt(_vcov_total(z, design))))
    return {"levels": levels, "estimate": np.asarray(ests, float),
            "se": np.asarray(ses, float), "df": design.degf}


def svyratio(num, den, design: SurveyDesign, level=95.0):
    """Design-based ratio ``R = Σ wᵢ numᵢ / Σ wᵢ denᵢ`` with Taylor-linearized SE.

    The ratio is a smooth function of two totals; its influence function is
    ``zᵢ = wᵢ (numᵢ − R·denᵢ) / Σ wⱼ denⱼ`` and ``V(R) = V(Σ zᵢ)`` under the
    ultimate-cluster estimator (identical machinery as ``svymean``).
    """
    nv = design.y_cols[num] if isinstance(num, str) else np.asarray(num, float)
    dv = design.y_cols[den] if isinstance(den, str) else np.asarray(den, float)
    w = design.weights
    tden = float((w * dv).sum())
    R = float((w * nv).sum() / tden)
    z = w * (nv - R * dv) / tden               # linearized total variable
    se = float(np.sqrt(_vcov_total(z, design)))
    df = design.degf
    crit = _crit(level, df)
    return {"estimate": R, "se": se, "df": df,
            "ci_lb": R - crit * se, "ci_ub": R + crit * se}


def svyciprop(y, design: SurveyDesign, level=95.0):
    """Confidence interval for a survey proportion via the **logit** method
    (R ``svyciprop(..., method="logit")``, the package default).

    ``survey`` fits an intercept-only ``quasibinomial`` ``svyglm`` and takes
    ``expit(β ± t·SE(β))``.  For an intercept-only model the sandwich reduces
    to the delta method around the weighted proportion, so equivalently:

        p        = weighted mean of the 0/1 indicator
        SE(p)    = linearized SE (svymean)
        β        = logit(p),   SE(β) = SE(p) / (p(1−p))
        CI       = expit(β ± qt(level, degf)·SE(β))

    The point estimate returned is ``p`` and its variance is ``V(p)`` (matching
    R's ``coef``/``attr(.,"var")``); the interval is on the logit scale.
    """
    yv = design.y_cols[y] if isinstance(y, str) else np.asarray(y, float)
    m = svymean(yv, design, level=level)
    p, se_p = m["estimate"], m["se"]
    df = design.degf
    crit = _crit(level, df)
    beta = np.log(p / (1.0 - p))               # logit(p)
    se_beta = se_p / (p * (1.0 - p))           # delta-method SE on logit scale
    lo = beta - crit * se_beta
    hi = beta + crit * se_beta
    expit = lambda x: 1.0 / (1.0 + np.exp(-x))
    return {"estimate": p, "var": se_p ** 2, "se": se_p, "df": df,
            "ci_lb": float(expit(lo)), "ci_ub": float(expit(hi))}
