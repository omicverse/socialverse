"""Pure-Python reconstruction of R **psych** (Revelle) classical-test-theory and
factor-analysis primitives.

Reference-driven port under the Omicverse-RebuildR protocol; parity-gated against
psych 2.6.5 (``tests/test_parity.py``) on the canonical ``psych::bfi`` first-5-items
fixture (complete cases).

Implemented, faithfully to psych's actual algorithms:

* :func:`cronbach_alpha` — ``alpha`` total row.  On the item covariance ``C`` and
  correlation ``R`` (n = #items)::

      raw_alpha = (1 - tr(C)/sum(C)) * n/(n-1)
      std_alpha = (1 - n/sum(R))    * n/(n-1)
      G6        = 1 - (n - sum(smc(R)))/sum(R)
      average_r = (sum(R) - n)/(n*(n-1))

  Keys are NOT reversed (psych's ``check.keys`` defaults to FALSE).

* :func:`fa_pa` — ``fa(nfactors=1, fm="pa")`` principal-axis factoring.  Exact
  replication of psych's ``fac`` PA loop: diagonal seeded with SMC, iterate the
  first-eigenvector loading ``λ = v₁·√e₁`` and reset the diagonal to the model
  communalities until ``|Σh² − Σh²_prev| < min.err`` (default 1e-3) or
  ``max.iter`` (default 50).  Column signs flipped so ``Σλ ≥ 0`` (psych convention).

* :func:`omega_total` — McDonald's ω_total from a one-factor solution::

      ω_tot = 1 - Σ(1 - h²) / sum(R)

  computed on the PA communalities.  (psych's ``omega()`` runs a *different*
  internal pipeline — automatic key reversal + GPArotation minres — so its
  ``omega.tot`` is not element-wise reproducible from the public ``fa(fm='pa')``
  solution; see the parity test's documented reference-tolerance note.)

``smc`` (squared multiple correlations) uses ``1 - 1/diag(R⁻¹)``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _sps

__all__ = ["smc", "cronbach_alpha", "fa_pa", "omega_total", "ICC", "corr_test"]


def _cov(X: np.ndarray) -> np.ndarray:
    """Sample covariance matrix (N-1 denominator), matching R ``cov``/``var``."""
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def _cov2cor(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def smc(R: np.ndarray) -> np.ndarray:
    """Squared multiple correlations, psych-style: ``1 - 1/diag(solve(R))``.

    ``R`` is treated as a correlation matrix (its diagonal is standardized first,
    exactly as psych's ``smc`` does via ``cov2cor``).
    """
    R = np.asarray(R, float)
    Rc = _cov2cor(R)
    Rinv = np.linalg.inv(Rc)
    return 1.0 - 1.0 / np.diag(Rinv)


def cronbach_alpha(X: np.ndarray) -> dict:
    """psych ``alpha`` total row from a raw item matrix (rows=subjects, cols=items).

    Returns ``raw_alpha``, ``std_alpha``, ``G6``, ``average_r``.  No key reversal.
    """
    X = np.asarray(X, float)
    n = X.shape[1]
    C = _cov(X)
    R = _cov2cor(C)
    trC = np.trace(C)
    sumC = C.sum()
    sumR = R.sum()
    raw_alpha = (1.0 - trC / sumC) * (n / (n - 1.0))
    std_alpha = (1.0 - n / sumR) * (n / (n - 1.0))
    G6 = 1.0 - (n - smc(R).sum()) / sumR
    average_r = (sumR - n) / (n * (n - 1.0))
    return {"raw_alpha": float(raw_alpha), "std_alpha": float(std_alpha),
            "G6": float(G6), "average_r": float(average_r)}


def fa_pa(R: np.ndarray, nfactors: int = 1, min_err: float = 1e-3,
          max_iter: int = 50) -> dict:
    """Principal-axis factor analysis, replicating psych ``fa(fm="pa")``.

    ``R`` : correlation matrix (p x p).  Returns single-column ``loadings`` (p,),
    ``communality`` (p,), ``uniqueness`` (p,) for ``nfactors=1``.

    Loop (psych ``fac`` PA branch): seed ``diag`` with SMC; each pass eigendecompose
    the current reduced matrix, take the leading ``nfactors`` loadings
    ``L = V[:,:k] · diag(√λ[:k])``, model communalities ``diag(L Lᵀ)``, reset the
    diagonal to them, and stop when the change in the communality *sum* drops below
    ``min_err`` (or after ``max_iter``).  Signs flipped so each column sums ≥ 0.
    """
    R = np.asarray(R, float)
    p = R.shape[0]
    k = nfactors
    r_mat = R.copy()
    np.fill_diagonal(r_mat, smc(R))

    comm = float(np.trace(r_mat))       # sum of communalities
    err = comm                          # ensures the loop runs at least once
    loadings = None
    it = 1
    while err > min_err:
        evals, evecs = np.linalg.eigh(r_mat)      # ascending order
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        top_vals = np.clip(evals[:k], 0.0, None)
        loadings = evecs[:, :k] * np.sqrt(top_vals)   # (p, k)
        model = loadings @ loadings.T
        new = np.diag(model).copy()
        comm1 = float(new.sum())
        np.fill_diagonal(r_mat, new)
        err = abs(comm - comm1)
        comm = comm1
        it += 1
        if it > max_iter:
            break

    # psych sign convention: flip each column so its column-sum is non-negative.
    signs = np.sign(loadings.sum(axis=0))
    signs[signs == 0] = 1.0
    loadings = loadings * signs

    communality = (loadings ** 2).sum(axis=1)
    uniqueness = 1.0 - communality
    return {"loadings": loadings.ravel() if k == 1 else loadings,
            "communality": communality, "uniqueness": uniqueness}


def omega_total(R: np.ndarray, communality: np.ndarray | None = None,
                nfactors: int = 1) -> float:
    """McDonald's ω_total from a one-factor solution.

    ``ω_tot = 1 - Σ(1 - h²)/sum(R)`` with ``h²`` the PA communalities.  If
    ``communality`` is not supplied it is computed from :func:`fa_pa` on ``R``.
    """
    R = np.asarray(R, float)
    if communality is None:
        communality = fa_pa(R, nfactors=nfactors)["communality"]
    communality = np.asarray(communality, float)
    return float((R.sum() - (1.0 - communality).sum()) / R.sum())


def ICC(ratings: np.ndarray, alpha: float = 0.05) -> dict:
    """Intraclass correlations, replicating psych ``ICC(x, lmer=FALSE)``.

    ``ratings`` : subjects x raters (judges) matrix (n rows, k columns), no
    missing values.  Runs the two-way ANOVA decomposition ``values ~ subjects +
    judges`` (Shrout & Fleiss) and returns the six ICC coefficients

    * ``ICC1``  — single, one-way random (absolute agreement, raters as a random
      one-way factor); uses ``MSW = (SS_j + SS_e)/(df_j + df_e)``.
    * ``ICC2``  — single, two-way random, absolute agreement.
    * ``ICC3``  — single, two-way mixed, consistency.
    * ``ICC1k/ICC2k/ICC3k`` — the corresponding average-of-k-raters forms.

    with, for each, the F ratio, its numerator/denominator df, its p value
    (``-expm1(pf(F, df1, df2, log.p=TRUE))`` = upper-tail), and the
    ``alpha``-level lower/upper confidence bounds — exactly psych's formulas.

    Returns a dict with keys ``type`` (list of names), ``ICC`` (6,), ``F`` (6,),
    ``df1`` (6,), ``df2`` (6,), ``p`` (6,), ``lower`` (6,), ``upper`` (6,), plus
    the ANOVA mean squares ``MSB``/``MSJ``/``MSE``/``MSW`` and ``n_obs``/``n_judge``.
    """
    x = np.asarray(ratings, float)
    n = x.shape[0]        # subjects
    nj = x.shape[1]       # judges / raters

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    # Two-way ANOVA sums of squares (values ~ subjects + judges).
    SSB = nj * ((row_means - grand) ** 2).sum()          # between subjects
    SSJ = n * ((col_means - grand) ** 2).sum()           # between judges
    SST = ((x - grand) ** 2).sum()
    SSE = SST - SSB - SSJ                                 # residual

    dfB = n - 1
    dfJ = nj - 1
    dfE = (n - 1) * (nj - 1)

    MSB = SSB / dfB
    MSJ = SSJ / dfJ
    MSE = SSE / dfE
    # one-way within-subject MS: pool judge + residual (psych aov path).
    MSW = (SSJ + SSE) / (dfJ + dfE)

    ICC1 = (MSB - MSW) / (MSB + (nj - 1) * MSW)
    ICC2 = (MSB - MSE) / (MSB + (nj - 1) * MSE + nj * (MSJ - MSE) / n)
    ICC3 = (MSB - MSE) / (MSB + (nj - 1) * MSE)
    ICC1k = (MSB - MSW) / MSB
    ICC2k = (MSB - MSE) / (MSB + (MSJ - MSE) / n)
    ICC3k = (MSB - MSE) / MSB

    F11 = MSB / MSW
    df11n = n - 1
    df11d = n * (nj - 1)
    p11 = -np.expm1(_sps.f.logcdf(F11, df11n, df11d))

    F21 = MSB / MSE
    df21n = n - 1
    df21d = (n - 1) * (nj - 1)
    p21 = -np.expm1(_sps.f.logcdf(F21, df21n, df21d))
    F31 = F21

    icc = np.array([ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k], float)
    Fv = np.array([F11, F21, F21, F11, F21, F21], float)
    df1 = np.array([df11n, df21n, df21n, df11n, df21n, df21n], float)
    df2 = np.array([df11d, df21d, df21d, df11d, df21d, df21d], float)
    pv = np.array([p11, p21, p21, p11, p21, p21], float)

    # --- confidence intervals (psych's closed forms) ---
    q = 1.0 - alpha / 2.0
    lower = np.full(6, np.nan)
    upper = np.full(6, np.nan)

    # ICC1 / ICC1k (one-way)
    F1L = F11 / _sps.f.ppf(q, df11n, df11d)
    F1U = F11 * _sps.f.ppf(q, df11d, df11n)
    L1 = (F1L - 1) / (F1L + (nj - 1))
    U1 = (F1U - 1) / (F1U + nj - 1)
    lower[0], upper[0] = L1, U1
    lower[3], upper[3] = 1 - 1 / F1L, 1 - 1 / F1U

    # ICC3 / ICC3k (two-way fixed)
    F3L = F31 / _sps.f.ppf(q, df21n, df21d)
    F3U = F31 * _sps.f.ppf(q, df21d, df21n)
    lower[2] = (F3L - 1) / (F3L + nj - 1)
    upper[2] = (F3U - 1) / (F3U + nj - 1)
    lower[5] = 1 - 1 / F3L
    upper[5] = 1 - 1 / F3U

    # ICC2 / ICC2k (two-way random, Satterthwaite df)
    Fj = MSJ / MSE
    vn = (nj - 1) * (n - 1) * ((nj * ICC2 * Fj + n * (1 + (nj - 1) * ICC2)
                               - nj * ICC2)) ** 2
    vd = ((n - 1) * nj ** 2 * ICC2 ** 2 * Fj ** 2
          + (n * (1 + (nj - 1) * ICC2) - nj * ICC2) ** 2)
    v = vn / vd
    F2U = _sps.f.ppf(q, n - 1, v)
    F2L = _sps.f.ppf(q, v, n - 1)
    L3 = n * (MSB - F2U * MSE) / (F2U * (nj * MSJ + (nj * n - nj - n) * MSE)
                                  + n * MSB)
    U3 = n * (F2L * MSB - MSE) / (nj * MSJ + (nj * n - nj - n) * MSE
                                  + n * F2L * MSB)
    lower[1], upper[1] = L3, U3
    lower[4] = L3 * nj / (1 + L3 * (nj - 1))
    upper[4] = U3 * nj / (1 + U3 * (nj - 1))

    return {
        "type": ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"],
        "ICC": icc, "F": Fv, "df1": df1, "df2": df2, "p": pv,
        "lower": lower, "upper": upper,
        "MSB": float(MSB), "MSJ": float(MSJ), "MSE": float(MSE),
        "MSW": float(MSW), "n_obs": int(n), "n_judge": int(nj),
    }


def corr_test(x: np.ndarray) -> dict:
    """Correlation matrix + pairwise n + raw p, replicating psych ``corr.test``.

    ``x`` : n x p data matrix (rows=observations, cols=variables), no missing
    values (so pairwise n is the constant sample size).  With psych's default
    ``method="pearson"`` and ``normal=TRUE``:

        r = cor(x)
        t = r * sqrt(n - 2) / sqrt(1 - r^2)
        p = -2 * expm1(pt(|t|, n - 2, log.p=TRUE))     # two-sided, RAW (unadjusted)
        se = sqrt((1 - r^2) / (n - 2))

    Diagonal r is 1, its t is +inf, and its raw p is 0 (psych's ``pt`` on |t|=inf).
    Returns ``r`` (p,p), ``n`` (scalar here), ``t`` (p,p), ``p`` (p,p RAW two-sided),
    ``se`` (p,p).  (psych's Holm-adjusted p and Fisher-z CIs are not gated here.)
    """
    X = np.asarray(x, float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / (n - 1)
    d = np.sqrt(np.diag(C))
    r = C / np.outer(d, d)
    np.fill_diagonal(r, 1.0)             # exact, as R's cor() self-correlation

    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(n - 2) / np.sqrt(1.0 - r ** 2)
        se = np.sqrt((1.0 - r * r) / (n - 2))
    # p = -2 * expm1(pt(|t|, n-2, log)) ; diagonal |t|=inf -> logcdf=0 -> p=0.
    p = -2.0 * np.expm1(_sps.t.logcdf(np.abs(t), n - 2))
    p = np.minimum(p, 1.0)

    return {"r": r, "n": int(n), "t": t, "p": p, "se": se}
