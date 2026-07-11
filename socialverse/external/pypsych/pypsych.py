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

__all__ = ["smc", "cronbach_alpha", "fa_pa", "omega_total"]


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
