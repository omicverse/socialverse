"""Pure-Python reconstruction of R **ergm** (Handcock/Hunter et al.) — the
maximum PSEUDO-likelihood estimator (MPLE) for **dyad-independent** Exponential
Random Graph Models.

Reference-driven port under the Omicverse-RebuildR protocol; parity-gated against
ergm (``tests/test_parity.py``) on the canonical Padgett Florentine marriage
network (``ergm::flomarriage``, undirected, 16 nodes) at 1e-6.

Scope
-----
For a dyad-independent ERGM the pseudo-likelihood factorizes exactly over dyads,
and the MPLE is **identical** to a logistic regression in which

* the response for dyad (i, j) is the observed tie indicator y_{ij} ∈ {0, 1};
* the predictor row is the **change statistic** Δ_{ij} = g(y⁺_{ij}) − g(y⁻_{ij}),
  i.e. the change in the network's sufficient statistics when the (i, j) dyad is
  toggled from absent to present.

``ergm(net ~ ... , estimate="MPLE")`` fits exactly this logistic regression via
IRLS and reports the model-based (inverse-Fisher-information) covariance. This
module replicates that: it builds the dyad design matrix from change statistics
and solves the weighted-least-squares Newton (IRLS) iteration to convergence.

Supported dyad-independent terms
--------------------------------
* ``edges``               — Δ = 1 for every dyad.
* ``nodecov(attr)``       — Δ = attr[i] + attr[j]  (main effect of a numeric
                            vertex covariate).
* ``nodematch(attr)``     — Δ = 1{attr[i] == attr[j]}  (uniform homophily).
* ``nodefactor``-style / other dyad-DEPENDENT terms (``triangle``, ``gwesp``,
  ``kstar``, …) are **out of scope**: their pseudo-likelihood no longer
  factorizes into a plain logistic regression and ergm fits them by stochastic
  MCMC-MLE (class-2, not deterministic — see module note / README).

Determinism
-----------
MPLE for dyad-independent terms is a convex logistic regression: fully
deterministic, gated element-wise at 1e-6 against ergm. MCMC-MLE and RSiena
SAOM are genuinely stochastic and are NOT reproduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import linalg

__all__ = [
    "dyads",
    "change_stats_edges",
    "change_stats_nodecov",
    "change_stats_nodematch",
    "build_design",
    "ergm_mple",
    "MPLEResult",
]


# --------------------------------------------------------------------------- #
# Dyad enumeration
# --------------------------------------------------------------------------- #
def dyads(n: int, directed: bool = False) -> np.ndarray:
    """All ordered/unordered dyad index pairs (i, j) of an ``n``-node network.

    Undirected: upper-triangle pairs i < j  → n(n-1)/2 rows.
    Directed  : all i != j ordered pairs    → n(n-1)   rows.
    """
    if directed:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return np.asarray(pairs, dtype=int)


# --------------------------------------------------------------------------- #
# Change statistics for dyad-independent terms
# --------------------------------------------------------------------------- #
def change_stats_edges(pairs: np.ndarray) -> np.ndarray:
    """``edges`` term: toggling any dyad on changes the edge count by 1."""
    return np.ones(pairs.shape[0], float)


def change_stats_nodecov(pairs: np.ndarray, attr: Sequence[float]) -> np.ndarray:
    """``nodecov(attr)``: Δ = attr[i] + attr[j] for dyad (i, j)."""
    a = np.asarray(attr, float)
    return a[pairs[:, 0]] + a[pairs[:, 1]]


def change_stats_nodematch(pairs: np.ndarray, attr: Sequence) -> np.ndarray:
    """``nodematch(attr)``: Δ = 1 if the two endpoints share the attribute."""
    a = np.asarray(attr)
    return (a[pairs[:, 0]] == a[pairs[:, 1]]).astype(float)


# term-name -> builder(pairs, adjacency, attrs) -> (column, label)
def build_design(
    adjacency: np.ndarray,
    terms: Sequence,
    directed: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble the MPLE logistic-regression design.

    Parameters
    ----------
    adjacency : (n, n) 0/1 array (symmetric for undirected networks).
    terms     : sequence of term specs. Each is either the string ``"edges"``
                or a tuple ``(kind, attr_values)`` where ``kind`` is
                ``"nodecov"`` or ``"nodematch"`` and ``attr_values`` is the
                length-n vertex-attribute vector.

    Returns
    -------
    X : (D, p) change-statistic design matrix over the D dyads.
    y : (D,)   observed tie indicators.
    labels : column labels.
    """
    A = np.asarray(adjacency, float)
    n = A.shape[0]
    pairs = dyads(n, directed=directed)
    y = A[pairs[:, 0], pairs[:, 1]].astype(float)

    cols: list[np.ndarray] = []
    labels: list[str] = []
    for term in terms:
        if term == "edges":
            cols.append(change_stats_edges(pairs))
            labels.append("edges")
        else:
            kind, attr = term
            if kind == "nodecov":
                cols.append(change_stats_nodecov(pairs, attr))
                labels.append("nodecov")
            elif kind == "nodematch":
                cols.append(change_stats_nodematch(pairs, attr))
                labels.append("nodematch")
            else:  # pragma: no cover - guard
                raise ValueError(f"unsupported dyad-independent term: {kind!r}")
    X = np.column_stack(cols)
    return X, y, labels


# --------------------------------------------------------------------------- #
# IRLS logistic regression (the MPLE)
# --------------------------------------------------------------------------- #
@dataclass
class MPLEResult:
    terms: list[str]
    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    n_iter: int
    loglik: float

    def summary(self) -> str:
        lines = [f"{t:>16s}  {b: .6f}  (se {s:.6f})"
                 for t, b, s in zip(self.terms, self.coef, self.se)]
        return "\n".join(lines)


def _irls_logit(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Newton / IRLS fit of a logistic regression (canonical link, no intercept
    column added — change statistics already span the model).

    Returns (beta, vcov, n_iter, loglik). ``vcov`` is the model-based inverse
    Fisher information (Xᵀ W X)⁻¹ — exactly what R's ``glm``/ergm-MPLE report.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    beta = np.zeros(p)

    def _mu(b):
        eta = X @ b
        # stable logistic
        return 0.5 * (1.0 + np.tanh(0.5 * eta))

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        mu = _mu(beta)
        Wd = w * mu * (1.0 - mu)                      # IRLS weights
        # working response gradient: Xᵀ W (z - Xβ) with z = Xβ + (y-mu)/(mu(1-mu))
        grad = X.T @ (w * (y - mu))
        XtWX = X.T @ (Wd[:, None] * X)
        step = linalg.solve(XtWX, grad, assume_a="sym")
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu = _mu(beta)
    Wd = w * mu * (1.0 - mu)
    XtWX = X.T @ (Wd[:, None] * X)
    vcov = linalg.inv(XtWX)
    eps = 1e-300
    loglik = float(np.sum(w * (y * np.log(mu + eps)
                               + (1.0 - y) * np.log(1.0 - mu + eps))))
    return beta, vcov, n_iter, loglik


def ergm_mple(
    adjacency: np.ndarray,
    terms: Sequence,
    directed: bool = False,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> MPLEResult:
    """Fit a dyad-independent ERGM by maximum pseudo-likelihood.

    Equivalent to ``ergm(net ~ <terms>, estimate="MPLE")`` for dyad-independent
    terms. ``adjacency`` is the observed 0/1 sociomatrix; ``terms`` follows the
    :func:`build_design` spec (e.g. ``["edges", ("nodecov", wealth)]``).
    """
    X, y, labels = build_design(adjacency, terms, directed=directed)
    beta, vcov, n_iter, ll = _irls_logit(X, y, max_iter=max_iter, tol=tol)
    se = np.sqrt(np.diag(vcov))
    return MPLEResult(terms=labels, coef=beta, se=se, vcov=vcov,
                      n_iter=n_iter, loglik=ll)
