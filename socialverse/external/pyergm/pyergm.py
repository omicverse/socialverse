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
    "summary_formula",
    "triad_census",
    "TRIAD_CENSUS_LABELS",
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


# --------------------------------------------------------------------------- #
# Observed sufficient statistics — summary(net ~ terms)
# --------------------------------------------------------------------------- #
# These reproduce ergm's ``summary(formula)`` for the observed network: the
# vector of sufficient statistics g(y) for a set of (dyad-independent + simple
# dyad-dependent) terms. Unlike the MPLE fit, these are exact integer/real
# counts — no estimation — so they are gated at 0 tolerance for counts.
#
# Supported term specs (each entry of ``terms``):
#   "edges"                        number of edges (undirected: sum/2; directed: sum)
#   "triangle"                     number of triangles (undirected)
#   ("degree", d) | ("degree", [d0, d1, ...])
#                                  #nodes of undirected degree exactly d
#   ("idegree", d|[...])           #nodes of in-degree exactly d  (directed)
#   ("odegree", d|[...])           #nodes of out-degree exactly d (directed)
#   ("kstar", k)                   sum_i C(deg_i, k)        (undirected k-stars)
#   ("istar", k)                   sum_i C(indeg_i, k)      (directed in-k-stars)
#   ("ostar", k)                   sum_i C(outdeg_i, k)     (directed out-k-stars)
#   "mutual"                       number of mutual (reciprocated) dyads (directed)
#   ("nodecov", attr)              sum over edges of (attr[i] + attr[j])
#   ("nodematch", attr)            number of edges with attr[i] == attr[j]
#
# Term labels follow ergm's naming (e.g. "degree3", "kstar2", "idegree1",
# "nodecov.<attr>", "nodematch.<attr>") when an attribute name is supplied.


def _degree_undirected(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=1)


def _indeg(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=0)


def _outdeg(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=1)


def _choose(n: np.ndarray, k: int) -> np.ndarray:
    """Vectorised integer binomial C(n, k) for the (small) k-star terms."""
    from math import comb

    n = np.asarray(n, dtype=np.int64)
    return np.array([comb(int(v), k) for v in n], dtype=np.int64)


def _edge_endpoints(A: np.ndarray, directed: bool) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) endpoint index arrays of the present ties.

    Undirected: upper triangle only (each edge once). Directed: all ordered
    present ties. Used by edge-summed covariate terms.
    """
    if directed:
        i, j = np.nonzero(A)
    else:
        i, j = np.nonzero(np.triu(A, 1))
    return i, j


def summary_formula(
    adjacency: np.ndarray,
    terms: Sequence,
    directed: bool = False,
    attr_name: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Observed network sufficient statistics — ``summary(net ~ terms)``.

    Reproduces R ``ergm``'s ``summary(formula)``: the exact vector of
    sufficient statistics g(y) of the *observed* sociomatrix ``adjacency`` for
    the requested ``terms``. No estimation is involved, so every element is an
    exact integer/real count (gated at 0 tolerance against ergm).

    Parameters
    ----------
    adjacency : (n, n) 0/1 array. Symmetric for undirected networks; for
        directed networks A[i, j] == 1 means i -> j.
    terms : sequence of term specs (see module doc for this section). ``"edges"``
        / ``"triangle"`` / ``"mutual"`` are bare strings; the rest are tuples
        ``(kind, arg)`` where ``arg`` is a scalar/list (degree, kstar, …) or a
        length-n attribute vector (nodecov, nodematch).
    directed : whether the network is directed.
    attr_name : optional attribute name used only for label formatting of
        ``nodecov``/``nodematch`` (e.g. ``"wealth"`` -> ``"nodecov.wealth"``).

    Returns
    -------
    stats : (p,) float array of the sufficient statistics, in the flattened
        order the terms expand to (degree/idegree/odegree with a list arg
        expand to one entry per value, matching ergm's ``degree(0:6)``).
    labels : matching list of ergm-style term labels.
    """
    A = np.asarray(adjacency)
    A = (A != 0).astype(np.int64)
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    stats: list[float] = []
    labels: list[str] = []

    def _as_list(arg) -> list[int]:
        if np.isscalar(arg):
            return [int(arg)]
        return [int(v) for v in arg]

    for term in terms:
        if term == "edges":
            e = A.sum() if directed else np.triu(A, 1).sum()
            stats.append(float(e))
            labels.append("edges")
        elif term == "triangle":
            # Undirected number of triangles: trace(A^3)/6 for symmetric 0/1 A.
            if directed:
                raise ValueError("triangle term is defined for undirected nets")
            A3 = A @ A @ A
            stats.append(float(np.trace(A3) // 6))
            labels.append("triangle")
        elif term == "mutual":
            # Number of mutual (reciprocated) dyads: sum_{i<j} A[i,j]*A[j,i].
            m = int(np.sum(np.triu(A * A.T, 1)))
            stats.append(float(m))
            labels.append("mutual")
        else:
            kind, arg = term
            if kind == "degree":
                deg = _degree_undirected(A)
                for d in _as_list(arg):
                    stats.append(float(np.sum(deg == d)))
                    labels.append(f"degree{d}")
            elif kind == "idegree":
                deg = _indeg(A)
                for d in _as_list(arg):
                    stats.append(float(np.sum(deg == d)))
                    labels.append(f"idegree{d}")
            elif kind == "odegree":
                deg = _outdeg(A)
                for d in _as_list(arg):
                    stats.append(float(np.sum(deg == d)))
                    labels.append(f"odegree{d}")
            elif kind == "kstar":
                deg = _degree_undirected(A)
                k = int(arg)
                stats.append(float(np.sum(_choose(deg, k))))
                labels.append(f"kstar{k}")
            elif kind == "istar":
                deg = _indeg(A)
                k = int(arg)
                stats.append(float(np.sum(_choose(deg, k))))
                labels.append(f"istar{k}")
            elif kind == "ostar":
                deg = _outdeg(A)
                k = int(arg)
                stats.append(float(np.sum(_choose(deg, k))))
                labels.append(f"ostar{k}")
            elif kind == "nodecov":
                a = np.asarray(arg, float)
                i, j = _edge_endpoints(A, directed)
                stats.append(float(np.sum(a[i] + a[j])))
                labels.append("nodecov" + (f".{attr_name}" if attr_name else ""))
            elif kind == "nodematch":
                a = np.asarray(arg)
                i, j = _edge_endpoints(A, directed)
                stats.append(float(np.sum(a[i] == a[j])))
                labels.append("nodematch" + (f".{attr_name}" if attr_name else ""))
            else:  # pragma: no cover - guard
                raise ValueError(f"unsupported summary term: {kind!r}")

    return np.asarray(stats, float), labels


# --------------------------------------------------------------------------- #
# Holland-Leinhardt directed triad census (sna::triad.census)
# --------------------------------------------------------------------------- #
# The 16 isomorphism classes of directed triads, labelled by the MAN code
# (number of Mutual, Asymmetric, Null dyads) plus a Davis-Leinhardt suffix
# (Down/Up/Cyclic/Transitive). Order matches sna's ``triad.census`` columns.
TRIAD_CENSUS_LABELS = [
    "003", "012", "102", "021D", "021U", "021C", "111D", "111U",
    "030T", "030C", "201", "120D", "120U", "120C", "210", "300",
]

# Map from (n_mutual, n_asymmetric, n_null, config_code) to census index.
# The classic Batagelj-Mrvar / Holland-Leinhardt "tricode" lookup: a triad is
# coded by its three ordered dyads (each 0 null / 1 asym / 2 mutual, oriented),
# yielding a 64-entry table that folds isomorphic codes into the 16 classes.
# We use the canonical tricode->class table (1-indexed classes 1..16 -> 0..15).
_TRICODE_TO_CLASS = np.array([
    1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11,
    2, 6, 4, 8, 5, 9, 9, 13, 6, 10, 9, 14, 7, 14, 12, 15,
    2, 5, 6, 7, 6, 9, 10, 14, 4, 9, 9, 12, 8, 13, 14, 15,
    3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15, 11, 15, 15, 16,
], dtype=int) - 1


def _tricode(A: np.ndarray, i: int, j: int, k: int) -> int:
    """Batagelj-Mrvar tricode of the ordered triad (i, j, k).

    Each of the three dyads contributes a bit pattern for its two directions;
    the weighted sum indexes ``_TRICODE_TO_CLASS``. Weights follow the standard
    reference implementation (Batagelj & Mrvar 2001; also igraph/sna).
    """
    return (
        1 * A[i, j] + 2 * A[j, i]
        + 4 * A[i, k] + 8 * A[k, i]
        + 16 * A[j, k] + 32 * A[k, j]
    )


def triad_census(adjacency: np.ndarray) -> np.ndarray:
    """Holland-Leinhardt 16-type directed triad census (``sna::triad.census``).

    Counts, over all C(n, 3) unordered node triples, how many fall into each of
    the 16 directed-triad isomorphism classes (003, 012, …, 300). Returns an
    integer-valued length-16 float vector in :data:`TRIAD_CENSUS_LABELS` order
    (identical to sna's column order). Exact counts — gated at 0 tolerance.

    ``adjacency`` is the directed 0/1 sociomatrix (A[i, j] == 1 means i -> j);
    the diagonal is ignored.
    """
    A = np.asarray(adjacency)
    A = (A != 0).astype(np.int64)
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    counts = np.zeros(16, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                code = _tricode(A, i, j, k)
                counts[_TRICODE_TO_CLASS[code]] += 1
    return counts.astype(float)
