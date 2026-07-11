"""``sv.tl._network2`` — statistical models *for* networks (ERGM + SAOM).

Two Python-native gap methods that have no first-class open-source home the way
R does (``statnet::ergm`` and ``RSiena::siena`` are the world champions). This
module fills that gap with honest, real computation:

- :func:`ergm` — an **Exponential Random Graph Model** fit by **MPLE**
  (maximum pseudo-likelihood): every directed dyad ``(i, j)`` contributes one
  Bernoulli observation whose linear predictor is a sum of *change statistics*
  (edges, mutual/reciprocity, transitive 2-paths). The pseudo-likelihood is then
  an ordinary logistic regression of ``tie ~ change-stats`` (``statsmodels.Logit``,
  numpy/IRLS fallback). MPLE is the classic Strauss–Ikeda (1990) / Frank–Strauss
  approximation to the intractable normalizing-constant MLE that ``ergm`` reaches
  by MCMC-MLE — **honestly labelled as an approximation** in every record.

- :func:`saom` — a **Stochastic Actor-Oriented Model** (Snijders' SIENA)
  *descriptive* two-wave summary: Jaccard tie-stability, creation/dissipation
  rates, Hamming distance, and — when actor behavior vectors are supplied — the
  cross-lagged (network→behavior, behavior→network) correlations that a full SAOM
  turns into rate/selection/influence parameters. **Honestly labelled as the
  descriptive/simplified layer, not the simulation-based SAOM estimation.**

Real backends (``statnet`` / ``RSiena`` via ``rpy2``) are treated as *optional
accelerators*; the pure ``statsmodels``/``numpy`` path below always runs and
recovers the planted structure (the toy DGP has reciprocity + transitivity, so
the MPLE ``mutual`` coefficient comes out positive). Deterministic (``seed=0``).
"""
from __future__ import annotations

import importlib
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["ergm", "network_statistics", "saom"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _edges_from(state: StudyState, kwargs: dict[str, Any], key: str) -> pd.DataFrame | None:
    """Resolve an edge table from a kwarg (``key``) else ``sources['datasets']``.

    A ``{name: DataFrame}`` mapping in the slot is unwrapped to its first frame.
    """
    df = kwargs.get(key)
    if df is None and key in ("edges", "wave1"):
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, (list, tuple)) and df:
        return pd.DataFrame(list(df))
    return None


def _endpoint_cols(df: pd.DataFrame, kwargs: dict[str, Any]) -> tuple[str, str]:
    """Pick ``(source, target)`` columns by kwarg, convention, else first two."""
    cols = list(df.columns)
    src, tgt = kwargs.get("source"), kwargs.get("target")
    if src is None or tgt is None:
        lower = {str(c).lower(): c for c in cols}
        for a, b in (("source", "target"), ("from", "to"), ("u", "v"),
                     ("i", "j"), ("ego", "alter")):
            if a in lower and b in lower:
                src, tgt = lower[a], lower[b]
                break
    if (src is None or tgt is None) and len(cols) >= 2:
        src, tgt = cols[0], cols[1]
    return src, tgt


def _adjacency(df: pd.DataFrame, kwargs: dict[str, Any],
               nodes: list | None = None) -> tuple[np.ndarray, list]:
    """Build a 0/1 directed adjacency matrix and its node ordering from an edgelist."""
    src, tgt = _endpoint_cols(df, kwargs)
    clean = df[[src, tgt]].dropna()
    if nodes is None:
        nodes = sorted(set(clean[src]).union(set(clean[tgt])), key=str)
    idx = {node: k for k, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u, v in zip(clean[src], clean[tgt]):
        if u in idx and v in idx and u != v:
            A[idx[u], idx[v]] = 1.0
    return A, nodes


def _logit_fit(y: np.ndarray, X: np.ndarray, colnames: list[str]) -> dict[str, Any]:
    """Fit logistic regression ``y ~ X`` — statsmodels if present, else numpy IRLS.

    Returns coefficients, std-errors, z, and (pseudo-)log-likelihood. This is the
    engine of MPLE: the ERGM pseudo-likelihood *is* a logistic regression of each
    dyad's tie on its change statistics.
    """
    sm = _try_import("statsmodels.api")
    if sm is not None:
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
            return {
                "backend": "statsmodels.Logit",
                "coef": {c: float(b) for c, b in zip(colnames, res.params)},
                "se": {c: float(s) for c, s in zip(colnames, res.bse)},
                "z": {c: float(z) for c, z in zip(colnames, res.tvalues)},
                "llf": float(res.llf),
                "n_obs": int(len(y)),
            }
        except Exception:
            pass
    # ---- pure-numpy IRLS (Newton) fallback -------------------------------
    beta = np.zeros(X.shape[1])
    XtX_inv = None
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        WX = X * w[:, None]
        H = X.T @ WX
        try:
            XtX_inv = np.linalg.pinv(H)
        except Exception:
            XtX_inv = np.linalg.pinv(H + 1e-8 * np.eye(H.shape[0]))
        step = XtX_inv @ (X.T @ (y - p))
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    llf = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) +
                       (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    se = np.sqrt(np.clip(np.diag(XtX_inv), 0, None)) if XtX_inv is not None \
        else np.full(len(beta), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return {
        "backend": "numpy-IRLS",
        "coef": {c: float(b) for c, b in zip(colnames, beta)},
        "se": {c: float(s) for c, s in zip(colnames, se)},
        "z": {c: float(v) for c, v in zip(colnames, z)},
        "llf": llf,
        "n_obs": int(len(y)),
    }


# Terms the parity-gated pyergm port can fit exactly (dyad-independent MPLE).
_PYERGM_TERMS = {"edges", "nodecov", "nodematch"}


def _node_attr_vector(kwargs: dict[str, Any], nodes: list, term: str):
    """Resolve a length-n vertex-attribute vector for a ``nodecov``/``nodematch`` term.

    Looks for ``kwargs['<term>']`` (or ``kwargs['<term>_attr']``) as either a
    ``{node: value}`` mapping or a sequence aligned to ``nodes``. dtypes are
    preserved: ``nodematch`` labels stay categorical (never numeric-coerced),
    only ``nodecov`` is cast to float (it is a numeric main effect).
    """
    attr = kwargs.get(term, kwargs.get(f"{term}_attr"))
    if attr is None:
        return None
    if isinstance(attr, dict):
        try:
            vals = [attr[node] for node in nodes]
        except KeyError:
            return None
    elif isinstance(attr, (list, tuple, np.ndarray, pd.Series)):
        seq = list(attr)
        if len(seq) != len(nodes):
            return None
        vals = seq
    else:
        return None
    if term == "nodecov":
        try:
            return np.asarray(vals, dtype=float)
        except (TypeError, ValueError):
            return None
    # nodematch: keep the raw (possibly string / categorical) labels as-is
    return np.asarray(vals, dtype=object)


def _ergm_mple_via_port(A: np.ndarray, terms: list[str], kwargs: dict[str, Any],
                        nodes: list, directed: bool = True) -> dict[str, Any] | None:
    """Delegate the MPLE fit to the parity-gated ``pyergm`` port.

    Returns a ``fit`` dict in the exact shape :func:`_logit_fit` produces
    (``backend``/``coef``/``se``/``z``/``llf``/``n_obs``) with ``backend`` set to
    ``"pyergm"``, or ``None`` if the requested terms fall outside the port's
    dyad-independent scope, an attribute cannot be resolved, or the port raises
    (so the caller can fall back to the pre-existing engine).
    """
    if not set(terms).issubset(_PYERGM_TERMS):
        return None
    try:
        from ..external.pyergm import ergm_mple

        port_terms: list[Any] = []
        for t in terms:
            if t == "edges":
                port_terms.append("edges")
            else:  # nodecov / nodematch — need an attribute vector
                vec = _node_attr_vector(kwargs, nodes, t)
                if vec is None:
                    return None
                port_terms.append((t, vec))

        res = ergm_mple(A, port_terms, directed=directed)
        # Map MPLEResult -> the exact _logit_fit dict shape, keyed by term name.
        coef = {t: float(b) for t, b in zip(terms, res.coef)}
        se = {t: float(s) for t, s in zip(terms, res.se)}
        with np.errstate(divide="ignore", invalid="ignore"):
            z = {t: (float(b) / float(s) if s else float("nan"))
                 for t, b, s in zip(terms, res.coef, res.se)}
        # dyad count = observations in the design (n(n-1) directed, n(n-1)/2 undirected)
        n = A.shape[0]
        n_obs = n * (n - 1) if directed else n * (n - 1) // 2
        return {
            "backend": "pyergm",
            "coef": coef,
            "se": se,
            "z": z,
            "llf": float(res.loglik),
            "n_obs": int(n_obs),
        }
    except Exception:
        return None


def _ergm_change_stats(A: np.ndarray, terms: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the dyadic design matrix of ERGM change statistics.

    For every ordered dyad ``(i, j)``, ``i != j``, one row: the response is the
    observed tie ``A[i, j]``; the predictors are the change in each network
    statistic from toggling that tie on. For a directed graph:

    - ``edges``       : +1 (baseline density / intercept)
    - ``mutual``      : +1 iff the reciprocal tie ``A[j, i]`` exists (reciprocity)
    - ``transitive``  : number of two-paths ``i→k→j`` closed by ``i→j``
                        (transitive triadic closure)
    """
    n = A.shape[0]
    rows_y: list[float] = []
    rows_X: list[list[float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            feats: list[float] = []
            if "edges" in terms:
                feats.append(1.0)
            if "mutual" in terms:
                feats.append(float(A[j, i]))
            if "transitive" in terms:
                # two-paths i -> k -> j closed by adding i -> j
                tp = float(np.sum(A[i, :] * A[:, j]))
                feats.append(tp)
            rows_y.append(float(A[i, j]))
            rows_X.append(feats)
    return np.asarray(rows_y), np.asarray(rows_X), list(terms)


def _observed_stats(A: np.ndarray) -> dict[str, float]:
    """Observed global network statistics (for the GoF comparison)."""
    n = A.shape[0]
    edges = float(A.sum())
    mutual = float(np.sum(A * A.T) / 2.0)          # unordered reciprocal pairs
    twopaths = float(np.sum(A @ A) - np.trace(A @ A))
    # transitive triads: i->k, k->j, i->j all present
    trans = float(np.sum((A @ A) * A))
    dens = edges / (n * (n - 1)) if n > 1 else 0.0
    return {
        "edges": edges,
        "mutual_pairs": mutual,
        "two_paths": twopaths,
        "transitive_triads": trans,
        "density": dens,
    }


def _ergm_gof(A: np.ndarray, coef: dict[str, float],
              terms: list[str], seed: int) -> dict[str, Any]:
    """Simple GoF: observed vs. model-expected global statistics.

    The fitted MPLE gives each dyad an independent tie probability
    ``p_ij = logit^{-1}(sum of change-stats · coef)``; we form the expected edge
    count analytically and simulate a small ensemble to compare mutual/transitive
    counts (the terms MPLE is known to under-fit). Honest, cheap GoF — not the
    full ``ergm::gof`` MCMC diagnostic.
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b_edges = coef.get("edges", 0.0)
    b_mut = coef.get("mutual", 0.0)
    b_tr = coef.get("transitive", 0.0)

    def _sim_once() -> np.ndarray:
        S = np.zeros((n, n))
        # sequential (conditional) simulation: use current S for mutual/transitive
        order = [(i, j) for i in range(n) for j in range(n) if i != j]
        rng.shuffle(order)
        for i, j in order:
            eta = b_edges
            if "mutual" in terms:
                eta += b_mut * S[j, i]
            if "transitive" in terms:
                eta += b_tr * float(np.sum(S[i, :] * S[:, j]))
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            S[i, j] = 1.0 if rng.uniform() < p else 0.0
        return S

    n_sim = 20
    sim_stats = [_observed_stats(_sim_once()) for _ in range(n_sim)]
    obs = _observed_stats(A)
    keys = ["edges", "mutual_pairs", "transitive_triads", "density"]
    exp = {k: float(np.mean([s[k] for s in sim_stats])) for k in keys}
    sd = {k: float(np.std([s[k] for s in sim_stats])) for k in keys}
    return {
        "method": "posterior-predictive (MPLE coef, sequential sim, n_sim=20)",
        "observed": {k: obs[k] for k in keys},
        "model_expected": exp,
        "model_sd": sd,
        "note": "MPLE≈MCMC-MLE 的近似;GoF 为简化的观测vs模型统计量对比,非 ergm::gof 完整诊断",
    }


# ---------------------------------------------------------------------------- ergm
@register(
    name="ergm",
    aliases=["ERGM", "指数随机图"],
    category="net",
    tier="pro",
    skill="(ERGM 缺口,Python 原生空白=高护城河)",
    languages=["Python"],
    key_tools=["statsmodels", "networkx"],
    description="指数随机图模型(ERGM),MPLE 伪似然估计边/互惠/传递闭合效应",
    requires={"sources": ["datasets"]},
    produces={"models": ["ergm"], "diagnostics": ["gof"]},
    prerequisites={"optional_functions": ["build_network"]},
    auto_fix="escalate",
)
def ergm(state: StudyState, **kwargs: Any) -> StudyState:
    """Fit an ERGM by maximum pseudo-likelihood (MPLE) on a directed edgelist.

    The edgelist is taken from ``edges=`` (or ``sources['datasets']``) and turned
    into a directed adjacency matrix. Each ordered dyad ``(i, j)`` becomes one
    Bernoulli observation whose change statistics are the requested ``terms``
    (default ``["edges", "mutual", "transitive"]``); a logistic regression of
    ``tie ~ change-stats`` *is* the pseudo-likelihood, so the coefficients are the
    ERGM parameters (log-odds contributions of reciprocity and transitive closure).

    Writes ``models['ergm']`` (coef / se / z / backend / observed statistics) and
    ``diagnostics['gof']`` (observed vs. model-expected global statistics). MPLE is
    an honest approximation to the MCMC-MLE that ``statnet::ergm`` computes — this
    is recorded on the result. Never raises: with no usable edge table it writes an
    empty-but-valid record so a resolver can chain past it.
    """
    seed = int(kwargs.get("seed", 0))
    terms = list(kwargs.get("terms", ["edges", "mutual", "transitive"]))
    if "edges" not in terms:
        terms = ["edges"] + terms  # intercept/density term is mandatory

    def _empty(note: str) -> StudyState:
        state.write("models", "ergm", {
            "method": "MPLE", "terms": terms, "coef": {}, "se": {}, "z": {},
            "n_nodes": 0, "n_edges": 0, "backend": None, "note": note,
        })
        state.write("diagnostics", "gof", {"note": note, "observed": {}, "model_expected": {}})
        return state

    df = _edges_from(state, kwargs, "edges")
    if df is None or df.empty:
        return _empty("缺少边表(edges= 或 sources['datasets']),无法拟合 ERGM")

    A, nodes = _adjacency(df, kwargs)
    n = A.shape[0]
    if n < 3 or A.sum() == 0:
        return _empty("网络过小或无边,无法拟合 ERGM")

    # Prefer the parity-gated pyergm port for dyad-independent terms (edges /
    # nodecov / nodematch); it reproduces statnet::ergm MPLE to 1e-6. If the
    # requested terms are dyad-dependent (mutual / transitive) or the port raises,
    # fall back to the pre-existing change-statistics + logistic-regression engine.
    fit = _ergm_mple_via_port(A, terms, kwargs, nodes, directed=True)
    if fit is None:
        y, X, colnames = _ergm_change_stats(A, terms)
        fit = _logit_fit(y, X, colnames)
    obs = _observed_stats(A)
    gof = _ergm_gof(A, fit["coef"], terms, seed)

    state.write("models", "ergm", {
        "method": "MPLE (maximum pseudo-likelihood)",
        "approximation": "MPLE ≈ MCMC-MLE;伪似然=逐 dyad change-stats 的 logistic 回归,"
                         "非 statnet::ergm 的 MCMC-MLE(intractable normalizing constant)",
        "terms": terms,
        "backend": fit["backend"],
        "coef": fit["coef"],
        "se": fit["se"],
        "z": fit["z"],
        "pseudo_llf": fit["llf"],
        "n_dyads": fit["n_obs"],
        "n_nodes": int(n),
        "n_edges": int(A.sum()),
        "observed_stats": obs,
        "note": "ERGM via MPLE:edges/mutual/transitive change statistics",
    })
    state.write("diagnostics", "gof", gof)
    return state


# ------------------------------------------------------ network_statistics helpers
def _normalize_summary_terms(raw: Any) -> list[Any]:
    """Coerce a user-supplied ``terms=`` spec into ``summary_formula`` term specs.

    Accepts a single term or a sequence of terms; each element is either a bare
    string (``"edges"`` / ``"mutual"`` / ``"triangle"``) or a ``(kind, arg)``
    pair (e.g. ``("idegree", [0, 1, 2])``). Lists inside a term (a degree spec)
    are preserved; only the *outer* container is treated as the term list.
    """
    if raw is None:
        return ["edges", "mutual"]
    # a lone bare-string term, or a single (kind, arg) tuple
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, tuple):
        return [raw]
    if isinstance(raw, (list,)):
        return list(raw)
    return [raw]


# ------------------------------------------------------------ network_statistics
@register(
    name="network_statistics",
    aliases=["network_stats", "网络统计量", "三元组普查"],
    category="net",
    tier="pro",
    skill="(ERGM 充分统计量 + Holland-Leinhardt 三元组普查,Python 原生空白)",
    languages=["Python"],
    key_tools=["numpy", "networkx"],
    description="有向网络的 ERGM 充分统计量(summary_formula)与 Holland-Leinhardt 16 型三元组普查(triad_census)",
    requires={"sources": ["datasets"]},
    produces={"models": ["network_stats"]},
    prerequisites={"optional_functions": ["build_network"]},
    auto_fix="escalate",
)
def network_statistics(state: StudyState, **kwargs: Any) -> StudyState:
    """Report observed ERGM sufficient statistics and the directed triad census.

    The directed network is read from ``edges=`` (or ``sources['datasets']``)
    and turned into a 0/1 adjacency matrix. Two exact (non-estimated) summaries
    are computed by delegating to the parity-gated ``pyergm`` port:

    - :func:`~socialverse.external.pyergm.summary_formula` — the vector of
      **observed sufficient statistics** ``g(y)`` for a requested ``terms`` set
      (default ``["edges", "mutual"]``), reproducing R ``ergm``'s
      ``summary(net ~ terms)``. Each term is either a bare string
      (``"edges"``/``"mutual"``/``"triangle"``) or a ``(kind, arg)`` pair such as
      ``("idegree", [0, 1, 2])`` or ``("nodematch", attr)``.
    - :func:`~socialverse.external.pyergm.triad_census` — the Holland-Leinhardt
      **16-type directed triad census** (``sna::triad.census``: 003, 012, …, 300).

    Writes ``models['network_stats']`` (summary statistics keyed by ergm-style
    label, plus the triad census keyed by MAN code). Never raises: with no usable
    edge table — or if the port rejects the requested terms — it writes an
    empty-but-valid record so a resolver can chain past it.
    """
    terms = _normalize_summary_terms(kwargs.get("terms"))
    attr_name = kwargs.get("attr_name")

    def _empty(note: str) -> StudyState:
        state.write("models", "network_stats", {
            "method": "observed sufficient statistics + directed triad census",
            "terms": [t if isinstance(t, str) else list(t)[:1][0] for t in terms],
            "summary": {}, "triad_census": {},
            "n_nodes": 0, "n_edges": 0, "backend": None, "note": note,
        })
        return state

    df = _edges_from(state, kwargs, "edges")
    if df is None or df.empty:
        return _empty("缺少边表(edges= 或 sources['datasets']),无法计算网络统计量")

    A, nodes = _adjacency(df, kwargs)
    n = A.shape[0]
    if n < 3 or A.sum() == 0:
        return _empty("网络过小或无边,无法计算三元组普查")

    try:
        from ..external.pyergm import (
            TRIAD_CENSUS_LABELS,
            summary_formula,
            triad_census,
        )

        stats, labels = summary_formula(A, terms, directed=True, attr_name=attr_name)
        census = triad_census(A)
    except Exception as exc:  # noqa: BLE001 — graceful: never crash the resolver
        return _empty(f"pyergm 端口未能计算网络统计量({type(exc).__name__}: {exc})")

    summary = {lab: float(v) for lab, v in zip(labels, stats)}
    triad = {lab: int(c) for lab, c in zip(TRIAD_CENSUS_LABELS, census)}

    state.write("models", "network_stats", {
        "method": "observed sufficient statistics (summary_formula) + "
                  "Holland-Leinhardt directed triad census (triad_census)",
        "backend": "pyergm",
        "directed": True,
        "terms": labels,
        "summary": summary,
        "triad_census": triad,
        "n_triads": int(sum(triad.values())),
        "n_nodes": int(n),
        "n_edges": int(A.sum()),
        "note": "充分统计量=精确计数(summary(net~terms));三元组普查=sna::triad.census 16 型 MAN 编码",
    })
    return state


# ------------------------------------------------------------------ saom helpers
def _tie_sets(A: np.ndarray) -> set[tuple[int, int]]:
    """The set of directed ties ``(i, j)`` present in an adjacency matrix."""
    ii, jj = np.nonzero(A)
    return set(zip(ii.tolist(), jj.tolist()))


def _cross_lag(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation of two aligned vectors (NaN-safe), or None."""
    if len(x) < 3:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------- saom
@register(
    name="saom",
    aliases=["SAOM", "RSiena", "网络行为共演化"],
    category="net",
    tier="pro",
    skill="(SAOM 缺口,Python 空白)",
    languages=["Python"],
    key_tools=["networkx", "numpy"],
    description="随机行动者导向模型(SAOM)两波共演化的描述性刻画:Jaccard/生成消失率/交叉滞后",
    requires={"sources": ["datasets"]},
    produces={"models": ["saom"], "diagnostics": ["coevolution"]},
    auto_fix="escalate",
)
def saom(state: StudyState, **kwargs: Any) -> StudyState:
    """Describe two-wave network (and behavior) co-evolution, SAOM-style.

    Given two panel waves of the same actors — ``wave1=`` / ``wave2=`` edgelists
    (``wave1`` falls back to ``sources['datasets']``) — computes the change
    structure a Stochastic Actor-Oriented Model is fit to:

    - **Jaccard index** of tie stability between waves (SIENA's key data-quality
      statistic; too-low Jaccard means the waves are too far apart to model);
    - **tie creation / dissipation / maintenance** counts and rates;
    - **Hamming distance** (number of differing dyads) between the two adjacencies.

    When actor behavior vectors are supplied (``behavior1=`` / ``behavior2=``),
    adds the two cross-lagged correlations SAOM decomposes into *influence* and
    *selection*: network-position (in/out-degree) at wave 1 vs. behavior change,
    and behavior at wave 1 vs. degree change.

    Writes ``models['saom']`` and ``diagnostics['coevolution']``. This is the
    **descriptive / simplified** layer — honestly *not* the simulation-based
    method-of-moments SAOM estimation that ``RSiena`` performs. Never raises.
    """
    def _empty(note: str) -> StudyState:
        state.write("models", "saom", {
            "method": "two-wave descriptive (SAOM-style)", "jaccard": None,
            "n_nodes": 0, "backend": None, "note": note,
        })
        state.write("diagnostics", "coevolution", {"note": note})
        return state

    df1 = _edges_from(state, kwargs, "wave1")
    df2 = _edges_from(state, kwargs, "wave2")
    if df1 is None or df1.empty:
        return _empty("缺少 wave1 边表,无法刻画共演化")
    if df2 is None or df2.empty:
        return _empty("缺少 wave2 边表(可对 wave1 加噪造第二波),无法刻画共演化")

    # shared node ordering across both waves
    s1, t1 = _endpoint_cols(df1, kwargs)
    s2, t2 = _endpoint_cols(df2, kwargs)
    nodes = sorted(
        set(df1[s1]).union(df1[t1]).union(df2[s2]).union(df2[t2]), key=str
    )
    A1, _ = _adjacency(df1, kwargs, nodes=nodes)
    A2, _ = _adjacency(df2, kwargs, nodes=nodes)
    n = len(nodes)

    ties1, ties2 = _tie_sets(A1), _tie_sets(A2)
    created = ties2 - ties1
    dropped = ties1 - ties2
    maintained = ties1 & ties2
    union = ties1 | ties2
    jaccard = float(len(maintained) / len(union)) if union else 0.0
    hamming = int(len(created) + len(dropped))
    n_dyads = n * (n - 1)

    coevo: dict[str, Any] = {
        "method": "two-wave descriptive co-evolution (SAOM-style)",
        "approximation": "描述性/简化版:两波间 Jaccard/生成消失/交叉滞后,"
                         "非 RSiena 基于模拟的矩量法 SAOM 估计",
        "n_nodes": int(n),
        "wave1_ties": int(len(ties1)),
        "wave2_ties": int(len(ties2)),
        "jaccard": jaccard,
        "hamming_distance": hamming,
        "ties_created": int(len(created)),
        "ties_dropped": int(len(dropped)),
        "ties_maintained": int(len(maintained)),
        "creation_rate": float(len(created) / n_dyads) if n_dyads else None,
        "dissipation_rate": (float(len(dropped) / len(ties1))
                             if ties1 else None),
        "note": "Jaccard 是 SIENA 数据质量指标;过低=两波相距过远不宜建模",
    }

    # optional behavior co-evolution (influence vs. selection cross-lags)
    b1 = kwargs.get("behavior1")
    b2 = kwargs.get("behavior2")
    if b1 is not None and b2 is not None:
        b1 = np.asarray(list(b1), dtype=float)
        b2 = np.asarray(list(b2), dtype=float)
        if len(b1) == n and len(b2) == n:
            outdeg1 = A1.sum(axis=1)
            indeg1 = A1.sum(axis=0)
            db = b2 - b1
            ddeg = (A2.sum(axis=1) + A2.sum(axis=0)) - (outdeg1 + indeg1)
            coevo["behavior"] = {
                "influence_proxy": _cross_lag(indeg1, db),
                "selection_proxy": _cross_lag(b1, ddeg),
                "behavior_change_mean": float(np.mean(db)),
                "note": "influence≈入度→行为变化;selection≈行为→度变化(交叉滞后代理,非结构估计)",
            }

    state.write("models", "saom", {
        "method": "two-wave descriptive (SAOM-style)",
        "backend": "numpy",
        "jaccard": jaccard,
        "hamming_distance": hamming,
        "ties_created": int(len(created)),
        "ties_dropped": int(len(dropped)),
        "ties_maintained": int(len(maintained)),
        "n_nodes": int(n),
        "has_behavior": bool(b1 is not None and b2 is not None),
        "note": "SAOM 共演化(描述性):Jaccard/生成消失率/(可选)行为交叉滞后",
    })
    state.write("diagnostics", "coevolution", coevo)
    return state
