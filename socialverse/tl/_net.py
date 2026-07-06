"""``sv.tl._net`` — registered implementation for the ``network-analysis`` skill.

The social-network-analysis (SNA) axis of ``socialverse``: turn an edge list into
a graph, then read its structure the way a sociologist does — who is central
(degree / betweenness / eigenvector), how densely tied is the whole, and which
sub-communities does modularity cut it into.

Real computation only: the graph is built with ``networkx.from_pandas_edgelist``
and every quantity — the three centralities, density, connected components, and
the greedy-modularity community partition — is computed on that graph, not
stubbed. When no edge table is supplied the function returns an empty-but-valid
network record rather than raising, so a resolver can still chain past it.

The registry contract wires this into the spine: ``build_network`` *requires*
``sources.datasets`` (an edge table) and *produces* ``models.network`` (the
structural summary) plus ``diagnostics.coverage`` (component structure) — the
slots a downstream reporting/figure step reads from.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_edges(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working edge table.

    Priority: explicit ``edges=`` kwarg, then ``data=`` kwarg, then
    ``sources['datasets']``. ``sources['datasets']`` may be a DataFrame or a
    ``{name: DataFrame}`` mapping; in the latter case the first frame is taken.
    """
    df = kwargs.get("edges")
    if df is None:
        df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _resolve_endpoints(
    df: pd.DataFrame, kwargs: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    """Pick the ``(source, target, weight)`` columns from kwargs, else by convention.

    Falls back to common names (source/target, from/to, u/v, i/j) and finally to
    the first two columns of the frame. ``weight`` is optional and only used when
    present.
    """
    cols = list(df.columns)
    src = kwargs.get("source")
    tgt = kwargs.get("target")
    wt = kwargs.get("weight")

    if src is None or tgt is None:
        candidates = [
            ("source", "target"),
            ("from", "to"),
            ("u", "v"),
            ("i", "j"),
            ("ego", "alter"),
        ]
        lower = {c.lower(): c for c in cols}
        for a, b in candidates:
            if a in lower and b in lower:
                src, tgt = lower[a], lower[b]
                break
    if (src is None or tgt is None) and len(cols) >= 2:
        src, tgt = cols[0], cols[1]

    if wt is None:
        for name in ("weight", "w", "value", "count"):
            if name in {c.lower() for c in cols}:
                wt = next(c for c in cols if c.lower() == name)
                break
    if wt is not None and wt not in cols:
        wt = None
    return src, tgt, wt


def _build_graph(
    nx, df: pd.DataFrame, src: str, tgt: str, wt: str | None, directed: bool
):
    """Construct a (di)graph from the edge frame, dropping null endpoints."""
    clean = df[[c for c in (src, tgt, wt) if c is not None]].dropna(subset=[src, tgt])
    create_using = nx.DiGraph if directed else nx.Graph
    edge_attr = wt if wt is not None else None
    return nx.from_pandas_edgelist(
        clean, source=src, target=tgt, edge_attr=edge_attr,
        create_using=create_using,
    )


def _top(scores: dict[Any, float], k: int) -> dict[str, float]:
    """The ``k`` highest-scoring nodes as a JSON-friendly ``{str(node): score}``."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {str(n): float(v) for n, v in ordered[:k]}


def _components(nx, G) -> dict[str, Any]:
    """Connected-component structure (weakly connected for digraphs)."""
    if G.number_of_nodes() == 0:
        return {"n_components": 0, "largest_cc_size": 0, "largest_cc_frac": None,
                "component_sizes": []}
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
        kind = "weakly_connected"
    else:
        comps = list(nx.connected_components(G))
        kind = "connected"
    sizes = sorted((len(c) for c in comps), reverse=True)
    n = G.number_of_nodes()
    return {
        "kind": kind,
        "n_components": len(sizes),
        "largest_cc_size": sizes[0],
        "largest_cc_frac": float(sizes[0] / n),
        "component_sizes": sizes[:20],
    }


# ------------------------------------------------------------------ build_network
@register(
    name="build_network",
    aliases=["建网络", "network", "SNA"],
    category="net",
    tier="plus",
    skill="network-analysis",
    languages=["Python"],
    key_tools=["networkx"],
    description="从边表构建社会网络并算中心性(度/介数/特征向量)与社群",
    requires={"sources": ["datasets"]},
    produces={"models": ["network"], "diagnostics": ["coverage"]},
    auto_fix="none",
)
def build_network(state: StudyState, **kwargs: Any) -> StudyState:
    """Build a social network from an edge table and read its structure.

    Constructs a graph with ``networkx.from_pandas_edgelist`` (columns resolved
    from ``source`` / ``target`` [/ ``weight``] kwargs, else by convention), then
    computes:

    - **centrality** — degree, betweenness, and eigenvector centrality per node
      (the top-``k`` of each are surfaced; full vectors would bloat the state);
    - **density** — realized ties over possible ties;
    - **communities** — a greedy-modularity partition with its modularity ``Q``.

    Writes ``models['network']`` (n_nodes / n_edges / density / directed /
    centrality / communities) and ``diagnostics['coverage']`` (component
    structure). With no usable edge table it writes an empty-but-valid record and
    returns — never raises — so a resolver can chain past a missing input.
    """
    nx = _try_import("networkx")
    top_k = int(kwargs.get("top_k", 10))
    directed = bool(kwargs.get("directed", False))
    seed = int(kwargs.get("seed", 0))

    def _empty(note: str) -> StudyState:
        state.write("models", "network", {
            "n_nodes": 0, "n_edges": 0, "density": None, "directed": directed,
            "centrality": {"degree": {}, "betweenness": {}, "eigenvector": {}},
            "communities": {"n_communities": 0, "modularity": None, "sizes": []},
            "note": note,
        })
        state.write("diagnostics", "coverage", {
            "n_components": 0, "largest_cc_size": 0, "largest_cc_frac": None,
            "component_sizes": [], "note": note,
        })
        return state

    if nx is None:
        return _empty("networkx 未安装,无法构建网络")

    df = _get_edges(state, kwargs)
    if df is None or df.empty:
        return _empty("缺少边表(sources['datasets'] 或 edges=),无法构建网络")

    src, tgt, wt = _resolve_endpoints(df, kwargs)
    if src is None or tgt is None or src not in df.columns or tgt not in df.columns:
        return _empty("无法识别 source/target 列(可用 source=/target= 指定)")

    G = _build_graph(nx, df, src, tgt, wt, directed)
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return _empty("边表清洗后为空(无有效端点)")

    weight_attr = wt if wt is not None else None

    # -- density -----------------------------------------------------------
    density = float(nx.density(G))

    # -- centrality (three canonical measures) -----------------------------
    deg = nx.degree_centrality(G)
    try:
        btw = nx.betweenness_centrality(G, weight=weight_attr, seed=seed)
    except Exception:
        btw = nx.betweenness_centrality(G, seed=seed)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000, weight=weight_attr)
    except Exception:
        # power iteration can fail to converge on disconnected / degenerate graphs;
        # fall back to the numpy eigensolver, then to degree as a last resort.
        try:
            eig = nx.eigenvector_centrality_numpy(G, weight=weight_attr)
        except Exception:
            eig = dict(deg)

    centrality = {
        "degree": _top(deg, top_k),
        "betweenness": _top(btw, top_k),
        "eigenvector": _top({k: float(v) for k, v in eig.items()}, top_k),
    }

    # -- communities (greedy modularity partition) -------------------------
    communities = {"n_communities": 0, "modularity": None, "sizes": []}
    try:
        from networkx.algorithms.community import (
            greedy_modularity_communities, modularity,
        )
        comm_G = G.to_undirected() if G.is_directed() else G
        parts = list(greedy_modularity_communities(comm_G, weight=weight_attr))
        if parts:
            sizes = sorted((len(c) for c in parts), reverse=True)
            try:
                q = float(modularity(comm_G, parts, weight=weight_attr))
            except Exception:
                q = None
            largest = max(parts, key=len)
            communities = {
                "n_communities": len(parts),
                "modularity": q,
                "sizes": sizes[:50],
                "largest_members": [str(x) for x in sorted(largest, key=str)[:top_k]],
            }
    except Exception:
        communities = {"n_communities": 0, "modularity": None, "sizes": [],
                       "note": "社群划分不可用"}

    coverage = _components(nx, G)
    coverage["note"] = "连通分量结构(有向图按弱连通)"

    state.write("models", "network", {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": density,
        "directed": directed,
        "weighted": weight_attr is not None,
        "avg_degree": float(2 * m / n) if not directed else float(m / n),
        "columns": {"source": src, "target": tgt, "weight": wt},
        "centrality": centrality,
        "communities": communities,
        "note": "SNA:度/介数/特征向量中心性 + greedy-modularity 社群",
    })
    state.write("diagnostics", "coverage", coverage)
    return state


__all__ = ["build_network"]
