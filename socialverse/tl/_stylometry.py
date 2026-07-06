"""``sv.tl._stylometry`` — computational stylometry / authorship attribution.

One registry entry backs the *stylometry* gap skill:

- :func:`stylometry` (对标 R 的 **stylo** 包,Maciej Eder 等) — Burrows's Delta
  authorship attribution: from the **most-frequent-words** (MFW) profile of each
  document, standardize per-word relative frequencies into z-scores across the
  corpus, compute the **Burrows's Delta** (mean of per-feature absolute z-score
  differences, i.e. an L1 / Manhattan distance on standardized frequencies)
  distance matrix, cluster it hierarchically (average linkage) into a
  dendrogram, and attribute each document to its nearest neighbour's author.

The whole thing is *really computed* on ``numpy`` + ``scipy`` — MFW selection,
z-scoring, the Delta metric, and the linkage are all exact. The champion
reference (R ``stylo``) is only conceptual; there is no Python dependency on it.
``scipy``'s hierarchical clustering (:mod:`scipy.cluster.hierarchy`) is used when
present and degrades to a self-contained pure-``numpy`` average-linkage
agglomerator otherwise, so the module loads and runs even without SciPy — never
raising at import time and never touching the network. The dendrogram PNG is
rendered with a headless ``matplotlib`` (Agg) and its path stored in
``artifacts['figures']``.

Speaks the 12-slot :class:`~socialverse._state.StudyState` vocabulary through the
``@register`` contract: it reads ``corpus['documents']`` and produces
``models['stylometry']`` + ``artifacts['figures']``, so a resolver can chain a
corpus-builder into it.
"""
from __future__ import annotations

import importlib
import os
import re
import tempfile
from typing import Any

import numpy as np

from .._registry import register
from .._state import StudyState

__all__ = ["stylometry"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft optional import — returns the module or ``None``."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _as_text(value: Any) -> str:
    """Stringify a document body (already-a-string fast path)."""
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _as_documents(documents: Any) -> dict[str, str]:
    """Coerce whatever arrived as ``documents`` into ``{doc_id: text}``.

    Accepts a mapping, or a sequence of ``(id, text)`` pairs / ``{"id","text"}``
    records / bare strings (auto-numbered). Mirrors the ``sv.tl._text`` coercion
    so the two corpus-consuming functions accept identical inputs.
    """
    if documents is None:
        return {}
    if isinstance(documents, dict):
        return {str(k): _as_text(v) for k, v in documents.items()}
    out: dict[str, str] = {}
    if isinstance(documents, (list, tuple)):
        for i, item in enumerate(documents):
            if isinstance(item, dict):
                did = str(item.get("id") or item.get("doc") or f"D{i + 1}")
                out[did] = _as_text(item.get("text") or item.get("content") or "")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = _as_text(item[1])
            else:
                out[f"D{i + 1}"] = _as_text(item)
        return out
    return {"D1": _as_text(documents)}


def _tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenization (Unicode word characters).

    Stylometry works on word tokens; punctuation and case are folded out so the
    frequency profile reflects lexical (function-word) habits, not typography.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _author_of(doc_id: str) -> str:
    """Infer an author label from a ``<author>_<k>`` document id.

    The toy corpus ids are ``austen_1`` … ``melville_3``; the author is the part
    before the final underscore-number. Ids without that shape return themselves.
    """
    m = re.match(r"^(.*)_\d+$", doc_id)
    return m.group(1) if m else doc_id


def _mfw_features(
    tokens_by_doc: dict[str, list[str]], n_mfw: int
) -> tuple[list[str], np.ndarray]:
    """Select the top-``n_mfw`` most-frequent words and build a relative-frequency
    matrix ``(n_docs, n_features)``.

    Frequency ranking is over the pooled corpus (the classic ``stylo`` MFW list);
    each cell is the word's within-document relative frequency (count / doc length),
    so documents of different lengths are comparable before standardization.
    """
    doc_ids = list(tokens_by_doc)

    # pooled frequency of every word across the whole corpus ------------------
    pooled: dict[str, int] = {}
    for toks in tokens_by_doc.values():
        for w in toks:
            pooled[w] = pooled.get(w, 0) + 1
    if not pooled:
        return [], np.zeros((len(doc_ids), 0))

    # rank by frequency (desc), ties broken alphabetically for determinism
    ranked = sorted(pooled, key=lambda w: (-pooled[w], w))
    features = ranked[: max(1, int(n_mfw))]

    # per-document relative frequencies over the selected features ------------
    mat = np.zeros((len(doc_ids), len(features)), dtype=float)
    fidx = {w: j for j, w in enumerate(features)}
    for i, did in enumerate(doc_ids):
        toks = tokens_by_doc[did]
        n = len(toks)
        if n == 0:
            continue
        for w in toks:
            j = fidx.get(w)
            if j is not None:
                mat[i, j] += 1.0
        mat[i] /= n
    return features, mat


def _zscore(freq: np.ndarray) -> np.ndarray:
    """Column-wise z-standardization across the corpus (Burrows's scaling).

    Each feature (word) is centred on its corpus mean and scaled by its corpus
    standard deviation, so every function word contributes on a common scale —
    the step that turns raw relative frequencies into a Delta-ready profile.
    Zero-variance columns (a word used identically everywhere) map to 0.
    """
    mu = freq.mean(axis=0)
    sd = freq.std(axis=0, ddof=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    z = (freq - mu) / sd_safe
    z[:, sd == 0] = 0.0
    return z


def _delta_matrix(z: np.ndarray) -> np.ndarray:
    """Burrows's Delta distance matrix from z-scored profiles.

    Delta(A, B) is the mean over features of ``|z_A - z_B|`` — the Manhattan
    (L1) distance on standardized frequencies, averaged by the number of
    features. Returns a symmetric ``(n_docs, n_docs)`` matrix with a zero
    diagonal.
    """
    n_docs, n_feat = z.shape
    d = np.zeros((n_docs, n_docs), dtype=float)
    denom = n_feat if n_feat else 1
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            dist = float(np.abs(z[i] - z[j]).sum() / denom)
            d[i, j] = d[j, i] = dist
    return d


def _condensed(dm: np.ndarray) -> np.ndarray:
    """Upper-triangle of a square distance matrix, in scipy ``pdist`` order."""
    n = dm.shape[0]
    iu = np.triu_indices(n, k=1)
    return dm[iu]


def _linkage_average(dm: np.ndarray):
    """Average-linkage hierarchical clustering.

    Uses :func:`scipy.cluster.hierarchy.linkage` (method=``average``) on the
    condensed Delta distances when SciPy is available; otherwise falls back to a
    self-contained pure-``numpy`` UPGMA agglomerator producing the identical
    scipy-format linkage matrix ``[idx_a, idx_b, dist, size]``. The fallback is
    honestly a from-scratch UPGMA — same math, no acceleration.
    """
    sch = _try_import("scipy.cluster.hierarchy")
    if sch is not None:
        return np.asarray(sch.linkage(_condensed(dm), method="average")), "scipy"
    return _upgma(dm), "numpy-upgma"


def _upgma(dm: np.ndarray) -> np.ndarray:
    """Pure-numpy UPGMA (average-linkage) → scipy-format linkage matrix.

    Maintains active clusters keyed by scipy's node numbering (leaves ``0..n-1``,
    internal nodes ``n, n+1, …``). At each step it merges the closest pair by the
    average distance between their members and emits ``[a, b, dist, size]``.
    """
    n = dm.shape[0]
    # cluster id -> list of original leaf indices it contains
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))
    Z = np.zeros((n - 1, 4), dtype=float) if n > 1 else np.zeros((0, 4))
    next_id = n

    def avg_dist(a: int, b: int) -> float:
        ma, mb = members[a], members[b]
        s = 0.0
        for p in ma:
            for q in mb:
                s += dm[p, q]
        return s / (len(ma) * len(mb))

    for step in range(n - 1):
        best = None
        best_pair = (active[0], active[1]) if len(active) >= 2 else None
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                dval = avg_dist(a, b)
                if best is None or dval < best:
                    best = dval
                    best_pair = (a, b)
        a, b = best_pair  # type: ignore[misc]
        lo, hi = sorted((a, b))
        Z[step] = [lo, hi, best if best is not None else 0.0,
                   len(members[a]) + len(members[b])]
        members[next_id] = members[a] + members[b]
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        next_id += 1
    return Z


def _nearest_neighbour_attribution(
    dm: np.ndarray, doc_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Attribute each document to the author of its single nearest neighbour.

    For every document, the closest *other* document by Delta distance is found;
    its inferred author is the predicted author. Compared against the document's
    own inferred author, this yields a leave-one-out attribution accuracy — the
    canonical stylometric validation.
    """
    n = dm.shape[0]
    out: dict[str, dict[str, Any]] = {}
    for i in range(n):
        d = dm[i].copy()
        d[i] = np.inf
        j = int(np.argmin(d)) if n > 1 else i
        out[doc_ids[i]] = {
            "true_author": _author_of(doc_ids[i]),
            "nearest": doc_ids[j],
            "predicted_author": _author_of(doc_ids[j]),
            "delta": float(dm[i, j]) if n > 1 else 0.0,
        }
    return out


def _dendrogram_png(
    Z: np.ndarray, doc_ids: list[str], kwargs: dict[str, Any]
) -> str | None:
    """Render the linkage as a dendrogram PNG (headless Agg). Returns its path.

    Uses ``scipy.cluster.hierarchy.dendrogram`` for the drawing coordinates when
    available; otherwise walks the linkage matrix with a small self-contained
    plotter. Never raises — returns ``None`` if matplotlib is unavailable.
    """
    mpl = _try_import("matplotlib")
    if mpl is None or Z.shape[0] == 0:
        return None
    mpl.use("Agg")
    plt = _try_import("matplotlib.pyplot")
    if plt is None:
        return None

    path = _out_path(kwargs, "delta_dendrogram")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sch = _try_import("scipy.cluster.hierarchy")
    try:
        if sch is not None:
            sch.dendrogram(Z, labels=doc_ids, ax=ax,
                           color_threshold=0.7 * float(Z[:, 2].max()))
        else:
            _plot_dendrogram_numpy(ax, Z, doc_ids)
        ax.set_title("Burrows's Delta — hierarchical clustering (average linkage)")
        ax.set_ylabel("Delta distance")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _plot_dendrogram_numpy(ax, Z: np.ndarray, doc_ids: list[str]) -> None:
    """Minimal dendrogram plotter for the no-scipy path.

    Recursively assigns each leaf an x-position (in-order), draws the classic
    U-shaped joins bottom-up from the linkage matrix, and labels the leaves.
    """
    n = len(doc_ids)
    # positions/heights for every node (leaves then internal)
    x_pos: dict[int, float] = {}
    height: dict[int, float] = {i: 0.0 for i in range(n)}
    order: list[int] = []

    def leaves(node: int) -> list[int]:
        if node < n:
            return [node]
        a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
        return leaves(a) + leaves(b)

    root = n + Z.shape[0] - 1
    order = leaves(root)
    for k, leaf in enumerate(order):
        x_pos[leaf] = float(k)

    def xcoord(node: int) -> float:
        if node in x_pos:
            return x_pos[node]
        a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
        x = (xcoord(a) + xcoord(b)) / 2.0
        x_pos[node] = x
        return x

    for m in range(Z.shape[0]):
        a, b = int(Z[m, 0]), int(Z[m, 1])
        h = float(Z[m, 2])
        node = n + m
        height[node] = h
        xa, xb = xcoord(a), xcoord(b)
        ha, hb = height[a], height[b]
        ax.plot([xa, xa, xb, xb], [ha, h, h, hb], color="C0", lw=1.4)

    ax.set_xticks(range(n))
    ax.set_xticklabels([doc_ids[i] for i in order], rotation=90, fontsize=8)
    ax.set_xlim(-0.5, n - 0.5)


def _out_path(kwargs: dict[str, Any], stem: str) -> str:
    """Resolve the output PNG path: explicit ``out=`` kwarg, else a scratch temp file.

    A directory-only ``out=`` is honored by joining ``<stem>.png`` onto it; the
    parent directory is created if missing. PNG throughout, so a tight bbox on
    ``savefig`` crops cleanly (raster).
    """
    out = kwargs.get("out")
    if out:
        out = os.path.expanduser(str(out))
        if os.path.isdir(out) or out.endswith(os.sep):
            out = os.path.join(out, f"{stem}.png")
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return out
    fd, path = tempfile.mkstemp(prefix=f"sv_{stem}_", suffix=".png")
    os.close(fd)
    return path


# ------------------------------------------------------------------ stylometry
@register(
    name="stylometry",
    aliases=["文体计量", "Delta", "作者归属"],
    category="stylometry",
    tier="plus",
    skill="(文体计量 缺口,对标 stylo)",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description="文体计量/作者归属:MFW→z-score→Burrows's Delta→层次聚类+最近邻归属",
    requires={"corpus": ["documents"]},
    produces={"models": ["stylometry"], "artifacts": ["figures"]},
    prerequisites={"optional_functions": ["build_corpus"]},
    auto_fix="escalate",
)
def stylometry(state: StudyState, **kwargs: Any) -> StudyState:
    """Burrows's Delta authorship attribution over a corpus of documents.

    Pipeline (all really computed on ``numpy`` / ``scipy``):

    1. read the corpus from ``texts=`` / ``documents=`` / ``data=`` kwargs or
       ``state.corpus['documents']`` (``{doc_id: text}``); tokenize (lowercase
       word tokens);
    2. **MFW** — rank words by pooled corpus frequency and keep the top
       ``n_mfw`` (default 100, capped at the vocabulary size) as features; build a
       per-document relative-frequency matrix;
    3. **z-score** — standardize each feature across the corpus (Burrows's
       scaling);
    4. **Delta** — the Burrows's Delta distance matrix (mean absolute z-score
       difference = averaged Manhattan distance);
    5. **cluster** — average-linkage hierarchical clustering
       (``scipy`` with a pure-numpy UPGMA fallback) → a dendrogram PNG saved to
       ``artifacts['figures']['dendrogram']``;
    6. **attribution** — nearest-neighbour author prediction per document, with a
       leave-one-out accuracy → ``models['stylometry']``.

    Documents arrive via ``texts=`` / ``documents=`` / ``data=`` or
    ``state.corpus['documents']``. Never raises for missing data: with an empty
    corpus it writes an empty-but-valid result.
    """
    raw = kwargs.get("texts")
    if raw is None:
        raw = kwargs.get("documents")
    if raw is None:
        raw = kwargs.get("data")
    if raw is None:
        raw = state.corpus.get("documents")
    docs = _as_documents(raw)

    doc_ids = list(docs)
    tokens_by_doc = {did: _tokenize(text) for did, text in docs.items()}

    # empty / degenerate corpus — write a valid empty result, never raise -----
    if len(doc_ids) < 2 or all(not t for t in tokens_by_doc.values()):
        result = {
            "method": "Burrows's Delta (MFW z-score, Manhattan)",
            "n_documents": len(doc_ids),
            "documents": doc_ids,
            "features": [],
            "distance_matrix": [],
            "attribution": {},
            "accuracy": None,
            "note": "empty or degenerate corpus — need ≥2 non-empty documents",
        }
        state.write("models", "stylometry", result)
        state.write("artifacts", "figures", {})
        return state

    n_mfw = int(kwargs.get("n_mfw", 100))
    features, freq = _mfw_features(tokens_by_doc, n_mfw)
    z = _zscore(freq)
    dm = _delta_matrix(z)

    Z, linkage_backend = _linkage_average(dm)
    attribution = _nearest_neighbour_attribution(dm, doc_ids)

    correct = sum(
        1 for r in attribution.values()
        if r["predicted_author"] == r["true_author"]
    )
    accuracy = correct / len(doc_ids)

    fig_path = _dendrogram_png(Z, doc_ids, kwargs)

    result = {
        "method": "Burrows's Delta (MFW z-score, Manhattan / average linkage)",
        "n_documents": len(doc_ids),
        "documents": doc_ids,
        "n_mfw": len(features),
        "features": features,
        "distance_matrix": dm.tolist(),
        "linkage": Z.tolist(),
        "linkage_backend": linkage_backend,
        "attribution": attribution,
        "accuracy": float(accuracy),
        "n_correct": int(correct),
    }

    state.write("models", "stylometry", result)
    state.write("artifacts", "figures",
                {"dendrogram": fig_path} if fig_path else {})
    return state
