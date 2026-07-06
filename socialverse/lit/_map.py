"""``sv.lit._map`` — registered implementations for the literature-map,
citation-management, and manuscript-review skills.

Three community-tier functions that close the literature loop of ``socialverse``:

- :func:`literature_map` reads the registered ``.bib`` and draws a **knowledge
  landscape** — schools of thought (clustered by shared keywords / co-citation
  via a real ``networkx`` community partition), seminal works, debate axes, and a
  time-line. Structured output (dicts/DataFrames), plus an optional quadrant
  figure rendered with the matplotlib **Agg** backend when it is available.
- :func:`citation_manage` formats a reference list into a target journal style —
  APA 7, Vancouver, or Nature — with three genuine template implementations, not
  placeholder strings.
- :func:`manuscript_review` audits a manuscript: it regex-extracts the in-text
  citations, cross-balances them against the already-``verified_bib`` (orphan /
  uncited), runs a hedge-mismatch first pass over causal / absolute language, and
  labels every claim ``supported`` / ``unsupported`` / ``over-claim`` — emitting a
  readiness verdict (``BLOCKER`` / ``MAJOR`` / ``MINOR`` / ``READY``).

Every heavy or optional dependency (matplotlib, bibtexparser) is imported lazily
and fails soft: a missing backend degrades the figure / parse step to a valid
skip, never an import-time error and never a network call. The registry contracts
below are the machine-readable spine — ``requires`` / ``produces`` must match the
skill spec exactly so :meth:`FunctionRegistry.resolve_plan` can chain these steps.
"""
from __future__ import annotations

import base64
import importlib
import io
import os
import re
import tempfile
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["literature_map", "citation_manage", "manuscript_review"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


#: English stop-words dropped before keyword clustering (small, self-contained).
_STOP = frozenset(
    """a an the of and or for to in on with without into from by at as is are be
    been being this that these those it its their our your his her they we you i
    using use used toward towards via new novel study studies analysis approach
    approaches method methods framework frameworks case cases evidence review
    research paper papers article articles digital data""".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")


def _get_bib(state: StudyState, kwargs: dict[str, Any]) -> list[dict]:
    """Resolve the working reference list.

    Priority: explicit ``bib=`` kwarg, then ``sources['bib']``. Accepts a list of
    record dicts, a single record dict, or a DataFrame (rows → records). Returns a
    normalized list; each record keeps ``id / title / authors / year / doi`` when
    present. Never raises on empty — returns ``[]`` so a resolver can chain past.
    """
    bib = kwargs.get("bib")
    if bib is None:
        bib = state.sources.get("bib")
    if bib is None:
        return []
    if isinstance(bib, pd.DataFrame):
        bib = bib.to_dict("records")
    if isinstance(bib, dict):
        bib = [bib]
    out: list[dict] = []
    for i, rec in enumerate(bib):
        if not isinstance(rec, dict):
            continue
        rec = dict(rec)
        rec.setdefault("id", rec.get("key") or rec.get("ID") or f"ref{i + 1}")
        out.append(rec)
    return out


def _authors_of(rec: dict) -> list[str]:
    """A record's author list, normalized to a list of strings."""
    a = rec.get("authors") or rec.get("author")
    if a is None:
        return []
    if isinstance(a, str):
        # "Braun, V. and Clarke, V." or "Braun, V.; Clarke, V."
        return [p.strip() for p in re.split(r"\s+and\s+|;", a) if p.strip()]
    return [str(x).strip() for x in a if str(x).strip()]


def _year_of(rec: dict) -> int | None:
    """Parse a publication year, tolerating strings / floats / ``None``."""
    y = rec.get("year")
    if y is None:
        return None
    try:
        return int(str(y)[:4])
    except (ValueError, TypeError):
        return None


def _keywords(rec: dict) -> list[str]:
    """Content keywords of a record: explicit ``keywords`` else title tokens."""
    kw = rec.get("keywords")
    if kw:
        if isinstance(kw, str):
            kw = re.split(r"[;,]", kw)
        toks = [str(k).strip().lower() for k in kw if str(k).strip()]
    else:
        toks = [m.group(0).lower() for m in _WORD.finditer(str(rec.get("title", "")))]
    return [t for t in toks if t not in _STOP and len(t) > 2]


def _last_name(author: str) -> str:
    """Best-effort surname from an author string (``"Braun, V."`` → ``"Braun"``)."""
    author = author.strip()
    if "," in author:
        return author.split(",", 1)[0].strip()
    parts = author.split()
    return parts[-1] if parts else author


# ------------------------------------------------------------------ literature_map
def _keyword_graph(nx, records: list[dict]) -> Any:
    """A record-record graph weighted by shared keywords (co-citation proxy).

    Nodes are reference ids; an edge joins two references sharing ≥1 keyword, its
    weight the number of shared keywords. This is the substrate the community
    partition cuts into "schools".
    """
    G = nx.Graph()
    kw_sets = {}
    for rec in records:
        rid = str(rec["id"])
        G.add_node(rid)
        kw_sets[rid] = set(_keywords(rec))
    ids = list(kw_sets)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shared = kw_sets[ids[i]] & kw_sets[ids[j]]
            if shared:
                G.add_edge(ids[i], ids[j], weight=len(shared))
    return G


def _partition(nx, G) -> list[list[str]]:
    """Greedy-modularity community partition, with connected-component fallback."""
    if G.number_of_nodes() == 0:
        return []
    try:
        from networkx.algorithms.community import greedy_modularity_communities

        parts = list(greedy_modularity_communities(G, weight="weight"))
        if parts:
            return [sorted(map(str, p)) for p in parts]
    except Exception:
        pass
    return [sorted(map(str, c)) for c in nx.connected_components(G)]


def _schools_from_partition(
    parts: list[list[str]], by_id: dict[str, dict], max_schools: int = 7
) -> list[dict]:
    """Turn a partition into 3–7 labelled schools of thought (structured)."""
    schools: list[dict] = []
    parts = sorted(parts, key=len, reverse=True)[:max_schools]
    for k, members in enumerate(parts, start=1):
        recs = [by_id[m] for m in members if m in by_id]
        kw_counts = Counter(w for r in recs for w in _keywords(r))
        label_terms = [w for w, _ in kw_counts.most_common(3)]
        authors = Counter(_last_name(a) for r in recs for a in _authors_of(r))
        years = [y for y in (_year_of(r) for r in recs) if y is not None]
        schools.append({
            "school_id": f"S{k}",
            "label": " / ".join(label_terms) or f"cluster {k}",
            "size": len(members),
            "members": members,
            "keywords": label_terms,
            "key_authors": [a for a, _ in authors.most_common(3)],
            "period": [min(years), max(years)] if years else None,
        })
    return schools


def _seminal_works(records: list[dict], top_n: int = 5) -> list[dict]:
    """Rank references by an influence proxy (explicit citations if present, else
    recency-adjusted keyword centrality)."""
    def score(rec: dict) -> float:
        cited = rec.get("cited_by") or rec.get("citations") or rec.get("n_citations")
        if cited is not None:
            try:
                return float(cited)
            except (ValueError, TypeError):
                pass
        # proxy: more content keywords + earlier year (foundational) score higher
        y = _year_of(rec)
        recency = (2100 - y) / 100.0 if y else 0.0
        return len(_keywords(rec)) + recency

    ranked = sorted(records, key=score, reverse=True)[:top_n]
    return [{
        "id": str(r["id"]),
        "title": r.get("title", ""),
        "authors": _authors_of(r),
        "year": _year_of(r),
        "influence": round(float(score(r)), 3),
    } for r in ranked]


def _debate_axes(schools: list[dict], max_axes: int = 4) -> list[dict]:
    """Contrast the largest schools pairwise into 2–4 debate axes (structured)."""
    axes: list[dict] = []
    top = schools[:4]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            axes.append({
                "axis": f"{a['label']}  ↔  {b['label']}",
                "pole_a": {"school": a["school_id"], "stance": a["label"],
                           "authors": a["key_authors"]},
                "pole_b": {"school": b["school_id"], "stance": b["label"],
                           "authors": b["key_authors"]},
                "tension": "分属不同关键词聚类,代表该文献场的一条论战轴线",
            })
            if len(axes) >= max_axes:
                return axes
    return axes


def _timeline(records: list[dict]) -> list[dict]:
    """Chronological pulse: reference count and exemplar per year."""
    by_year: dict[int, list[dict]] = {}
    for r in records:
        y = _year_of(r)
        if y is not None:
            by_year.setdefault(y, []).append(r)
    out = []
    for y in sorted(by_year):
        recs = by_year[y]
        out.append({
            "year": y,
            "n": len(recs),
            "exemplar": recs[0].get("title", ""),
        })
    return out


def _quadrant_figure(schools: list[dict]) -> dict | None:
    """Render a school-landscape quadrant scatter with the matplotlib Agg backend.

    x = temporal midpoint (older ↔ newer), y = school size. Returns a dict with
    the saved PNG path and a base64 data-URI, or ``None`` when matplotlib is
    unavailable (fail-soft: the caller simply omits the figure).
    """
    mpl = _try_import("matplotlib")
    if mpl is None or not schools:
        return None
    try:
        mpl.use("Agg", force=True)
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return None

    try:
        periods = [s["period"] for s in schools if s.get("period")]
        if periods:
            lo = min(p[0] for p in periods)
            hi = max(p[1] for p in periods)
        else:
            lo, hi = 2000, 2020
        span = max(hi - lo, 1)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for s in schools:
            p = s.get("period")
            x = ((p[0] + p[1]) / 2 - lo) / span if p else 0.5
            y = s["size"]
            ax.scatter(x, y, s=120 + 40 * s["size"], alpha=0.6, edgecolor="k",
                       linewidth=0.5)
            ax.annotate(s["label"], (x, y), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.axvline(0.5, color="gray", lw=0.6, ls="--")
        if schools:
            ax.axhline(np.median([s["size"] for s in schools]),
                       color="gray", lw=0.6, ls="--")
        ax.set_xlabel("time  (older  →  newer)")
        ax.set_ylabel("school size  (# works)")
        ax.set_title("Knowledge landscape — schools of thought")
        ax.set_xlim(-0.05, 1.05)
        fig.tight_layout()

        out_dir = os.path.join(tempfile.gettempdir(), "socialverse_figures")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "literature_landscape.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")  # PNG → tight per house style

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"path": path, "format": "png", "data_uri": data_uri,
                "caption": "文献知识地形图(象限:时间×流派规模)"}
    except Exception:
        return None


@register(
    name="literature_map",
    aliases=["文献地图", "literature_map"],
    category="literature",
    tier="community",
    skill="literature-map",
    languages=["无代码(方法论)"],
    key_tools=["co-citation clustering"],
    description="主题知识地形图:流派/代表学者/论战轴线/时间脉络(结构化)",
    requires={"sources": ["bib"]},
    produces={"evidence": ["landscape"], "artifacts": ["figures"]},
    auto_fix="auto",
)
def literature_map(state: StudyState, **kwargs: Any) -> StudyState:
    """Draw a knowledge landscape from the registered ``.bib``.

    Reads ``sources['bib']`` (or a ``bib=`` kwarg), builds a keyword co-occurrence
    graph as a co-citation proxy, and cuts it into **schools of thought** with a
    real ``networkx`` greedy-modularity partition. Also surfaces **seminal works**
    (ranked by an influence proxy), **debate axes** (largest schools contrasted
    pairwise), and a **timeline**. Writes ``evidence['landscape']`` (the full
    structured map) and, when matplotlib is present, an optional quadrant figure to
    ``artifacts['figures']``. With no usable ``.bib`` it writes an empty-but-valid
    landscape and returns — never raises.
    """
    max_schools = int(kwargs.get("max_schools", 7))
    records = _get_bib(state, kwargs)

    if not records:
        state.write("evidence", "landscape", {
            "n_references": 0, "schools": [], "seminal_works": [],
            "debate_axes": [], "timeline": [],
            "note": "缺少 sources['bib'](或 bib=),无法绘制知识地形图",
        })
        state.write("artifacts", "figures", {})
        return state

    by_id = {str(r["id"]): r for r in records}
    nx = _try_import("networkx")

    if nx is not None:
        G = _keyword_graph(nx, records)
        parts = _partition(nx, G)
        method = "networkx greedy-modularity(keyword co-occurrence)"
    else:
        # fail-soft: group by dominant keyword when networkx is unavailable
        buckets: dict[str, list[str]] = {}
        for r in records:
            kws = _keywords(r)
            buckets.setdefault(kws[0] if kws else "misc", []).append(str(r["id"]))
        parts = list(buckets.values())
        method = "keyword-bucket fallback(networkx 未安装)"

    schools = _schools_from_partition(parts, by_id, max_schools=max_schools)
    landscape = {
        "n_references": len(records),
        "n_schools": len(schools),
        "cluster_method": method,
        "schools": schools,
        "seminal_works": _seminal_works(records, top_n=int(kwargs.get("top_n", 5))),
        "debate_axes": _debate_axes(schools),
        "timeline": _timeline(records),
        "note": "知识地形图:流派(共现聚类)/代表学者/论战轴线/时间脉络",
    }
    state.write("evidence", "landscape", landscape)

    fig = _quadrant_figure(schools)
    state.write("artifacts", "figures",
                {"landscape": fig} if fig else {})
    return state


# ------------------------------------------------------------------ citation_manage
def _fmt_authors_apa(authors: list[str]) -> str:
    """APA-7 author string: ``Braun, V., & Clarke, V.`` (up to 20, then ellipsis)."""
    if not authors:
        return "Anonymous"
    norm = [_apa_name(a) for a in authors]
    if len(norm) == 1:
        return norm[0]
    if len(norm) <= 20:
        return ", ".join(norm[:-1]) + ", & " + norm[-1]
    return ", ".join(norm[:19]) + ", … " + norm[-1]


def _apa_name(author: str) -> str:
    """One author → ``Surname, I. I.`` (APA/Nature initials style)."""
    if "," in author:
        surname, given = [p.strip() for p in author.split(",", 1)]
    else:
        parts = author.split()
        surname, given = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (author, "")
    initials = " ".join(
        f"{g[0].upper()}." for g in re.split(r"[\s.]+", given) if g
    )
    return f"{surname}, {initials}".strip().rstrip(",") if initials else surname


def _fmt_authors_vancouver(authors: list[str]) -> str:
    """Vancouver author string: ``Braun V, Clarke V`` (≤6, else 6 + et al.)."""
    def one(a: str) -> str:
        if "," in a:
            surname, given = [p.strip() for p in a.split(",", 1)]
        else:
            parts = a.split()
            surname, given = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (a, "")
        inits = "".join(g[0].upper() for g in re.split(r"[\s.]+", given) if g)
        return f"{surname} {inits}".strip()

    names = [one(a) for a in authors] or ["Anonymous"]
    if len(names) > 6:
        return ", ".join(names[:6]) + ", et al"
    return ", ".join(names)


def _fmt_authors_nature(authors: list[str]) -> str:
    """Nature author string: ``Braun, V. & Clarke, V.`` (≤5, else first + et al.)."""
    norm = [_apa_name(a) for a in authors] or ["Anonymous"]
    if len(norm) == 1:
        return norm[0]
    if len(norm) > 5:
        return norm[0] + " et al."
    return ", ".join(norm[:-1]) + " & " + norm[-1]


def _format_reference(rec: dict, style: str) -> str:
    """Format one reference record into the requested journal style.

    Implements APA 7, Vancouver, and Nature templates against the
    ``id/title/authors/year/doi`` record shape (with ``journal/volume/pages``
    used when present).
    """
    style_l = style.strip().lower()
    authors = _authors_of(rec)
    year = _year_of(rec)
    title = str(rec.get("title", "")).strip().rstrip(".")
    journal = str(rec.get("journal", "") or "").strip()
    volume = str(rec.get("volume", "") or "").strip()
    pages = str(rec.get("pages", "") or "").strip()
    doi = rec.get("doi")
    doi_txt = f" https://doi.org/{doi}" if doi else ""

    if style_l in {"apa", "apa7", "apa 7"}:
        auth = _fmt_authors_apa(authors)
        yr = f"({year})" if year else "(n.d.)"
        jrnl = f" {journal}" if journal else ""
        vol = f", {volume}" if volume else ""
        pg = f", {pages}" if pages else ""
        tail = f".{jrnl}{vol}{pg}." if journal else "."
        return f"{auth} {yr}. {title}{tail}{doi_txt}".strip()

    if style_l in {"vancouver", "nlm", "icmje"}:
        auth = _fmt_authors_vancouver(authors)
        yr = f" {year}" if year else ""
        jrnl = f" {journal}." if journal else ""
        vp = ""
        if volume or pages:
            vp = f"{yr};{volume}:{pages}." if journal else f"{yr};{volume}:{pages}."
        else:
            vp = f"{yr}." if yr else "."
        doi_v = f" doi:{doi}" if doi else ""
        return f"{auth}. {title}.{jrnl}{vp}{doi_v}".replace("  ", " ").strip()

    if style_l in {"nature", "nat"}:
        auth = _fmt_authors_nature(authors)
        jrnl = f" {journal}" if journal else ""
        vol = f" {volume}," if volume else ""
        pg = f" {pages}" if pages else ""
        yr = f" ({year})" if year else ""
        tail = f".{jrnl}{vol}{pg}{yr}." if journal else f".{yr}."
        return f"{auth} {title}{tail}{doi_txt}".replace("  ", " ").strip()

    # unknown style → fall back to APA
    return _format_reference(rec, "APA")


@register(
    name="citation_manage",
    aliases=["引用管理", "citation_management"],
    category="literature",
    tier="community",
    skill="citation-management",
    languages=["无代码(方法论)"],
    key_tools=["CSL", "BibTeX"],
    description="按目标期刊风格(APA/Vancouver/Nature)格式化参考文献列表",
    requires={"sources": ["bib"]},
    produces={"evidence": ["citations"], "artifacts": ["tables"]},
    auto_fix="none",
)
def citation_manage(state: StudyState, **kwargs: Any) -> StudyState:
    """Format the registered ``.bib`` into a target journal style.

    Reads ``sources['bib']`` (or ``bib=``) and formats every record with the
    ``style=`` template — ``"APA"`` (7th ed.), ``"Vancouver"``, or ``"Nature"``.
    Writes ``evidence['citations']`` (the ordered, formatted list + the resolved
    style) and ``artifacts['tables']`` (a per-reference DataFrame). With no ``.bib``
    it writes an empty list and returns — never raises.
    """
    style = str(kwargs.get("style", "APA"))
    records = _get_bib(state, kwargs)

    if not records:
        state.write("evidence", "citations",
                    {"style": style, "n": 0, "formatted": [],
                     "note": "缺少 sources['bib'](或 bib=),无引用可格式化"})
        state.write("artifacts", "tables",
                    pd.DataFrame(columns=["id", "formatted", "style"]))
        return state

    # APA sorts by author surname; numbered styles keep input (citation) order.
    style_l = style.strip().lower()
    if style_l in {"apa", "apa7", "apa 7"}:
        records = sorted(
            records,
            key=lambda r: (
                (_last_name(_authors_of(r)[0]).lower() if _authors_of(r) else "zzz"),
                _year_of(r) or 9999,
            ),
        )

    formatted: list[dict] = []
    rows: list[dict] = []
    for n, rec in enumerate(records, start=1):
        text = _format_reference(rec, style)
        formatted.append({"n": n, "id": str(rec["id"]), "reference": text})
        rows.append({"n": n, "id": str(rec["id"]),
                     "formatted": text, "style": style})

    state.write("evidence", "citations", {
        "style": style,
        "n": len(formatted),
        "formatted": formatted,
        "note": f"按 {style} 风格格式化的参考文献列表",
    })
    state.write("artifacts", "tables",
                pd.DataFrame(rows, columns=["n", "id", "formatted", "style"]))
    return state


# ------------------------------------------------------------------ manuscript_review
#: in-text citation patterns: APA ``(Braun & Clarke, 2006)`` / narrative
#: ``Braun (2006)`` / numbered ``[12]`` or ``[3,4]``.
_CITE_PAREN = re.compile(r"\(([^()]*?\b(?:19|20)\d{2}[a-z]?[^()]*?)\)")
_CITE_NARR = re.compile(r"\b([A-Z][A-Za-z’'-]+(?:\s+(?:et al\.|and|&)\s+[A-Z][A-Za-z’'-]+)?)"
                        r"\s+\((?:19|20)\d{2}[a-z]?\)")
_CITE_NUM = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")
_YEAR_IN = re.compile(r"(?:19|20)\d{2}")

#: hedge cues (appropriately cautious) vs. over-claim cues (causal / absolute).
_HEDGE = frozenset(
    """may might could suggests suggest indicates indicate appears seem seems
    possibly likely potentially associated correlated tends tend consistent
    可能 或许 提示 倾向 相关 关联""".split()
)
_OVERCLAIM = re.compile(
    r"\b(cause[sd]?|causes|causal|proves?|proven|demonstrates? that|"
    r"always|never|all|none|every|guarantee[sd]?|undeniabl[ey]|"
    r"clearly shows?|definitively|无疑|必然|证明|导致|总是|从不|所有|一定)\b",
    re.IGNORECASE,
)


def _verified_ids(verified: Any) -> set[str]:
    """Extract the set of reference ids from an ``evidence['verified_bib']`` payload.

    Tolerates several shapes: a list of records, a ``{id: record}`` map, a
    DataFrame, or a dict wrapping a ``records`` / ``verified`` list.
    """
    if verified is None:
        return set()
    if isinstance(verified, pd.DataFrame):
        col = next((c for c in ("id", "key", "ID") if c in verified.columns), None)
        return set(map(str, verified[col])) if col else set()
    if isinstance(verified, dict):
        for k in ("records", "verified", "entries", "bib"):
            if isinstance(verified.get(k), (list, tuple)):
                return _verified_ids(verified[k])
        # otherwise assume an {id: record} mapping
        return {str(k) for k in verified}
    ids: set[str] = set()
    for rec in verified:
        if isinstance(rec, dict):
            ids.add(str(rec.get("id") or rec.get("key") or rec.get("ID")))
        else:
            ids.add(str(rec))
    return {i for i in ids if i and i != "None"}


def _verified_surnames(verified: Any) -> set[str]:
    """Author surnames present in the verified bib — for narrative-cite matching."""
    surnames: set[str] = set()
    recs: list = []
    if isinstance(verified, pd.DataFrame):
        recs = verified.to_dict("records")
    elif isinstance(verified, dict):
        for k in ("records", "verified", "entries", "bib"):
            if isinstance(verified.get(k), (list, tuple)):
                recs = list(verified[k])
                break
        else:
            recs = [v for v in verified.values() if isinstance(v, dict)]
    elif isinstance(verified, (list, tuple)):
        recs = list(verified)
    for r in recs:
        if isinstance(r, dict):
            for a in _authors_of(r):
                surnames.add(_last_name(a).lower())
    return surnames


def _extract_citations(text: str) -> list[dict]:
    """Regex-extract in-text citations with their character span and kind.

    Narrative cites (``Finlay (2002)``) are matched first; a parenthetical whose
    span sits *inside* a narrative match is that same citation's trailing year, so
    it is dropped to avoid a spurious duplicate / orphan. The result is sorted by
    position so sentence attribution stays stable.
    """
    cites: list[dict] = []
    narr_spans: list[tuple[int, int]] = []
    for m in _CITE_NARR.finditer(text):
        y = _YEAR_IN.search(m.group(0))
        narr_spans.append((m.start(), m.end()))
        cites.append({"kind": "narrative", "raw": m.group(0),
                      "span": [m.start(), m.end()],
                      "year": y.group(0) if y else None,
                      "surnames": _surnames_in(m.group(1))})
    for m in _CITE_PAREN.finditer(text):
        if any(a <= m.start() and m.end() <= b for a, b in narr_spans):
            continue  # the year-paren of a narrative cite already captured above
        y = _YEAR_IN.search(m.group(1))
        cites.append({"kind": "parenthetical", "raw": m.group(0),
                      "span": [m.start(), m.end()],
                      "year": y.group(0) if y else None,
                      "surnames": _surnames_in(m.group(1))})
    for m in _CITE_NUM.finditer(text):
        nums = [n.strip() for n in re.split(r"[,\-]", m.group(1)) if n.strip()]
        cites.append({"kind": "numeric", "raw": m.group(0),
                      "span": [m.start(), m.end()], "refs": nums})
    cites.sort(key=lambda c: c["span"][0])
    return cites


def _surnames_in(fragment: str) -> list[str]:
    """Capitalized surname-like tokens inside a citation fragment."""
    return [w.lower() for w in re.findall(r"[A-Z][A-Za-z’'-]+", fragment)
            if w.lower() not in {"et", "al", "and"}]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence split (adequate for claim auditing; keeps CJK segments)."""
    parts = re.split(r"(?<=[.!?。!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _cite_supported(cite: dict, ver_ids: set[str], ver_surnames: set[str]) -> bool:
    """Is one extracted citation matched by something in the verified bib?"""
    if cite["kind"] == "numeric":
        return any(r in ver_ids for r in cite.get("refs", [])) if ver_ids else False
    sn = set(cite.get("surnames", []))
    return bool(sn & ver_surnames)


@register(
    name="manuscript_review",
    aliases=["稿件审校", "manuscript_review"],
    category="literature",
    tier="community",
    skill="manuscript-review",
    languages=["Python"],
    key_tools=["regex", "claim-evidence"],
    description="稿件审校:格式配平 + 逐条 claim-evidence 支撑审计 + 就绪裁决",
    requires={"sources": ["datasets"], "evidence": ["verified_bib"]},
    produces={"evidence": ["claim_evidence"],
              "diagnostics": ["coverage"],
              "artifacts": ["tables"]},
    prerequisites={"functions": ["verify_citations"]},
    auto_fix="auto",
)
def manuscript_review(state: StudyState, **kwargs: Any) -> StudyState:
    """Audit a manuscript against its verified references.

    Regex-extracts the in-text citations from ``manuscript=`` (or
    ``sources['datasets']`` when it holds the text), cross-balances them against
    ``evidence['verified_bib']`` (``orphan`` = cited but unverified, ``uncited`` =
    verified but never cited), runs a hedge-mismatch first pass over causal /
    absolute language, and labels every claim ``supported`` / ``unsupported`` /
    ``over-claim``.

    Writes ``evidence['claim_evidence']`` (per-claim ledger), ``diagnostics[
    'coverage']`` (``supported_ratio`` + a ``BLOCKER/MAJOR/MINOR/READY`` verdict),
    and ``artifacts['tables']`` (the issue list). The ``requires`` — including
    ``evidence['verified_bib']`` — are enforced by the registry wrapper *before*
    this body runs, so a missing verified bib raises ``RegistryError`` upstream.
    """
    verified = state.evidence.get("verified_bib")
    ver_ids = _verified_ids(verified)
    ver_surnames = _verified_surnames(verified)

    # resolve manuscript text: kwarg first, then sources['datasets'] if textual.
    text = kwargs.get("manuscript") or kwargs.get("text")
    if text is None:
        ds = state.sources.get("datasets")
        if isinstance(ds, str):
            text = ds
        elif isinstance(ds, dict):
            text = ds.get("manuscript") or ds.get("text")
    text = text if isinstance(text, str) else ""

    cites = _extract_citations(text)

    # -- format balance: orphan (cited∉verified) vs uncited (verified∉cited) --
    cited_ids: set[str] = set()
    cited_surnames: set[str] = set()
    for c in cites:
        cited_ids.update(c.get("refs", []))
        cited_surnames.update(c.get("surnames", []))
    orphans = sorted(
        c["raw"] for c in cites if not _cite_supported(c, ver_ids, ver_surnames)
    )
    uncited_ids = sorted(ver_ids - cited_ids) if ver_ids else []
    uncited_authors = sorted(ver_surnames - cited_surnames) if ver_surnames else []

    # -- per-claim claim-evidence audit --------------------------------------
    sentences = _split_sentences(text)
    claim_ledger: list[dict] = []
    issues: list[dict] = []
    for idx, sent in enumerate(sentences, start=1):
        sent_cites = [c for c in cites
                      if idx == _sentence_of(c["span"][0], sentences, text)]
        has_cite = bool(sent_cites)
        has_support = any(_cite_supported(c, ver_ids, ver_surnames) for c in sent_cites)
        overclaim = bool(_OVERCLAIM.search(sent))
        hedged = any(w in _HEDGE for w in re.findall(r"[A-Za-z一-鿿]+", sent.lower()))
        empirical = has_cite or _looks_empirical(sent)

        if not empirical:
            continue  # non-empirical prose (aims, transitions) — not a claim

        if overclaim and not has_support:
            status = "over-claim"
        elif has_support:
            status = "supported"
        else:
            status = "unsupported"

        # over-claim also flags a causal/absolute claim that *is* cited but unhedged
        if status == "supported" and overclaim and not hedged:
            status = "over-claim"

        record = {
            "claim_id": f"C{idx}",
            "sentence": sent[:300],
            "status": status,
            "cited": has_cite,
            "citations": [c["raw"] for c in sent_cites],
            "hedged": hedged,
            "causal_language": overclaim,
        }
        claim_ledger.append(record)
        if status != "supported":
            issues.append({
                "claim_id": f"C{idx}",
                "type": status,
                "severity": "MAJOR" if status == "over-claim" else "MINOR",
                "excerpt": sent[:200],
                "hint": ("因果/绝对措辞缺乏支撑或对冲" if status == "over-claim"
                         else "论断未见已核验引文支撑"),
            })

    for raw in orphans:
        issues.append({"claim_id": None, "type": "orphan_citation",
                       "severity": "MAJOR", "excerpt": raw,
                       "hint": "引注未在 verified_bib 中找到匹配"})
    for a in uncited_authors:
        issues.append({"claim_id": None, "type": "uncited_reference",
                       "severity": "MINOR", "excerpt": a,
                       "hint": "已核验参考文献从未被正文引用"})

    # -- coverage + readiness verdict ----------------------------------------
    n_claims = len(claim_ledger)
    n_supported = sum(1 for c in claim_ledger if c["status"] == "supported")
    n_overclaim = sum(1 for c in claim_ledger if c["status"] == "over-claim")
    supported_ratio = round(n_supported / n_claims, 4) if n_claims else 1.0

    if orphans or n_overclaim > 0:
        verdict = "BLOCKER" if (len(orphans) > 2 or n_overclaim > 2) else "MAJOR"
    elif supported_ratio < 0.6:
        verdict = "MAJOR"
    elif supported_ratio < 0.85 or uncited_authors:
        verdict = "MINOR"
    else:
        verdict = "READY"

    state.write("evidence", "claim_evidence", {
        "n_claims": n_claims,
        "claims": claim_ledger,
        "balance": {
            "n_citations": len(cites),
            "orphans": orphans,
            "uncited_reference_ids": uncited_ids,
            "uncited_authors": uncited_authors,
        },
        "note": "逐条 claim→verified_bib 支撑审计 + 引注配平",
    })
    state.write("diagnostics", "coverage", {
        "n_claims": n_claims,
        "n_supported": n_supported,
        "n_unsupported": n_claims - n_supported - n_overclaim,
        "n_overclaim": n_overclaim,
        "supported_ratio": supported_ratio,
        "n_orphan_citations": len(orphans),
        "n_uncited_references": len(uncited_authors),
        "verdict": verdict,
        "note": "就绪裁决:BLOCKER/MAJOR/MINOR/READY",
    })
    state.write("artifacts", "tables", pd.DataFrame(
        issues, columns=["claim_id", "type", "severity", "excerpt", "hint"],
    ))
    return state


def _sentence_of(char_idx: int, sentences: list[str], text: str) -> int:
    """1-based index of the sentence containing character ``char_idx``."""
    pos = 0
    for i, sent in enumerate(sentences, start=1):
        found = text.find(sent, pos)
        if found == -1:
            found = pos
        end = found + len(sent)
        if char_idx < end:
            return i
        pos = end
    return len(sentences)


def _looks_empirical(sent: str) -> bool:
    """Heuristic: does a sentence make an empirical / findings claim?

    Numbers, percentages, statistical tokens, or reporting verbs mark a sentence
    as a factual claim worth auditing (vs. aims / definitions / transitions).
    """
    if re.search(r"\d", sent):
        return True
    empirical_cues = (
        "found", "show", "shows", "showed", "demonstrate", "report", "reported",
        "reveal", "result", "results", "significant", "effect", "increase",
        "decrease", "associated", "correlated", "evidence", "data",
        "发现", "表明", "显示", "结果", "显著", "效应", "证据", "数据", "关联",
    )
    low = sent.lower()
    return any(cue in low for cue in empirical_cues)
