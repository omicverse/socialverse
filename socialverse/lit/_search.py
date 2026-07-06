"""``sv.lit._search`` — literature-retrieval registrations for the *search* phase.

Registered implementations of two literature skills that seed and refine the
``sources['bib']`` slot — the raw pool from which downstream verification
(``sv.lit._verify``) and mapping (``sv.lit._map``) work.

* :func:`search_free` — ``literature-search-free``: a **default-offline** first
  pass over the free, keyless bibliographic APIs (NCBI E-utilities, Crossref).
  It never touches the network unless the caller *explicitly* passes
  ``online=True`` **and** ``requests`` is importable; otherwise it consumes a
  caller-supplied record list (``records=[{title, authors, year, doi}, ...]``)
  or a ``resolver`` callable, normalizes every record to a canonical schema, and
  writes both the ``sources['bib']`` pool and a screening-ready
  ``evidence['citations']`` candidate ledger.
* :func:`zotero_bridge` — ``zotero-bridge``: a personal-library fan-out that
  scores the existing ``sources['bib']`` pool against a ``query`` under **five
  strategies** (title / author / tag / full-text / annotation), fuses the
  per-strategy scores, de-duplicates by DOI-then-title, and writes back the
  ranked, resolved subset. With no Zotero MCP available it degrades to pure
  in-process text matching; ``bibtexparser`` is imported lazily and only when a
  raw ``.bib`` string must be parsed into records.

The scoring / fusion / dedup / ranking chain is *real* computation over plain
Python + difflib (no heavy dependency at import time, and the module never
reaches the network unless asked). Everything is deterministic — ranking ties
break on a stable key and any randomness is avoided entirely.
"""
from __future__ import annotations

import importlib
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

from .._registry import register
from .._state import StudyState

__all__ = ["search_free", "zotero_bridge"]


def _try_import(name: str) -> Any | None:
    """Import ``name`` lazily, returning ``None`` if unavailable (never raises)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# --------------------------------------------------------------------- helpers
_WORD = re.compile(r"[一-鿿]|[A-Za-zÀ-ɏ0-9][A-Za-zÀ-ɏ0-9'-]*")


def _tokens(text: str) -> list[str]:
    """Lowercased word/CJK-char tokens of ``text`` (empty list for non-strings)."""
    if not text:
        return []
    return _WORD.findall(str(text).lower())


def _norm_authors(authors: Any) -> list[str]:
    """Coerce an ``authors`` value into a clean list of author-name strings."""
    if authors is None:
        return []
    if isinstance(authors, str):
        parts = re.split(r"\s*(?:;| and |,\s*(?=[A-Z一-鿿]))\s*", authors)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(authors, Iterable):
        out: list[str] = []
        for a in authors:
            if isinstance(a, dict):
                name = a.get("name") or " ".join(
                    str(a.get(k, "")) for k in ("given", "family")
                ).strip()
                if name:
                    out.append(str(name).strip())
            elif a is not None and str(a).strip():
                out.append(str(a).strip())
        return out
    return [str(authors).strip()]


def _norm_doi(doi: Any) -> str:
    """Normalize a DOI to a bare lowercase ``10.xxxx/...`` form (or ``''``)."""
    if not doi:
        return ""
    s = str(doi).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()


def _norm_record(rec: Any, idx: int) -> dict[str, Any]:
    """Coerce one raw record into the canonical bib schema used across ``sv.lit``.

    Guarantees the keys ``id / title / authors / year / doi / venue / abstract /
    tags / source`` on the returned dict. Accepts a dict (any casing of the
    common field aliases) or a bare title string.
    """
    if isinstance(rec, dict):
        g = {str(k).lower(): v for k, v in rec.items()}
        title = g.get("title") or g.get("name") or ""
        year = g.get("year") or g.get("date") or g.get("published")
        try:
            year_val: int | None = int(str(year)[:4]) if year not in (None, "") else None
        except (ValueError, TypeError):
            year_val = None
        doi = _norm_doi(g.get("doi"))
        rid = str(g.get("id") or g.get("key") or (doi or f"rec{idx:04d}"))
        return {
            "id": rid,
            "title": str(title).strip(),
            "authors": _norm_authors(g.get("authors") or g.get("author")),
            "year": year_val,
            "doi": doi,
            "venue": str(g.get("venue") or g.get("journal") or g.get("container-title") or "").strip(),
            "abstract": str(g.get("abstract") or g.get("summary") or "").strip(),
            "tags": [str(t).strip() for t in (g.get("tags") or g.get("keywords") or []) if str(t).strip()],
            "source": str(g.get("source") or "input"),
        }
    # bare scalar → treat as a title
    return {
        "id": f"rec{idx:04d}",
        "title": str(rec).strip() if rec is not None else "",
        "authors": [],
        "year": None,
        "doi": "",
        "venue": "",
        "abstract": "",
        "tags": [],
        "source": "input",
    }


def _normalize_records(records: Any) -> list[dict[str, Any]]:
    """Coerce a raw ``records`` value into a list of canonical bib dicts."""
    if records is None:
        return []
    if isinstance(records, dict):
        records = [records]
    if isinstance(records, str):
        records = [records]
    out: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        out.append(_norm_record(r, i))
    return out


def _dedup_key(rec: dict[str, Any]) -> str:
    """Stable dedup key: DOI when present, else a normalized-title fingerprint."""
    doi = rec.get("doi")
    if doi:
        return f"doi:{doi}"
    title_toks = _tokens(rec.get("title", ""))
    return "title:" + " ".join(title_toks) if title_toks else f"id:{rec.get('id')}"


def _dedup(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate by DOI-then-title, keeping the first (richest-order) record."""
    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = _dedup_key(rec)
        if key not in seen:
            seen[key] = rec
        else:
            # merge: prefer non-empty fields from the incoming record
            kept = seen[key]
            for f in ("abstract", "venue", "doi", "year"):
                if not kept.get(f) and rec.get(f):
                    kept[f] = rec[f]
            for t in rec.get("tags", []):
                if t not in kept["tags"]:
                    kept["tags"].append(t)
    return list(seen.values())


def _parse_bib(bib_text: str) -> list[dict[str, Any]]:
    """Parse a BibTeX string into raw records (lazy ``bibtexparser``, regex fallback)."""
    bp = _try_import("bibtexparser")
    if bp is not None:
        try:
            db = bp.loads(bib_text)
            return list(getattr(db, "entries", []) or [])
        except Exception:
            pass
    # minimal regex fallback so a .bib still yields records without the dependency.
    # Split into top-level ``@type{...}`` entries (single- or multi-line), then
    # pull each ``field = {value}`` / ``field = "value"`` pair out of the body.
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]*)\s*,(.*?)\}\s*(?=@|\Z)", bib_text, re.DOTALL):
        key, body = m.group(1).strip(), m.group(2)
        fields: dict[str, Any] = {"key": key} if key else {}
        for fm in re.finditer(
            r"(\w+)\s*=\s*(?:\{(.*?)\}|\"(.*?)\")\s*,?", body, re.DOTALL
        ):
            val = fm.group(2) if fm.group(2) is not None else fm.group(3)
            fields[fm.group(1).strip().lower()] = re.sub(r"\s+", " ", val or "").strip()
        if fields:
            out.append(fields)
    return out


# --------------------------------------------------------------------- search
@register(
    name="search_free",
    aliases=["文献检索", "literature_search"],
    category="literature",
    tier="community",
    skill="literature-search-free",
    languages=["Python"],
    key_tools=["requests", "NCBI E-utilities", "Crossref"],
    description="免费无鉴权 API 检索初筛(离线友好:可传入 records/resolver,默认不联网)",
    requires={},
    produces={"sources": ["bib"], "evidence": ["citations"]},
    auto_fix="none",
)
def search_free(state: StudyState, **kwargs: Any) -> StudyState:
    """First-pass literature retrieval over free, keyless APIs — **offline by default**.

    Populates the ``sources['bib']`` pool and a screening-ready
    ``evidence['citations']`` candidate ledger. The function is deliberately
    network-silent: it only issues HTTP requests when the caller *explicitly*
    passes ``online=True`` **and** ``requests`` is importable. In every other
    mode it consumes caller-supplied data, so it is fully reproducible and safe
    to run in a sandbox.

    Parameters (via ``kwargs``)
    ---------------------------
    records : list[dict], optional
        Pre-fetched records ``[{title, authors, year, doi, ...}]``. Used directly
        (normalized). This is the default offline path.
    resolver : callable, optional
        ``resolver(query) -> records`` — a caller-supplied fetcher (e.g. a cached
        API client or an OmicOS tool). Preferred over network access.
    query : str, optional
        The search query — passed to ``resolver`` and recorded on the ledger.
    online : bool, default False
        When True *and* ``requests`` is available, query Crossref (keyless) live.
        Any failure degrades silently to whatever ``records`` were supplied.
    limit : int, default 25
        Cap on the number of live/resolved records fetched.
    """
    query = str(kwargs.get("query", "")).strip()
    limit = int(kwargs.get("limit", 25))
    resolver: Callable[[str], Any] | None = kwargs.get("resolver")
    online = bool(kwargs.get("online", False))

    raw: list[Any] = []
    mode = "records"

    # 1) explicit pre-fetched records (the default, fully offline)
    if kwargs.get("records") is not None:
        raw.extend(kwargs["records"] if isinstance(kwargs["records"], list) else [kwargs["records"]])

    # 2) caller-supplied resolver callable (offline-friendly, no network here)
    if resolver is not None and callable(resolver):
        try:
            got = resolver(query)
            if got:
                raw.extend(got if isinstance(got, (list, tuple)) else [got])
                mode = "resolver"
        except Exception:
            pass

    # 3) live network — ONLY when explicitly requested and `requests` present
    if online:
        requests = _try_import("requests")
        if requests is not None and query:
            try:
                resp = requests.get(
                    "https://api.crossref.org/works",
                    params={"query": query, "rows": limit},
                    timeout=15,
                )
                items = resp.json().get("message", {}).get("items", [])
                for it in items:
                    raw.append(
                        {
                            "title": (it.get("title") or [""])[0],
                            "authors": it.get("author", []),
                            "year": (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
                            "doi": it.get("DOI", ""),
                            "container-title": (it.get("container-title") or [""])[0],
                            "abstract": it.get("abstract", ""),
                            "source": "crossref",
                        }
                    )
                mode = "crossref"
            except Exception:
                pass  # degrade silently to whatever we already have

    records = _dedup(_normalize_records(raw))[: max(limit, len(raw)) if limit else None]

    # candidate ledger — screening-ready, one row per candidate
    citations = [
        {
            "id": r["id"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
            "doi": r["doi"],
            "venue": r["venue"],
            "source": r["source"],
            "screen": "pending",  # include | exclude | pending
        }
        for r in records
    ]

    state.write("sources", "bib", records)
    state.write(
        "evidence",
        "citations",
        {
            "query": query,
            "mode": mode,
            "online": online,
            "n_candidates": len(citations),
            "candidates": citations,
        },
    )
    return state


# --------------------------------------------------------------------- zotero
_STRATEGIES = ("title", "author", "tag", "fulltext", "annotation")
_STRATEGY_WEIGHTS = {
    "title": 1.0,
    "author": 0.7,
    "tag": 0.6,
    "fulltext": 0.5,
    "annotation": 0.4,
}


def _score_field(query_toks: set[str], field_text: str) -> float:
    """Fraction of query tokens present in ``field_text`` (token overlap in [0,1])."""
    if not query_toks:
        return 0.0
    ftoks = set(_tokens(field_text))
    if not ftoks:
        return 0.0
    return len(query_toks & ftoks) / len(query_toks)


def _fuzzy_title(query: str, title: str) -> float:
    """difflib similarity ratio between query and title (whole-string, in [0,1])."""
    if not query or not title:
        return 0.0
    return SequenceMatcher(None, query.lower(), str(title).lower()).ratio()


@register(
    name="zotero_bridge",
    aliases=["Zotero桥", "zotero"],
    category="literature",
    tier="plus",
    skill="zotero-bridge",
    languages=["Python"],
    key_tools=["bibtexparser", "Zotero MCP"],
    description="个人文献库五策略扇出检索+打分排序去重(无 MCP 降级纯文本解析 .bib)",
    requires={"sources": ["bib"]},
    produces={"sources": ["bib"]},
    auto_fix="auto",
)
def zotero_bridge(state: StudyState, **kwargs: Any) -> StudyState:
    """Personal-library five-strategy fan-out search over ``sources['bib']``.

    Scores every record in the ``sources['bib']`` pool against a ``query`` under
    five strategies — **title / author / tag / full-text / annotation** — fuses
    the weighted per-strategy scores into one relevance score, de-duplicates by
    DOI-then-title, ranks descending (stable ties on title then id), and writes
    the resolved subset back to ``sources['bib']``. With no Zotero MCP present it
    is pure in-process text matching; a raw ``.bib`` string is parsed lazily via
    ``bibtexparser`` (regex fallback) and folded into the pool first.

    Parameters (via ``kwargs``)
    ---------------------------
    query : str, optional
        The search query fanned out across the five strategies. When empty every
        record scores 0 and the pool is returned de-duplicated but unranked.
    bib : list[dict], optional
        Records overriding ``state.sources['bib']``.
    bibtex : str, optional
        A raw ``.bib`` string to parse and merge into the pool before scoring.
    annotations : dict, optional
        ``{record_id: annotation_text}`` — the annotation strategy's corpus.
    top_k : int, default 25
        Cap on the number of ranked records written back.
    min_score : float, default 0.0
        Records scoring at or below this are dropped from the resolved subset
        (kept only if nothing else clears the bar).
    """
    query = str(kwargs.get("query", "")).strip()
    top_k = int(kwargs.get("top_k", 25))
    min_score = float(kwargs.get("min_score", 0.0))
    annotations = kwargs.get("annotations") or {}

    pool = _normalize_records(kwargs.get("bib", state.sources.get("bib")) or [])
    if isinstance(kwargs.get("bibtex"), str) and kwargs["bibtex"].strip():
        pool.extend(_normalize_records(_parse_bib(kwargs["bibtex"])))
    pool = _dedup(pool)

    query_toks = set(_tokens(query))

    ranked: list[dict[str, Any]] = []
    for rec in pool:
        ann_text = ""
        if isinstance(annotations, dict):
            ann_text = str(annotations.get(rec["id"]) or annotations.get(rec["doi"]) or "")

        per_strategy = {
            "title": max(
                _score_field(query_toks, rec["title"]),
                _fuzzy_title(query, rec["title"]),
            ),
            "author": _score_field(query_toks, " ".join(rec["authors"])),
            "tag": _score_field(query_toks, " ".join(rec["tags"])),
            "fulltext": _score_field(
                query_toks, " ".join([rec["title"], rec["abstract"], rec["venue"]])
            ),
            "annotation": _score_field(query_toks, ann_text),
        }
        fused = sum(_STRATEGY_WEIGHTS[s] * per_strategy[s] for s in _STRATEGIES)
        total_w = sum(_STRATEGY_WEIGHTS.values())
        score = round(fused / total_w, 6) if total_w else 0.0

        out = dict(rec)
        out["_score"] = score
        out["_strategy_scores"] = {s: round(per_strategy[s], 6) for s in _STRATEGIES}
        out["_matched_strategies"] = sorted(
            s for s in _STRATEGIES if per_strategy[s] > 0.0
        )
        ranked.append(out)

    # rank: score desc, then stable tie-break (title, id)
    ranked.sort(key=lambda r: (-r["_score"], r["title"].lower(), r["id"]))

    resolved = [r for r in ranked if r["_score"] > min_score]
    if not resolved and ranked:  # nothing cleared the bar → keep the best anyway
        resolved = ranked
    resolved = resolved[: top_k if top_k > 0 else None]

    state.write("sources", "bib", resolved)
    return state
