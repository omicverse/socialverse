"""``sv.lit._verify`` — the ``citation-verify`` skill registration.

A single registered implementation, :func:`verify_citations`, that vets a
``.bib`` reference list one record at a time and classifies each into one of four
integrity statuses — ``verified`` / ``suspicious`` / ``not_found`` / ``chimeric``
— the failure modes that matter when an LLM-drafted manuscript cites the
literature. The design goal is **offline-first, triangulation-second**:

* *Offline* (the default). Every record is judged by deterministic rules that
  need no network: a syntactically valid DOI plus complete metadata →
  ``verified``; a missing / malformed DOI → ``suspicious``; a title that a
  *known-truth* table maps to a **different** author set → ``chimeric`` (a real
  paper's title/DOI welded onto the wrong authors — the classic hallucinated
  citation); nothing to check it against and no DOI → ``not_found``.
* *Triangulation*. The ground truth can be supplied three ways, in priority
  order: a user ``resolver`` callable ``record -> truth-dict``; a ``known``
  reference table (``DataFrame`` or list of dicts) matched by DOI then by fuzzy
  title (``difflib``); or, **only** when ``online=True`` *and* ``urllib`` is
  importable, a best-effort Crossref/OpenAlex lookup. The three sources are
  consulted in that order and the first hit wins — three libraries triangulating
  one claim.

Everything is deterministic and the module never touches the network unless the
caller explicitly opts in with ``online=True``; even then a lookup failure
degrades silently to the offline verdict rather than raising. No heavy optional
dependency is imported at module load — ``pandas`` is the only hard import, and
``urllib`` / ``requests`` are reached lazily inside the online branch.
"""
from __future__ import annotations

import importlib
import re
from difflib import SequenceMatcher
from typing import Any, Callable

import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["verify_citations"]


def _try_import(name: str) -> Any | None:
    """Import ``name`` lazily, returning ``None`` if unavailable (never raises)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# --------------------------------------------------------------------- parsing
# A DOI is ``10.<registrant>/<suffix>`` — the canonical Crossref shape.
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
# non-alphanumeric run, for title/author normalization
_NONWORD = re.compile(r"[^0-9a-z一-鿿]+")


def _clean_doi(doi: Any) -> str | None:
    """Normalize a DOI string, stripping a ``https://doi.org/`` / ``doi:`` prefix.

    Returns the bare DOI (lowercased) or ``None`` if the value is missing.
    """
    if doi is None:
        return None
    s = str(doi).strip()
    if not s:
        return None
    for pref in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                 "doi.org/", "doi:", "DOI:"):
        if s.lower().startswith(pref.lower()):
            s = s[len(pref):]
            break
    return s.strip().lower()


def _valid_doi(doi: str | None) -> bool:
    """True if ``doi`` matches the canonical DOI syntax."""
    return bool(doi) and bool(_DOI_RE.match(doi))


def _norm_title(title: Any) -> str:
    """Lowercased, punctuation-stripped title for fuzzy comparison."""
    return _NONWORD.sub(" ", str(title or "").lower()).strip()


def _authors_list(rec: Any) -> list[str]:
    """Coerce a record's ``authors`` / ``author`` field into a list of strings."""
    if isinstance(rec, dict):
        val = rec.get("authors", rec.get("author"))
    else:
        val = rec
    if val is None:
        return []
    if isinstance(val, str):
        # split a "A and B and C" / "A; B" / "A, B & C" author string
        parts = re.split(r"\s+and\s+|;|&", val)
        return [p.strip() for p in parts if p.strip()]
    return [str(a).strip() for a in val if str(a).strip()]


def _author_key(name: str) -> str:
    """Comparable surname key for one author name (``Braun, V.`` -> ``braun``)."""
    n = str(name).strip().lower()
    if "," in n:
        surname = n.split(",", 1)[0]
    else:
        surname = n.split()[-1] if n.split() else n
    return _NONWORD.sub("", surname)


def _author_set(rec_or_authors: Any) -> frozenset[str]:
    """Set of comparable surname keys for a record / author list."""
    return frozenset(_author_key(a) for a in _authors_list(rec_or_authors) if _author_key(a))


def _ratio(a: str, b: str) -> float:
    """Deterministic fuzzy string similarity in ``[0, 1]`` (difflib)."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _complete(rec: dict[str, Any]) -> list[str]:
    """Names of the required bib fields (title / authors / year) that are absent."""
    missing: list[str] = []
    if not _norm_title(rec.get("title")):
        missing.append("title")
    if not _authors_list(rec):
        missing.append("authors")
    if rec.get("year") in (None, "") or str(rec.get("year")).strip() == "":
        missing.append("year")
    return missing


# ---------------------------------------------------------------- known table
def _known_records(known: Any) -> list[dict[str, Any]]:
    """Coerce a ``known`` truth table (DataFrame / list / dict) into record dicts."""
    if known is None:
        return []
    if isinstance(known, pd.DataFrame):
        return known.to_dict("records")
    if isinstance(known, dict):
        # {doi_or_id: {title, authors, ...}} OR a single record dict
        if all(isinstance(v, dict) for v in known.values()) and known:
            out = []
            for k, v in known.items():
                r = dict(v)
                r.setdefault("doi", k)
                out.append(r)
            return out
        return [known]
    return [dict(r) for r in known if isinstance(r, dict)]


def _match_known(
    rec: dict[str, Any], known: list[dict[str, Any]], *, title_cut: float
) -> dict[str, Any] | None:
    """Find the ground-truth record for ``rec`` in ``known``.

    Matches by DOI first (exact, normalized); failing that, by the best fuzzy
    title match at or above ``title_cut``. Returns the truth record or ``None``.
    """
    doi = _clean_doi(rec.get("doi"))
    if doi:
        for k in known:
            if _clean_doi(k.get("doi")) == doi:
                return k
    qt = _norm_title(rec.get("title"))
    if not qt:
        return None
    best: dict[str, Any] | None = None
    best_r = 0.0
    for k in known:
        r = _ratio(qt, _norm_title(k.get("title")))
        if r > best_r:
            best_r, best = r, k
    return best if best_r >= title_cut else None


def _online_lookup(rec: dict[str, Any], timeout: float) -> dict[str, Any] | None:
    """Best-effort Crossref/OpenAlex metadata lookup (only when ``online=True``).

    Reached lazily and wrapped so any network / import failure degrades to
    ``None`` (the caller then falls back to the offline verdict). Never raises.
    """
    doi = _clean_doi(rec.get("doi"))
    if not doi:
        return None
    ureq = _try_import("urllib.request")
    ujson = _try_import("json")
    if ureq is None or ujson is None:
        return None
    for url in (
        f"https://api.crossref.org/works/{doi}",
        f"https://api.openalex.org/works/https://doi.org/{doi}",
    ):
        try:
            req = ureq.Request(url, headers={"User-Agent": "socialverse-citation-verify/1.0"})
            with ureq.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - opt-in only
                payload = ujson.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            continue
        msg = payload.get("message", payload) if isinstance(payload, dict) else {}
        title = msg.get("title")
        if isinstance(title, list):
            title = title[0] if title else None
        authors = []
        for a in msg.get("author", []) or []:
            if isinstance(a, dict):
                fam = a.get("family") or a.get("last") or a.get("name", "")
                authors.append(str(fam))
        if title or authors:
            return {"title": title, "authors": authors, "doi": doi, "source": "online"}
    return None


# ------------------------------------------------------------------- classify
def _classify(
    rec: dict[str, Any],
    truth: dict[str, Any] | None,
    *,
    author_cut: float,
    title_cut: float,
) -> tuple[str, str]:
    """Return ``(status, reason)`` for one record given its (optional) ground truth.

    The four statuses:

    * ``chimeric``   — a ground truth exists whose title matches but whose author
      set is disjoint/low-overlap: a real paper's identity welded onto the wrong
      authors (the signature hallucinated citation).
    * ``verified``   — a syntactically valid DOI **and** complete metadata, and
      no contradicting ground truth.
    * ``suspicious`` — complete-ish but the DOI is missing or malformed (cannot be
      resolved), so the citation cannot be trusted as-is.
    * ``not_found``  — nothing to verify against: no valid DOI and no matching
      ground-truth record.
    """
    doi = _clean_doi(rec.get("doi"))
    doi_ok = _valid_doi(doi)
    missing = _complete(rec)

    # --- chimera check: title agrees with truth but authorship does not --------
    if truth is not None:
        t_ratio = _ratio(_norm_title(rec.get("title")), _norm_title(truth.get("title")))
        a_rec = _author_set(rec)
        a_truth = _author_set(truth)
        if a_rec and a_truth:
            overlap = len(a_rec & a_truth) / len(a_rec | a_truth)
        else:
            overlap = 1.0  # cannot contradict without both author sets
        if t_ratio >= title_cut and overlap < author_cut:
            return (
                "chimeric",
                f"标题与已知文献匹配(sim={t_ratio:.2f})但作者不符"
                f"(overlap={overlap:.2f}<{author_cut}):疑似张冠李戴的杜撰引文",
            )
        # title + author both agree with a real record -> trust it
        if t_ratio >= title_cut and overlap >= author_cut:
            if doi_ok or _clean_doi(truth.get("doi")):
                return ("verified", f"三角核验命中已知文献(标题 sim={t_ratio:.2f},作者一致)")
            return ("suspicious", f"作者/标题与已知文献一致但无有效 DOI(sim={t_ratio:.2f})")

    # --- offline rule ladder (no contradicting truth) --------------------------
    if doi_ok and not missing:
        return ("verified", "DOI 语法有效且题录字段完整")
    if doi_ok and missing:
        return ("suspicious", f"DOI 有效但字段缺失:{', '.join(missing)}")
    if not doi_ok and not missing:
        return ("suspicious", "题录完整但缺少有效 DOI,无法解析核验")
    return ("not_found", f"无有效 DOI 且字段缺失({', '.join(missing) or '空记录'}),查无实据")


# --------------------------------------------------------------------- register
@register(
    name="verify_citations",
    aliases=["引文核验", "citation_verify"],
    category="literature",
    tier="community",
    skill="citation-verify",
    languages=["Python"],
    key_tools=["urllib", "difflib", "Crossref", "OpenAlex"],
    description="逐条核验参考文献(离线友好):三库三角+模糊标题→verified/suspicious/not_found/chimeric",
    requires={"sources": ["bib"]},
    produces={"evidence": ["verified_bib"]},
    auto_fix="escalate",
)
def verify_citations(state: StudyState, **kwargs: Any) -> StudyState:
    """Verify a ``.bib`` reference list record-by-record (offline-first).

    Reads the reference list from ``state.sources['bib']`` (or a ``bib`` kwarg),
    resolves each record against — in priority order — a user ``resolver``
    callable, a ``known`` ground-truth table (DOI then fuzzy title), and (only if
    ``online=True``) a Crossref/OpenAlex lookup, then classifies it into one of
    ``verified`` / ``suspicious`` / ``not_found`` / ``chimeric``. Writes the
    annotated records plus a status tally to ``evidence['verified_bib']``.

    Parameters (via ``kwargs``)
    ---------------------------
    bib : list[dict] | DataFrame, optional
        Reference records overriding ``state.sources['bib']``. Each is a dict with
        ``title`` / ``authors`` / ``year`` / ``doi`` (and an optional ``id``).
    resolver : callable, optional
        ``record -> truth-dict | None`` — the highest-priority ground-truth
        source. Any exception it raises is swallowed (treated as no match).
    known : DataFrame | list[dict] | dict, optional
        A local reference table of trusted records used to triangulate by DOI and
        fuzzy title. The offline analog of a bibliographic database.
    online : bool, default False
        When ``True`` *and* ``urllib`` is importable, consult Crossref/OpenAlex as
        a last resort. Never reached otherwise; failures degrade silently.
    title_cut : float, default 0.85
        Fuzzy-title similarity threshold for a ground-truth match / chimera check.
    author_cut : float, default 0.34
        Author-set Jaccard overlap below which a title-matched record is flagged
        ``chimeric``.
    timeout : float, default 5.0
        Per-request network timeout (seconds) for the opt-in online branch.
    """
    bib_in = kwargs.get("bib", state.sources.get("bib"))
    if isinstance(bib_in, pd.DataFrame):
        records = bib_in.to_dict("records")
    elif isinstance(bib_in, dict):
        records = [bib_in]
    elif bib_in is None:
        records = []
    else:
        records = [dict(r) if isinstance(r, dict) else {"title": str(r)} for r in bib_in]

    resolver: Callable[[dict[str, Any]], Any] | None = kwargs.get("resolver")
    known = _known_records(kwargs.get("known"))
    online = bool(kwargs.get("online", False))
    title_cut = float(kwargs.get("title_cut", 0.85))
    author_cut = float(kwargs.get("author_cut", 0.34))
    timeout = float(kwargs.get("timeout", 5.0))

    out_records: list[dict[str, Any]] = []
    tally: dict[str, int] = {"verified": 0, "suspicious": 0, "not_found": 0, "chimeric": 0}

    for i, raw in enumerate(records):
        rec = dict(raw) if isinstance(raw, dict) else {"title": str(raw)}
        rec_id = str(rec.get("id", rec.get("ID", f"ref{i:03d}")))

        # --- triangulate ground truth: resolver > known table > online ---------
        truth: dict[str, Any] | None = None
        source = "offline"
        if callable(resolver):
            try:
                cand = resolver(rec)
            except Exception:
                cand = None
            if isinstance(cand, dict):
                truth, source = cand, "resolver"
        if truth is None and known:
            cand = _match_known(rec, known, title_cut=title_cut)
            if cand is not None:
                truth, source = cand, "known"
        if truth is None and online:
            cand = _online_lookup(rec, timeout=timeout)
            if cand is not None:
                truth, source = cand, "online"

        status, reason = _classify(
            rec, truth, author_cut=author_cut, title_cut=title_cut
        )
        tally[status] = tally.get(status, 0) + 1

        doi = _clean_doi(rec.get("doi"))
        out_records.append(
            {
                "id": rec_id,
                "title": rec.get("title"),
                "authors": _authors_list(rec),
                "year": rec.get("year"),
                "doi": doi,
                "doi_valid": _valid_doi(doi),
                "missing_fields": _complete(rec),
                "status": status,
                "reason": reason,
                "resolved_via": source,
                "matched_title": (truth or {}).get("title") if truth else None,
            }
        )

    n = len(out_records)
    n_ok = tally["verified"]
    summary = {
        "n_records": n,
        "n_verified": tally["verified"],
        "n_suspicious": tally["suspicious"],
        "n_not_found": tally["not_found"],
        "n_chimeric": tally["chimeric"],
        "pass_rate": round(n_ok / n, 4) if n else 0.0,
        "flagged_ids": [r["id"] for r in out_records if r["status"] != "verified"],
        "online_used": bool(online),
        "ground_truth": "resolver"
        if callable(resolver)
        else ("known" if known else ("online" if online else "rules-only")),
        "note": (
            "全部离线规则核验" if not (known or callable(resolver) or online)
            else "结合本地/在线真值三角核验"
        ),
    }

    verified_bib = {"records": out_records, "summary": summary, "tally": dict(tally)}
    state.write("evidence", "verified_bib", verified_bib)
    return state
