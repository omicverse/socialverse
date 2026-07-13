"""``sv.tl._qual`` — qualitative-analysis registrations for the *analyze* phase.

Registered implementations of three qualitative skills that operate on the
addressable coding **units** produced by ``sv.pp.build_corpus`` (each unit a
``{unit_id, doc_id, start, end, text}`` dict). Together they form a small
reflexive-thematic-analysis (Braun & Clarke) pipeline plus its audit spine:

* :func:`code_themes` — ``qualitative-coding``: apply a code lexicon (or an
  auto-seeded one from corpus term frequency) to every unit, producing a coding
  ledger (``codebook``), the matched ``segments``, aggregated ``themes``, and a
  code **co-occurrence** ``theme_map`` (a networkx graph exported as an adjacency
  dict). Each theme carries its claim→supporting-unit evidence scaffold.
* :func:`trace_quotes` — ``quote-traceability``: attach an offset provenance
  stamp ``(doc_id, unit_id, start, end)`` to every coded segment, slice the
  original corpus text back to *verify* the quote matches, and audit orphans
  (codes with no quote / units with no code). Bidirectional claim⇄quote index.
* :func:`reflexive_memo` — ``reflexive-memo`` (no-code methodology): structure
  the researcher's interpretive trail into an auditable memo — a positionality
  statement (three axes), a four-part log entry per theme (observation /
  reaction / bias / adjustment), and AI-vs-human interpretation authorship.

The core coding / co-occurrence / traceability chain is *real* computation over
pandas + networkx; the reflexive memo produces a **structured** protocol (nested
dicts), not placeholder prose. Everything is deterministic (seed fixed where any
ordering could otherwise be ambiguous) and no optional heavy dependency is
imported at module load — ``networkx`` is imported lazily and degrades to a plain
adjacency dict if it is somehow unavailable, and the module never reaches the
network.
"""
from __future__ import annotations

import importlib
import re
from collections import Counter
from typing import Any

import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["code_themes", "trace_quotes", "reflexive_memo", "code_analysis", "coding_reliability"]


def _try_import(name: str) -> Any | None:
    """Import ``name`` lazily, returning ``None`` if unavailable (never raises)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# --------------------------------------------------------------------- helpers
def _as_units(units: Any) -> list[dict[str, Any]]:
    """Coerce a ``corpus['units']`` value into a list of unit dicts.

    Accepts the canonical list-of-dicts, a single unit dict, or a bare list of
    strings (auto-numbered). Every returned unit is guaranteed the keys
    ``unit_id / doc_id / start / end / text``.
    """
    if units is None:
        return []
    if isinstance(units, dict):
        units = [units]
    out: list[dict[str, Any]] = []
    for i, u in enumerate(units):
        if isinstance(u, dict):
            text = "" if u.get("text") is None else str(u.get("text"))
            uid = str(u.get("unit_id", f"u{i:04d}"))
            doc = str(u.get("doc_id", uid.split(":", 1)[0]))
            out.append(
                {
                    "unit_id": uid,
                    "doc_id": doc,
                    "start": u.get("start"),
                    "end": u.get("end"),
                    "text": text,
                }
            )
        else:  # bare string / other scalar
            text = "" if u is None else str(u)
            out.append(
                {
                    "unit_id": f"u{i:04d}",
                    "doc_id": f"u{i:04d}",
                    "start": None,
                    "end": None,
                    "text": text,
                }
            )
    return out


# token = run of CJK chars OR a Latin/greek word of length >= 2
_TOKEN = re.compile(r"[一-鿿]+|[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'-]+")

# very small multilingual stoplist — enough to keep auto-seeded codes meaningful
_STOP: frozenset[str] = frozenset(
    {
        "the", "and", "that", "this", "with", "for", "was", "were", "are", "have",
        "has", "had", "not", "but", "you", "they", "she", "his", "her", "him",
        "our", "their", "from", "which", "what", "when", "will", "would", "there",
        "been", "then", "than", "them", "these", "those", "into", "about", "just",
        "也", "的", "了", "和", "是", "在", "我", "有", "就", "不", "都", "而", "及",
        "与", "或", "这", "那", "他", "她", "它", "们", "个", "上", "下", "很", "并",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercased content tokens of ``text``, minus a small stoplist."""
    toks = _TOKEN.findall(text.lower())
    out: list[str] = []
    for t in toks:
        # split CJK runs into single characters so co-occurrence stays meaningful
        if "一" <= t[0] <= "鿿":
            out.extend(ch for ch in t if ch not in _STOP)
        elif t not in _STOP:
            out.append(t)
    return out


def _auto_lexicon(units: list[dict[str, Any]], n_codes: int) -> dict[str, list[str]]:
    """Seed a code lexicon from the most frequent content terms across units.

    A deterministic fallback for when the user supplies no ``lexicon``: the top
    ``n_codes`` terms (by document frequency, ties broken alphabetically) each
    become a single-keyword code named ``code_<term>``.
    """
    df: Counter[str] = Counter()
    for u in units:
        for tok in set(_tokenize(u["text"])):
            df[tok] += 1
    ranked = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
    return {f"code_{term}": [term] for term, _ in ranked[:n_codes] if _ > 0}


def _normalize_lexicon(lexicon: Any) -> dict[str, list[str]]:
    """Coerce a user ``lexicon`` into ``{code: [lowercased keyword, ...]}``."""
    out: dict[str, list[str]] = {}
    if not isinstance(lexicon, dict):
        return out
    for code, kws in lexicon.items():
        if kws is None:
            words = [str(code)]
        elif isinstance(kws, str):
            words = [kws]
        else:
            words = [str(w) for w in kws]
        out[str(code)] = sorted({w.lower() for w in words if str(w).strip()})
    return out


def _match_codes(text: str, lexicon: dict[str, list[str]]) -> list[str]:
    """Codes whose any keyword appears (case-insensitively, substring) in text."""
    low = text.lower()
    hits: list[str] = []
    for code, kws in lexicon.items():
        if any(kw and kw in low for kw in kws):
            hits.append(code)
    return hits


def _theme_of(code: str) -> str:
    """Default theme grouping for a code (strip a ``code_`` / ``T#_`` prefix)."""
    for pref in ("code_", "theme_"):
        if code.startswith(pref):
            return code[len(pref):]
    return code


# --------------------------------------------------------------------- codes
@register(
    name="code_themes",
    aliases=["主题编码", "qualitative_coding"],
    category="qual",
    tier="plus",
    skill="qualitative-coding",
    languages=["Python"],
    key_tools=["pandas", "networkx", "Braun&Clarke"],
    description="Braun&Clarke 反身主题分析:对 units 编码,建编码台账+主题地图",
    requires={"corpus": ["units"]},
    produces={
        "codes": ["codebook", "segments", "themes", "theme_map"],
        "evidence": ["claim_evidence"],
        "artifacts": ["tables"],
    },
    auto_fix="escalate",
)
def code_themes(state: StudyState, **kwargs: Any) -> StudyState:
    """Reflexive thematic coding of ``corpus['units']`` (Braun & Clarke phases 2–4).

    Applies a code ``lexicon`` to every unit, records the matched ``segments``,
    aggregates codes into ``themes``, and builds a code **co-occurrence** map
    (``theme_map``) as a networkx graph exported to an adjacency dict.

    Parameters (via ``kwargs``)
    ---------------------------
    units : list[dict], optional
        Coding units overriding ``state.corpus['units']``.
    lexicon : dict, optional
        ``{code: [keyword, ...]}``. When absent, a lexicon is auto-seeded from
        the most frequent corpus terms (``n_auto_codes`` of them).
    themes : dict, optional
        ``{theme: [code, ...]}`` grouping of codes into higher-order themes.
        When absent each code maps to a same-named theme.
    n_auto_codes : int, default 12
        Number of codes to auto-seed when no ``lexicon`` is given.
    """
    units = _as_units(kwargs.get("units", state.corpus.get("units")))
    n_auto = int(kwargs.get("n_auto_codes", 12))

    lexicon = _normalize_lexicon(kwargs.get("lexicon"))
    if not lexicon:
        lexicon = _auto_lexicon(units, n_auto)

    # --- phase: coding — match each unit against the lexicon -----------------
    segments: list[dict[str, Any]] = []
    code_counts: Counter[str] = Counter()
    code_units: dict[str, list[str]] = {c: [] for c in lexicon}
    per_unit_codes: dict[str, list[str]] = {}

    for u in units:
        hits = _match_codes(u["text"], lexicon)
        per_unit_codes[u["unit_id"]] = hits
        for code in hits:
            code_counts[code] += 1
            code_units.setdefault(code, []).append(u["unit_id"])
            segments.append(
                {
                    "unit_id": u["unit_id"],
                    "doc_id": u["doc_id"],
                    "code": code,
                    "start": u.get("start"),
                    "end": u.get("end"),
                    "text": u["text"],
                }
            )

    # --- phase: theming — group codes -> themes ------------------------------
    theme_groups = kwargs.get("themes")
    if isinstance(theme_groups, dict) and theme_groups:
        code_to_theme = {
            str(c): str(theme)
            for theme, codes in theme_groups.items()
            for c in (codes or [])
        }
    else:
        code_to_theme = {c: _theme_of(c) for c in lexicon}
    # any code not explicitly grouped keeps its own default theme
    for c in lexicon:
        code_to_theme.setdefault(c, _theme_of(c))

    themes: dict[str, dict[str, Any]] = {}
    for code, theme in code_to_theme.items():
        t = themes.setdefault(
            theme, {"codes": [], "n_segments": 0, "unit_ids": []}
        )
        if code not in t["codes"]:
            t["codes"].append(code)
        t["n_segments"] += code_counts.get(code, 0)
        for uid in code_units.get(code, []):
            if uid not in t["unit_ids"]:
                t["unit_ids"].append(uid)

    # --- codebook ledger -----------------------------------------------------
    rows = [
        {
            "code": code,
            "count": code_counts.get(code, 0),
            "theme": code_to_theme.get(code, _theme_of(code)),
            "def": "keywords: " + ", ".join(lexicon.get(code, [])),
        }
        for code in lexicon
    ]
    codebook = pd.DataFrame(rows, columns=["code", "count", "theme", "def"])
    if not codebook.empty:
        codebook = codebook.sort_values(
            ["count", "code"], ascending=[False, True]
        ).reset_index(drop=True)

    # --- theme_map: code co-occurrence graph (per unit) ----------------------
    codes_sorted = list(lexicon.keys())
    cooc: Counter[tuple[str, str]] = Counter()
    for hits in per_unit_codes.values():
        uniq = sorted(set(hits))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                cooc[(uniq[i], uniq[j])] += 1

    nx = _try_import("networkx")
    if nx is not None:
        graph = nx.Graph()
        graph.add_nodes_from(
            (c, {"count": code_counts.get(c, 0), "theme": code_to_theme.get(c, _theme_of(c))})
            for c in codes_sorted
        )
        for (a, b), w in cooc.items():
            graph.add_edge(a, b, weight=w)
        adjacency = {
            n: {nbr: int(d.get("weight", 1)) for nbr, d in graph.adj[n].items()}
            for n in graph.nodes
        }
        density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0
        theme_map = {
            "nodes": codes_sorted,
            "adjacency": adjacency,
            "n_edges": graph.number_of_edges(),
            "density": round(density, 4),
        }
    else:  # graceful degradation without networkx
        adjacency = {c: {} for c in codes_sorted}
        for (a, b), w in cooc.items():
            adjacency[a][b] = int(w)
            adjacency[b][a] = int(w)
        theme_map = {
            "nodes": codes_sorted,
            "adjacency": adjacency,
            "n_edges": len(cooc),
            "density": 0.0,
        }

    # --- claim_evidence: theme -> supporting unit_ids ------------------------
    claim_evidence = {
        theme: {
            "claim": f"主题「{theme}」得到 {len(info['unit_ids'])} 个编码单元支撑",
            "codes": list(info["codes"]),
            "support_unit_ids": list(info["unit_ids"]),
            "n_support": len(info["unit_ids"]),
        }
        for theme, info in themes.items()
    }

    state.write("codes", "codebook", codebook)
    state.write("codes", "segments", segments)
    state.write("codes", "themes", themes)
    state.write("codes", "theme_map", theme_map)
    state.write("evidence", "claim_evidence", claim_evidence)
    state.write("artifacts", "tables", {"codebook": codebook})
    return state


# --------------------------------------------------------------------- trace
@register(
    name="trace_quotes",
    aliases=["引语溯源", "quote_traceability"],
    category="qual",
    tier="community",
    skill="quote-traceability",
    languages=["Python"],
    key_tools=["stdlib"],
    description="建论断⇄quote 双向溯源索引:offset 溯源戳 + slice 回校 + 孤儿审计",
    requires={"corpus": ["units"], "codes": ["segments"]},
    produces={"evidence": ["quote_index"]},
    prerequisites={"functions": ["code_themes"]},
    auto_fix="escalate",
)
def trace_quotes(state: StudyState, **kwargs: Any) -> StudyState:
    """Build a bidirectional claim⇄quote traceability index with offset verification.

    For each coded ``segment`` it stamps a provenance tuple
    ``(doc_id, unit_id, start, end)`` and, when the original document text is
    available, slices ``documents[doc_id][start:end]`` back and checks that the
    recovered quote matches the segment text (``verified``). It then audits
    **orphans**: codes with no supporting quote and units carrying no code.

    Parameters (via ``kwargs``)
    ---------------------------
    segments : list[dict], optional
        Coded segments overriding ``state.codes['segments']``.
    units : list[dict], optional
        Coding units overriding ``state.corpus['units']``.
    documents : dict, optional
        ``{doc_id: text}`` overriding ``state.corpus['documents']`` (used for the
        slice-back verification).
    """
    segments = kwargs.get("segments", state.codes.get("segments")) or []
    units = _as_units(kwargs.get("units", state.corpus.get("units")))
    documents = kwargs.get("documents", state.corpus.get("documents")) or {}
    unit_by_id = {u["unit_id"]: u for u in units}

    entries: list[dict[str, Any]] = []
    codes_seen: set[str] = set()
    units_with_code: set[str] = set()
    n_verified = 0
    n_checkable = 0

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        uid = str(seg.get("unit_id", ""))
        code = str(seg.get("code", ""))
        doc_id = str(seg.get("doc_id", unit_by_id.get(uid, {}).get("doc_id", "")))
        start = seg.get("start", unit_by_id.get(uid, {}).get("start"))
        end = seg.get("end", unit_by_id.get(uid, {}).get("end"))
        quote = str(seg.get("text", unit_by_id.get(uid, {}).get("text", "")))

        codes_seen.add(code)
        units_with_code.add(uid)

        # slice-back verification against the original document text
        recovered: str | None = None
        verified: bool | None = None
        doc_text = documents.get(doc_id)
        if isinstance(doc_text, str) and isinstance(start, int) and isinstance(end, int):
            n_checkable += 1
            recovered = doc_text[start:end]
            verified = recovered == quote
            if verified:
                n_verified += 1

        entries.append(
            {
                "code": code,
                "unit_id": uid,
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "quote": quote,
                "recovered": recovered,
                "verified": verified,
            }
        )

    # --- orphan audit --------------------------------------------------------
    all_codes = set()
    codebook = state.codes.get("codebook")
    if isinstance(codebook, pd.DataFrame) and "code" in codebook.columns:
        all_codes = set(map(str, codebook["code"].tolist()))
    themes = state.codes.get("themes")
    if isinstance(themes, dict):
        for info in themes.values():
            all_codes.update(map(str, (info or {}).get("codes", [])))

    all_unit_ids = {u["unit_id"] for u in units}
    orphan_codes = sorted(all_codes - codes_seen)
    orphan_units = sorted(all_unit_ids - units_with_code)

    coverage = {
        "n_segments": len(entries),
        "n_codes_with_quote": len(codes_seen & all_codes) if all_codes else len(codes_seen),
        "n_units_coded": len(units_with_code & all_unit_ids),
        "n_units_total": len(all_unit_ids),
        "unit_coverage": round(
            len(units_with_code & all_unit_ids) / len(all_unit_ids), 4
        )
        if all_unit_ids
        else 0.0,
        "n_verified": n_verified,
        "n_checkable": n_checkable,
        "verify_rate": round(n_verified / n_checkable, 4) if n_checkable else None,
    }

    quote_index = {
        "entries": entries,
        "orphans": {"codes_without_quote": orphan_codes, "units_without_code": orphan_units},
        "coverage": coverage,
    }

    state.write("evidence", "quote_index", quote_index)
    return state


# ------------------------------------------------------------- CAQDAS content analysis
@register(
    name="code_analysis",
    aliases=["编码分析", "caqdas_analysis", "code_index"],
    category="qual",
    tier="plus",
    skill="qualitative-coding",
    languages=["Python"],
    key_tools=["pandas", "networkx"],
    description="从已编码片段做 CAQDAS 内容分析:代码×文档矩阵/频次/共现/编码累积曲线 + 引文核验(不重新编码)",
    requires={},
    produces={
        "codes": ["codebook", "segments", "theme_map"],
        "diagnostics": ["code_matrix", "code_frequency", "saturation", "coverage"],
        "evidence": ["quote_index"],
    },
    auto_fix="none",
)
def code_analysis(state: StudyState, **kwargs: Any) -> StudyState:
    """CAQDAS content-analysis + provenance over **already-coded** segments.

    Distinct from :func:`code_themes` (which *applies a lexicon to discover* codes):
    this takes segments an analyst/agent has *already* coded and computes the
    descriptive layer of computer-assisted qualitative analysis **without
    re-coding** — deliberately paradigm-NEUTRAL (these are content-analysis
    operations; they make no thematic/grounded-theory claim):

      * a **code × document** matrix + code **frequency**;
      * a code **co-occurrence** graph (``theme_map``: codes sharing a document);
      * a code-**accumulation** curve over document order — how many *new* codes each
        added document introduces. This is a descriptive coverage curve, explicitly
        NOT a claim of theoretical saturation (which is a paradigm-committed concept
        the caller must not assert on the analyst's behalf);
      * **quote verification** — slice each segment's offsets back out of the source
        document and confirm the quote matches (delegates to :func:`trace_quotes`).

    This is the SINGLE contract-checked entrypoint the interactive Coding tool and
    the methodology skill both call, so the two paths cannot compute divergent
    results and both survive socialverse version drift.

    Parameters (via ``kwargs``)
    ---------------------------
    segments : list[dict]
        Already-coded segments, each ``{doc_id, code, quote, start, end}``
        (``start``/``end`` = character offsets in that document's text). Falls back
        to ``state.codes['segments']``.
    documents : dict, optional
        ``{doc_id: text}`` — for quote slice-back verification and the accumulation
        curve's document order. Falls back to ``state.corpus['documents']``.
    doc_order : list, optional
        Explicit document order for the accumulation curve.
    """
    from collections import Counter as _Counter

    raw_segs = kwargs.get("segments")
    if raw_segs is None:
        raw_segs = state.codes.get("segments") or []
    documents = kwargs.get("documents") or state.corpus.get("documents") or {}

    # normalize to the canonical segment shape (a synthetic unit_id per span)
    segs: list[dict[str, Any]] = []
    for s in raw_segs:
        if not isinstance(s, dict):
            continue
        code = str(s.get("code", "")).strip()
        if not code:
            continue
        doc_id = str(s.get("doc_id", ""))
        quote = str(s.get("quote", s.get("text", "")))
        start, end = s.get("start"), s.get("end")
        # unit = the document (coarse but honest); verification uses the segment's
        # own char offsets against the document text, independent of unit grain.
        segs.append({
            "unit_id": doc_id,
            "doc_id": doc_id, "code": code,
            "start": start, "end": end, "text": quote,
        })

    if documents:
        state.write("corpus", "documents", dict(documents))
        # seed document-level units so trace_quotes' `requires: corpus.units`
        # contract is satisfied (and unit_coverage = share of documents coded)
        if not state.corpus.get("units"):
            state.write("corpus", "units", [
                {"unit_id": did, "doc_id": did, "start": 0,
                 "end": len(str(txt)), "text": str(txt)}
                for did, txt in documents.items()
            ])

    # --- codebook (code -> count) --------------------------------------------
    cnt = _Counter(s["code"] for s in segs)
    codes = list(cnt.keys())
    codebook = pd.DataFrame(
        [{"code": c, "count": int(n)} for c, n in cnt.most_common()],
        columns=["code", "count"],
    )
    state.write("codes", "segments", segs)
    state.write("codes", "codebook", codebook)

    # --- code x document matrix + frequency ----------------------------------
    doc_ids = list(dict.fromkeys(
        kwargs.get("doc_order") or list(documents.keys()) or [s["doc_id"] for s in segs]
    ))
    mat = pd.DataFrame(0, index=codes, columns=doc_ids, dtype=int)
    for s in segs:
        if s["doc_id"] in mat.columns and s["code"] in mat.index:
            mat.at[s["code"], s["doc_id"]] += 1
    code_matrix = mat.reset_index().rename(columns={"index": "code"})
    state.write("diagnostics", "code_matrix", code_matrix)
    state.write("diagnostics", "code_frequency", codebook.copy())

    # --- code co-occurrence (codes sharing a document) -> theme_map ----------
    by_doc: dict[str, set] = {}
    for s in segs:
        by_doc.setdefault(s["doc_id"], set()).add(s["code"])
    cooc: _Counter = _Counter()
    for cs in by_doc.values():
        ordered = sorted(cs)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                cooc[(ordered[i], ordered[j])] += 1
    nx = _try_import("networkx")
    if nx is not None:
        g = nx.Graph()
        g.add_nodes_from((c, {"count": int(cnt[c])}) for c in codes)
        for (a, b), w in cooc.items():
            g.add_edge(a, b, weight=int(w))
        adjacency = {n: {nb: int(d.get("weight", 1)) for nb, d in g.adj[n].items()} for n in g.nodes}
        density = float(nx.density(g)) if g.number_of_nodes() > 1 else 0.0
        theme_map = {"nodes": codes, "adjacency": adjacency,
                     "n_edges": g.number_of_edges(), "density": round(density, 4)}
    else:
        adjacency = {c: {} for c in codes}
        for (a, b), w in cooc.items():
            adjacency[a][b] = int(w)
            adjacency[b][a] = int(w)
        theme_map = {"nodes": codes, "adjacency": adjacency, "n_edges": len(cooc), "density": 0.0}
    state.write("codes", "theme_map", theme_map)

    # --- code-accumulation curve over document order (NOT saturation) --------
    seen: set = set()
    rows = []
    for i, did in enumerate(doc_ids, 1):
        dcodes = by_doc.get(did, set())
        new = dcodes - seen
        seen |= dcodes
        rows.append({"documents": i, "doc_id": did,
                     "unique_codes": len(seen), "new_codes": len(new)})
    saturation = pd.DataFrame(rows, columns=["documents", "doc_id", "unique_codes", "new_codes"])
    state.write("diagnostics", "saturation", saturation)

    # --- quote verification (best-effort) ------------------------------------
    coverage: dict[str, Any] = {
        "n_documents": len(doc_ids),
        "n_documents_coded": len(by_doc),
        "doc_coverage": round(len(by_doc) / len(doc_ids), 4) if doc_ids else 0.0,
    }
    try:
        trace_quotes(state, documents=documents)
        qi = state.evidence.get("quote_index") or {}
        cov = qi.get("coverage") or {}
        coverage.update({
            "n_verified": cov.get("n_verified"),
            "n_checkable": cov.get("n_checkable"),
            "verify_rate": cov.get("verify_rate"),
        })
    except Exception as exc:  # pragma: no cover - defensive
        coverage["verify_error"] = str(exc)
    state.write("diagnostics", "coverage", coverage)
    return state


# ----------------------------------------------------------- inter-coder reliability
def _unit_spans(text: str, unit: str) -> list[tuple[int, int]]:
    """Split ``text`` into (start, end) unit spans (sentence or paragraph)."""
    text = text or ""
    spans: list[tuple[int, int]] = []
    if unit == "paragraph":
        start = 0
        for m in re.finditer(r"\n[ \t]*\n", text):
            end = m.start()
            if text[start:end].strip():
                spans.append((start, end))
            start = m.end()
        if text[start:].strip():
            spans.append((start, len(text)))
    else:  # sentence
        start = 0
        for m in re.finditer(r"[.!?。!?！？]+(?:\s+|$)", text):
            end = m.end()
            if text[start:end].strip():
                spans.append((start, end))
            start = end
        if start < len(text) and text[start:].strip():
            spans.append((start, len(text)))
    return spans or ([(0, len(text))] if text.strip() else [])


@register(
    name="coding_reliability",
    aliases=["编码者间信度", "intercoder_reliability", "coding_agreement"],
    category="qual",
    tier="plus",
    skill="qualitative-coding",
    languages=["Python"],
    key_tools=["pandas", "numpy"],
    description="两名及以上编码者对同一语料的编码者间信度:投影到单元网格→Cohen/Fleiss κ + Krippendorff α + 百分比一致 + 逐单元分歧",
    requires={},
    produces={"diagnostics": ["interrater", "coding_disagreements"], "sources": ["datasets"]},
    auto_fix="none",
)
def coding_reliability(state: StudyState, **kwargs: Any) -> StudyState:
    """Inter-coder reliability across TWO+ independent codings of the SAME corpus.

    Each coder brings free-span segments ``{doc_id, code, start, end}``. Reliability
    needs a shared coding UNIT, so this projects every coder onto a common unit grid
    (sentences by default): each unit gets one nominal label per coder — the code
    whose span most overlaps it, or ``'∅'`` (uncoded). It then builds the
    subjects×raters frame and runs :func:`~socialverse.tl.interrater` — percent
    agreement, Cohen's κ (2 coders), Fleiss' κ (N coders), Krippendorff's α — and
    audits per-unit disagreements.

    **Paradigm note.** Inter-coder reliability is a **content-analysis / positivist-
    qualitative** quality criterion. Reflexive thematic analysis (Braun & Clarke)
    deliberately does NOT use it (the analyst is the instrument). Report it only when
    the chosen paradigm calls for it.

    kwargs
    ------
    coders : dict ``{coder_name: [segment, ...]}`` — two or more coders' segments.
    documents : dict ``{doc_id: text}`` — the shared corpus (else ``corpus['documents']``).
    unit : ``'sentence'`` (default) or ``'paragraph'`` — the reliability unit.
    """
    coders = kwargs.get("coders") or {}
    documents = kwargs.get("documents") or state.corpus.get("documents") or {}
    unit = kwargs.get("unit", "sentence")
    UNC = "∅"

    if len(coders) < 2:
        state.write("diagnostics", "interrater", {
            "note": f"编码者间信度需要 ≥2 名编码者(收到 {len(coders)} 名)",
            "n_raters": len(coders), "n_subjects": 0,
        })
        return state

    coder_names = list(coders.keys())
    # build the shared unit grid per document
    units: list[dict[str, Any]] = []
    for doc_id, text in documents.items():
        for (us, ue) in _unit_spans(str(text), unit):
            units.append({"doc_id": str(doc_id), "start": us, "end": ue,
                          "text": str(text)[us:ue].strip()})

    def _label(segs: list, doc_id: str, us: int, ue: int) -> str:
        best, best_ov = UNC, 0
        for s in segs or []:
            if str(s.get("doc_id", "")) != doc_id:
                continue
            ss, se = s.get("start"), s.get("end")
            if not isinstance(ss, int) or not isinstance(se, int):
                continue
            ov = max(0, min(ue, se) - max(us, ss))
            if ov > best_ov:
                best_ov, best = ov, str(s.get("code", UNC))
        return best

    rows = []
    disagreements = []
    for u in units:
        labels = {c: _label(coders[c], u["doc_id"], u["start"], u["end"]) for c in coder_names}
        row = {"doc_id": u["doc_id"], "unit": u["text"]}
        row.update({f"rater_{c}": labels[c] for c in coder_names})
        rows.append(row)
        if len(set(labels.values())) > 1:
            disagreements.append({"doc_id": u["doc_id"], "unit": u["text"], **{c: labels[c] for c in coder_names}})

    df = pd.DataFrame(rows)
    rater_cols = [f"rater_{c}" for c in coder_names]
    # drop units nobody coded (all ∅) — they only inflate agreement
    if not df.empty:
        coded_mask = df[rater_cols].apply(lambda r: any(v != UNC for v in r), axis=1)
        df_scored = df[coded_mask].reset_index(drop=True)
    else:
        df_scored = df

    # delegate the reliability battery to the interrater entrypoint
    from ._interrater import interrater as _interrater
    state.write("sources", "datasets", df_scored[rater_cols] if not df_scored.empty else df_scored)
    _interrater(state, raters=rater_cols)
    ir = state.diagnostics.get("interrater") or {}
    ir = dict(ir)
    ir["coders"] = coder_names
    ir["unit"] = unit
    ir["n_units_total"] = len(units)
    ir["n_units_scored"] = int(len(df_scored))
    ir["n_disagreements"] = len(disagreements)
    state.write("diagnostics", "interrater", ir)
    state.write("diagnostics", "coding_disagreements", disagreements[:200])
    return state


# --------------------------------------------------------------------- memo
@register(
    name="reflexive_memo",
    aliases=["反身备忘", "reflexive_memo"],
    category="qual",
    tier="community",
    skill="reflexive-memo",
    languages=["无代码(方法论)"],
    key_tools=["reflexive TA", "positionality"],
    description="把研究者解释轨迹结构化为可审计反身备忘:立场声明+四段日志+AI/人归属",
    requires={"codes": ["themes"], "corpus": ["units"]},
    produces={"evidence": ["provenance"], "governance": ["ethics"]},
    auto_fix="none",
)
def reflexive_memo(state: StudyState, **kwargs: Any) -> StudyState:
    """Structure the researcher's interpretive trail into an auditable reflexive memo.

    Produces a **structured** protocol (nested dicts), not prose: a three-axis
    positionality statement, a four-part reflexive log entry per theme
    (observation / reaction / bias / adjustment), and an explicit
    AI-vs-human interpretation-authorship split. Written to
    ``evidence['provenance']`` (the interpretive audit chain) and
    ``governance['ethics']`` (reflexivity-declaration status).

    Parameters (via ``kwargs``)
    ---------------------------
    positionality : dict, optional
        Overrides for the three axes ``social_location`` / ``field_relation`` /
        ``stakes`` (each a free-text string the researcher supplies).
    log : dict, optional
        ``{theme: {observation, reaction, bias, adjustment}}`` overriding the
        auto-scaffolded four-part entries.
    authorship : dict, optional
        ``{ai: [...], human: [...]}`` attributing interpretive moves.
    researcher : str, optional
        Name / id recorded as the memo's author.
    """
    themes = state.codes.get("themes") or {}
    theme_names = list(themes.keys()) if isinstance(themes, dict) else list(themes)
    units = _as_units(kwargs.get("units", state.corpus.get("units")))
    researcher = str(kwargs.get("researcher", "researcher"))

    # --- positionality: three axes ------------------------------------------
    pos_in = kwargs.get("positionality") or {}
    positionality = {
        "social_location": str(
            pos_in.get(
                "social_location",
                "研究者的社会位置(阶层/性别/族裔/学科训练)尚未声明 — 待填写",
            )
        ),
        "field_relation": str(
            pos_in.get(
                "field_relation",
                "研究者与田野/受访者的关系(内部人/外部人、权力位差)尚未声明 — 待填写",
            )
        ),
        "stakes": str(
            pos_in.get(
                "stakes",
                "研究发现对研究者及被研究者的利害关系尚未声明 — 待填写",
            )
        ),
    }
    positionality_declared = any(k in pos_in for k in ("social_location", "field_relation", "stakes"))

    # --- four-part reflexive log, one entry per theme ------------------------
    log_in = kwargs.get("log") or {}

    def _entry(theme: str) -> dict[str, str]:
        info = themes.get(theme) if isinstance(themes, dict) else None
        n_codes = len((info or {}).get("codes", [])) if isinstance(info, dict) else 0
        n_units = len((info or {}).get("unit_ids", [])) if isinstance(info, dict) else 0
        override = log_in.get(theme) if isinstance(log_in, dict) else None
        if isinstance(override, dict):
            return {
                "observation": str(override.get("observation", "")),
                "reaction": str(override.get("reaction", "")),
                "bias": str(override.get("bias", "")),
                "adjustment": str(override.get("adjustment", "")),
            }
        return {
            "observation": f"主题「{theme}」由 {n_codes} 个编码、{n_units} 个单元构成。",
            "reaction": f"研究者对「{theme}」的直觉反应与初始解读 — 待研究者补写。",
            "bias": f"可能影响对「{theme}」判读的先见/立场偏差 — 待研究者识别。",
            "adjustment": f"据反身审视对「{theme}」编码/解读所做的调整 — 待研究者记录。",
        }

    log = {theme: _entry(theme) for theme in theme_names}

    # --- interpretation authorship: AI vs human -----------------------------
    auth_in = kwargs.get("authorship") or {}
    interpretation_authorship = {
        "ai": list(
            auth_in.get(
                "ai",
                ["auto-seeded code lexicon", "code co-occurrence theme_map", "memo scaffolding"],
            )
        ),
        "human": list(
            auth_in.get(
                "human",
                ["theme naming & interpretation", "positionality statement", "final claim–evidence judgement"],
            )
        ),
        "policy": "reflexive TA: 生成式辅助编码与结构脚手架由 AI 承担,解释与主题命名归研究者;须在方法与致谢中披露。",
    }

    # --- provenance: interpretive audit chain --------------------------------
    provenance = {
        "step": "reflexive_memo",
        "researcher": researcher,
        "framework": "Braun & Clarke reflexive thematic analysis",
        "positionality": positionality,
        "log": log,
        "interpretation_authorship": interpretation_authorship,
        "n_themes": len(theme_names),
        "n_units": len(units),
    }

    # --- governance/ethics: reflexivity-declaration status -------------------
    ethics = {
        "reflexivity_declared": bool(positionality_declared),
        "positionality_complete": bool(positionality_declared),
        "ai_use_disclosed": True,
        "log_entries": len(log),
        "status": "declared" if positionality_declared else "pending-researcher-input",
        "note": (
            "反身性声明已提供"
            if positionality_declared
            else "反身性声明为自动脚手架,研究者须补全立场三轴后方可视为完成"
        ),
    }

    state.write("evidence", "provenance", provenance)
    state.write("governance", "ethics", ethics)
    return state
