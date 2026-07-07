"""``sv.tl._text`` — registered implementations for the *text-as-data* /
philology domain.

Two registry entries back the textual-scholarship skills:

- :func:`philology_collate` (``philology-variant-reading``) — textual criticism:
  collate multiple witnesses of a work against a base text, classify the variant
  readings into a critical *apparatus*, and reconstruct a *stemma codicum* by the
  method of shared error (agreement in variants ⇒ common descent).
- :func:`tei_encode` (``tei-encoding``) — encode transcribed text as valid
  TEI-P5 XML: a real ``teiHeader`` (title / responsibility statement) plus a
  structured ``text/body`` with paragraph ``<p>`` and line-beginning ``<lb/>``
  milestones, well-formedness-checked.

Both speak the 12-slot :class:`~socialverse._state.StudyState` vocabulary through
the ``@register`` contract, so the resolver can chain ``tei_encode`` (produces
``corpus.tei``) into downstream corpus work, and ``philology_collate`` off a
registered ``corpus.documents``.

The collation core is *really computed* with :mod:`difflib` (sequence alignment)
and :mod:`networkx` (the stemma as a minimum-spanning tree over a shared-error
distance). The heavy XML validator (``lxml``) is imported lazily and degrades to
the stdlib :mod:`xml.etree.ElementTree` parser, so the module loads and runs even
when ``lxml`` is absent — never raising at import time and never touching the
network.
"""
from __future__ import annotations

import difflib
import importlib
import re
from typing import Any

from .._registry import register
from .._state import StudyState

__all__ = ["philology_collate", "tei_encode"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft optional import — returns the module or ``None``."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _as_documents(documents: Any) -> dict[str, str]:
    """Coerce whatever arrived as ``documents`` into ``{witness_id: text}``.

    Accepts a mapping, or a sequence of ``(id, text)`` pairs / ``{"id","text"}``
    records / bare strings (auto-numbered). Non-string texts are stringified.
    """
    if documents is None:
        return {}
    if isinstance(documents, dict):
        return {str(k): _as_text(v) for k, v in documents.items()}
    out: dict[str, str] = {}
    if isinstance(documents, (list, tuple)):
        for i, item in enumerate(documents):
            if isinstance(item, dict):
                wid = str(item.get("id") or item.get("witness") or f"W{i + 1}")
                out[wid] = _as_text(item.get("text") or item.get("content") or "")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = _as_text(item[1])
            else:
                out[f"W{i + 1}"] = _as_text(item)
        return out
    # single bare string
    return {"W1": _as_text(documents)}


def _as_text(value: Any) -> str:
    """Stringify a document body (already-a-string fast path)."""
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _tokenize(text: str) -> list[str]:
    """Word-level tokenization for collation (words + standalone punctuation).

    Word-level (not char-level) tokens keep the apparatus loci human-legible —
    a variant reading is a swapped/added/dropped *word*, as an editor records it.
    """
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _synthetic_witnesses(seed: int = 0) -> dict[str, str]:
    """A deterministic 4-witness tradition with planted shared errors.

    The fallback when no ``documents`` are supplied. Witnesses B and C share the
    error ``bright→bryght`` (⇒ they should cluster on the stemma), while D is
    closest to the base A — so the reconstructed tree is non-trivial and stable.
    """
    del seed  # tradition is fixed; seed kept for signature symmetry
    base = "the moon was bright and the sea was calm that night"
    return {
        "A": base,
        "B": "the moon was bryght and the sea was calm that nyght",
        "C": "the moon was bryght and the ocean was calm that night",
        "D": "the moon was bright and the sea was calm that night indeed",
    }


# ------------------------------------------------------------------ collation
def _collate_pair(
    base_tokens: list[str], other_tokens: list[str]
) -> list[dict[str, Any]]:
    """Align two token streams and emit the loci where they diverge.

    Uses :class:`difflib.SequenceMatcher` opcodes: ``replace`` → substitution,
    ``delete`` → omission in the witness, ``insert`` → addition in the witness.
    Each locus is anchored to the base token index so loci from different
    witnesses can be merged.
    """
    sm = difflib.SequenceMatcher(a=base_tokens, b=other_tokens, autojunk=False)
    loci: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        lemma = " ".join(base_tokens[i1:i2])
        reading = " ".join(other_tokens[j1:j2])
        loci.append(
            {
                "base_start": i1,
                "base_end": i2,
                "type": {"replace": "sub", "delete": "om", "insert": "add"}[tag],
                "lemma": lemma,       # "" for a pure addition
                "reading": reading,   # "" for a pure omission
            }
        )
    return loci


def _build_apparatus(
    base_id: str, base_tokens: list[str], witnesses: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Merge per-witness variant loci into a positional critical apparatus.

    Returns a list of entries ``{locus, base_span, lemma, readings:{witness:...}}``
    sorted by position in the base text — the shape an editor prints at the foot
    of a critical edition.
    """
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for wid, wtokens in witnesses.items():
        if wid == base_id:
            continue
        for v in _collate_pair(base_tokens, wtokens):
            key = (v["base_start"], v["base_end"])
            entry = merged.setdefault(
                key,
                {
                    "base_start": v["base_start"],
                    "base_end": v["base_end"],
                    "lemma": v["lemma"] or "∅",
                    "readings": {},
                    "types": {},
                },
            )
            entry["readings"][wid] = v["reading"] or "∅"
            entry["types"][wid] = v["type"]

    apparatus: list[dict[str, Any]] = []
    for i, key in enumerate(sorted(merged), start=1):
        e = merged[key]
        # witnesses that agree with the base at this locus (the positive apparatus)
        agree = [w for w in witnesses if w != base_id and w not in e["readings"]]
        apparatus.append(
            {
                "locus": i,
                "base_span": [e["base_start"], e["base_end"]],
                "lemma": e["lemma"],
                "readings": e["readings"],
                "witness_types": e["types"],
                "base_witness": base_id,
                "agree_with_base": agree,
            }
        )
    return apparatus


def _variant_profiles(
    apparatus: list[dict[str, Any]], witness_ids: list[str], base_id: str
) -> dict[str, list[str]]:
    """A per-witness reading signature over all loci (base reading = lemma).

    Two witnesses that share the *same* non-base reading at a locus committed a
    common error — the signal the stemma is reconstructed from.
    """
    profiles: dict[str, list[str]] = {w: [] for w in witness_ids}
    for entry in apparatus:
        readings = entry["readings"]
        for w in witness_ids:
            if w == base_id or w not in readings:
                profiles[w].append(entry["lemma"])  # agrees with base
            else:
                profiles[w].append(readings[w])
    return profiles


def _shared_error_distance(a: list[str], b: list[str]) -> int:
    """Hamming distance over aligned reading profiles (# of disagreeing loci)."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_stemma(
    profiles: dict[str, list[str]], base_id: str
) -> dict[str, Any]:
    """Reconstruct a stemma as a minimum-spanning tree over shared-error distance.

    Witnesses that agree in error sit close in the distance metric, so the MST
    groups them — an operational (unrooted) proxy for the genealogical tree of
    the method of common errors. Rooted nominally at the base witness.
    """
    import networkx as nx  # optional backend — imported lazily so the module registers without it

    ids = list(profiles)
    g = nx.Graph()
    g.add_nodes_from(ids)
    for i, u in enumerate(ids):
        for v in ids[i + 1:]:
            g.add_edge(u, v, weight=_shared_error_distance(profiles[u], profiles[v]))

    tree = nx.minimum_spanning_tree(g, weight="weight") if g.number_of_edges() else g
    adjacency = {u: sorted(tree.neighbors(u)) for u in ids}
    edges = [
        {"from": u, "to": v, "shared_error_distance": int(d["weight"])}
        for u, v, d in tree.edges(data=True)
    ]
    return {
        "root": base_id,
        "nodes": ids,
        "edges": edges,
        "adjacency": adjacency,
        "method": "minimum-spanning-tree over shared-error (Hamming) distance",
    }


@register(
    name="philology_collate",
    aliases=["校勘", "collate", "异文"],
    category="text",
    tier="plus",
    skill="philology-variant-reading",
    languages=["Python"],
    key_tools=["difflib", "networkx"],
    description="校勘异文:多见证本对勘→分类异文→apparatus→共同错误法重建谱系",
    requires={"corpus": ["documents"]},
    produces={
        "models": ["stemma"],
        "evidence": ["provenance"],
        "artifacts": ["apparatus"],
    },
    auto_fix="escalate",
)
def philology_collate(state: StudyState, **kwargs: Any) -> StudyState:
    """Collate witnesses, build a critical apparatus, and reconstruct the stemma.

    Pipeline (all really computed):

    1. read ``corpus['documents']`` (``{witness_id: text}``) or ``data=`` /
       ``documents=`` kwargs; pick a *base* witness (``base=`` kwarg, else the
       first) as the collation reference;
    2. **pairwise collation** — align each witness against the base with
       :class:`difflib.SequenceMatcher` and record substitutions / omissions /
       additions;
    3. **critical apparatus** — merge the per-witness loci positionally into a
       list of ``{locus, lemma, readings:{witness:variant}}`` entries
       → ``artifacts['apparatus']``;
    4. **stemma** — from each witness's reading signature, build a minimum-
       spanning tree over shared-error (Hamming) distance
       → ``models['stemma']`` (nodes / edges / adjacency);
    5. **provenance** — a witness-level manifest (id, length, variant count vs.
       base) → ``evidence['provenance']``.

    Documents arrive via ``documents=`` / ``data=`` or ``state.corpus['documents']``;
    if none are present a deterministic 4-witness tradition with planted shared
    errors is used, so the whole chain is exercisable without external data.
    Never raises for missing data — degrades gracefully.
    """
    raw = kwargs.get("documents")
    if raw is None:
        raw = kwargs.get("data")
    if raw is None:
        raw = state.corpus.get("documents")
    docs = _as_documents(raw)
    if not docs:
        docs = _synthetic_witnesses(seed=0)

    witness_ids = list(docs)
    base_id = kwargs.get("base") if kwargs.get("base") in docs else witness_ids[0]

    tokens = {wid: _tokenize(text) for wid, text in docs.items()}
    base_tokens = tokens[base_id]

    apparatus = _build_apparatus(base_id, base_tokens, tokens)
    profiles = _variant_profiles(apparatus, witness_ids, base_id)
    stemma = _build_stemma(profiles, base_id)

    # per-witness provenance manifest (variant count vs. base) -----------------
    per_witness = {}
    for wid in witness_ids:
        n_var = sum(1 for e in apparatus if wid in e["readings"])
        per_witness[wid] = {
            "witness": wid,
            "n_tokens": len(tokens[wid]),
            "n_variants_vs_base": int(n_var),
            "is_base": wid == base_id,
        }
    provenance = {
        "base_witness": base_id,
        "n_witnesses": len(witness_ids),
        "n_loci": len(apparatus),
        "witnesses": per_witness,
        "method": "difflib.SequenceMatcher pairwise collation against base",
    }

    state.write("artifacts", "apparatus", apparatus)
    state.write("models", "stemma", stemma)
    state.write("evidence", "provenance", provenance)
    return state


# ------------------------------------------------------------------ TEI encoding
def _xml_escape(text: str) -> str:
    """Escape the five XML predefined entities for character data / attributes."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _split_paragraphs(text: str) -> list[list[str]]:
    """Split a transcription into paragraphs (blank-line separated) of lines.

    Returns ``[[line, line, ...], ...]`` — the structure needed to emit ``<p>``
    blocks with ``<lb/>`` milestones between physical lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n[ \t]*\n", text.strip())
    out: list[list[str]] = []
    for para in paras:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if lines:
            out.append(lines)
    return out


def _encode_body(paragraphs: list[list[str]]) -> str:
    """Render paragraph/line structure as TEI ``<body>`` content.

    Each paragraph is a ``<p>``; physical line breaks after the first line of a
    paragraph are marked with a self-closing ``<lb/>`` milestone (TEI P5 practice).
    """
    blocks: list[str] = []
    for para in paragraphs:
        parts: list[str] = []
        for i, line in enumerate(para):
            if i:
                parts.append("<lb/>")
            parts.append(_xml_escape(line))
        blocks.append("        <p>" + "".join(parts) + "</p>")
    if not blocks:
        blocks.append("        <p/>")
    return "\n".join(blocks)


def _build_tei(text: str, *, title: str, author: str, responsibility: str) -> str:
    """Assemble a complete, well-formed TEI-P5 document string.

    A real ``teiHeader`` (``fileDesc`` → ``titleStmt`` with ``title`` / ``author``
    / ``respStmt``, and a minimal ``publicationStmt`` / ``sourceDesc``) plus a
    ``text/body`` carrying the structured transcription.
    """
    paragraphs = _split_paragraphs(text)
    body = _encode_body(paragraphs)
    t = _xml_escape(title)
    a = _xml_escape(author)
    r = _xml_escape(responsibility)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">\n'
        "  <teiHeader>\n"
        "    <fileDesc>\n"
        "      <titleStmt>\n"
        f"        <title>{t}</title>\n"
        f"        <author>{a}</author>\n"
        "        <respStmt>\n"
        "          <resp>encoded by</resp>\n"
        f"          <name>{r}</name>\n"
        "        </respStmt>\n"
        "      </titleStmt>\n"
        "      <publicationStmt>\n"
        "        <p>Encoded with socialverse (TEI P5).</p>\n"
        "      </publicationStmt>\n"
        "      <sourceDesc>\n"
        "        <p>Born-digital transcription.</p>\n"
        "      </sourceDesc>\n"
        "    </fileDesc>\n"
        "  </teiHeader>\n"
        "  <text>\n"
        "    <body>\n"
        f"{body}\n"
        "    </body>\n"
        "  </text>\n"
        "</TEI>\n"
    )


def _validate_wellformed(xml: str) -> dict[str, Any]:
    """Well-formedness check via ``lxml`` if present, else stdlib ElementTree.

    Returns ``{well_formed, parser, error}`` — never raises. ``lxml`` additionally
    reports the TEI root element name; the stdlib path strips the default
    namespace for the local-name check.
    """
    lxml_etree = _try_import("lxml.etree")
    if lxml_etree is not None:
        try:
            root = lxml_etree.fromstring(xml.encode("utf-8"))
            local = lxml_etree.QName(root.tag).localname
            return {"well_formed": True, "parser": "lxml", "root": local, "error": None}
        except Exception as exc:  # lxml.etree.XMLSyntaxError et al.
            return {"well_formed": False, "parser": "lxml", "root": None, "error": str(exc)}

    ET = _try_import("xml.etree.ElementTree")
    if ET is not None:
        try:
            root = ET.fromstring(xml)
            local = root.tag.rsplit("}", 1)[-1]
            return {"well_formed": True, "parser": "etree", "root": local, "error": None}
        except Exception as exc:
            return {"well_formed": False, "parser": "etree", "root": None, "error": str(exc)}

    return {"well_formed": None, "parser": "none", "root": None, "error": "no XML parser"}


@register(
    name="tei_encode",
    aliases=["TEI编码", "tei_encoding"],
    category="text",
    tier="plus",
    skill="tei-encoding",
    languages=["XML(TEI)", "Python"],
    key_tools=["lxml", "TEI-P5"],
    description="按 TEI P5 把已转录文本编码为合法 TEI-XML(teiHeader+结构化正文)",
    requires={"corpus": ["documents"]},
    produces={
        "corpus": ["tei"],
        "artifacts": ["xml"],
        "evidence": ["provenance"],
    },
    auto_fix="escalate",
)
def tei_encode(state: StudyState, **kwargs: Any) -> StudyState:
    """Encode a transcribed document as valid TEI-P5 XML.

    Pipeline:

    1. read the transcription from ``text=`` / ``document=`` / ``data=`` kwargs or
       the first entry of ``corpus['documents']``; metadata (``title`` / ``author``
       / ``responsibility``) from kwargs, with sensible defaults;
    2. **structure** the text into paragraphs (blank-line separated) and physical
       lines;
    3. **emit** a complete TEI-P5 string: ``teiHeader`` (``titleStmt`` with
       ``title`` / ``author`` / ``respStmt``) + ``text/body`` with ``<p>`` blocks
       and ``<lb/>`` line milestones;
    4. **validate** well-formedness with ``lxml`` (lazy) or the stdlib
       ``xml.etree`` fallback;
    5. write the XML to ``corpus['tei']`` and ``artifacts['xml']`` and a
       validation record to ``evidence['provenance']``.

    Never raises for missing data: with no transcription a short deterministic
    placeholder line is encoded so the emitted document is always well-formed TEI.
    """
    text = kwargs.get("text") or kwargs.get("document") or kwargs.get("data")
    if text is None:
        documents = _as_documents(state.corpus.get("documents"))
        if documents:
            first = kwargs.get("witness") if kwargs.get("witness") in documents else None
            first = first or next(iter(documents))
            text = documents[first]
    text = _as_text(text) if text is not None else ""
    if not text.strip():
        text = "Untitled transcription."

    title = str(kwargs.get("title") or "Untitled")
    author = str(kwargs.get("author") or "Anonymous")
    responsibility = str(kwargs.get("responsibility") or kwargs.get("resp") or "socialverse")

    xml = _build_tei(text, title=title, author=author, responsibility=responsibility)
    validation = _validate_wellformed(xml)

    provenance = {
        "encoding": "TEI P5",
        "title": title,
        "author": author,
        "responsibility": responsibility,
        "n_chars": len(text),
        "validation": validation,
    }

    state.write("corpus", "tei", xml)
    state.write("artifacts", "xml", xml)
    state.write("evidence", "provenance", provenance)
    return state
