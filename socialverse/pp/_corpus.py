"""``sv.pp._corpus`` — corpus-building registrations for the *prepare* phase.

Registered implementations of the ``corpus-builder`` and ``ocr-tei-encoding``
skills. Both turn raw text-as-data inputs (already ingested into
``state.sources``) into the canonical ``corpus`` slot that every downstream text
tool (dfm construction, topic models, quote-tracing) speaks in:

* :func:`build_corpus` — Unicode-normalize raw texts (NFC) and segment them into
  addressable coding **units** carrying stable ``unit_id`` + character offsets,
  plus a ``manifest`` DataFrame and a provenance record of the segmentation.
* :func:`ocr_tei` — layout-aware OCR of page scans (via Tesseract when
  available, otherwise accepting text already present) encoded into a minimal
  **TEI-P5** skeleton written to both ``corpus['tei']`` and ``artifacts['xml']``.

Heavy / optional dependencies (``pytesseract`` + Pillow for OCR) are imported
lazily and degrade gracefully — the module never imports them at load time and
never reaches the network. All segmentation is deterministic.
"""
from __future__ import annotations

import importlib
import re
import unicodedata
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["build_corpus", "ocr_tei"]


def _try_import(name: str) -> Any | None:
    """Import ``name`` lazily, returning ``None`` if unavailable (never raises)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _as_doc_map(corpora: Any) -> dict[str, str]:
    """Coerce ``sources['corpora']`` into an ordered ``{doc_id: text}`` mapping.

    Accepts a ``{doc_id: text}`` dict, a list/tuple of texts (auto-numbered),
    or a single string. Non-string values are stringified; ``None`` → empty.
    """
    if corpora is None:
        return {}
    if isinstance(corpora, dict):
        return {str(k): ("" if v is None else str(v)) for k, v in corpora.items()}
    if isinstance(corpora, (list, tuple)):
        return {f"doc{i:04d}": ("" if t is None else str(t)) for i, t in enumerate(corpora)}
    return {"doc0000": str(corpora)}


# paragraph = one or more blank lines; sentence = terminal punctuation (CN + EN)
_PARA_SPLIT = re.compile(r"\n[ \t]*\n+")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])(?=\s)")


def _segment(text: str, unit: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, span_text)`` triples over ``text`` in char offsets.

    ``unit='paragraph'`` splits on blank lines; ``unit='sentence'`` further
    splits each paragraph on terminal punctuation; ``unit='document'`` keeps the
    whole text as one unit. Offsets index the *normalized* text and are exact
    (empty/whitespace-only spans are dropped, real offsets preserved).
    """
    if not text:
        return []
    if unit == "document":
        return [(0, len(text), text)]

    spans: list[tuple[int, int, str]] = []
    pos = 0
    for para in _PARA_SPLIT.split(text):
        start = text.find(para, pos)
        if start < 0:  # pragma: no cover - defensive
            start = pos
        end = start + len(para)
        pos = end
        if unit == "paragraph":
            if para.strip():
                spans.append((start, end, para))
            continue
        # sentence granularity within this paragraph
        spos = start
        for piece in _SENT_SPLIT.split(para):
            if not piece:
                continue
            s = text.find(piece, spos)
            if s < 0:  # pragma: no cover - defensive
                s = spos
            e = s + len(piece)
            spos = e
            if piece.strip():
                spans.append((s, e, piece))
    return spans


@register(
    name="build_corpus",
    aliases=["语料构建", "corpus_builder"],
    category="prep",
    tier="community",
    skill="corpus-builder",
    languages=["Python"],
    key_tools=["pandas", "regex", "unicodedata"],
    description="清洗规范化(NFC)原始文本并切成带 unit_id 与字符偏移的可编码单元及 manifest",
    requires={"sources": ["corpora"]},
    produces={"corpus": ["documents", "units", "manifest"], "evidence": ["provenance"]},
    auto_fix="escalate",
)
def build_corpus(state: StudyState, **kwargs: Any) -> StudyState:
    """Normalize and segment ``sources['corpora']`` into addressable units.

    Parameters (via ``kwargs``)
    ---------------------------
    data : dict | list | str, optional
        Raw corpora, overriding ``state.sources['corpora']``.
    unit : {'sentence', 'paragraph', 'document'}, default 'paragraph'
        Segmentation granularity.
    normalize : str, default 'NFC'
        Unicode normalization form applied before segmentation.
    """
    corpora = kwargs.get("data", state.sources.get("corpora"))
    unit = str(kwargs.get("unit", "paragraph")).lower()
    if unit not in {"sentence", "paragraph", "document"}:
        unit = "paragraph"
    form = str(kwargs.get("normalize", "NFC")).upper()
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        form = "NFC"

    doc_map = _as_doc_map(corpora)
    documents: dict[str, str] = {}
    units: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for doc_id, raw in doc_map.items():
        text = unicodedata.normalize(form, raw)
        documents[doc_id] = text
        spans = _segment(text, unit)
        for start, end, span_text in spans:
            units.append(
                {
                    "unit_id": f"{doc_id}:{start}-{end}",
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                    "text": span_text,
                }
            )
        rows.append({"doc_id": doc_id, "n_units": len(spans), "n_chars": len(text)})

    manifest = pd.DataFrame(rows, columns=["doc_id", "n_units", "n_chars"])

    state.write("corpus", "documents", documents)
    state.write("corpus", "units", units)
    state.write("corpus", "manifest", manifest)
    state.write(
        "evidence",
        "provenance",
        {
            "step": "build_corpus",
            "normalize": form,
            "segmentation": unit,
            "n_documents": len(documents),
            "n_units": len(units),
        },
    )
    return state


def _tei_p5(doc_id: str, text: str, title: str = "") -> str:
    """Wrap ``text`` in a minimal, well-formed TEI-P5 document skeleton.

    Paragraphs (blank-line separated) become ``<p>`` elements; if the text has
    no paragraph breaks the whole body is a single ``<p>``. All content is
    XML-escaped, so arbitrary OCR output is safe to embed.
    """
    heading = escape(title or doc_id)
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()] or [text.strip()]
    body = "\n".join(f"        <p>{escape(p)}</p>" for p in paras if p)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">\n'
        "  <teiHeader>\n"
        "    <fileDesc>\n"
        f"      <titleStmt><title>{heading}</title></titleStmt>\n"
        "      <publicationStmt><p>Encoded by socialverse ocr-tei-encoding.</p></publicationStmt>\n"
        f"      <sourceDesc><p>OCR / text of source {escape(doc_id)}.</p></sourceDesc>\n"
        "    </fileDesc>\n"
        "  </teiHeader>\n"
        "  <text>\n"
        "    <body>\n"
        f"{body}\n"
        "    </body>\n"
        "  </text>\n"
        "</TEI>"
    )


def _ocr_page(image: Any, pytesseract: Any) -> str:
    """Run Tesseract on one PIL image; return '' on any failure."""
    try:
        return str(pytesseract.image_to_string(image))
    except Exception:
        return ""


def _as_scan_map(scans: Any) -> dict[str, Any]:
    """Coerce ``sources['scans']`` into an ordered ``{doc_id: page_list}`` map.

    A page is either a text string (already-OCR'd / plain text) or an image
    handle (path or PIL image). Accepts a ``{doc_id: pages}`` dict, a flat list
    (single synthetic document), or a single page.
    """
    if scans is None:
        return {}
    if isinstance(scans, dict):
        out: dict[str, Any] = {}
        for k, v in scans.items():
            out[str(k)] = list(v) if isinstance(v, (list, tuple)) else [v]
        return out
    if isinstance(scans, (list, tuple)):
        return {"doc0000": list(scans)}
    return {"doc0000": [scans]}


@register(
    name="ocr_tei",
    aliases=["ocr", "OCR转TEI"],
    category="prep",
    tier="plus",
    skill="ocr-tei-encoding",
    languages=["Python", "XML(TEI)"],
    key_tools=["pytesseract", "Tesseract", "TEI-P5"],
    description="版面感知 OCR 扫描件(缺引擎则接受已有文本)并编码为 TEI-P5 骨架",
    requires={"sources": ["scans"]},
    produces={"corpus": ["documents", "tei"], "artifacts": ["xml"], "evidence": ["provenance"]},
    auto_fix="auto",
)
def ocr_tei(state: StudyState, **kwargs: Any) -> StudyState:
    """OCR page scans (when Tesseract is present) and encode them as TEI-P5.

    Parameters (via ``kwargs``)
    ---------------------------
    data : dict | list, optional
        Scans overriding ``state.sources['scans']``. Each page may be a text
        string (used verbatim) or an image (path / PIL image, OCR'd if possible).
    lang : str, default 'eng'
        Tesseract language code, when OCR is used.
    titles : dict, optional
        ``{doc_id: title}`` used for the TEI ``<title>`` element.
    """
    scans = kwargs.get("data", state.sources.get("scans"))
    lang = str(kwargs.get("lang", "eng"))
    titles = kwargs.get("titles") or {}

    pytesseract = _try_import("pytesseract")
    PIL = _try_import("PIL.Image")
    engine_ok = pytesseract is not None and PIL is not None

    scan_map = _as_scan_map(scans)
    documents: dict[str, str] = {}
    tei_map: dict[str, str] = {}
    xml_map: dict[str, str] = {}
    ocr_used = False

    for doc_id, pages in scan_map.items():
        page_texts: list[str] = []
        for page in pages:
            if isinstance(page, str) and "\n" not in page and _looks_like_path(page):
                # a path string: OCR it if we can, else keep the path text out
                if engine_ok:
                    img = _open_image(page, PIL)
                    if img is not None:
                        page_texts.append(_ocr_kwarg(pytesseract, img, lang))
                        ocr_used = True
                        continue
                # no engine → treat the bare path string as literal text fallback
                page_texts.append(page)
            elif isinstance(page, str):
                page_texts.append(page)  # already text
            else:
                # an image handle (PIL image object)
                if engine_ok:
                    page_texts.append(_ocr_kwarg(pytesseract, page, lang))
                    ocr_used = True
                else:
                    page_texts.append("")

        text = "\n\n".join(t for t in page_texts if t is not None)
        documents[doc_id] = text
        tei = _tei_p5(doc_id, text, str(titles.get(doc_id, "")))
        tei_map[doc_id] = tei
        xml_map[doc_id] = tei

    state.write("corpus", "documents", documents)
    state.write("corpus", "tei", tei_map)
    state.write("artifacts", "xml", xml_map)
    state.write(
        "evidence",
        "provenance",
        {
            "step": "ocr_tei",
            "engine": "tesseract" if (engine_ok and ocr_used) else "text-passthrough",
            "ocr_available": engine_ok,
            "lang": lang,
            "n_documents": len(documents),
        },
    )
    return state


def _looks_like_path(s: str) -> bool:
    """Heuristic: a short single-line string ending in an image extension."""
    return len(s) <= 260 and s.lower().rsplit(".", 1)[-1] in {
        "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"
    }


def _open_image(path: str, PIL: Any) -> Any | None:
    try:
        return PIL.open(path)
    except Exception:
        return None


def _ocr_kwarg(pytesseract: Any, image: Any, lang: str) -> str:
    try:
        return str(pytesseract.image_to_string(image, lang=lang))
    except Exception:
        return _ocr_page(image, pytesseract)
