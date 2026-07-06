"""sv.pp._redact — registered implementation of the ``pii-redaction`` skill.

Detect and remove personally identifying information (PII) from a text corpus
before any downstream analysis or sharing. The core detectors are deterministic
regex passes (email / phone / long numeric IDs such as 身份证号) that always run;
person names are additionally picked up by a lazily-imported spaCy NER model when
one is installed (gracefully skipped otherwise — the module never imports spaCy at
load time and never touches the network).

Redaction is *consistent* and *reversible*: every distinct raw entity is mapped to
a stable pseudonym (``[EMAIL_1]``, ``[PHONE_2]``, ``[PERSON_1]`` …) so the same
person reads as the same token everywhere, and a ``crosswalk`` dict (pseudonym →
original) is retained so an authorized holder can re-identify if governance allows.

Writes the scrubbed text back into ``corpus['documents']`` and records a
``governance['pii_status']`` receipt ``{"method", "crosswalk_size", "date"}`` — the
compliance breadcrumb the evidence spine expects.
"""
from __future__ import annotations

import importlib
import re
from datetime import date
from typing import Any

from .._registry import register
from .._state import StudyState

__all__ = ["redact_pii"]


# --------------------------------------------------------------------------- utils
def _try_import(name: str) -> Any | None:
    """Lazily import ``name``; return the module or ``None`` if unavailable."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# Regex detectors, ordered so that greedier/composite patterns match first.
# Each entry: (label, compiled pattern). Longer numeric IDs before phones so an
# 18-digit 身份证 is not chewed up by the phone matcher.
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Mainland China resident ID: a fixed 18-char block (17 digits + digit/X), no
# separators — matched before phones so it is labelled ID, not PHONE.
_ID_CN = re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])")
# Phones: optional +country code then grouped separators; run before the generic
# long-ID pass so a "+86 138-1234-5678" is labelled PHONE, not a numeric ID.
_PHONE = re.compile(r"(?<![\w.+])\+?\d[\d\s\-().]{5,}\d(?![\w])")
# Generic long numeric identifier (>= 13 contiguous-ish digits, allowing spaced or
# hyphenated runs) — a catch-all for account/case numbers phones did not claim.
_LONG_ID = re.compile(r"(?<![\d\-])\d(?:[ \-]?\d){12,}(?![\d])")

_REGEX_DETECTORS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", _EMAIL),
    ("ID", _ID_CN),
    ("PHONE", _PHONE),
    ("ID", _LONG_ID),
]


def _digit_count(s: str) -> int:
    return sum(c.isdigit() for c in s)


class _Pseudonymizer:
    """Assign stable per-label pseudonyms and keep a reverse crosswalk."""

    def __init__(self) -> None:
        # original text -> pseudonym (so a repeated entity reuses its token)
        self._forward: dict[str, str] = {}
        # pseudonym -> original text (the reversible crosswalk)
        self.crosswalk: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, label: str, original: str) -> str:
        original = original.strip()
        if original in self._forward:
            return self._forward[original]
        self._counters[label] = self._counters.get(label, 0) + 1
        token = f"[{label}_{self._counters[label]}]"
        self._forward[original] = token
        self.crosswalk[token] = original
        return token


def _redact_regex(text: str, pseudo: _Pseudonymizer) -> str:
    """Replace every regex-detected PII span with a consistent pseudonym."""
    for label, pattern in _REGEX_DETECTORS:

        def _sub(m: "re.Match[str]", _label: str = label) -> str:
            raw = m.group(0)
            # Phones need enough digits to be plausible; skip short number runs.
            if _label == "PHONE" and _digit_count(raw) < 7:
                return raw
            return pseudo.token_for(_label, raw)

        text = pattern.sub(_sub, text)
    return text


def _redact_names(text: str, pseudo: _Pseudonymizer, nlp: Any) -> str:
    """Replace spaCy PERSON entities with pseudonyms (longest span first)."""
    try:
        doc = nlp(text)
        persons = [
            (ent.start_char, ent.end_char, ent.text)
            for ent in doc.ents
            if ent.label_ in {"PERSON", "PER"}
        ]
    except Exception:
        return text
    # Replace from the end so earlier offsets stay valid.
    for start, end, raw in sorted(persons, key=lambda t: t[0], reverse=True):
        token = pseudo.token_for("PERSON", raw)
        text = text[:start] + token + text[end:]
    return text


def _load_spacy() -> Any | None:
    """Return a loaded spaCy pipeline, or ``None`` if unavailable.

    Tries a real NER model first, then a blank pipeline (regex-only names). Never
    downloads: a missing model just means the name pass is skipped.
    """
    spacy = _try_import("spacy")
    if spacy is None:
        return None
    for model in ("zh_core_web_sm", "en_core_web_sm"):
        try:
            return spacy.load(model)
        except Exception:
            continue
    return None


# ------------------------------------------------------------------------ register
@register(
    name="redact_pii",
    aliases=["去标识", "pii_redaction"],
    category="prep",
    tier="community",
    skill="pii-redaction",
    languages=["Python"],
    key_tools=["regex", "spaCy"],
    description="检测并去除语料 PII(邮箱/电话/身份证/姓名),一致假名替换并保留可逆 crosswalk",
    requires={"corpus": ["documents"]},
    produces={"corpus": ["documents"], "governance": ["pii_status"]},
    auto_fix="auto",
)
def redact_pii(state: StudyState, **kwargs: Any) -> StudyState:
    """Scrub PII from ``corpus['documents']`` with consistent, reversible pseudonyms.

    Parameters (via ``kwargs``)
    ---------------------------
    documents:
        Optional explicit documents to redact. Accepts a mapping ``{doc_id: text}``,
        a sequence of strings, or a single string. Defaults to
        ``state.corpus['documents']`` (guaranteed present by ``requires``).
    use_ner:
        If ``True`` (default) and spaCy + a model are installed, add a person-name
        pass on top of the always-on regex detectors.
    """
    use_ner: bool = bool(kwargs.get("use_ner", True))

    docs = kwargs.get("documents", state.corpus["documents"])

    # Normalize input shape to an ordered mapping of id -> text.
    if isinstance(docs, str):
        items: list[tuple[Any, str]] = [(0, docs)]
        was_scalar, was_mapping = True, False
    elif isinstance(docs, dict):
        items = list(docs.items())
        was_scalar, was_mapping = False, True
    else:  # sequence of strings (or of objects coerced to str)
        items = list(enumerate(docs or []))
        was_scalar, was_mapping = False, False

    pseudo = _Pseudonymizer()
    nlp = _load_spacy() if use_ner else None

    redacted: list[tuple[Any, str]] = []
    for doc_id, raw in items:
        text = "" if raw is None else str(raw)
        text = _redact_regex(text, pseudo)
        if nlp is not None:
            text = _redact_names(text, pseudo, nlp)
        redacted.append((doc_id, text))

    # Rebuild the documents container in its original shape.
    if was_scalar:
        out_documents: Any = redacted[0][1] if redacted else ""
    elif was_mapping:
        out_documents = {k: v for k, v in redacted}
    else:
        out_documents = [v for _, v in redacted]

    method = "regex+spaCy-NER" if nlp is not None else "regex"
    pii_status: dict[str, Any] = {
        "method": method,
        "crosswalk_size": len(pseudo.crosswalk),
        "date": date.today().isoformat(),
    }

    state.write("corpus", "documents", out_documents)
    # Retain the reversible mapping alongside the compliance receipt; keeping it
    # inside governance means re-identification is gated by the same slot that
    # records that redaction happened at all.
    state.write("governance", "pii_status", pii_status)
    state.write("governance", "pii_crosswalk", pseudo.crosswalk)
    return state
