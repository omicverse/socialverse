"""``sv.gov._gates`` — registered implementations for the *governance* gates.

Three registry entries back the governance skills — the macro checkpoints that a
social-science study must clear before (and after) it runs:

- :func:`ethics_check` (``ethics-check``) — the ethics macro-gate: IRB
  classification, informed-consent status, re-identifiability (a *real*
  **k-anonymity** computation over declared quasi-identifiers via
  ``pandas.groupby().size().min()``), and data-minimization → a ``PASS / FIX /
  NO-GO`` verdict.
- :func:`data_use_check` (``data-use-check``) — per-source copyright/licence
  triage into the *five buckets* (public-domain / Creative-Commons / publisher
  TDM / GLAM / platform-ToS), yielding scrape and redistribution decisions with
  the ``UNKNOWN → strictest`` default.
- :func:`ai_use_disclosure` (``ai-use-disclosure``) — audits an AI
  contribution log for the *accepted-but-unverified* red line and renders a
  paste-ready disclosure paragraph from the target journal's policy family
  (ICMJE / COPE / …).

All three speak the 12-slot :class:`~socialverse._state.StudyState` vocabulary
through the ``@register`` contract, so the resolver can gate a plan on them.
Every computation is real (k-anonymity over a DataFrame, deterministic bucket
triage, log audit + table assembly) — no placeholder strings. Optional deps are
imported lazily and degraded gracefully; nothing here imports heavy libraries or
touches the network at module import time.
"""
from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["ethics_check", "data_use_check", "ai_use_disclosure"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import — returns the module or ``None`` if unavailable."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _as_frame(data: Any) -> pd.DataFrame | None:
    """Coerce a dataset (DataFrame / dict-of-frames / mapping / array) to a frame."""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        # dict of frames → first frame; else treat as column mapping
        inner = next(iter(data.values()), None)
        if isinstance(inner, pd.DataFrame):
            return inner
        try:
            return pd.DataFrame(data)
        except Exception:
            return None
    try:
        return pd.DataFrame(data)
    except Exception:
        return None


def _k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str]) -> tuple[int, dict[str, Any]]:
    """Smallest equivalence-class size over the quasi-identifier columns.

    k = ``df.groupby(QIs).size().min()`` — a real re-identifiability metric: every
    record shares its QI-tuple with at least ``k−1`` others, so no combination of
    the declared quasi-identifiers isolates fewer than ``k`` people.
    """
    cols = [c for c in quasi_identifiers if c in df.columns]
    if not cols or df.empty:
        return -1, {"quasi_identifiers": cols, "n_records": int(len(df)),
                    "note": "no valid quasi_identifier columns present"}
    grp = df.groupby(cols, dropna=False, observed=True).size()
    k = int(grp.min())
    n_unique = int((grp == 1).sum())
    return k, {
        "quasi_identifiers": cols,
        "n_records": int(len(df)),
        "n_equivalence_classes": int(grp.shape[0]),
        "n_unique_records": n_unique,      # equivalence classes of size 1 (k=1)
        "share_unique": round(n_unique / float(len(df)), 4) if len(df) else float("nan"),
    }


def _verdict(checks: list[dict]) -> str:
    """Aggregate a list of ``{status: PASS|FIX|NO-GO}`` checks into a gate verdict."""
    statuses = {c.get("status") for c in checks}
    if "NO-GO" in statuses:
        return "NO-GO"
    if "FIX" in statuses:
        return "FIX"
    return "PASS"


# ===================================================================== ethics
@register(
    name="ethics_check",
    aliases=["伦理审查", "ethics_check"],
    category="governance",
    tier="community",
    skill="ethics-check",
    languages=["Python"],
    key_tools=["pandas", "k-anonymity"],
    description="伦理宏观闸门:IRB 分类/知情同意/可识别性(k-匿名)/最小化 → go/fix/no-go",
    requires={"design": ["unit"]},
    produces={"governance": ["ethics"]},
    auto_fix="escalate",
)
def ethics_check(state: StudyState, **kwargs: Any) -> StudyState:
    """Ethics macro-gate over the study design and (optionally) the microdata.

    Runs four checks and folds them into a single ``PASS / FIX / NO-GO`` verdict:

    1. **IRB** — human-subjects classification (exempt / expedited / full) and
       whether a determination has been recorded.
    2. **Consent** — informed-consent status for the unit of analysis.
    3. **Re-identifiability** — a real **k-anonymity** computation over the
       declared ``quasi_identifiers`` against the required threshold ``k_threshold``.
    4. **Minimization** — whether direct identifiers were dropped / the analysis
       is restricted to the variables it needs.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame | dict, optional
        Microdata for the k-anonymity computation (defaults to
        ``state.sources['datasets']``).
    quasi_identifiers : list[str], optional
        Columns whose combination could re-identify a subject (e.g. age, ZIP, sex).
    k_threshold : int, optional
        Minimum acceptable equivalence-class size (default ``5``).
    human_subjects : bool, optional
        Whether the study involves human subjects (default inferred: ``True``
        unless the analysis unit is clearly non-human, e.g. ``country``/``firm``).
    irb : str, optional
        Recorded IRB determination — ``exempt`` / ``expedited`` / ``full`` /
        ``approved`` / ``pending`` / ``none``.
    consent : str, optional
        Consent basis — ``informed`` / ``waiver`` / ``public`` / ``broad`` /
        ``none``.
    minimized : bool, optional
        Whether direct identifiers have been removed / data minimized.
    """
    unit = state.design.get("unit")
    k_threshold = int(kwargs.get("k_threshold", 5))
    checks: list[dict] = []

    # 1. IRB / human-subjects classification --------------------------------
    non_human_units = {"country", "firm", "organization", "org", "state", "nation",
                       "region", "county", "municipality", "document", "text", "corpus"}
    human_subjects = kwargs.get("human_subjects")
    if human_subjects is None:
        human_subjects = str(unit).lower() not in non_human_units
    irb = str(kwargs.get("irb", state.governance.get("irb") or "none")).lower()
    if not human_subjects:
        checks.append({"check": "irb", "status": "PASS",
                       "detail": f"unit '{unit}' is not human subjects — IRB review typically N/A"})
    elif irb in {"exempt", "expedited", "full", "approved"}:
        checks.append({"check": "irb", "status": "PASS",
                       "detail": f"IRB determination on record: {irb}"})
    elif irb == "pending":
        checks.append({"check": "irb", "status": "FIX",
                       "detail": "IRB determination pending — obtain before data collection"})
    else:
        checks.append({"check": "irb", "status": "NO-GO",
                       "detail": "human subjects but no IRB determination recorded"})

    # 2. Informed consent ----------------------------------------------------
    consent = str(kwargs.get("consent", state.governance.get("consent") or "none")).lower()
    if not human_subjects:
        checks.append({"check": "consent", "status": "PASS",
                       "detail": "no human subjects — consent N/A"})
    elif consent in {"informed", "broad", "public"}:
        checks.append({"check": "consent", "status": "PASS",
                       "detail": f"consent basis: {consent}"})
    elif consent == "waiver":
        checks.append({"check": "consent", "status": "FIX",
                       "detail": "consent waiver claimed — document IRB-approved justification"})
    else:
        checks.append({"check": "consent", "status": "NO-GO",
                       "detail": "human subjects but no consent basis recorded"})

    # 3. Re-identifiability — real k-anonymity ------------------------------
    data = kwargs.get("data")
    if data is None:
        data = state.sources.get("datasets")
    df = _as_frame(data)
    quasi = kwargs.get("quasi_identifiers") or kwargs.get("qi") or []
    if isinstance(quasi, str):
        quasi = [quasi]
    quasi = list(quasi)

    if df is None or not quasi:
        k = None
        k_detail = {"quasi_identifiers": quasi,
                    "note": "no microdata or quasi_identifiers supplied — k-anonymity not computed"}
        checks.append({"check": "k_anonymity", "status": "FIX",
                       "detail": "declare quasi_identifiers + supply microdata to test re-identifiability"})
    else:
        k, k_detail = _k_anonymity(df, quasi)
        k_detail["k_threshold"] = k_threshold
        if k < 0:
            checks.append({"check": "k_anonymity", "status": "FIX",
                           "detail": "quasi_identifier columns not found in data"})
        elif k >= k_threshold:
            checks.append({"check": "k_anonymity", "status": "PASS",
                           "detail": f"k={k} ≥ threshold {k_threshold}"})
        elif k <= 1:
            checks.append({"check": "k_anonymity", "status": "NO-GO",
                           "detail": f"k={k}: unique records are directly re-identifiable "
                                     f"({k_detail.get('n_unique_records')} singletons)"})
        else:
            checks.append({"check": "k_anonymity", "status": "FIX",
                           "detail": f"k={k} < threshold {k_threshold}: generalize/suppress QIs "
                                     f"or coarsen to reach k≥{k_threshold}"})
    k_detail["k"] = k

    # 4. Data minimization ---------------------------------------------------
    minimized = kwargs.get("minimized")
    if minimized is None:
        minimized = bool(state.governance.get("pii_status"))
    if minimized:
        checks.append({"check": "minimization", "status": "PASS",
                       "detail": "direct identifiers removed / analysis restricted to needed variables"})
    else:
        checks.append({"check": "minimization", "status": "FIX",
                       "detail": "confirm direct identifiers dropped and variables minimized to need"})

    verdict = _verdict(checks)
    ethics = {
        "verdict": verdict,
        "checks": checks,
        "k_anonymity": k_detail,
        "unit": unit,
        "human_subjects": bool(human_subjects),
        "policy": "escalate: FIX/NO-GO require human review before proceeding",
    }
    state.write("governance", "ethics", ethics)
    return state


# =================================================================== data-use
#: the five licence buckets, ordered strict → permissive, with default rights.
_LICENCE_BUCKETS: dict[str, dict[str, Any]] = {
    "platform_tos": {
        "label": "Platform Terms-of-Service (社交/商业平台)",
        "can_scrape": False, "redistribution": "prohibited", "attribution": True,
        "note": "governed by ToS/API contract — scraping usually breaches ToS; "
                "share derived aggregates only, never raw content",
    },
    "publisher_tdm": {
        "label": "Publisher Text-and-Data-Mining licence",
        "can_scrape": True, "redistribution": "derived_only", "attribution": True,
        "note": "TDM rights under subscription/licence — mine locally, redistribute "
                "only non-consumptive derivatives, never the full text",
    },
    "glam": {
        "label": "GLAM (Galleries/Libraries/Archives/Museums)",
        "can_scrape": True, "redistribution": "conditional", "attribution": True,
        "note": "check per-item rights statement (rightsstatements.org); metadata "
                "often open, digitized objects may carry viral/NC terms",
    },
    "cc": {
        "label": "Creative Commons",
        "can_scrape": True, "redistribution": "share_alike_or_by", "attribution": True,
        "note": "redistribution allowed under the specific CC terms — honor BY / "
                "SA / NC / ND obligations of the exact version",
    },
    "public_domain": {
        "label": "Public Domain / CC0",
        "can_scrape": True, "redistribution": "unrestricted", "attribution": False,
        "note": "no copyright restrictions — reuse and redistribute freely "
                "(scholarly attribution still good practice)",
    },
}

#: substring signals → bucket, most-specific first.
_LICENCE_SIGNALS: list[tuple[tuple[str, ...], str]] = [
    (("cc0", "public domain", "publicdomain", "pd-", "no known copyright",
      "us government work"), "public_domain"),
    (("cc-by", "cc by", "creativecommons", "creative commons", "cc-sa", "cc-nc",
      "cc-nd", "attribution-share"), "cc"),
    (("tdm", "text and data mining", "text-and-data-mining", "non-consumptive",
      "publisher licen", "subscription"), "publisher_tdm"),
    (("glam", "gallery", "library", "archive", "museum", "europeana", "dpla",
      "rightsstatements"), "glam"),
    (("terms of service", "tos", "api terms", "developer policy", "twitter",
      "facebook", "meta ", "reddit", "tiktok", "platform"), "platform_tos"),
]


def _classify_licence(text: str) -> str:
    """Map a free-text licence string to one of the five buckets (UNKNOWN → strictest)."""
    t = (text or "").strip().lower()
    if not t:
        return "platform_tos"  # UNKNOWN defaults to the strictest bucket
    for signals, bucket in _LICENCE_SIGNALS:
        if any(s in t for s in signals):
            return bucket
    return "platform_tos"


def _triage_source(name: str, licence: str) -> dict[str, Any]:
    """Triage one source into a bucket and derive its scrape/redistribution rights."""
    known = bool((licence or "").strip())
    bucket = _classify_licence(licence)
    rights = dict(_LICENCE_BUCKETS[bucket])
    flags: list[str] = []
    if not known:
        flags.append("UNKNOWN licence — defaulted to strictest (platform_tos); "
                     "resolve rights before scraping or sharing")
    lic_l = (licence or "").lower()
    if "nc" in lic_l or "noncommercial" in lic_l or "non-commercial" in lic_l:
        flags.append("NonCommercial clause — no commercial reuse")
    if "nd" in lic_l or "noderiv" in lic_l:
        flags.append("NoDerivatives clause — cannot redistribute modified versions")
    if "sa" in lic_l or "sharealike" in lic_l or "share-alike" in lic_l:
        flags.append("ShareAlike clause — derivatives inherit the same licence")
    return {
        "source": name,
        "licence": licence or "(unspecified)",
        "bucket": bucket,
        "bucket_label": rights["label"],
        "can_scrape": bool(rights["can_scrape"]),
        "redistribution": rights["redistribution"],
        "attribution": bool(rights["attribution"]),
        "note": rights["note"],
        "flags": flags,
    }


@register(
    name="data_use_check",
    aliases=["数据合规", "data_use"],
    category="governance",
    tier="community",
    skill="data-use-check",
    languages=["无代码(方法论)"],
    key_tools=["五桶许可分诊"],
    description="逐源版权/许可分诊:五桶(公有域/CC/出版商TDM/GLAM/平台ToS)+抓取决策+再分发义务",
    requires={"sources": ["datasets"]},
    produces={"governance": ["data_use"]},
    auto_fix="escalate",
)
def data_use_check(state: StudyState, **kwargs: Any) -> StudyState:
    """Per-source copyright/licence triage into the five buckets.

    Each declared source is classified into one of
    ``public_domain / cc / publisher_tdm / glam / platform_tos`` from its licence
    string, yielding ``can_scrape`` / ``redistribution`` / ``attribution``
    decisions and obligation flags (NC / ND / SA). An unspecified licence defaults
    to the **strictest** bucket (``platform_tos``) rather than assuming permission.

    Parameters (via ``kwargs``)
    ---------------------------
    license : str | dict | list, optional
        Licence descriptor(s). A single string classifies one source; a mapping
        ``{source: licence}`` or a list of ``{"source":…, "license":…}`` records
        classifies many. Defaults to ``state.sources['datasets']`` when it carries
        licence hints, else a single UNKNOWN source.
    records : list[dict], optional
        Alias for a list of ``{source, license}`` records.
    sources : list[str], optional
        Source names (paired positionally with a list ``license`` if given).
    """
    licence_arg = kwargs.get("license", kwargs.get("licence"))
    records = kwargs.get("records")
    src_names = kwargs.get("sources")

    triaged: list[dict[str, Any]] = []

    if records:
        for r in records:
            triaged.append(_triage_source(
                str(r.get("source", r.get("name", f"source_{len(triaged)+1}"))),
                str(r.get("license", r.get("licence", ""))),
            ))
    elif isinstance(licence_arg, dict):
        for name, lic in licence_arg.items():
            triaged.append(_triage_source(str(name), str(lic or "")))
    elif isinstance(licence_arg, (list, tuple)):
        names = list(src_names) if isinstance(src_names, (list, tuple)) else []
        for i, lic in enumerate(licence_arg):
            if isinstance(lic, dict):
                triaged.append(_triage_source(
                    str(lic.get("source", lic.get("name", f"source_{i+1}"))),
                    str(lic.get("license", lic.get("licence", ""))),
                ))
            else:
                nm = str(names[i]) if i < len(names) else f"source_{i+1}"
                triaged.append(_triage_source(nm, str(lic or "")))
    elif isinstance(licence_arg, str):
        nm = str(src_names[0]) if isinstance(src_names, (list, tuple)) and src_names else "source_1"
        triaged.append(_triage_source(nm, licence_arg))
    else:
        # Fall back to the registered sources; classify their licence hint if any.
        datasets = state.sources.get("datasets")
        lic_hint = ""
        if isinstance(datasets, dict):
            lic_hint = str(datasets.get("license", datasets.get("licence", "")))
            name = str(datasets.get("name", "dataset_1"))
        else:
            name = "dataset_1"
        triaged.append(_triage_source(name, lic_hint))

    # Whole-plan rights = intersection of per-source rights (weakest link wins).
    can_scrape = all(t["can_scrape"] for t in triaged) if triaged else False
    redist_rank = {"prohibited": 0, "derived_only": 1, "conditional": 2,
                   "share_alike_or_by": 3, "unrestricted": 4}
    redistribution = min((t["redistribution"] for t in triaged),
                         key=lambda r: redist_rank.get(r, 0)) if triaged else "prohibited"
    flags = sorted({f for t in triaged for f in t["flags"]})
    buckets = sorted({t["bucket"] for t in triaged})

    data_use = {
        "bucket": buckets[0] if len(buckets) == 1 else buckets,
        "can_scrape": bool(can_scrape),
        "redistribution": redistribution,
        "attribution": any(t["attribution"] for t in triaged),
        "flags": flags,
        "per_source": triaged,
        "n_sources": len(triaged),
        "policy": "escalate: UNKNOWN or platform_tos sources need rights clearance",
    }
    state.write("governance", "data_use", data_use)
    return state


# ============================================================= ai-disclosure
#: journal-policy families → a paste-ready disclosure template.
_DISCLOSURE_TEMPLATES: dict[str, str] = {
    "icmje": (
        "During the preparation of this work the author(s) used {tools} in order to "
        "{purposes}. After using {tool_names}, the author(s) reviewed and edited the "
        "content as needed and take(s) full responsibility for the content of the "
        "publication. Generative AI was not listed as an author."
    ),
    "cope": (
        "The author(s) declare(s) the use of generative AI and AI-assisted "
        "technologies in the writing process. Specifically, {tools} were used to "
        "{purposes}. All AI-assisted output was verified by the author(s), who "
        "take(s) full responsibility for the integrity of the work. AI tools do not "
        "meet authorship criteria and are not credited as authors."
    ),
    "nature": (
        "The author(s) used {tools} to {purposes}. The use of these tools is "
        "documented in the Methods section. Large language models do not satisfy "
        "the criteria for authorship and are not listed as authors; the author(s) "
        "take(s) full responsibility for all content."
    ),
    "none": (
        "AI-use disclosure: {tools} were used to {purposes}. All output was "
        "human-verified and the author(s) take(s) full responsibility."
    ),
}

#: policy aliases → template family.
_POLICY_ALIAS: dict[str, str] = {
    "icmje": "icmje", "jama": "icmje", "nejm": "icmje", "bmj": "icmje",
    "cope": "cope", "elsevier": "cope", "sage": "cope", "wiley": "cope",
    "nature": "nature", "springer": "nature", "science": "nature",
    "": "none", "none": "none", "generic": "none",
}


def _render_disclosure(policy: str, log: list[dict]) -> str:
    """Render a paste-ready disclosure paragraph from the policy family and log."""
    family = _POLICY_ALIAS.get(str(policy or "").strip().lower(), "none")
    template = _DISCLOSURE_TEMPLATES[family]
    tool_names = sorted({str(e.get("tool", "an AI assistant")) for e in log}) or ["an AI assistant"]
    stages = sorted({str(e.get("stage", "the writing process")) for e in log}) or ["the writing process"]
    tools_phrase = ", ".join(tool_names)
    purposes_phrase = "assist with " + ", ".join(stages)
    if not log:
        return "No generative-AI use to disclose for this work."
    return template.format(tools=tools_phrase, tool_names=tools_phrase,
                           purposes=purposes_phrase)


@register(
    name="ai_use_disclosure",
    aliases=["AI披露", "ai_disclosure"],
    category="governance",
    tier="community",
    skill="ai-use-disclosure",
    languages=["Python"],
    key_tools=["csv", "ICMJE/COPE"],
    description="AI 使用披露:贡献日志 audit + 按期刊政策族渲染 paste-ready 声明",
    requires={},
    produces={
        "governance": ["ai_disclosure"],
        "artifacts": ["tables"],
        "evidence": ["provenance"],
    },
    auto_fix="escalate",
)
def ai_use_disclosure(state: StudyState, **kwargs: Any) -> StudyState:
    """Audit an AI-contribution log and render a journal-ready disclosure.

    Reads a per-stage AI-use log, audits it for the **accepted-but-unverified**
    red line (content accepted into the manuscript without human verification —
    an integrity escalation), and renders a paste-ready disclosure paragraph from
    the target journal's policy family (ICMJE / COPE / Nature / generic).

    Parameters (via ``kwargs``)
    ---------------------------
    ai_log : list[dict], optional
        Records ``{stage, tool, accepted: bool, verified: bool}`` — one per point
        where a generative-AI tool contributed. Defaults to
        ``state.governance['ai_log']`` if present, else empty.
    policy : str, optional
        Target journal / policy family — ``ICMJE`` / ``COPE`` / ``Nature`` / … .
        Defaults to ``ICMJE``.
    """
    log = kwargs.get("ai_log")
    if log is None:
        log = state.governance.get("ai_log") or []
    log = [dict(e) for e in log] if log else []
    policy = kwargs.get("policy", "ICMJE")

    # -- audit: the accepted-but-unverified red line ------------------------
    rows: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for i, e in enumerate(log):
        accepted = bool(e.get("accepted", False))
        verified = bool(e.get("verified", False))
        red_line = accepted and not verified
        row = {
            "stage": e.get("stage", f"stage_{i+1}"),
            "tool": e.get("tool", "unspecified"),
            "accepted": accepted,
            "verified": verified,
            "flag": "accepted-but-unverified" if red_line else "",
        }
        rows.append(row)
        if red_line:
            unverified.append(row)

    if unverified:
        status = "ESCALATE"
        detail = (f"{len(unverified)} AI contribution(s) accepted without verification — "
                  "verify or remove before submission (research-integrity red line)")
    elif log:
        status = "PASS"
        detail = "all accepted AI contributions were human-verified"
    else:
        status = "PASS"
        detail = "no generative-AI use recorded — nothing to disclose"

    statement = _render_disclosure(policy, log)
    audit = {
        "status": status,
        "detail": detail,
        "n_entries": len(log),
        "n_accepted": int(sum(r["accepted"] for r in rows)),
        "n_unverified_accepted": len(unverified),
        "unverified": unverified,
    }

    ai_disclosure = {
        "audit": audit,
        "policy": str(policy),
        "policy_family": _POLICY_ALIAS.get(str(policy or "").strip().lower(), "none"),
        "statement": statement,
    }

    log_table = pd.DataFrame(
        rows, columns=["stage", "tool", "accepted", "verified", "flag"]
    )
    provenance = {
        "kind": "ai_use_disclosure",
        "policy": str(policy),
        "statement": statement,
        "audit_status": status,
    }

    state.write("governance", "ai_disclosure", ai_disclosure)
    state.write("artifacts", "tables", log_table)
    state.write("evidence", "provenance", provenance)
    return state
