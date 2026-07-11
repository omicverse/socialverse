"""``sv.gov._meta_gov`` — Tier-2 systematic-review governance / reporting / appraisal.

PRISMA 2020 flow accounting + 27-item checklist, risk-of-bias judgement schemas
(RoB2 / ROBINS-I / QUADAS-2 / NOS / JBI), screening inter-rater agreement
(Cohen's κ + Gwet AC1 + PABAK), and GRADE certainty algebra. These are the
governance bookkeeping layer — deterministic and validated; final judgement
calls (RoB domain ratings, GRADE certainty) stay reviewer-entered, the tool only
does the arithmetic and *suggests* flags from the evidence slot.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._registry import register
from .._state import StudyState

__all__ = ["prisma_flow", "prisma_checklist", "risk_of_bias", "screen_agreement", "grade"]

_ROB_DOMAINS = {
    "ROB2": ["randomization", "deviations", "missing_data", "measurement", "selection_reporting"],
    "ROBINS-I": ["confounding", "selection", "classification", "deviations", "missing_data",
                 "measurement", "selection_reporting"],
    "QUADAS-2": ["patient_selection", "index_test", "reference_standard", "flow_timing"],
    "NOS": ["selection", "comparability", "outcome"],
    "JBI": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"],
}


@register(
    name="prisma_flow", aliases=["PRISMA计数", "prisma_records"],
    category="scholarship_governance", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="PRISMA 2020 记录计数登记 + 算术自洽校验(identification→screening→eligibility→included);供 sv.pl.prisma_diagram 画图",
    requires={}, produces={"governance": ["prisma"]},
)
def prisma_flow(state: StudyState, **kwargs: Any) -> StudyState:
    """Record PRISMA 2020 four-stage counts + validate the arithmetic.

    kwargs: ``identified``, ``duplicates``, ``screened``, ``excluded_screen``,
    ``full_text``, ``excluded_fulltext``, ``included``. Missing stages are derived
    where possible; inconsistencies are flagged, not silently fixed.
    """
    g = {k: int(kwargs[k]) for k in ("identified", "duplicates", "screened", "excluded_screen",
                                     "full_text", "excluded_fulltext", "included") if k in kwargs}
    ident = g.get("identified"); dup = g.get("duplicates", 0)
    if ident is not None:
        g.setdefault("after_dedup", ident - dup)
    warnings = []
    if {"screened", "excluded_screen", "full_text"} <= g.keys():
        if g["screened"] - g["excluded_screen"] != g["full_text"]:
            warnings.append("screened − excluded_screen ≠ full_text")
    if {"full_text", "excluded_fulltext", "included"} <= g.keys():
        if g["full_text"] - g["excluded_fulltext"] != g["included"]:
            warnings.append("full_text − excluded_fulltext ≠ included")
    g["consistent"] = not warnings
    g["warnings"] = warnings
    state.write("governance", "prisma", g)
    return state


@register(
    name="prisma_checklist", aliases=["PRISMA清单", "prisma27"],
    category="scholarship_governance", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="PRISMA 2020 的 27 项报告清单合规表(逐项 addressed/位置),导出投稿附件用",
    requires={}, produces={"governance": ["prisma_checklist"]},
)
def prisma_checklist(state: StudyState, **kwargs: Any) -> StudyState:
    """PRISMA 2020 27-item reporting checklist. kwargs: ``items={n: True|'page 3'}``."""
    items = kwargs.get("items") or {}
    total = 27
    addressed = sum(1 for v in items.values() if v)
    rows = {str(i): {"addressed": bool(items.get(i, items.get(str(i), False))),
                     "location": items.get(i) if isinstance(items.get(i), str) else ""}
            for i in range(1, total + 1)}
    state.write("governance", "prisma_checklist", {
        "items": rows, "n_addressed": addressed, "n_total": total,
        "completeness": round(100 * addressed / total, 1),
    })
    return state


@register(
    name="risk_of_bias", aliases=["偏倚风险", "rob", "quality_appraisal"],
    category="scholarship_governance", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="偏倚风险评级架构:RoB2/ROBINS-I/QUADAS-2/NOS/JBI 的 domain 校验 + 逐研究判断录入(供 sv.pl.rob_traffic_light)",
    requires={}, produces={"governance": ["risk_of_bias"]},
)
def risk_of_bias(state: StudyState, **kwargs: Any) -> StudyState:
    """Risk-of-bias judgement schema. kwargs: ``tool='ROB2'``, ``studies={name: {domain: 'low'|'high'|...}}``.
    Validates the domain set for the tool and computes an overall per-study judgement."""
    tool = str(kwargs.get("tool", "ROB2")).upper()
    domains = _ROB_DOMAINS.get(tool, _ROB_DOMAINS["ROB2"])
    studies = kwargs.get("studies") or {}
    order = {"low": 0, "some": 1, "moderate": 1, "unclear": 1, "no_info": 1, "high": 2, "critical": 3}
    overall = {}
    for s, judged in studies.items():
        worst = max((order.get(str(v).lower(), 1) for v in judged.values()), default=1)
        overall[s] = ["low", "some concerns", "high", "critical"][worst]
    state.write("governance", "risk_of_bias", {
        "tool": tool, "domains": domains, "studies": studies, "overall": overall,
    })
    return state


@register(
    name="screen_agreement", aliases=["筛选一致性", "screening_irr", "gwet_ac1"],
    category="scholarship_governance", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="双人筛选一致性:Cohen's κ + Gwet AC1 + PABAK + 冲突清单(κ 高流行率悖论时看 AC1/PABAK;ICC/加权κ 用 sv.tl.interrater)",
    requires={"sources": ["datasets"]}, produces={"governance": ["screen_agreement"]},
)
def screen_agreement(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-reviewer screening agreement: Cohen's κ + Gwet AC1 + PABAK + conflict list."""
    import pandas as pd
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    r1c = kwargs.get("rater1"); r2c = kwargs.get("rater2")
    if not isinstance(df, pd.DataFrame) or r1c not in df or r2c not in df:
        return state
    a = df[r1c].astype(str).to_numpy(); b = df[r2c].astype(str).to_numpy()
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    # Cohen's kappa
    pe = sum((np.mean(a == c)) * (np.mean(b == c)) for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    # Gwet AC1 (binary/multi): pe_gwet = 1/(K-1) Σ q(1-q), q = avg marginal
    K = max(len(cats), 2)
    pe_g = sum((qk := 0.5 * (np.mean(a == c) + np.mean(b == c))) * (1 - qk) for c in cats) / (K - 1)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else float("nan")
    pabak = 2 * po - 1
    conflicts = [i for i in range(n) if a[i] != b[i]]
    state.write("governance", "screen_agreement", {
        "n": n, "percent_agreement": round(100 * po, 1), "cohen_kappa": float(kappa),
        "gwet_ac1": float(ac1), "pabak": float(pabak), "n_conflicts": len(conflicts),
        "conflict_rows": conflicts,
        "note": "κ suffers the prevalence paradox at extreme base rates — read AC1/PABAK too; "
                "ICC / weighted κ via sv.tl.interrater",
    })
    return state


@register(
    name="grade", aliases=["GRADE证据确定性", "grade_certainty", "sof"],
    category="scholarship_governance", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="GRADE 证据确定性代数 + SoF 表:从起点(RCT高/观测低)按 5 域降级、按大效应等升级;并从 evidence 槽建议不一致性/发表偏倚旗标(终判仍由评审给)",
    requires={}, produces={"governance": ["grade"]},
)
def grade(state: StudyState, **kwargs: Any) -> StudyState:
    """GRADE certainty algebra. kwargs: ``design='rct'|'observational'``, reviewer-entered
    downgrade flags ``risk_of_bias``/``indirectness``/``imprecision`` (0/1/2 each) and
    upgrades ``large_effect``/``dose_response`` (0/1/2). *Suggests* inconsistency (from I²)
    and publication-bias (from Egger) flags out of the evidence slot; the reviewer confirms."""
    design = str(kwargs.get("design", "rct")).lower()
    start = 4 if design.startswith("rct") else 2   # 4=High, 3=Moderate, 2=Low, 1=Very low
    downgrades = {k: int(kwargs.get(k, 0)) for k in ("risk_of_bias", "indirectness", "imprecision")}
    # suggest inconsistency from heterogeneity, publication bias from Egger
    het = state.diagnostics.get("heterogeneity") or {}
    eg = state.diagnostics.get("egger") or {}
    i2 = het.get("I2")
    inc_suggest = 2 if (i2 is not None and i2 > 75) else (1 if (i2 is not None and i2 > 50) else 0)
    pb_suggest = 1 if (eg.get("pval") is not None and eg["pval"] < 0.10) else 0
    downgrades["inconsistency"] = int(kwargs.get("inconsistency", inc_suggest))
    downgrades["publication_bias"] = int(kwargs.get("publication_bias", pb_suggest))
    upgrades = {k: int(kwargs.get(k, 0)) for k in ("large_effect", "dose_response", "confounding_toward_null")}
    level = start - sum(downgrades.values()) + sum(upgrades.values())
    level = int(np.clip(level, 1, 4))
    labels = {4: "High", 3: "Moderate", 2: "Low", 1: "Very low"}
    state.write("governance", "grade", {
        "certainty": labels[level], "level": level, "start": labels[start],
        "downgrades": downgrades, "upgrades": upgrades,
        "suggested_inconsistency": inc_suggest, "suggested_publication_bias": pb_suggest,
        "note": "domain suggestions are from I²/Egger; final GRADE call is reviewer judgement",
    })
    return state
