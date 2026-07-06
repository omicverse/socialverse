"""Canonical StudyState slot vocabulary — the social-science analog of AnnData's
``obs / var / obsm / obsp / varm / layers / uns``.

Every registered function speaks ``requires`` / ``produces`` in terms of these
12 slots (and named keys *inside* each slot). This is the vocabulary that makes
the dependency graph machine-readable: a resolver can chain functions purely by
matching one function's ``produces`` against another's ``requires`` — exactly the
mechanism that lets an agent plan a valid analysis without hallucinating.

The discipline mirrors omicverse's registry, where ``requires={'layers':['scaled']}``
means ``adata.layers['scaled']`` must exist. Here ``requires={'design':['weights']}``
means ``state.design['weights']`` must exist.
"""
from __future__ import annotations

# slot -> (one-line meaning, typical keys held inside the slot)
SLOTS: dict[str, tuple[str, list[str]]] = {
    "sources": (
        "已登记的原始输入:数据集/语料/手稿/.bib/扫描件",
        ["datasets", "corpora", "bib", "scans", "manifest"],
    ),
    "design": (
        "研究设计:抽样框/权重/分层/聚类/panel_id/time/处理时点/分析单元",
        ["unit", "panel_id", "time", "treatment", "first_treated",
         "weights", "strata", "psu", "sampling_frame"],
    ),
    "variables": (
        "codebook:变量定义/类型/测量层次/量表题项",
        ["codebook", "outcome", "exposure", "controls", "scales", "constructs"],
    ),
    "corpus": (
        "文本即数据态:文档/分词/dfm/OCR文本/TEI/编码单元",
        ["documents", "units", "dfm", "tei", "tokens", "manifest"],
    ),
    "codes": (
        "质性编码态:质性codebook/编码片段/主题/主题地图",
        ["codebook", "segments", "themes", "theme_map"],
    ),
    "estimand": (
        "目标量:ATT/患病率/关联/目标总体(通常由用户/研究问题给定)",
        ["target", "population", "effect", "quantity"],
    ),
    "identification": (
        "识别假设:DAG/平行趋势/IV有效性/排他性/正值性",
        ["strategy", "assumptions", "dag", "parallel_trends", "iv_validity"],
    ),
    "models": (
        "拟合结果:回归表/DID/FE/event-study/加权估计/主题模型/网络",
        ["did", "twfe", "event_study", "weighted_reg", "cox", "topic",
         "network", "field_map", "ideal_type", "stemma"],
    ),
    "diagnostics": (
        "检验:pretrend/平衡性/稳健性矩阵/信度α/拟合/敏感性",
        ["pretrend", "balance", "robustness", "reliability", "ph_test",
         "power", "sensitivity", "coverage"],
    ),
    "evidence": (
        "证据链:claim→引语/引文/quote-trace索引/已核验.bib/来源span",
        ["citations", "verified_bib", "quote_index", "claim_evidence",
         "landscape", "provenance"],
    ),
    "governance": (
        "伦理合规:IRB/知情同意/PII去标识/数据使用许可/AI使用披露",
        ["ethics", "data_use", "pii_status", "ai_disclosure", "consent"],
    ),
    "artifacts": (
        "交付物:图/docx-pdf稿件/表/TEI-XML/apparatus",
        ["figures", "tables", "docx", "pdf", "xml", "apparatus", "scripts"],
    ),
}

#: the frozen set of valid slot names — anything outside this is a registry error
VALID_SLOTS: frozenset[str] = frozenset(SLOTS)

#: valid keys for a ``prerequisites`` declaration (function-level, not slots)
VALID_PREREQ_KEYS: frozenset[str] = frozenset({"functions", "optional_functions"})


def validate_slot_map(mapping: dict | None, *, field: str) -> dict[str, list[str]]:
    """Validate a ``requires`` / ``produces`` declaration against the vocabulary.

    Returns a normalized ``{slot: [keys...]}`` dict. Raises ``ValueError`` on any
    slot outside :data:`VALID_SLOTS` — this is the discipline that keeps the whole
    registry queryable (the omicverse ``valid_keys`` guard, ported to social science).
    """
    if not mapping:
        return {}
    out: dict[str, list[str]] = {}
    for slot, keys in mapping.items():
        if slot not in VALID_SLOTS:
            raise ValueError(
                f"Invalid {field} slot {slot!r}. Valid slots: {sorted(VALID_SLOTS)}"
            )
        if keys is None:
            out[slot] = []
        elif isinstance(keys, str):
            out[slot] = [keys]
        else:
            out[slot] = list(keys)
    return out


def validate_prerequisites(mapping: dict | None) -> dict[str, list[str]]:
    """Validate a ``prerequisites`` declaration (``functions`` / ``optional_functions``)."""
    if not mapping:
        return {}
    out: dict[str, list[str]] = {}
    for key, funcs in mapping.items():
        if key not in VALID_PREREQ_KEYS:
            raise ValueError(
                f"Invalid prerequisites key {key!r}. Valid keys: {sorted(VALID_PREREQ_KEYS)}"
            )
        out[key] = list(funcs or [])
    return out
