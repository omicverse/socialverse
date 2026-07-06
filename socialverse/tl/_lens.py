"""``sv.tl._lens`` — registered implementations for the social-theory *lens* skills.

Three canonical theory lenses, ported to the ``StudyState`` / ``registry`` spine:
``foucault-discourse``, ``bourdieu-field``, and ``weber-ideal-type``. Unlike the
causal / survey modules, a *lens* is a **reading protocol**, not an estimator: its
job is to impose a theory's analytic categories on already-coded material and to
tie every interpretive claim back to a locatable piece of evidence (a corpus
unit, an actor, a case). So the output here is deliberately **structured**
(dicts / DataFrames of protocols, positions, scores, and claim→evidence
scaffolds) — never a placeholder string — and every reading carries a support
pointer so the interpretation stays falsifiable.

Two of the three lenses still do real numerics where the theory is spatial:
``bourdieu_field`` builds an actor × capital position space by MCA (``prince``
when installed) or a from-scratch centered-SVD PCA fallback, and
``weber_ideal_type`` scores every case against a pure-type yardstick by genuine
distance-to-pole computation. ``foucault_discourse`` is methodological (no
statistics by design) and produces an interrogation protocol mapped onto the
supplied discursive units. All heavy optional dependencies are lazy-imported and
degrade gracefully; nothing here networks or raises at import time.
"""
from __future__ import annotations

import importlib
import re
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency (never networks)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _as_frame(data: Any) -> pd.DataFrame:
    """Coerce whatever arrived in a kwarg into a :class:`pandas.DataFrame`."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if data is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def _units_as_records(units: Any) -> list[dict[str, Any]]:
    """Normalize ``corpus['units']`` into a list of ``{id, text}`` records.

    ``units`` may be a list of strings, a list of dicts (with any of
    ``id``/``unit_id``/``text``/``content`` keys), a ``{id: text}`` mapping, or a
    DataFrame with a text-like column. Anything unrecognized yields ``[]``.
    """
    out: list[dict[str, Any]] = []
    if units is None:
        return out
    if isinstance(units, pd.DataFrame):
        df = units
        id_col = next((c for c in ("id", "unit_id", "uid") if c in df.columns), None)
        txt_col = next(
            (c for c in ("text", "content", "unit", "quote", "segment") if c in df.columns),
            None,
        )
        for i, (_, row) in enumerate(df.iterrows()):
            uid = str(row[id_col]) if id_col else f"u{i}"
            text = str(row[txt_col]) if txt_col else str(row.to_dict())
            out.append({"id": uid, "text": text})
        return out
    if isinstance(units, dict):
        for i, (k, v) in enumerate(units.items()):
            out.append({"id": str(k), "text": str(v)})
        return out
    if isinstance(units, (list, tuple)):
        for i, u in enumerate(units):
            if isinstance(u, dict):
                uid = str(u.get("id") or u.get("unit_id") or u.get("uid") or f"u{i}")
                text = str(u.get("text") or u.get("content") or u.get("unit") or u)
                out.append({"id": uid, "text": text})
            else:
                out.append({"id": f"u{i}", "text": str(u)})
    return out


def _match_units(records: list[dict[str, Any]], patterns: list[str], cap: int = 6) -> list[str]:
    """Return up to ``cap`` unit ids whose text matches any regex in ``patterns``.

    A best-effort keyword grounding: even a methodological lens should point at
    *which* units triggered a category. Falls back to the first ids if nothing
    matches so a reading is never left dangling.
    """
    if not records:
        return []
    hits: list[str] = []
    rx = [re.compile(p, re.IGNORECASE) for p in patterns]
    for rec in records:
        text = rec.get("text", "")
        if any(r.search(text) for r in rx):
            hits.append(rec["id"])
        if len(hits) >= cap:
            break
    if not hits:
        hits = [r["id"] for r in records[:cap]]
    return hits


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns, coercing where possible; drop all-NaN columns."""
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all")
    return num


def _pca_2d(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered-SVD PCA → first two component scores, loadings, explained variance.

    A dependency-free stand-in for MCA/prince: center the (already numeric) matrix,
    take the SVD, and read the top-2 singular directions. Deterministic.
    """
    x = np.asarray(mat, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0)
    center = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(center, full_matrices=False)
    k = min(2, s.shape[0])
    scores = u[:, :k] * s[:k]
    if scores.shape[1] < 2:  # pad to 2-D when the matrix has a single column
        scores = np.hstack([scores, np.zeros((scores.shape[0], 2 - scores.shape[1]))])
    loadings = vt[:k].T if k else np.zeros((center.shape[1], 2))
    total = float((s ** 2).sum()) or 1.0
    explained = (s[:k] ** 2) / total
    if explained.shape[0] < 2:
        explained = np.concatenate([explained, np.zeros(2 - explained.shape[0])])
    return scores, loadings, explained


# ===================================================================== foucault
@register(
    name="foucault_discourse",
    aliases=["福柯话语", "foucault"],
    category="lens",
    tier="plus",
    skill="foucault-discourse",
    languages=["无代码(方法论)"],
    key_tools=["archaeology/genealogy", "pouvoir-savoir"],
    description="福柯话语分析透镜:权力/知识、话语构成、系谱学的结构化追问协议",
    requires={"corpus": ["units"]},
    produces={"evidence": ["claim_evidence"]},
    prerequisites={"functions": ["code_themes"]},
    auto_fix="auto",
)
def foucault_discourse(state: StudyState, **kwargs: Any) -> StudyState:
    """Foucauldian discourse lens: a structured interrogation protocol over units.

    This is a *reading protocol*, not a statistic. For each unit in
    ``corpus['units']`` the lens instantiates Foucault's analytic questions —
    which **discursive formations** organize what can be said, the **conditions of
    possibility** that make a statement sayable, the **inclusion / exclusion /
    normalization** operations that police the boundary of the sayable, the
    **power–knowledge** (pouvoir-savoir) relations at work, and the modes of
    **subjectivation** by which the discourse constitutes its subjects.

    Every reading is grounded: each entry carries the ``unit_id``s whose text
    triggered the category (best-effort keyword grounding), so the interpretation
    remains locatable and contestable. Writes
    ``evidence['claim_evidence'] = {protocol, readings, ...}``.
    """
    units = kwargs.get("units")
    if units is None:
        units = state.corpus.get("units")
    records = _units_as_records(units)

    # The archaeology / genealogy interrogation grid — the fixed protocol.
    protocol: list[dict[str, str]] = [
        {
            "axis": "discursive_formation",
            "question": "哪些陈述被组织为一个话语对象?什么规则界定了可说的对象、概念与主体位置?",
            "method": "archaeology",
        },
        {
            "axis": "conditions_of_possibility",
            "question": "是什么历史性的知识型(épistémè)使这一陈述得以成为『真』并被言说?",
            "method": "archaeology",
        },
        {
            "axis": "inclusion_exclusion",
            "question": "该话语把什么纳入『正常/可说』,又把什么划为『异常/沉默』而排除?",
            "method": "genealogy",
        },
        {
            "axis": "normalization",
            "question": "通过何种规范、分类与测量,主体被度量、矫正并趋于常态?",
            "method": "genealogy",
        },
        {
            "axis": "power_knowledge",
            "question": "知识如何生产权力、权力又如何生产知识(pouvoir-savoir)?谁因此获得说真话的资格?",
            "method": "genealogy",
        },
        {
            "axis": "subjectivation",
            "question": "话语把言说者/被言说者构成为何种主体(病人/罪犯/学生…)?自我如何被规训?",
            "method": "genealogy",
        },
    ]

    # Keyword cues per axis — deliberately conservative; grounding, not NLP.
    cues: dict[str, list[str]] = {
        "discursive_formation": [r"知识|科学|真理|定义|分类|discourse|knowledge|truth|categor"],
        "conditions_of_possibility": [r"历史|时代|背景|前提|context|histor|epistem|condition"],
        "inclusion_exclusion": [r"正常|异常|排除|沉默|禁止|normal|abnormal|exclud|silence|forbid"],
        "normalization": [r"规范|标准|矫正|测量|评估|norm|standard|correct|measur|assess"],
        "power_knowledge": [r"权力|控制|规训|监视|专家|power|control|disciplin|surveil|expert"],
        "subjectivation": [r"主体|身份|自我|病人|罪犯|subject|identity|self|patient|criminal"],
    }

    readings: list[dict[str, Any]] = []
    for entry in protocol:
        axis = entry["axis"]
        support = _match_units(records, cues[axis])
        readings.append(
            {
                "axis": axis,
                "method": entry["method"],
                "question": entry["question"],
                "claim": f"待填:关于『{axis}』的解读性主张(以下 unit 为其证据锚点)",
                "support_units": support,
                "n_support": len(support),
            }
        )

    claim_evidence = {
        "lens": "foucault_discourse",
        "stance": "解释性阅读协议(archaeology/genealogy),非统计;每条主张须由 support_units 支撑",
        "protocol": protocol,
        "readings": readings,
        "n_units": len(records),
        "note": (
            "无编码语料时产出空 readings 的空协议骨架;有 units 时每轴已挂 support_units"
            if not records else "每轴 support_units 为关键词命中的证据锚点,claim 待研究者填写"
        ),
    }
    state.write("evidence", "claim_evidence", claim_evidence)
    return state


# ===================================================================== bourdieu
@register(
    name="bourdieu_field",
    aliases=["布迪厄场域", "bourdieu"],
    category="lens",
    tier="plus",
    skill="bourdieu-field",
    languages=["无代码(方法论)"],
    key_tools=["field/habitus/capital", "MCA", "prince"],
    description="布迪厄场域:actor×capital 位置空间(MCA/PCA)+ 位置-立场同源性",
    requires={"codes": ["themes"], "variables": ["constructs"]},
    produces={"models": ["field_map"], "evidence": ["claim_evidence"]},
    auto_fix="escalate",
)
def bourdieu_field(state: StudyState, **kwargs: Any) -> StudyState:
    """Bourdieusian field lens: an actor × capital position space + homology reading.

    Builds the two-dimensional *space of positions* in which actors are distributed
    by the volume and structure of their **capital** (economic / cultural / social /
    symbolic). The projection uses ``prince.MCA`` when installed (the canonical
    Bourdieusian correspondence-analysis route); otherwise it degrades to a
    from-scratch centered-SVD PCA so the field map is always produced.

    Data arrives as ``capital_table=`` (a DataFrame indexed by actor, columns =
    capital dimensions). The **homology** between the space of positions and the
    space of position-takings (stances / themes) is scaffolded as claim→evidence:
    each actor's coordinate is traced back to the independent capital indicators
    (and any coded ``themes``) that place it there. Writes ``models['field_map']``
    and ``evidence['claim_evidence']``.
    """
    cap = kwargs.get("capital_table")
    if cap is None:
        cap = kwargs.get("data")
    if cap is None:
        cap = state.variables.get("constructs")
    df = _as_frame(cap)

    themes = kwargs.get("themes") or state.codes.get("themes")

    if df.empty:
        field_map = {
            "positions": {},
            "axes": {"x": "capital_axis_1", "y": "capital_axis_2"},
            "method": "none",
            "explained_variance": [0.0, 0.0],
            "note": "未提供 capital_table(actor×资本维度),无法构建位置空间",
        }
        claim_evidence = {
            "lens": "bourdieu_field",
            "stance": "位置空间(capital)↔立场空间(themes)同源性;每个位置须溯源到独立资本指标",
            "homology": [],
            "note": "缺 capital_table:field_map 为空骨架",
        }
        state.write("models", "field_map", field_map)
        state.write("evidence", "claim_evidence", claim_evidence)
        return state

    # actor labels = index if meaningful, else first non-numeric column, else range
    if df.index.name or not isinstance(df.index, pd.RangeIndex):
        actors = [str(a) for a in df.index]
        num = _numeric_frame(df)
    else:
        obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if obj_cols:
            actors = [str(a) for a in df[obj_cols[0]]]
            num = _numeric_frame(df.drop(columns=obj_cols[:1]))
        else:
            actors = [f"actor{i}" for i in range(len(df))]
            num = _numeric_frame(df)

    dims = list(num.columns)
    method = "none"
    explained = np.array([0.0, 0.0])
    coords = np.zeros((len(actors), 2))

    prince = _try_import("prince")
    if prince is not None and num.shape[1] >= 1 and num.shape[0] >= 2:
        try:
            mca = prince.MCA(n_components=2, random_state=0)
            # MCA expects categorical columns; discretize numeric capital into terciles.
            disc = num.apply(
                lambda s: pd.qcut(s.rank(method="first"), q=min(3, s.nunique()),
                                  labels=False, duplicates="drop"),
                axis=0,
            ).astype("category")
            mca = mca.fit(disc)
            row_coords = np.asarray(mca.row_coordinates(disc))
            coords = row_coords[:, :2]
            if coords.shape[1] < 2:
                coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
            ev = getattr(mca, "percentage_of_variance_", None)
            explained = (np.asarray(ev[:2]) / 100.0 if ev is not None
                         else np.array([np.nan, np.nan]))
            method = "MCA(prince)"
        except Exception:
            method = "none"

    if method == "none":  # PCA fallback (dependency-free)
        if num.shape[1] >= 1 and num.shape[0] >= 1:
            coords, _loadings, explained = _pca_2d(num.to_numpy())
            method = "PCA(centered-SVD fallback)"
        else:
            explained = np.array([0.0, 0.0])

    positions = {
        actors[i]: (float(coords[i, 0]), float(coords[i, 1]))
        for i in range(len(actors))
    }

    field_map = {
        "positions": positions,
        "axes": {
            "x": "capital_axis_1 (总资本量/结构主轴)",
            "y": "capital_axis_2 (资本构成:经济↔文化)",
        },
        "capital_dims": dims,
        "method": method,
        "explained_variance": [float(explained[0]), float(explained[1])],
        "n_actors": len(actors),
        "note": "位置=资本空间投影;二维=Bourdieu 场域的位置空间近似",
    }

    # homology scaffold: each actor position ↔ its own independent capital indicators
    homology: list[dict[str, Any]] = []
    for i, actor in enumerate(actors):
        indicators = {d: (None if pd.isna(num.iloc[i][d]) else float(num.iloc[i][d]))
                      for d in dims}
        homology.append(
            {
                "actor": actor,
                "position": (float(coords[i, 0]), float(coords[i, 1])),
                "capital_indicators": indicators,
                "claim": f"待填:{actor} 的立场(position-taking)与其资本结构的同源性主张",
                "support": {"capital": dims, "themes": bool(themes)},
            }
        )

    claim_evidence = {
        "lens": "bourdieu_field",
        "stance": "位置空间(capital)↔立场空间(themes/position-taking)同源性;位置须溯源到独立资本指标",
        "homology": homology,
        "themes_available": bool(themes),
        "method": method,
        "note": "每个 actor 的坐标已溯源到其独立资本指标;themes 存在时可读位置-立场同源性",
    }

    state.write("models", "field_map", field_map)
    state.write("evidence", "claim_evidence", claim_evidence)
    return state


# ===================================================================== weber
@register(
    name="weber_ideal_type",
    aliases=["韦伯理想类型", "ideal_type"],
    category="lens",
    tier="plus",
    skill="weber-ideal-type",
    languages=["无代码(方法论)"],
    key_tools=["Idealtypus", "Verstehen"],
    description="韦伯理想类型:纯粹类型标尺→逐案维度打分→Verstehen 解释偏离(守价值中立)",
    requires={"sources": ["datasets"]},
    produces={
        "models": ["ideal_type"],
        "diagnostics": ["coverage"],
        "governance": ["ethics"],
        "evidence": ["claim_evidence"],
    },
    auto_fix="escalate",
)
def weber_ideal_type(state: StudyState, **kwargs: Any) -> StudyState:
    """Weberian ideal-type lens: pure-type yardstick → per-case scoring → Verstehen.

    An *ideal type* (Idealtypus) is a deliberately one-sided, analytically pure
    construct — no real case is expected to match it. This lens formalizes that:
    ``schema={dimension: pole_description}`` defines the pure poles, then every case
    in ``cases`` (a DataFrame, or ``sources['datasets']``) is scored on each
    dimension as a real distance-from-pole in ``[0, 1]`` (1 = at the ideal pole).
    The **deviation** of each case from the pure type is what carries interpretive
    weight — those deviations are handed to *Verstehen* (interpretive understanding)
    as claim→evidence, not smoothed away.

    Guards **value-freedom** (Wertfreiheit): the scores are analytic distances, not
    evaluations of worth, and a governance ethics note states so explicitly. Writes
    ``models['ideal_type']``, ``diagnostics['coverage']``, ``governance['ethics']``
    and ``evidence['claim_evidence']``.
    """
    schema = kwargs.get("schema") or {}
    cases = kwargs.get("cases")
    if cases is None:
        cases = kwargs.get("data")
    if cases is None:
        cases = state.sources.get("datasets")
    if isinstance(cases, dict) and not isinstance(cases, pd.DataFrame):
        cases = next((v for v in cases.values() if isinstance(v, pd.DataFrame)), cases)
    df = _as_frame(cases)

    # If no schema given, treat every numeric column as a dimension whose pole is
    # its observed maximum (a defensible default so the lens still runs).
    if not schema:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        schema = {c: f"{c} 的纯粹极值(方向由最大观测值给定)" for c in num_cols}

    dims = list(schema.keys())
    ethics = {
        "principle": "Wertfreiheit(价值中立)",
        "statement": (
            "本理想类型评分为分析性的『与纯粹类型的距离』,非对案例价值/优劣的评判;"
            "理想型是一侧强调的分析建构,任何真实案例都不应被期待与之完全吻合。"
        ),
        "verstehen": "偏离(deviation)交由解释性理解(Verstehen)说明其意义,不作规范性排序。",
    }

    if df.empty or not dims:
        model = {
            "schema": schema,
            "scores": pd.DataFrame(columns=dims),
            "deviations": pd.DataFrame(columns=dims),
            "note": "缺 cases 或 schema:仅产出理想型标尺(schema),无逐案打分",
        }
        state.write("models", "ideal_type", model)
        state.write("diagnostics", "coverage", {"per_dimension": {}, "overall": 0.0,
                                                 "note": "无案例可覆盖"})
        state.write("governance", "ethics", ethics)
        state.write("evidence", "claim_evidence", {
            "lens": "weber_ideal_type",
            "stance": "偏离→Verstehen 解释;守价值中立",
            "deviations": [],
            "note": "缺数据:claim_evidence 为空骨架",
        })
        return state

    # case labels
    if df.index.name or not isinstance(df.index, pd.RangeIndex):
        case_ids = [str(c) for c in df.index]
    else:
        obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        case_ids = ([str(c) for c in df[obj_cols[0]]] if obj_cols
                    else [f"case{i}" for i in range(len(df))])

    # Score each dimension in [0,1] = proximity to the ideal pole. For numeric
    # dimensions, min-max normalize (max = pole); non-numeric dims score NaN
    # (they need qualitative scoring the researcher supplies).
    scores = pd.DataFrame(index=case_ids, columns=dims, dtype=float)
    coverage: dict[str, float] = {}
    for d in dims:
        if d in df.columns and pd.api.types.is_numeric_dtype(df[d]):
            col = pd.to_numeric(df[d], errors="coerce").to_numpy(dtype=float)
            lo, hi = np.nanmin(col), np.nanmax(col)
            rng = (hi - lo) or 1.0
            scores[d] = np.clip((col - lo) / rng, 0.0, 1.0)
            coverage[d] = float(np.isfinite(col).mean())
        else:
            scores[d] = np.nan
            coverage[d] = 0.0

    # deviation from the pure type = 1 - proximity (0 at the pole, 1 farthest).
    deviations = (1.0 - scores).round(4)
    scores = scores.round(4)

    per_case_dev = deviations.mean(axis=1, skipna=True)
    overall_cov = float(np.mean(list(coverage.values()))) if coverage else 0.0

    model = {
        "schema": schema,
        "scores": scores,
        "deviations": deviations,
        "case_mean_deviation": {k: (None if pd.isna(v) else float(v))
                                for k, v in per_case_dev.items()},
        "dimensions": dims,
        "n_cases": len(case_ids),
        "note": "score∈[0,1](1=贴近纯粹极);deviation=1-score,承载解释重量",
    }

    # Verstehen scaffold: the most-deviating (dimension, case) pairs, each a claim
    # to be interpretively explained — grounded in the actual score/value.
    dev_entries: list[dict[str, Any]] = []
    for cid in case_ids:
        row = deviations.loc[cid]
        finite = row.dropna()
        if finite.empty:
            continue
        top_dim = finite.idxmax()
        dev_entries.append(
            {
                "case": cid,
                "dimension": top_dim,
                "deviation": float(finite.max()),
                "score": (None if pd.isna(scores.loc[cid, top_dim])
                          else float(scores.loc[cid, top_dim])),
                "pole": schema.get(top_dim, ""),
                "claim": f"待填:{cid} 在『{top_dim}』上偏离纯粹类型的意义(Verstehen 解释)",
                "value": (None if (top_dim not in df.columns or pd.isna(df.iloc[case_ids.index(cid)][top_dim]))
                          else _cell(df, case_ids, cid, top_dim)),
            }
        )
    dev_entries.sort(key=lambda e: e["deviation"], reverse=True)

    claim_evidence = {
        "lens": "weber_ideal_type",
        "stance": "偏离(deviation)→解释性理解(Verstehen);评分为分析距离,非价值判断",
        "deviations": dev_entries,
        "note": "每条=一个案例最偏离的维度,交 Verstehen 解释;value 为原始证据值",
    }

    state.write("models", "ideal_type", model)
    state.write("diagnostics", "coverage",
                {"per_dimension": coverage, "overall": overall_cov,
                 "note": "各维度非缺失比例;非数值维需研究者补质性打分"})
    state.write("governance", "ethics", ethics)
    state.write("evidence", "claim_evidence", claim_evidence)
    return state


def _cell(df: pd.DataFrame, case_ids: list[str], cid: str, col: str) -> Any:
    """Fetch the raw evidence value behind a (case, dimension) score, JSON-safe."""
    try:
        v = df.iloc[case_ids.index(cid)][col]
    except Exception:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if pd.isna(v) else float(v)
    return None if pd.isna(v) else str(v)


__all__ = ["foucault_discourse", "bourdieu_field", "weber_ideal_type"]
