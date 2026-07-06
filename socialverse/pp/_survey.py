"""sv.pp._survey — registered implementations for the *ingest* step and the
``complex-survey-analysis`` skill's design-declaration step.

Two functions live here, both in the "prepare" phase:

* :func:`ingest` — register tabular data (DataFrame / dict / list-of-records)
  into ``state.sources['datasets']`` so downstream contracts have something to
  ``require``. This is the community-tier entry point of every study.
* :func:`declare_design` — declare the survey/panel **design** columns (panel
  id, time, treatment, first-treated cohort, weights, strata, PSU, unit) into
  ``state.design``. Column names are validated against the registered dataset
  when one is present; a missing column is a *warning*, never an exception —
  the design vocabulary is what the causal / survey estimators later read.

Both are thin, deterministic, dependency-free (stdlib + pandas) and follow the
in-place ``state`` convention: read via slots, write via :meth:`StudyState.write`,
return ``state``. The ``@register`` wrapper enforces ``requires`` and records
provenance automatically — these bodies only do the work.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["ingest", "declare_design"]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _coerce_frame(data: Any) -> pd.DataFrame:
    """Best-effort coercion of ``data`` to a DataFrame without ever raising.

    Accepts an existing DataFrame (passed through), a Series, a dict of columns,
    a list of record-dicts, or anything ``pd.DataFrame`` can swallow. Falls back
    to an empty frame if coercion fails, so ingest is always non-fatal.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    if data is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(data)
    except Exception:
        # last resort: wrap the object as a single opaque cell
        return pd.DataFrame({"value": [data]})


def _describe(obj: Any) -> dict[str, Any]:
    """Shape/length metadata for whatever got ingested (frame-aware)."""
    if isinstance(obj, pd.DataFrame):
        return {"kind": "DataFrame", "shape": tuple(obj.shape),
                "columns": list(map(str, obj.columns))}
    try:
        return {"kind": type(obj).__name__, "len": len(obj)}
    except Exception:
        return {"kind": type(obj).__name__}


# --------------------------------------------------------------------------- #
# ingest                                                                       #
# --------------------------------------------------------------------------- #
@register(
    name="ingest",
    aliases=["读取数据", "load_data"],
    category="prep",
    tier="community",
    skill="(ingest)",
    languages=["Python"],
    key_tools=["pandas"],
    description="把表格/DataFrame 数据登记进 state.sources",
    requires={},
    produces={"sources": ["datasets"]},
    auto_fix="none",
)
def ingest(state: StudyState, **kwargs: Any) -> StudyState:
    """Register a tabular dataset into ``state.sources['datasets']``.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame | Series | dict | list[dict] | None
        The raw table. Coerced to a DataFrame; ``None`` yields an empty
        placeholder frame so the ``sources['datasets']`` contract is satisfied
        even before real data arrives.
    name : str, optional
        A human label for the dataset (defaults to ``"dataset"``).
    """
    data = kwargs.get("data", None)
    name = kwargs.get("name", "dataset")

    frame = _coerce_frame(data)
    state.write("sources", "datasets", frame)
    state.write("sources", "dataset_name", str(name))
    state.write("sources", "dataset_meta", _describe(frame))
    return state


# --------------------------------------------------------------------------- #
# declare_design                                                              #
# --------------------------------------------------------------------------- #
#: design keys this function can populate (order = declaration order)
_DESIGN_KEYS: tuple[str, ...] = (
    "panel_id", "time", "treatment", "first_treated",
    "weights", "strata", "psu", "unit",
)


@register(
    name="declare_design",
    aliases=["声明设计", "set_design"],
    category="prep",
    tier="plus",
    skill="complex-survey-analysis",
    languages=["Python"],
    key_tools=["pandas"],
    description="声明研究设计变量(面板id/时间/处理/处理时点/权重/分层/PSU/单元)",
    requires={"sources": ["datasets"]},
    produces={"design": ["panel_id", "time", "treatment", "first_treated",
                          "weights", "strata", "psu", "unit"]},
    auto_fix="escalate",
)
def declare_design(state: StudyState, **kwargs: Any) -> StudyState:
    """Declare survey/panel design **column names** into ``state.design``.

    Each of ``panel_id / time / treatment / first_treated / weights / strata /
    psu / unit`` is taken from ``kwargs`` as a *column-name string* (only the
    keys actually provided are written). When ``state.sources['datasets']`` is a
    DataFrame, each declared column is checked for existence; unknown columns are
    reported in the returned ``design['warnings']`` list rather than raising —
    the declaration is advisory metadata, not a hard gate.
    """
    datasets = state.sources.get("datasets")
    known_cols: set[str] = set()
    if isinstance(datasets, pd.DataFrame):
        known_cols = set(map(str, datasets.columns))

    warnings: list[str] = []
    for key in _DESIGN_KEYS:
        if key not in kwargs or kwargs[key] is None:
            continue
        col = kwargs[key]
        if known_cols and isinstance(col, str) and col not in known_cols:
            warnings.append(f"design.{key}: column {col!r} not found in datasets")
        state.write("design", key, col)

    if warnings:
        state.write("design", "warnings", warnings)
    return state
