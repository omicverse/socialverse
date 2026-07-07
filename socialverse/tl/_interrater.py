"""``sv.tl._interrater`` — registered implementation for the measurement gap:
**inter-rater / inter-coder reliability**.

The workhorse of quantitative content analysis and any human-coding pipeline:
given the same set of subjects coded independently by two or more raters, how
much of the observed agreement is real rather than chance? This module ports the
standard reliability battery to the ``StudyState`` / ``registry`` spine with
*real*, from-scratch numpy/scipy estimators (recovers the known ``agree=0.8``
DGP), no placeholders.

Champion packages this file mirrors
-----------------------------------
* R ``irr::agree`` — raw **percentage agreement**, averaged over all rater pairs.
* R ``irr::kappa2`` / SPSS ``KAPPA`` — **Cohen's κ** for exactly two raters
  (chance-corrected pairwise agreement on nominal categories).
* R ``irr::kappam.fleiss`` — **Fleiss' κ** for any number of raters on a nominal
  scale (extends Cohen's κ to >2 raters via per-subject category counts).
* R ``irr::kripp.alpha`` (``method='nominal'``) — **Krippendorff's α**, the
  most general chance-corrected coefficient: it tolerates missing ratings, an
  arbitrary number of raters, and a chosen difference metric (nominal here), and
  is defined through observed vs expected coincidence.

Both Fleiss' κ and Krippendorff's α are hand-implemented from their coincidence /
category-count definitions — no dependency on an external ``irr``-style package —
so the notebook runs on a bare numpy/scipy environment. ``factor_analyzer`` is
not needed here; the only optional import is a lazy cross-check hook that is
never required.

The registry contract: ``interrater`` ``requires`` a working
``sources['datasets']`` ratings frame and ``produces`` the
``diagnostics['interrater']`` block — so a resolver can refuse to report, say, a
qualitative coding result as "reliable" until κ / α have actually been computed.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["interrater"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional dependency (used only for cross-check)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_datasets(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working frame: explicit ``data=`` kwarg, else ``sources['datasets']``.

    ``sources['datasets']`` may be a DataFrame or a ``{name: DataFrame}`` mapping;
    in the latter case the first frame is taken.
    """
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _pick_rater_columns(
    df: pd.DataFrame, kwargs: dict[str, Any]
) -> list[str]:
    """Resolve the rater columns: explicit ``raters=`` kwarg, else columns whose
    name starts with ``rater`` (case-insensitive), else all columns."""
    raters = kwargs.get("raters")
    if raters:
        return [c for c in raters if c in df.columns]
    hit = [c for c in df.columns if str(c).lower().startswith("rater")]
    if hit:
        return hit
    return list(df.columns)


# ---------------------------------------------------------- reliability engines
def _percent_agreement(mat: np.ndarray) -> float | None:
    """Mean pairwise raw agreement over all rater pairs (R ``irr::agree``).

    ``mat`` is a (subjects × raters) object array; ``None``/NaN entries are
    missing. For each unordered rater pair we count the fraction of subjects on
    which both rated and agreed, then average across pairs.
    """
    n_sub, n_rat = mat.shape
    pair_scores: list[float] = []
    for a in range(n_rat):
        for b in range(a + 1, n_rat):
            agree = 0
            both = 0
            for i in range(n_sub):
                va, vb = mat[i, a], mat[i, b]
                if _is_missing(va) or _is_missing(vb):
                    continue
                both += 1
                if va == vb:
                    agree += 1
            if both:
                pair_scores.append(agree / both)
    if not pair_scores:
        return None
    return float(np.mean(pair_scores))


def _cohen_kappa(mat: np.ndarray) -> float | None:
    """Cohen's κ for exactly two raters on nominal categories (SPSS ``KAPPA``).

    Only subjects rated by *both* raters contribute. ``κ = (p_o - p_e)/(1 - p_e)``
    with ``p_o`` the observed agreement and ``p_e`` the chance agreement from the
    two raters' marginal category distributions.
    """
    if mat.shape[1] != 2:
        return None
    pairs = [
        (mat[i, 0], mat[i, 1])
        for i in range(mat.shape[0])
        if not _is_missing(mat[i, 0]) and not _is_missing(mat[i, 1])
    ]
    if not pairs:
        return None
    cats = sorted({v for p in pairs for v in p}, key=lambda x: str(x))
    idx = {c: j for j, c in enumerate(cats)}
    k = len(cats)
    conf = np.zeros((k, k), dtype=float)
    for a, b in pairs:
        conf[idx[a], idx[b]] += 1.0
    total = conf.sum()
    p_o = float(np.trace(conf) / total)
    row = conf.sum(axis=1) / total
    col = conf.sum(axis=0) / total
    p_e = float(np.dot(row, col))
    if p_e >= 1.0:
        return None
    return float((p_o - p_e) / (1.0 - p_e))


def _fleiss_kappa(mat: np.ndarray) -> float | None:
    """Fleiss' κ for N raters on nominal categories (R ``irr::kappam.fleiss``).

    Implemented from the category-count definition. Build the (subjects ×
    categories) count matrix ``n_ij`` = number of raters that assigned category
    ``j`` to subject ``i``. Only subjects rated by ≥ 2 raters are used, and (per
    Fleiss) every used subject must have the *same* number of ratings ``m``; rows
    with fewer than the modal count are dropped so the fixed-``m`` formula holds.

    ``P_i = (Σ_j n_ij² - m) / (m (m - 1))`` is subject ``i``'s agreement,
    ``P_bar = mean_i P_i``; ``p_j`` is the overall proportion in category ``j`` and
    ``P_e = Σ_j p_j²``. ``κ = (P_bar - P_e)/(1 - P_e)``.
    """
    n_sub, n_rat = mat.shape
    # per-subject rating count (non-missing)
    counts_per_sub = np.array(
        [sum(0 if _is_missing(mat[i, r]) else 1 for r in range(n_rat))
         for i in range(n_sub)]
    )
    used = counts_per_sub >= 2
    if not used.any():
        return None
    # Fleiss assumes a fixed number of ratings per subject; use the modal count.
    modal = int(np.bincount(counts_per_sub[used]).argmax())
    if modal < 2:
        return None
    rows = [i for i in range(n_sub) if counts_per_sub[i] == modal]
    if not rows:
        return None

    cats = sorted(
        {mat[i, r] for i in rows for r in range(n_rat) if not _is_missing(mat[i, r])},
        key=lambda x: str(x),
    )
    idx = {c: j for j, c in enumerate(cats)}
    k = len(cats)
    if k < 1:
        return None

    m = modal
    N = len(rows)
    nij = np.zeros((N, k), dtype=float)
    for ri, i in enumerate(rows):
        for r in range(n_rat):
            v = mat[i, r]
            if _is_missing(v):
                continue
            nij[ri, idx[v]] += 1.0

    # overall category proportions
    p_j = nij.sum(axis=0) / (N * m)
    P_e = float(np.sum(p_j ** 2))
    # per-subject agreement
    P_i = (np.sum(nij ** 2, axis=1) - m) / (m * (m - 1))
    P_bar = float(np.mean(P_i))
    denom = 1.0 - P_e
    if denom <= 0:
        return 1.0 if P_bar >= 1.0 else None
    return float((P_bar - P_e) / denom)


def _krippendorff_alpha_nominal(mat: np.ndarray) -> float | None:
    """Krippendorff's α, nominal metric (R ``irr::kripp.alpha``, method='nominal').

    Computed from the **coincidence matrix**. For every subject (unit) rated by
    ``m_u ≥ 2`` raters, each ordered pair of that subject's ratings contributes
    ``1/(m_u - 1)`` to coincidence cell ``(c, c')``. Let ``o_cc'`` be the
    coincidence matrix, ``n_c = Σ_c' o_cc'`` the value marginals and
    ``n = Σ_c n_c`` the total number of pairable values.

    ``D_o = Σ_{c≠c'} o_cc'`` (observed nominal disagreement) and
    ``D_e = (Σ_{c≠c'} n_c n_c') / (n - 1)`` (expected). Then
    ``α = 1 - D_o / D_e``. Missing ratings are naturally tolerated: a subject with
    only one non-missing rating contributes nothing.
    """
    n_sub, n_rat = mat.shape
    values: list[Any] = []
    per_unit: list[list[Any]] = []
    for i in range(n_sub):
        vals = [mat[i, r] for r in range(n_rat) if not _is_missing(mat[i, r])]
        if len(vals) >= 2:
            per_unit.append(vals)
            values.extend(vals)
    if not per_unit:
        return None

    cats = sorted(set(values), key=lambda x: str(x))
    idx = {c: j for j, c in enumerate(cats)}
    k = len(cats)
    coincidence = np.zeros((k, k), dtype=float)
    for vals in per_unit:
        mu = len(vals)
        w = 1.0 / (mu - 1)
        for a in range(mu):
            for b in range(mu):
                if a == b:
                    continue
                coincidence[idx[vals[a]], idx[vals[b]]] += w

    n_c = coincidence.sum(axis=1)          # value marginals
    n = float(n_c.sum())                    # total pairable values
    if n <= 1:
        return None

    # observed nominal disagreement = off-diagonal coincidence mass
    D_o = float(coincidence.sum() - np.trace(coincidence))
    # expected nominal disagreement
    D_e = float((n * n - np.sum(n_c ** 2)) / (n - 1.0))
    if D_e <= 0:
        # no expected disagreement (all one category) → perfect by convention
        return 1.0
    return float(1.0 - D_o / D_e)


def _is_missing(v: Any) -> bool:
    """True for None / NaN sentinel entries in the object rating matrix."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _to_object_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Ratings frame → (subjects × raters) object matrix, categories as strings.

    Using string labels makes the nominal metric robust to mixed int/str/NaN
    codes without accidental ordinal ordering.
    """
    sub = df[cols]
    mat = np.empty(sub.shape, dtype=object)
    for j, c in enumerate(cols):
        col = sub[c].to_numpy()
        for i in range(len(col)):
            v = col[i]
            mat[i, j] = None if _is_missing(v) else str(v)
    return mat


# ------------------------------------------------------------------- interrater
@register(
    name="interrater",
    aliases=["评分者间信度", "编码者间信度", "inter-rater", "inter-coder",
             "kappa", "krippendorff"],
    category="qual",
    tier="plus",
    skill="(测量缺口)",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description="评分者间/编码者间信度:百分比一致 + Cohen κ(2 rater)+ Fleiss κ + "
                "Krippendorff α(名义,容忍缺失)",
    requires={"sources": ["datasets"]},
    produces={"diagnostics": ["interrater"]},
    auto_fix="escalate",
)
def interrater(state: StudyState, **kwargs: Any) -> StudyState:
    """Inter-rater / inter-coder reliability battery on a subjects × raters frame.

    Given a ratings frame where each row is a subject and each ``rater_*`` column
    is one coder's nominal category, computes the full reliability battery:

    * **percent_agreement** — mean raw agreement over all rater pairs.
    * **cohen_kappa** — Cohen's κ, only when there are exactly two raters
      (``None`` otherwise, matching SPSS ``KAPPA``'s two-rater scope).
    * **fleiss_kappa** — Fleiss' κ for any number of raters (nominal).
    * **krippendorff_alpha** — Krippendorff's α (nominal metric, missing-tolerant).

    All four are chance-corrected apart from ``percent_agreement``. Both Fleiss' κ
    and Krippendorff's α are computed from scratch (numpy) and agree on the
    ``agree=0.8`` toy DGP (κ, α ≈ 0.5–0.85 = substantial agreement).

    kwargs
    ------
    ratings : pd.DataFrame, optional
        Explicit subjects × raters frame (overrides ``sources['datasets']``).
    raters : list[str], optional
        Rater column names. Default: columns whose name starts with ``rater``.
    """
    # allow ratings= as an alias for data=
    if kwargs.get("ratings") is not None and kwargs.get("data") is None:
        kwargs = {**kwargs, "data": kwargs["ratings"]}

    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        state.write("diagnostics", "interrater", {
            "percent_agreement": None,
            "cohen_kappa": None,
            "fleiss_kappa": None,
            "krippendorff_alpha": None,
            "n_raters": 0,
            "n_subjects": 0,
            "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法计算评分者间信度")

    cols = _pick_rater_columns(df, kwargs)
    if len(cols) < 2:
        return _empty(f"评分者列不足 2 个(找到 {len(cols)} 列),无法计算一致性")

    mat = _to_object_matrix(df, cols)
    # drop subjects with fewer than 2 non-missing ratings (nothing pairable)
    keep = np.array(
        [sum(0 if _is_missing(mat[i, j]) else 1 for j in range(mat.shape[1])) >= 2
         for i in range(mat.shape[0])]
    )
    mat = mat[keep]
    if mat.shape[0] == 0:
        return _empty("没有被 ≥2 个评分者共同评定的主体")

    n_subjects, n_raters = mat.shape

    percent = _percent_agreement(mat)
    cohen = _cohen_kappa(mat) if n_raters == 2 else None
    fleiss = _fleiss_kappa(mat)
    kripp = _krippendorff_alpha_nominal(mat)

    note = ("百分比一致(两两平均);Cohen κ 仅在恰 2 个评分者时给出;"
            "Fleiss κ / Krippendorff α 均为名义、手写实现(α 容忍缺失)")
    state.write("diagnostics", "interrater", {
        "percent_agreement": percent,
        "cohen_kappa": cohen,
        "fleiss_kappa": fleiss,
        "krippendorff_alpha": kripp,
        "n_raters": int(n_raters),
        "n_subjects": int(n_subjects),
        "raters": list(cols),
        "estimator": "numpy: pairwise agreement + Cohen/Fleiss κ + Krippendorff α(名义)",
        "note": note,
    })
    return state
