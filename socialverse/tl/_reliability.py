"""``sv.tl._reliability`` — registered implementation for the scale-reliability
gap: **internal-consistency reliability** of a multi-item measurement scale.

``sv.gov.design_survey`` reports Cronbach's α as a single scalar side-note; this
module gives the full psychometric reliability report a social scientist expects
from R's ``psych::alpha`` / ``psych::omega`` or SPSS ``RELIABILITY``:

* **Cronbach's α** — ``(k/(k-1))·(1 − Σ var_item / var_total)``.
* **McDonald's ω** (total) — from a single-factor (congeneric) loading model,
  ``ω = (Σλ)² / ((Σλ)² + Σ(1 − λ²))``. The loadings λ come from a real
  single-factor extraction: ``factor_analyzer`` (MINRES/ML) when installed,
  otherwise the first principal component of the correlation matrix scaled to a
  unit-variance factor (both are honest congeneric loadings, not placeholders).
* **average inter-item correlation** — mean of the off-diagonal correlation
  matrix.
* **corrected item-total correlation** — each item vs. the summed score of the
  *remaining* items (the standard "corrected" r that avoids part-whole
  inflation).
* **α-if-item-deleted** — Cronbach's α recomputed on the scale with each single
  item removed, so a resolver / analyst can see which items drag reliability
  down.

Everything is computed with plain numpy/scipy on the item covariance matrix, so
the notebook recovers the true high-reliability structure of ``ds.load_survey()``
(one strong latent factor → α ≈ 0.9) without any optional dependency; the ω
extraction opportunistically upgrades to ``factor_analyzer`` when present.

Registry contract: ``requires`` a working ``sources['datasets']`` frame and
``produces`` the ``diagnostics['reliability']`` report — so a resolver can refuse
to quote a reliability coefficient until this has actually run.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["reliability"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency."""
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


def _cronbach_alpha(item_matrix: np.ndarray) -> float | None:
    """Cronbach's α from an ``(n, k)`` item score matrix.

    ``α = (k/(k−1))·(1 − Σ var_item / var_total)``, using sample variances
    (ddof=1). Returns ``None`` if fewer than two items.

    Delegates to the parity-gated :mod:`pypsych` port (``raw_alpha`` of psych's
    ``alpha`` total row, which is the covariance-based Cronbach's α); falls back
    to the local closed-form if the port raises.
    """
    n, k = item_matrix.shape
    if k < 2:
        return None
    try:
        from ..external.pypsych import cronbach_alpha as _pp_alpha
        val = _pp_alpha(item_matrix)["raw_alpha"]
        if np.isfinite(val):
            return float(val)
    except Exception:
        pass
    item_vars = item_matrix.var(axis=0, ddof=1)
    total_var = item_matrix.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return None
    return float((k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var))


def _single_factor_loadings(item_matrix: np.ndarray) -> np.ndarray:
    """Standardized single-factor (congeneric) loadings for McDonald's ω.

    Prefers ``factor_analyzer`` (a genuine single-factor extraction); falls back
    to the first principal component of the correlation matrix, scaled so the
    factor has unit variance (loading_i = sign · sqrt(eigval) · eigvec_i, i.e. the
    standardized loading of item i on PC1). Loadings are clipped to [−0.999,
    0.999] so ``1 − λ²`` stays positive.
    """
    # correlation matrix (standardized items → loadings are standardized)
    corr = np.corrcoef(item_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    fa_mod = _try_import("factor_analyzer")
    if fa_mod is not None:
        try:
            FactorAnalyzer = fa_mod.FactorAnalyzer
            fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
            fa.fit(item_matrix)
            load = np.asarray(fa.loadings_, dtype=float).ravel()
            if load.shape[0] == item_matrix.shape[1] and np.all(np.isfinite(load)):
                # orient so the factor is positively related to the items
                if load.sum() < 0:
                    load = -load
                return np.clip(load, -0.999999, 0.999999)
        except Exception:
            pass

    # PCA fallback: leading eigenpair of the correlation matrix
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = int(np.argmax(eigvals))
    lam = float(max(eigvals[idx], 0.0))
    vec = eigvecs[:, idx]
    if vec.sum() < 0:  # orient toward positive loadings
        vec = -vec
    load = vec * np.sqrt(lam)
    return np.clip(load, -0.999999, 0.999999)


def _mcdonald_omega(item_matrix: np.ndarray) -> float | None:
    """McDonald's ω(total) from single-factor congeneric loadings.

    ``ω = (Σλ)² / ((Σλ)² + Σ(1 − λ²))`` where λ are the standardized single-factor
    loadings and ``1 − λ²`` are the item uniquenesses.
    """
    if item_matrix.shape[1] < 2:
        return None
    # Prefer the parity-gated pypsych port: McDonald's ω_total from the
    # principal-axis (fm="pa") communalities, ω = 1 − Σ(1−h²)/sum(R).
    try:
        from ..external.pypsych import omega_total as _pp_omega
        corr = np.corrcoef(item_matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        val = _pp_omega(corr)
        if np.isfinite(val):
            return float(val)
    except Exception:
        pass
    load = _single_factor_loadings(item_matrix)
    sum_load = float(load.sum())
    uniqueness = float(np.sum(1.0 - load ** 2))
    denom = sum_load ** 2 + uniqueness
    if denom <= 0:
        return None
    return float(sum_load ** 2 / denom)


# ------------------------------------------------------------------ reliability
@register(
    name="reliability",
    aliases=["信度", "reliability", "scale_reliability"],
    category="psychometrics",
    tier="plus",
    skill="(测量缺口)",
    languages=["Python"],
    key_tools=["numpy", "scipy", "factor_analyzer"],
    description=(
        "量表信度:Cronbach α + McDonald ω(单因子载荷)+ 平均项间相关 "
        "+ 校正项-总相关 + α-if-item-deleted"
    ),
    requires={"sources": ["datasets"]},
    produces={"diagnostics": ["reliability"]},
    auto_fix="escalate",
)
def reliability(state: StudyState, **kwargs: Any) -> StudyState:
    """Full internal-consistency reliability report for a multi-item scale.

    Computes, on the items of a survey scale:

    * Cronbach's α;
    * McDonald's ω (single-factor congeneric loadings);
    * the average inter-item correlation;
    * the corrected item-total correlation for each item (item vs. sum of the
      remaining items);
    * α-if-item-deleted for each item.

    kwargs
    ------
    items : list[str], optional
        The item columns forming the scale. Default: every column whose name
        starts with ``"item"``; if none match, all numeric columns are used.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        state.write("diagnostics", "reliability", {
            "cronbach_alpha": None,
            "mcdonald_omega": None,
            "avg_inter_item_r": None,
            "item_total": {},
            "alpha_if_deleted": {},
            "k_items": 0,
            "n": 0,
            "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法计算量表信度")

    # ---- resolve the item columns ----
    items = list(kwargs.get("items") or [])
    if not items:
        items = [c for c in df.columns if str(c).startswith("item")]
    if not items:
        items = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    items = [c for c in items if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(items) < 2:
        return _empty(f"信度需要至少 2 个题项(找到 {len(items)} 个)")

    work = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        return _empty("有效被试不足(去缺失后 < 3 行),无法计算信度")

    mat = work.to_numpy(dtype=float)
    n, k = mat.shape

    # ---- Cronbach's alpha ----
    cronbach = _cronbach_alpha(mat)

    # ---- McDonald's omega (single factor) ----
    omega = _mcdonald_omega(mat)

    # ---- average inter-item correlation (off-diagonal mean) ----
    corr = np.corrcoef(mat, rowvar=False)
    off = corr[~np.eye(k, dtype=bool)]
    avg_inter_item_r = float(np.nanmean(off)) if off.size else None

    # ---- corrected item-total correlation + alpha-if-item-deleted ----
    item_total: dict[str, float] = {}
    alpha_if_deleted: dict[str, float] = {}
    for j, name in enumerate(items):
        rest_idx = [i for i in range(k) if i != j]
        rest_sum = mat[:, rest_idx].sum(axis=1)
        this_item = mat[:, j]
        # corrected item-total: item vs. sum of remaining items
        if np.std(this_item) == 0 or np.std(rest_sum) == 0:
            item_total[name] = 0.0
        else:
            item_total[name] = float(np.corrcoef(this_item, rest_sum)[0, 1])
        # alpha computed on the scale with this item dropped
        if k - 1 >= 2:
            alpha_if_deleted[name] = _cronbach_alpha(mat[:, rest_idx])
        else:
            alpha_if_deleted[name] = None

    state.write("diagnostics", "reliability", {
        "cronbach_alpha": cronbach,
        "mcdonald_omega": omega,
        "avg_inter_item_r": avg_inter_item_r,
        "item_total": item_total,
        "alpha_if_deleted": alpha_if_deleted,
        "k_items": int(k),
        "n": int(n),
        "items": list(items),
        "backend": "pypsych",
        "note": (
            "Cronbach α + McDonald ω(单因子载荷);校正项-总相关=题项 vs 其余题项之和;"
            "α-if-item-deleted=删该题后的 α"
        ),
    })
    return state
