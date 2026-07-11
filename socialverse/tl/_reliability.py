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

__all__ = ["reliability", "icc", "correlation_test"]


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


# -------------------------------------------------------------------------- icc
@register(
    name="icc",
    aliases=["组内相关", "icc", "intraclass_correlation"],
    category="psychometrics",
    tier="plus",
    skill="(测量缺口)",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description=(
        "组内相关系数 ICC:被试×评分者矩阵的两因素方差分解,给出 "
        "ICC1/ICC2/ICC3 单评分者与 ICC1k/ICC2k/ICC3k 平均 k 评分者六型 "
        "(F/df/p/置信区间),复现 R psych::ICC"
    ),
    requires={"sources": ["datasets"]},
    produces={"models": ["icc"]},
    auto_fix="escalate",
)
def icc(state: StudyState, **kwargs: Any) -> StudyState:
    """Intraclass correlations (ICC1/2/3 + k-forms) for a subjects×raters matrix.

    Rows are subjects (targets), columns are raters (judges). Delegates the
    two-way ANOVA decomposition + the six Shrout & Fleiss ICC coefficients to the
    parity-gated :func:`pypsych.ICC` port (replicates ``psych::ICC``).

    kwargs
    ------
    raters : list[str], optional
        The rater/judge columns forming the ratings matrix. Default: every
        column whose name starts with ``"rater"`` or ``"judge"``; if none match,
        all numeric columns are used.
    alpha : float, optional
        Significance level for the confidence bounds (default ``0.05``).
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        state.write("models", "icc", {
            "type": ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"],
            "ICC": {},
            "F": {},
            "df1": {},
            "df2": {},
            "p": {},
            "lower": {},
            "upper": {},
            "MSB": None,
            "MSJ": None,
            "MSE": None,
            "MSW": None,
            "n_subjects": 0,
            "n_raters": 0,
            "raters": [],
            "backend": "pypsych",
            "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法计算 ICC")

    # ---- resolve the rater columns ----
    raters = list(kwargs.get("raters") or [])
    if not raters:
        raters = [c for c in df.columns
                  if str(c).startswith("rater") or str(c).startswith("judge")]
    if not raters:
        raters = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    raters = [c for c in raters
              if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(raters) < 2:
        return _empty(f"ICC 需要至少 2 个评分者列(找到 {len(raters)} 个)")

    work = df[raters].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        return _empty("有效被试不足(去缺失后 < 3 行),无法计算 ICC")

    mat = work.to_numpy(dtype=float)

    try:
        alpha = float(kwargs.get("alpha", 0.05))
    except Exception:
        alpha = 0.05

    try:
        from ..external.pypsych import ICC as _pp_icc
        res = _pp_icc(mat, alpha=alpha)
    except Exception as exc:  # pragma: no cover - graceful degradation
        return _empty(f"ICC 计算失败:{exc}")

    types = list(res["type"])

    def _named(arr) -> dict[str, float]:
        vals = np.asarray(arr, dtype=float).ravel()
        return {t: float(v) for t, v in zip(types, vals)}

    state.write("models", "icc", {
        "type": types,
        "ICC": _named(res["ICC"]),
        "F": _named(res["F"]),
        "df1": _named(res["df1"]),
        "df2": _named(res["df2"]),
        "p": _named(res["p"]),
        "lower": _named(res["lower"]),
        "upper": _named(res["upper"]),
        "MSB": float(res["MSB"]),
        "MSJ": float(res["MSJ"]),
        "MSE": float(res["MSE"]),
        "MSW": float(res["MSW"]),
        "n_subjects": int(res["n_obs"]),
        "n_raters": int(res["n_judge"]),
        "alpha": float(alpha),
        "raters": list(raters),
        "backend": "pypsych",
        "note": (
            "组内相关:两因素方差分解(被试+评分者);ICC1/2/3=单评分者,"
            "ICC1k/2k/3k=平均 k 评分者;每型附 F/df1/df2/p 及置信区间"
        ),
    })
    return state


# --------------------------------------------------------------- correlation_test
@register(
    name="correlation_test",
    aliases=["相关检验", "correlation_test", "corr_test"],
    category="psychometrics",
    tier="plus",
    skill="(测量缺口)",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description=(
        "相关矩阵显著性检验:Pearson r 矩阵 + 成对样本量 n + t 值 + 双尾原始 p "
        "+ 标准误,复现 R psych::corr.test(未校正 p)"
    ),
    requires={"sources": ["datasets"]},
    produces={"models": ["corr_test"]},
    auto_fix="escalate",
)
def correlation_test(state: StudyState, **kwargs: Any) -> StudyState:
    """Correlation matrix + pairwise n + p for a set of numeric variables.

    Delegates to the parity-gated :func:`pypsych.corr_test` port (replicates
    ``psych::corr.test`` with ``method="pearson"``): returns the Pearson r
    matrix, the (constant, after listwise deletion) sample size n, the t-value
    matrix, the two-sided RAW (unadjusted) p matrix, and the standard-error
    matrix.

    kwargs
    ------
    vars : list[str], optional
        The variable columns to correlate. Default: all numeric columns.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        state.write("models", "corr_test", {
            "vars": [],
            "r": {},
            "t": {},
            "p": {},
            "se": {},
            "n": 0,
            "k_vars": 0,
            "backend": "pypsych",
            "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法计算相关检验")

    # ---- resolve the variable columns ----
    variables = list(kwargs.get("vars") or [])
    if not variables:
        variables = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    variables = [c for c in variables
                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(variables) < 2:
        return _empty(f"相关检验需要至少 2 个数值变量(找到 {len(variables)} 个)")

    work = df[variables].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        return _empty("有效样本不足(去缺失后 < 3 行),无法计算相关检验")

    mat = work.to_numpy(dtype=float)

    try:
        from ..external.pypsych import corr_test as _pp_corr
        res = _pp_corr(mat)
    except Exception as exc:  # pragma: no cover - graceful degradation
        return _empty(f"相关检验计算失败:{exc}")

    def _matrix(arr) -> dict[str, dict[str, float]]:
        a = np.asarray(arr, dtype=float)
        return {
            variables[i]: {variables[j]: float(a[i, j])
                           for j in range(len(variables))}
            for i in range(len(variables))
        }

    state.write("models", "corr_test", {
        "vars": list(variables),
        "r": _matrix(res["r"]),
        "t": _matrix(res["t"]),
        "p": _matrix(res["p"]),
        "se": _matrix(res["se"]),
        "n": int(res["n"]),
        "k_vars": int(len(variables)),
        "backend": "pypsych",
        "note": (
            "Pearson 相关矩阵 + 双尾原始(未校正)p;t=r·√(n−2)/√(1−r²);"
            "对角 r=1、p=0;成对 n 为去缺失后的常数样本量"
        ),
    })
    return state
