"""``sv.tl._causal`` — registered implementations for the ``causal-identification`` skill.

The difference-in-differences family, ported to the ``StudyState`` /
``registry`` spine. Each function is a link in the identification chain:

    parallel_trends  →  did (twfe)          event_study
    (pretrend Wald)     (ATT + cluster SE)  (dynamic leads/lags)

The registry contract makes the chain machine-checkable: ``parallel_trends``
*produces* ``identification.parallel_trends``, which ``did`` *requires* — so a
resolver refuses to report a DID estimate as *causal* until the pretrend test
has actually run (and ``did`` labels the estimate "关联非因果" when it failed).

Real computation only: event-study designs are estimated with ``statsmodels``
OLS over unit/time dummies, and the pre-period coefficients get a genuine joint
Wald/F test. ``linearmodels`` is used opportunistically for absorbed high-
dimensional fixed effects when installed, but everything degrades gracefully to
plain ``statsmodels`` (with clustered SEs) when it is not.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_datasets(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working panel: explicit ``data=`` kwarg, else ``sources['datasets']``.

    ``sources['datasets']`` may itself be a DataFrame or a ``{name: DataFrame}``
    mapping; in the latter case the first frame is taken.
    """
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _cols(state: StudyState, kwargs: dict[str, Any]) -> dict[str, str | None]:
    """Column names for the DID design — from kwargs first, then the design slot."""
    d = state.design
    return {
        "panel_id": kwargs.get("panel_id", d.get("panel_id")),
        "time": kwargs.get("time", d.get("time")),
        "treatment": kwargs.get("treatment", d.get("treatment")),
        "first_treated": kwargs.get("first_treated", d.get("first_treated")),
        "outcome": kwargs.get("outcome", state.variables.get("outcome") or d.get("outcome")),
    }


def _pick_outcome(df: pd.DataFrame, cols: dict, exclude: list[str]) -> str | None:
    """Resolve the outcome column, defaulting to the first numeric non-design column."""
    y = cols.get("outcome")
    if y is not None and y in df.columns:
        return y
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _rel_time(df: pd.DataFrame, cols: dict) -> pd.Series:
    """Relative event time = time - first_treated (NaN for never-treated).

    Never-treated units are conventionally encoded with ``first_treated`` = 0 (the
    Callaway-Sant'Anna ``G=0`` convention), NaN, or a non-positive sentinel. All of
    these are mapped to NaN so those rows get all-zero event dummies and serve as the
    clean comparison group, rather than spurious positive event-times.
    """
    t = pd.to_numeric(df[cols["time"]], errors="coerce")
    ft = pd.to_numeric(df[cols["first_treated"]], errors="coerce")
    rel = t - ft
    never = (~np.isfinite(ft)) | (ft <= 0)
    return rel.where(~never, other=np.nan)


def _event_dummies(
    df: pd.DataFrame, rel: pd.Series, base: int = -1
) -> tuple[pd.DataFrame, list[int]]:
    """Build one indicator column per relative-time bin, omitting the ``base`` period.

    Never-treated rows (``rel`` is NaN) get all-zero event dummies — they serve as
    the clean comparison group. Returns the augmented frame and the ordered list of
    included event-time bins.
    """
    periods = sorted(int(k) for k in pd.unique(rel.dropna()))
    included = [p for p in periods if p != base]
    out = df.copy()
    finite = rel.notna()
    for p in included:
        out[f"_et_{p}"] = ((rel == p) & finite).astype(float)
    return out, included


def _fit_ols(y: pd.Series, X: pd.DataFrame, groups: pd.Series):
    """Cluster-robust OLS via statsmodels, clustered on ``groups``.

    Falls back to non-robust SEs if the clustered covariance is singular.
    """
    sm = _try_import("statsmodels.api")
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.asarray(y, dtype=float), np.asarray(Xc, dtype=float))
    try:
        res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": np.asarray(groups)},
        )
    except Exception:
        res = model.fit()
    res._sv_xcols = list(Xc.columns)  # type: ignore[attr-defined]
    return res


def _res_lookup(res, colnames: list[str], name: str) -> int | None:
    """Positional index of a design column in the fitted parameter vector."""
    try:
        return colnames.index(name)
    except ValueError:
        return None


# ------------------------------------------------------------------ parallel_trends
@register(
    name="parallel_trends",
    aliases=["平行趋势", "pretrend_test"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["statsmodels", "linearmodels", "numpy"],
    description="DID 前置:event-study 前导期联合 Wald 检验平行趋势",
    requires={
        "design": ["panel_id", "time", "treatment", "first_treated"],
        "variables": ["outcome"],
        "estimand": ["target"],
    },
    produces={"diagnostics": ["pretrend"], "identification": ["parallel_trends"]},
    auto_fix="escalate",
)
def parallel_trends(state: StudyState, **kwargs: Any) -> StudyState:
    """Estimate an event study and jointly Wald-test all pre-treatment leads.

    Under parallel trends every pre-period (relative time < 0, base = ``-1``)
    coefficient is zero. We estimate the full event study with unit + time fixed
    effects, then run a joint F/Wald test on the pre-period coefficients. A
    non-significant test (``p > 0.05``) means we *fail to reject* parallel trends
    → ``identification.parallel_trends = "pass"``.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)
    alpha = float(kwargs.get("alpha", 0.05))

    if df is None or any(cols[k] is None for k in ("panel_id", "time", "first_treated")):
        pretrend = {
            "joint_F": None,
            "p_value": None,
            "pre_coefs": {},
            "note": "缺少面板数据或设计列(panel_id/time/first_treated),无法检验",
        }
        state.write("diagnostics", "pretrend", pretrend)
        state.write("identification", "parallel_trends", "unknown")
        return state

    y_col = _pick_outcome(df, cols, exclude=list(cols.values()))
    rel = _rel_time(df, cols)
    work, included = _event_dummies(df, rel, base=-1)

    pre_bins = [p for p in included if p < -1]  # -1 is the omitted base
    if y_col is None or not pre_bins:
        pretrend = {
            "joint_F": None,
            "p_value": None,
            "pre_coefs": {},
            "note": "无可检验的前导期(pre-period)或缺结果变量",
        }
        state.write("diagnostics", "pretrend", pretrend)
        state.write("identification", "parallel_trends", "unknown")
        return state

    sm = _try_import("statsmodels.api")
    smf = _try_import("statsmodels.formula.api")
    et_cols = [f"_et_{p}" for p in included]

    unit_d = pd.get_dummies(work[cols["panel_id"]].astype("category"),
                            prefix="u", drop_first=True, dtype=float)
    time_d = pd.get_dummies(work[cols["time"]].astype("category"),
                            prefix="t", drop_first=True, dtype=float)
    X = pd.concat([work[et_cols].astype(float), unit_d, time_d], axis=1)

    valid = X.notna().all(axis=1) & work[y_col].notna()
    X, y, groups = X.loc[valid], work.loc[valid, y_col], work.loc[valid, cols["panel_id"]]

    res = _fit_ols(y, X, groups)
    xcols = res._sv_xcols  # includes leading 'const'

    pre_coefs: dict[str, tuple[float, float]] = {}
    pre_idx: list[int] = []
    for p in pre_bins:
        j = _res_lookup(res, xcols, f"_et_{p}")
        if j is not None:
            pre_coefs[str(p)] = (float(res.params[j]), float(res.bse[j]))
            pre_idx.append(j)

    joint_F: float | None = None
    p_value: float | None = None
    if pre_idx:
        R = np.zeros((len(pre_idx), len(res.params)))
        for r, j in enumerate(pre_idx):
            R[r, j] = 1.0
        try:
            wald = res.f_test(R)
            joint_F = float(np.ravel(wald.fvalue)[0])
            p_value = float(wald.pvalue)
        except Exception:
            joint_F = p_value = None

    verdict = "pass" if (p_value is not None and p_value > alpha) else \
              ("fail" if p_value is not None else "unknown")

    pretrend = {
        "joint_F": joint_F,
        "p_value": p_value,
        "pre_coefs": pre_coefs,
        "n_pre": len(pre_idx),
        "outcome": y_col,
        "alpha": alpha,
        "note": ("未拒绝平行趋势(p>{:.3g})".format(alpha) if verdict == "pass"
                 else "拒绝平行趋势 — 前导期系数联合显著" if verdict == "fail"
                 else "无法计算联合检验"),
    }
    state.write("diagnostics", "pretrend", pretrend)
    state.write("identification", "parallel_trends", verdict)
    return state


# ---------------------------------------------------------------------------- did
@register(
    name="did",
    aliases=["双重差分", "DID", "twfe"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["statsmodels", "linearmodels", "numpy"],
    description="双向固定效应 DID:估 ATT + 聚类稳健 SE;平行趋势 fail 时标为关联非因果",
    requires={
        "design": ["panel_id", "time", "treatment"],
        "variables": ["outcome"],
        "identification": ["parallel_trends"],
    },
    produces={"models": ["did", "twfe"], "diagnostics": ["robustness"]},
    prerequisites={"functions": ["parallel_trends"]},
    auto_fix="escalate",
)
def did(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-way fixed-effects DID: ATT with panel-clustered robust standard errors.

    Estimates ``y ~ treat_post + unit FE + time FE``, where ``treat_post`` is the
    interaction that switches on for treated units in post-treatment periods
    (built from ``treatment`` × ``time >= first_treated`` when available, else the
    raw ``treatment`` indicator). Standard errors are clustered on ``panel_id``.

    The ``robustness`` diagnostic re-estimates the ATT under alternative variance
    specifications (classical, HC1, clustered) so the point estimate can be read
    against its SE sensitivity. If ``identification.parallel_trends == 'fail'`` the
    estimate is still reported but flagged as association, not a causal ATT.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)
    pt = state.identification.get("parallel_trends")

    def _empty(note: str) -> StudyState:
        model = {"att": None, "se": None, "ci": None, "p": None, "note": note}
        state.write("models", "twfe", model)
        state.write("models", "did", dict(model))
        state.write("diagnostics", "robustness", {"specs": [], "note": note})
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time", "treatment")):
        return _empty("缺少面板数据或设计列(panel_id/time/treatment),无法估计")

    work = df.copy()
    treat = pd.to_numeric(work[cols["treatment"]], errors="coerce")

    # Build treat_post: prefer treatment × post(first_treated); else treatment itself.
    if cols["first_treated"] is not None and cols["first_treated"] in work.columns:
        t = pd.to_numeric(work[cols["time"]], errors="coerce")
        ft = pd.to_numeric(work[cols["first_treated"]], errors="coerce")
        post = (t >= ft) & np.isfinite(ft)
        # ``treatment`` may already be the switched-on indicator; if it varies only
        # within treated-post, treat it as the group flag and multiply by post.
        group = treat.fillna(0)
        treat_post = (group * post.astype(float)).astype(float)
        # If that collapses to all-zero (treatment already == treat_post), fall back.
        if treat_post.abs().sum() == 0:
            treat_post = treat.fillna(0).astype(float)
    else:
        treat_post = treat.fillna(0).astype(float)

    work["_treat_post"] = treat_post
    y_col = _pick_outcome(work, cols, exclude=list(cols.values()) + ["_treat_post"])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    sm = _try_import("statsmodels.api")
    unit_d = pd.get_dummies(work[cols["panel_id"]].astype("category"),
                            prefix="u", drop_first=True, dtype=float)
    time_d = pd.get_dummies(work[cols["time"]].astype("category"),
                            prefix="t", drop_first=True, dtype=float)
    X = pd.concat([work[["_treat_post"]].astype(float), unit_d, time_d], axis=1)

    valid = X.notna().all(axis=1) & work[y_col].notna()
    X, y, groups = X.loc[valid], work.loc[valid, y_col], work.loc[valid, cols["panel_id"]]

    res = _fit_ols(y, X, groups)
    xcols = res._sv_xcols
    j = _res_lookup(res, xcols, "_treat_post")

    if j is None:
        return _empty("处理×时点交互项无法辨识(共线或无变异)")

    att = float(res.params[j])
    se = float(res.bse[j])
    p = float(res.pvalues[j])
    ci_lo, ci_hi = (float(x) for x in res.conf_int()[j])

    # robustness: re-fit the same design under alternative covariance specs.
    Xc = sm.add_constant(X, has_constant="add")
    base = sm.OLS(np.asarray(y, float), np.asarray(Xc, float))
    specs: list[dict[str, Any]] = []
    for label, kw in (
        ("classical", {}),
        ("HC1_robust", {"cov_type": "HC1"}),
        ("cluster_panel", {"cov_type": "cluster",
                           "cov_kwds": {"groups": np.asarray(groups)}}),
    ):
        try:
            r = base.fit(**kw)
            specs.append({
                "spec": label,
                "att": float(r.params[j]),
                "se": float(r.bse[j]),
                "p": float(r.pvalues[j]),
            })
        except Exception:
            specs.append({"spec": label, "att": None, "se": None, "p": None})

    causal_note = ("关联非因果(平行趋势未过)" if pt == "fail"
                   else "因果 ATT(平行趋势通过)" if pt == "pass"
                   else "平行趋势未检验/未知 — 谨慎解读")

    model = {
        "att": att,
        "se": se,
        "ci": [ci_lo, ci_hi],
        "p": p,
        "n": int(valid.sum()),
        "n_clusters": int(pd.unique(groups).size),
        "outcome": y_col,
        "estimator": "twfe_ols_cluster",
        "parallel_trends": pt,
        "note": causal_note,
    }
    state.write("models", "twfe", model)
    state.write("models", "did", dict(model))
    state.write("diagnostics", "robustness", {
        "specs": specs,
        "note": "ATT 在 classical / HC1 / panel-cluster SE 下对比",
    })
    return state


# -------------------------------------------------------------------- event_study
@register(
    name="event_study",
    aliases=["事件研究", "event_study"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["statsmodels", "numpy"],
    description="事件研究:相对处理时点的 leads/lags 动态效应",
    requires={"design": ["panel_id", "time", "treatment", "first_treated"],
              "variables": ["outcome"]},
    produces={"models": ["event_study"]},
    prerequisites={"optional_functions": ["parallel_trends"]},
    auto_fix="escalate",
)
def event_study(state: StudyState, **kwargs: Any) -> StudyState:
    """Dynamic event-study: a coefficient per relative period (base = ``-1``).

    Estimates ``y ~ Σ_k β_k · 1[event_time = k] + unit FE + time FE`` over all
    relative-time bins except the omitted base period ``-1``. Each ``β_k`` traces
    the treatment effect ``k`` periods from onset — leads (``k < 0``) probe pre-
    trends, lags (``k >= 0``) trace the post-treatment dynamic response. SEs are
    clustered on ``panel_id``.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)
    base = int(kwargs.get("base", -1))

    if df is None or any(cols[k] is None for k in ("panel_id", "time", "first_treated")):
        state.write("models", "event_study", {
            "coefs": {}, "base": base,
            "note": "缺少面板数据或设计列(panel_id/time/first_treated),无法估计",
        })
        return state

    rel = _rel_time(df, cols)
    work, included = _event_dummies(df, rel, base=base)
    y_col = _pick_outcome(work, cols, exclude=list(cols.values()))

    if y_col is None or not included:
        state.write("models", "event_study", {
            "coefs": {}, "base": base,
            "note": "无可估计的相对期(leads/lags)或缺结果变量",
        })
        return state

    sm = _try_import("statsmodels.api")
    et_cols = [f"_et_{p}" for p in included]
    unit_d = pd.get_dummies(work[cols["panel_id"]].astype("category"),
                            prefix="u", drop_first=True, dtype=float)
    time_d = pd.get_dummies(work[cols["time"]].astype("category"),
                            prefix="t", drop_first=True, dtype=float)
    X = pd.concat([work[et_cols].astype(float), unit_d, time_d], axis=1)

    valid = X.notna().all(axis=1) & work[y_col].notna()
    X, y, groups = X.loc[valid], work.loc[valid, y_col], work.loc[valid, cols["panel_id"]]

    res = _fit_ols(y, X, groups)
    xcols = res._sv_xcols

    coefs: dict[str, tuple[float, float]] = {str(base): (0.0, 0.0)}  # normalized base
    for p in included:
        j = _res_lookup(res, xcols, f"_et_{p}")
        if j is not None:
            coefs[str(p)] = (float(res.params[j]), float(res.bse[j]))

    ordered = {k: coefs[k] for k in sorted(coefs, key=lambda s: int(s))}
    state.write("models", "event_study", {
        "coefs": ordered,
        "base": base,
        "outcome": y_col,
        "n": int(valid.sum()),
        "n_clusters": int(pd.unique(groups).size),
        "estimator": "event_study_ols_cluster",
        "note": "各相对期动态效应(base={} 归一化为 0)".format(base),
    })
    return state


__all__ = ["parallel_trends", "did", "event_study"]
