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


def _within_fit(y, D, unit, time, groups):
    """Two-way fixed-effects regression of ``y`` on the columns of ``D``, absorbing
    unit and time effects by within-transformation.

    Explicit unit dummies are infeasible once a panel has thousands of units — the
    design matrix becomes huge and rank-deficient and OLS fails (``did`` /
    ``event_study`` / ``parallel_trends`` all shared this ceiling). This routine
    demeans ``y`` and each column of ``D`` by unit and time via alternating
    projections (which handles unbalanced panels), then regresses the residualised
    outcome on the residualised regressors, exactly as ``reghdfe`` / ``fixest`` do.
    It returns the coefficient vector plus classical, HC1 and cluster-robust
    covariance matrices (clustered on ``groups``) with the standard small-sample
    corrections, so it reproduces the dummy-OLS estimates at scale. ``D`` may be 1-D
    (single regressor, e.g. ``treat_post``) or 2-D (event-time dummies). Returns
    ``None`` if the regressors have no residual variation after absorbing the FE.
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float)
    if D.ndim == 1:
        D = D[:, None]
    n, p = D.shape
    u = pd.factorize(np.asarray(unit))[0]
    t = pd.factorize(np.asarray(time))[0]
    n_u, n_t = int(u.max()) + 1, int(t.max()) + 1
    uc = np.bincount(u).astype(float)
    tc = np.bincount(t).astype(float)

    def _demean(v: np.ndarray) -> np.ndarray:
        v = v.astype(float).copy()
        for _ in range(5000):
            v0 = v
            v = v - (np.bincount(u, v) / uc)[u]
            v = v - (np.bincount(t, v) / tc)[t]
            if np.max(np.abs(v - v0)) < 1e-11:
                break
        return v

    yt = _demean(y)
    Dt = np.column_stack([_demean(D[:, c]) for c in range(p)])
    XtX = Dt.T @ Dt
    if np.linalg.matrix_rank(XtX) < p:
        return None
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Dt.T @ yt)
    e = yt - Dt @ beta
    k = p + (n_u - 1) + (n_t - 1)  # regressors + absorbed FE parameters
    dof = max(n - k, 1)

    sigma2 = float(e @ e) / dof
    V_classical = sigma2 * XtX_inv

    Xe = Dt * e[:, None]
    V_hc1 = XtX_inv @ (Xe.T @ Xe) @ XtX_inv * (n / dof)

    g = pd.factorize(np.asarray(groups))[0]
    n_g = int(g.max()) + 1
    Sg = np.column_stack([np.bincount(g, Xe[:, c], minlength=n_g) for c in range(p)])
    adj = (n_g / (n_g - 1.0)) * ((n - 1.0) / (n - k)) if n_g > 1 else 1.0
    V_cluster = XtX_inv @ (Sg.T @ Sg) @ XtX_inv * adj

    return {
        "beta": beta,
        "V_classical": V_classical,
        "V_hc1": V_hc1,
        "V_cluster": V_cluster,
        "dof": dof,
        "df_cluster": n_g - 1,
        "n": n,
        "n_clusters": n_g,
    }


def _within_coef(fit: dict, i: int, which: str = "V_cluster") -> dict:
    """Point estimate + SE + two-sided p + 95% CI for coefficient ``i`` of a
    :func:`_within_fit` result under covariance spec ``which``."""
    from scipy import stats

    b = float(fit["beta"][i])
    se = float(np.sqrt(fit[which][i, i]))
    df = fit["df_cluster"] if which == "V_cluster" else fit["dof"]
    if se > 0:
        p = float(2 * stats.t.sf(abs(b / se), df))
        tcrit = float(stats.t.ppf(0.975, df))
        ci = [b - tcrit * se, b + tcrit * se]
    else:
        p, ci = float("nan"), [float("nan"), float("nan")]
    return {"coef": b, "se": se, "p": p, "ci": ci}


# ------------------------------------------------------- pydid (Callaway-Sant'Anna)
def _cs_inputs(df: pd.DataFrame, cols: dict, y_col: str) -> dict | None:
    """Reshape the StudyState panel into the column-dict ``pydid.att_gt`` expects.

    Callaway & Sant'Anna needs a first-treatment (``gname``) column with ``0`` for
    the never-treated control group. We derive it from ``first_treated`` (mapping
    NaN / non-positive sentinels to ``0``, matching the ``G=0`` convention used by
    :func:`_rel_time`). ``idname`` / ``tname`` are kept as-is (NOT numeric-coerced
    beyond what CS requires) so grouping labels are not corrupted. Returns ``None``
    when the design lacks the columns the port needs (so the caller can fall back).
    """
    if cols.get("first_treated") is None or cols["first_treated"] not in df.columns:
        return None
    if cols.get("panel_id") is None or cols.get("time") is None:
        return None

    t = pd.to_numeric(df[cols["time"]], errors="coerce")
    ft = pd.to_numeric(df[cols["first_treated"]], errors="coerce")
    # never-treated / invalid onset -> 0 (CS never-treated code); post -> first-treat period
    g = ft.where(np.isfinite(ft) & (ft > 0), other=0.0).astype(float)

    ok = t.notna() & df[y_col].notna()
    if not ok.any():
        return None
    # unit ids kept verbatim (do NOT numeric-coerce label columns)
    ids = df[cols["panel_id"]].to_numpy()[ok.to_numpy()]
    return {
        "yname": "y", "tname": "t", "idname": "id", "gname": "g",
        "data": {
            "y": np.asarray(df[y_col], float)[ok.to_numpy()],
            "t": t.to_numpy()[ok.to_numpy()],
            "id": ids,
            "g": g.to_numpy()[ok.to_numpy()],
        },
    }


def _cs_estimate(
    df: pd.DataFrame, cols: dict, y_col: str, control_group: str = "nevertreated"
) -> dict | None:
    """Run the parity-verified pydid backend and return CS point estimates.

    Returns a dict with the overall (``simple``) ATT, the ``dynamic`` event-time
    path (``egt`` -> ATT), and the ``group`` (per-cohort) and ``calendar`` (per-
    period) aggregations, or ``None`` if the port is unavailable / not applicable
    (raises are swallowed by the caller's try/except regardless). Point estimates
    only — the port's bootstrap SEs are stochastic and NOT parity-gated, so SEs are
    left to the existing TWFE path.

    ``control_group`` selects the comparison group for ``att_gt`` and mirrors
    ``did::att_gt`` — ``'nevertreated'`` (default, back-compatible) uses only the
    never-treated units; ``'notyettreated'`` also borrows the not-yet-treated units
    as controls. For ``'notyettreated'`` a never-treated control is not required.
    """
    from ..external.pydid import att_gt, aggte  # parity-gated port

    if control_group not in ("nevertreated", "notyettreated"):
        control_group = "nevertreated"

    payload = _cs_inputs(df, cols, y_col)
    if payload is None:
        return None
    # need at least one treated group; a never-treated control is required only for
    # the 'nevertreated' comparison (not-yet-treated can borrow later cohorts).
    gvals = np.asarray(payload["data"]["g"], float)
    if not np.any(gvals > 0):
        return None
    if control_group == "nevertreated" and not np.any(gvals == 0):
        return None

    res = att_gt(control_group=control_group, **payload)  # est_method='reg', varying base
    simple = aggte(res, type="simple")
    dynamic = aggte(res, type="dynamic")
    egt = {int(e): float(a) for e, a in zip(dynamic.egt, dynamic.att_egt)}

    # group aggregation: one ATT per treatment cohort g
    by_group: dict[int, float] = {}
    group_overall: float | None = None
    try:
        grp = aggte(res, type="group")
        by_group = {int(g): float(a) for g, a in zip(grp.egt, grp.att_egt)}
        group_overall = float(grp.overall_att)
    except Exception:
        by_group, group_overall = {}, None

    # calendar aggregation: one ATT per calendar period t
    by_calendar: dict[int, float] = {}
    calendar_overall: float | None = None
    try:
        cal = aggte(res, type="calendar")
        by_calendar = {int(t): float(a) for t, a in zip(cal.egt, cal.att_egt)}
        calendar_overall = float(cal.overall_att)
    except Exception:
        by_calendar, calendar_overall = {}, None

    return {
        "overall_att": float(simple.overall_att),
        "event": egt,
        "dynamic_overall": float(dynamic.overall_att),
        "group": by_group,
        "group_overall": group_overall,
        "calendar": by_calendar,
        "calendar_overall": calendar_overall,
        "control_group": control_group,
    }


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
    et_cols = [f"_et_{p}" for p in included]

    pre_coefs: dict[str, tuple[float, float]] = {}
    joint_F: float | None = None
    p_value: float | None = None
    n_pre = 0

    n_fe = int(work[cols["panel_id"]].nunique()) + int(work[cols["time"]].nunique())
    if n_fe > 150:
        # High-dimensional FE: absorb by within-transformation, then jointly test
        # the pre-period leads with a cluster-robust Wald F.
        valid = work[et_cols].notna().all(axis=1) & work[y_col].notna()
        w = work.loc[valid]
        fit = _within_fit(w[y_col], w[et_cols].to_numpy(float),
                          w[cols["panel_id"]], w[cols["time"]], w[cols["panel_id"]])
        pre_pos = [i for i, pbin in enumerate(included) if pbin < -1]
        if fit is not None:
            for i in pre_pos:
                c = _within_coef(fit, i, "V_cluster")
                pre_coefs[str(included[i])] = (c["coef"], c["se"])
            n_pre = len(pre_pos)
            if pre_pos:
                from scipy import stats
                b = fit["beta"][pre_pos]
                V = fit["V_cluster"][np.ix_(pre_pos, pre_pos)]
                try:
                    wald = float(b @ np.linalg.solve(V, b))
                    df1, df2 = len(pre_pos), fit["df_cluster"]
                    joint_F = wald / df1
                    p_value = float(stats.f.sf(joint_F, df1, df2))
                except Exception:
                    joint_F = p_value = None
    else:
        unit_d = pd.get_dummies(work[cols["panel_id"]].astype("category"),
                                prefix="u", drop_first=True, dtype=float)
        time_d = pd.get_dummies(work[cols["time"]].astype("category"),
                                prefix="t", drop_first=True, dtype=float)
        X = pd.concat([work[et_cols].astype(float), unit_d, time_d], axis=1)

        valid = X.notna().all(axis=1) & work[y_col].notna()
        X, y, groups = X.loc[valid], work.loc[valid, y_col], work.loc[valid, cols["panel_id"]]

        res = _fit_ols(y, X, groups)
        xcols = res._sv_xcols  # includes leading 'const'

        pre_idx: list[int] = []
        for p in pre_bins:
            j = _res_lookup(res, xcols, f"_et_{p}")
            if j is not None:
                pre_coefs[str(p)] = (float(res.params[j]), float(res.bse[j]))
                pre_idx.append(j)
        n_pre = len(pre_idx)

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
        "n_pre": n_pre,
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

    # High-dimensional fixed effects (many units) make explicit dummies infeasible —
    # the design matrix is huge and rank-deficient and OLS fails. Above a threshold
    # of absorbed levels, switch to within-transformation (reghdfe/fixest-style);
    # small panels keep the explicit-dummy path so nothing about existing behaviour
    # changes for the common toy/teaching cases.
    n_units = int(work[cols["panel_id"]].nunique())
    n_times = int(work[cols["time"]].nunique())
    use_within = (n_units + n_times) > 150

    if use_within:
        valid = work["_treat_post"].notna() & work[y_col].notna()
        w = work.loc[valid]
        fit = _within_fit(w[y_col], w["_treat_post"].to_numpy(float),
                          w[cols["panel_id"]], w[cols["time"]], w[cols["panel_id"]])
        if fit is None:
            return _empty("处理×时点交互项无法辨识(共线或无变异)")
        cl = _within_coef(fit, 0, "V_cluster")
        att, se, p = cl["coef"], cl["se"], cl["p"]
        ci_lo, ci_hi = cl["ci"]
        n_used, n_clusters = fit["n"], fit["n_clusters"]
        estimator = "twfe_within_absorb_cluster"
        specs = [
            {"spec": "classical", "att": att, **{k: _within_coef(fit, 0, "V_classical")[k]
                                                 for k in ("se", "p")}},
            {"spec": "HC1_robust", "att": att, **{k: _within_coef(fit, 0, "V_hc1")[k]
                                                  for k in ("se", "p")}},
            {"spec": "cluster_panel", "att": att, "se": se, "p": p},
        ]
    else:
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
        n_used = int(valid.sum())
        n_clusters = int(pd.unique(groups).size)
        estimator = "twfe_ols_cluster"

        # robustness: re-fit the same design under alternative covariance specs.
        Xc = sm.add_constant(X, has_constant="add")
        base = sm.OLS(np.asarray(y, float), np.asarray(Xc, float))
        specs = []
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

    # Replace the ad-hoc TWFE point estimate with the parity-verified Callaway-
    # Sant'Anna overall ATT (``did::att_gt`` + ``aggte(type='simple')`` via the
    # pydid port). The port's point estimate is R-faithful to 1e-6; its bootstrap
    # SE is stochastic and NOT gated, so we keep the TWFE cluster-robust ``se`` and
    # re-centre the CI on the CS estimate. Guarded so any failure leaves the
    # pre-existing TWFE estimate fully intact.
    # ``control_group`` selects the CS comparison group: 'nevertreated' (default,
    # back-compatible) or 'notyettreated' (also borrows not-yet-treated cohorts).
    control_group = str(kwargs.get("control_group", "nevertreated"))
    if control_group not in ("nevertreated", "notyettreated"):
        control_group = "nevertreated"

    backend = "twfe"
    att_by_group: dict[int, float] = {}
    att_by_calendar: dict[int, float] = {}
    att_group_overall: float | None = None
    att_calendar_overall: float | None = None
    cs_control_group: str | None = None
    try:
        cs = _cs_estimate(work, cols, y_col, control_group=control_group)
        if cs is not None and np.isfinite(cs["overall_att"]):
            att = float(cs["overall_att"])
            backend = "pydid"
            cs_control_group = cs.get("control_group")
            att_by_group = cs.get("group", {}) or {}
            att_by_calendar = cs.get("calendar", {}) or {}
            att_group_overall = cs.get("group_overall")
            att_calendar_overall = cs.get("calendar_overall")
            if se is not None and np.isfinite(se) and se > 0:
                from scipy import stats as _st
                _tcrit = float(_st.t.ppf(0.975, max(n_clusters - 1, 1)))
                ci_lo, ci_hi = att - _tcrit * se, att + _tcrit * se
                p = float(2 * _st.t.sf(abs(att / se), max(n_clusters - 1, 1)))
    except Exception:
        backend = "twfe"  # pydid unavailable / not applicable — keep TWFE estimate

    model = {
        "att": att,
        "se": se,
        "ci": [ci_lo, ci_hi],
        "p": p,
        "n": n_used,
        "n_clusters": n_clusters,
        "outcome": y_col,
        "estimator": estimator,
        "parallel_trends": pt,
        "backend": backend,
        "control_group": cs_control_group,
        "att_by_group": att_by_group,
        "att_group_overall": att_group_overall,
        "att_by_calendar": att_by_calendar,
        "att_calendar_overall": att_calendar_overall,
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
    coefs: dict[str, tuple[float, float]] = {str(base): (0.0, 0.0)}  # normalized base

    n_fe = int(work[cols["panel_id"]].nunique()) + int(work[cols["time"]].nunique())
    if n_fe > 150:
        # High-dimensional FE: absorb by within-transformation (dummies infeasible).
        valid = work[et_cols].notna().all(axis=1) & work[y_col].notna()
        w = work.loc[valid]
        fit = _within_fit(w[y_col], w[et_cols].to_numpy(float),
                          w[cols["panel_id"]], w[cols["time"]], w[cols["panel_id"]])
        if fit is not None:
            for i, pbin in enumerate(included):
                c = _within_coef(fit, i, "V_cluster")
                coefs[str(pbin)] = (c["coef"], c["se"])
        n_used, n_clusters = (fit["n"], fit["n_clusters"]) if fit else (int(valid.sum()), 0)
        estimator = "event_study_within_absorb_cluster"
    else:
        unit_d = pd.get_dummies(work[cols["panel_id"]].astype("category"),
                                prefix="u", drop_first=True, dtype=float)
        time_d = pd.get_dummies(work[cols["time"]].astype("category"),
                                prefix="t", drop_first=True, dtype=float)
        X = pd.concat([work[et_cols].astype(float), unit_d, time_d], axis=1)

        valid = X.notna().all(axis=1) & work[y_col].notna()
        X, y, groups = X.loc[valid], work.loc[valid, y_col], work.loc[valid, cols["panel_id"]]

        res = _fit_ols(y, X, groups)
        xcols = res._sv_xcols

        for p in included:
            j = _res_lookup(res, xcols, f"_et_{p}")
            if j is not None:
                coefs[str(p)] = (float(res.params[j]), float(res.bse[j]))
        n_used, n_clusters = int(valid.sum()), int(pd.unique(groups).size)
        estimator = "event_study_ols_cluster"

    # Replace the ad-hoc TWFE event-time point estimates with the parity-verified
    # Callaway-Sant'Anna dynamic ATT(e) (``did::att_gt`` + ``aggte(type='dynamic')``
    # via the pydid port). Only the point estimate per relative period is swapped;
    # the TWFE cluster-robust SE for that period is retained (CS bootstrap SEs are
    # stochastic and NOT gated). Guarded so any failure leaves the TWFE path intact.
    backend = "twfe"
    try:
        cs = _cs_estimate(work, cols, y_col)
        if cs is not None and cs["event"]:
            backend = "pydid"
            for e, att_e in cs["event"].items():
                if not np.isfinite(att_e) or int(e) == base:
                    continue  # never overwrite the normalized base period
                key = str(int(e))
                se_prev = coefs.get(key, (0.0, 0.0))[1]  # keep TWFE SE for this period
                coefs[key] = (float(att_e), float(se_prev))
    except Exception:
        backend = "twfe"  # pydid unavailable / not applicable — keep TWFE estimates

    ordered = {k: coefs[k] for k in sorted(coefs, key=lambda s: int(s))}
    state.write("models", "event_study", {
        "coefs": ordered,
        "base": base,
        "outcome": y_col,
        "n": n_used,
        "n_clusters": n_clusters,
        "estimator": estimator,
        "backend": backend,
        "note": "各相对期动态效应(base={} 归一化为 0)".format(base),
    })
    return state


__all__ = ["parallel_trends", "did", "event_study"]
