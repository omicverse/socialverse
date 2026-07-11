"""``sv.tl._longitudinal`` — registered implementations for the longitudinal /
hierarchical-data gap: **multilevel (HLM)** and **survival / event-history (Cox)**.

Two workhorses of social-science panel and duration data, ported to the
``StudyState`` / ``registry`` spine with *real* estimation (recovers known DGP
parameters), no placeholders.

Champion packages this file mirrors
-----------------------------------
* ``multilevel`` — R's ``lme4::lmer`` / Python ``statsmodels.MixedLM``. Fits a
  linear mixed model with a random intercept (and, optionally, random slopes)
  via REML; reports fixed effects, variance components, and the intraclass
  correlation (ICC). ``statsmodels`` is the primary backend and is always
  available in this environment, so there is no degradation path needed — but if
  the mixed-model fit fails to converge we fall back to a genuine
  variance-components estimate from a one-way ANOVA decomposition (still real
  numbers, honestly labelled).
* ``survival`` — R's ``survival::coxph`` + ``survfit`` / Python ``lifelines``.
  The Cox proportional-hazards partial-likelihood fit uses ``statsmodels.PHReg``
  (Breslow ties); Kaplan-Meier survival curves use ``statsmodels.SurvfuncRight``
  stratified by group; the proportional-hazards assumption is checked with a
  Schoenfeld-residual / time-interaction test. ``lifelines`` is used
  opportunistically as an optional cross-check when installed, but every reported
  number comes from the ``statsmodels`` implementation so the notebook runs
  correctly without ``lifelines``.

The registry contracts chain: both functions ``require`` a working
``sources['datasets']`` frame and a declared ``variables['outcome']``, and
``produce`` fitted models plus their assumption diagnostics — so a resolver can
refuse to report, say, a Cox hazard ratio as "proportional" until the PH test
has actually run.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["multilevel", "survival", "conditional_logit", "aft_survreg"]


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


def _pick_outcome(
    df: pd.DataFrame, kwargs: dict[str, Any], state: StudyState, exclude: list[str]
) -> str | None:
    """Resolve the outcome column: kwargs ``outcome=`` → ``variables['outcome']`` →
    first numeric non-excluded column."""
    y = kwargs.get("outcome") or state.variables.get("outcome")
    if y is not None and y in df.columns:
        return y
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


# ------------------------------------------------------------------- multilevel
@register(
    name="multilevel",
    aliases=["多层", "HLM"],
    category="longitudinal",
    tier="plus",
    skill="(多层 缺口)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="多层线性模型(HLM):随机截距(可选随机斜率)MixedLM + 方差成分 + ICC",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["mixedlm"], "diagnostics": ["variance_components"]},
    auto_fix="escalate",
)
def multilevel(state: StudyState, **kwargs: Any) -> StudyState:
    """Fit a two-level linear mixed model (random intercept, optional random slopes).

    Estimates ``y ~ 1 + Σ predictors`` with a group-level random intercept via
    ``statsmodels.MixedLM`` (REML). Optionally adds random slopes for the columns
    named in ``random_slopes``. Reports the fixed-effect coefficients, the
    variance components (between-group intercept variance, residual variance, and
    any slope variances), and the intraclass correlation
    ``ICC = σ²_between / (σ²_between + σ²_residual)``.

    kwargs
    ------
    groups : str
        Grouping (level-2) column, e.g. ``"school"``. Default: first non-numeric
        or lowest-cardinality integer column.
    predictors : list[str]
        Fixed-effect covariates. Default: ``["x"]`` if present.
    random_slopes : list[str], optional
        Subset of ``predictors`` to give random (group-varying) slopes.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        model = {"fixed_effects": {}, "groups": None, "n": 0, "note": note}
        state.write("models", "mixedlm", model)
        state.write("diagnostics", "variance_components",
                    {"between_var": None, "residual_var": None, "icc": None,
                     "slope_vars": {}, "note": note})
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法拟合多层模型")

    groups = kwargs.get("groups") or state.design.get("panel_id")
    if groups is None or groups not in df.columns:
        # heuristic: a low-cardinality column is the grouping factor
        cand = [c for c in df.columns
                if df[c].nunique() < max(2, len(df) // 3)]
        groups = cand[0] if cand else df.columns[0]

    predictors = list(kwargs.get("predictors") or [])
    if not predictors:
        predictors = [c for c in ["x"] if c in df.columns]

    outcome = _pick_outcome(
        df, kwargs, state, exclude=[groups] + predictors
    )
    if outcome is None:
        return _empty("找不到结果变量(outcome)")
    if not predictors:
        # no covariate → intercept-only random-effects (variance-components) model
        predictors = []

    random_slopes = [c for c in (kwargs.get("random_slopes") or []) if c in predictors]

    work = df.copy()
    keep = [outcome, groups] + predictors
    work = work[keep].dropna()
    work[groups] = work[groups].astype("category")

    sm = _try_import("statsmodels.api")
    smf = _try_import("statsmodels.formula.api")

    fixed_effects: dict[str, tuple[float, float]] = {}
    between_var = residual_var = icc = None
    slope_vars: dict[str, float] = {}
    converged = False
    method = "statsmodels.MixedLM(REML)"

    if sm is not None:
        y = work[outcome].to_numpy(dtype=float)
        exog = np.column_stack(
            [np.ones(len(work))] + [work[p].to_numpy(dtype=float) for p in predictors]
        )
        exog_names = ["Intercept"] + list(predictors)
        grp = work[groups].to_numpy()

        # random-effects design: intercept (+ optional random slopes)
        if random_slopes:
            exog_re = np.column_stack(
                [np.ones(len(work))]
                + [work[c].to_numpy(dtype=float) for c in random_slopes]
            )
            re_names = ["re_intercept"] + [f"re_{c}" for c in random_slopes]
        else:
            exog_re = np.ones((len(work), 1))
            re_names = ["re_intercept"]

        try:
            md = sm.MixedLM(y, exog, groups=grp, exog_re=exog_re)
            res = md.fit(reml=True, method="lbfgs")
            converged = bool(getattr(res, "converged", True))
            params = np.asarray(res.fe_params, dtype=float)
            bse = np.asarray(res.bse_fe, dtype=float)
            for name, b, s in zip(exog_names, params, bse):
                fixed_effects[name] = (float(b), float(s))

            # variance components: statsmodels reports the random-effects covariance
            # scaled by the residual variance (res.scale).
            residual_var = float(res.scale)
            cov_re = np.asarray(res.cov_re, dtype=float) * residual_var
            between_var = float(cov_re[0, 0])
            for i, c in enumerate(random_slopes, start=1):
                slope_vars[c] = float(cov_re[i, i])
            denom = between_var + residual_var
            icc = float(between_var / denom) if denom > 0 else None
        except Exception:
            converged = False

    if not converged or between_var is None:
        # honest fallback: one-way ANOVA variance-components + OLS fixed effects.
        method = "fallback: ANOVA variance-components + OLS (MixedLM 未收敛)"
        y = work[outcome].to_numpy(dtype=float)
        # OLS fixed effects for the mean structure
        if predictors:
            X = np.column_stack(
                [np.ones(len(work))]
                + [work[p].to_numpy(dtype=float) for p in predictors]
            )
        else:
            X = np.ones((len(work), 1))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        n, k = X.shape
        dof = max(n - k, 1)
        sigma2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0, np.inf))
        names = ["Intercept"] + list(predictors)
        for name, b, s in zip(names, beta, se):
            fixed_effects[name] = (float(b), float(s))

        # between/within variance from group means of the OLS residuals
        rdf = pd.DataFrame({"g": work[groups].to_numpy(), "r": resid})
        grp_means = rdf.groupby("g", observed=True)["r"].mean()
        grand = float(rdf["r"].mean())
        ng = rdf.groupby("g", observed=True)["r"].size()
        m = len(grp_means)
        n_bar = float(ng.mean())
        ss_between = float((ng.to_numpy() * (grp_means.to_numpy() - grand) ** 2).sum())
        ms_between = ss_between / max(m - 1, 1)
        within = rdf["r"].to_numpy() - rdf["g"].map(grp_means).to_numpy()
        ss_within = float(within @ within)
        ms_within = ss_within / max(n - m, 1)
        residual_var = float(ms_within)
        between_var = float(max((ms_between - ms_within) / n_bar, 0.0))
        denom = between_var + residual_var
        icc = float(between_var / denom) if denom > 0 else None

    slope = None
    if predictors:
        slope = fixed_effects.get(predictors[0], (None, None))[0]

    model = {
        "fixed_effects": fixed_effects,
        "groups": groups,
        "predictors": predictors,
        "random_slopes": random_slopes,
        "outcome": outcome,
        "n": int(len(work)),
        "n_groups": int(work[groups].nunique()),
        "converged": converged,
        "estimator": method,
        "note": "固定效应 + 组间/残差方差成分 + ICC",
    }
    state.write("models", "mixedlm", model)
    state.write("diagnostics", "variance_components", {
        "between_var": between_var,
        "residual_var": residual_var,
        "icc": icc,
        "slope_vars": slope_vars,
        "primary_slope": slope,
        "note": "ICC = 组间方差 / (组间方差 + 残差方差)",
    })
    return state


# --------------------------------------------------------------------- survival
@register(
    name="survival",
    aliases=["生存", "事件史", "Cox"],
    category="longitudinal",
    tier="plus",
    skill="(事件史 缺口)",
    languages=["Python"],
    key_tools=["statsmodels", "lifelines"],
    description="事件史分析:Cox 比例风险(PHReg)log-HR + KM 生存曲线(分组)+ PH 假设检验",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["cox", "km"], "diagnostics": ["ph_test"]},
    auto_fix="escalate",
)
def survival(state: StudyState, **kwargs: Any) -> StudyState:
    """Cox proportional-hazards fit + Kaplan-Meier curves + PH-assumption test.

    Fits ``statsmodels.PHReg`` (Cox partial likelihood, Breslow ties) of the
    duration ``time`` with censoring indicator ``event`` on the ``covariates``,
    reporting each log hazard ratio with its SE and hazard ratio ``exp(β)``.
    Kaplan-Meier survival curves (``statsmodels.SurvfuncRight``) are estimated
    overall and stratified by the group covariate. The proportional-hazards
    assumption is tested via a time-interaction test: each covariate is
    interacted with ``log(time)`` and the added terms are jointly (and
    individually) tested — a Schoenfeld-residual-equivalent global PH check.

    kwargs
    ------
    time : str
        Duration column. Default ``"time"``.
    event : str
        Event indicator (1 = observed, 0 = censored). Default ``"event"``.
    covariates : list[str]
        Covariates for the Cox model. Default: numeric columns other than
        ``time`` / ``event``.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        model = {"log_hr": {}, "hr": {}, "n": 0, "n_events": 0, "note": note}
        state.write("models", "cox", model)
        state.write("models", "km", {"overall": None, "by_group": {}, "note": note})
        state.write("diagnostics", "ph_test",
                    {"global_p": None, "per_covariate": {}, "note": note})
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法进行生存分析")

    # ``duration`` is accepted as an alias for ``time`` (more intuitive). The
    # event indicator falls back to the declared outcome, since for survival
    # ``variables['outcome']`` conventionally names the event column.
    time_col = kwargs.get("time") or kwargs.get("duration") or "time"
    event_col = kwargs.get("event") or state.variables.get("outcome") or "event"
    if time_col not in df.columns and state.design.get("duration") in df.columns:
        time_col = state.design.get("duration")
    if time_col == event_col:
        return _empty(
            f"time 与 event 不能是同一列('{time_col}')— 用 time=/duration= 指定生存时间、"
            f"event= 或 variables['outcome'] 指定事件指示"
        )
    if time_col not in df.columns or event_col not in df.columns:
        return _empty(f"缺少生存列(time='{time_col}' / event='{event_col}')")

    covariates = list(kwargs.get("covariates") or [])
    if not covariates:
        covariates = [
            c for c in df.columns
            if c not in (time_col, event_col) and pd.api.types.is_numeric_dtype(df[c])
        ]
    covariates = [c for c in covariates if c in df.columns]
    if not covariates:
        return _empty("没有可用协变量")

    # keep the KM grouping column (and, for time-varying Cox, the interval-start
    # column) even when they are not Cox covariates.
    _grp = kwargs.get("group")
    start_col = kwargs.get("start") or kwargs.get("entry")   # counting-process (start, stop]
    extra = [c for c in [_grp, start_col]
             if c and c in df.columns and c not in ([time_col, event_col] + covariates)]
    work = df[[time_col, event_col] + covariates + extra].copy().dropna()
    time_varying = bool(start_col and start_col in work.columns)
    if time_varying:
        work = work[work[time_col] > work[start_col]]       # valid (start, stop] intervals
    else:
        work = work[work[time_col] > 0]
    if work.empty:
        return _empty("有效生存时间为空(time <= 0 或 stop <= start)")

    dur = work[time_col].to_numpy(dtype=float)
    status = work[event_col].to_numpy(dtype=float)
    X = work[covariates].to_numpy(dtype=float)
    entry = work[start_col].to_numpy(dtype=float) if time_varying else None

    sm = _try_import("statsmodels.api")

    log_hr: dict[str, tuple[float, float, float]] = {}
    cox_note = (
        "statsmodels.PHReg — Andersen-Gill 时变协变量(计数过程 (start,stop],左截断 entry=start)"
        if time_varying else "statsmodels.PHReg (Cox, Breslow ties)"
    )
    try:
        if time_varying:
            # ``entry`` left-truncates each interval so it enters the risk set at
            # its start — the Andersen-Gill formulation for time-varying
            # covariates (not covered by the pysurvival right-censored port).
            ph = sm.PHReg(dur, X, status=status, entry=entry, ties="breslow")
            res = ph.fit()
            params = np.asarray(res.params, dtype=float)
            bse = np.asarray(res.bse, dtype=float)
            pvals = np.asarray(res.pvalues, dtype=float)
            for name, b, s, p in zip(covariates, params, bse, pvals):
                log_hr[name] = (float(b), float(s), float(p))
        else:
            # Parity-gated Cox reconstruction (external/pysurvival) — R coxph's
            # default Efron tie-handling, Newton-Raphson partial likelihood,
            # matched to survival 3.8.3 at 1e-6. Supersedes the Breslow PHReg fit.
            from ..external.pysurvival import coxph as _coxph
            rc = _coxph(dur, status, X, ties="efron")
            cox_note = "pysurvival Cox PH (Efron ties, survival::coxph parity)"
            for i, name in enumerate(covariates):
                log_hr[name] = (float(rc.coef[i]), float(rc.se[i]), float(rc.pval[i]))
    except Exception as exc:  # pragma: no cover
        if time_varying:
            # the numpy fallback ignores left truncation → wrong for AG; refuse.
            return _empty(f"Andersen-Gill 时变 Cox 需 statsmodels.PHReg(未可用:{exc!s})")
        # honest fallback: Newton-Raphson on the Breslow partial likelihood.
        cox_note = f"fallback: numpy Breslow partial-likelihood Newton ({exc!s})"
        beta = _cox_newton(dur, status, X)
        se = _cox_se(dur, status, X, beta)
        for name, b, s in zip(covariates, beta, se):
            z = b / s if s > 0 else 0.0
            from math import erfc, sqrt
            p = float(erfc(abs(z) / sqrt(2.0)))
            log_hr[name] = (float(b), float(s), p)

    hr = {k: float(np.exp(v[0])) for k, v in log_hr.items()}

    # -- Kaplan-Meier (overall + stratified by a group covariate) --------------
    km_overall = _km_curve(sm, dur, status)
    km_by_group: dict[str, Any] = {}
    group_col = kwargs.get("group")
    if group_col is None:
        # pick a low-cardinality integer covariate as the stratifier
        for c in covariates:
            u = work[c].nunique()
            if 2 <= u <= 6:
                group_col = c
                break
    logrank: dict[str, Any] | None = None
    if group_col is not None and group_col in work.columns:
        for gval, sub in work.groupby(group_col, observed=True):
            km_by_group[str(gval)] = _km_curve(
                sm, sub[time_col].to_numpy(float), sub[event_col].to_numpy(float)
            )
        # log-rank (Mantel-Cox) test that the group survival curves are equal
        if work[group_col].nunique() >= 2:
            try:
                from statsmodels.duration.survfunc import survdiff

                chi2, p = survdiff(dur, status, work[group_col].to_numpy())
                logrank = {"chi2": float(chi2), "p": float(p),
                           "df": int(work[group_col].nunique() - 1), "group": group_col,
                           "note": "log-rank(Mantel-Cox):各组生存曲线是否相同"}
            except Exception:
                logrank = None

    # -- proportional-hazards test (Grambsch-Therneau / Schoenfeld residuals) --
    ph_test = _ph_schoenfeld_test(sm, dur, status, X, covariates)

    model = {
        "log_hr": log_hr,
        "hr": hr,
        "covariates": covariates,
        "n": int(len(work)),
        "n_events": int(status.sum()),
        "estimator": cox_note,
        "note": "log-HR(协变量) = Cox 偏似然系数;HR = exp(log-HR)",
    }
    state.write("models", "cox", model)
    state.write("models", "km", {
        "overall": km_overall,
        "by_group": km_by_group,
        "group_col": group_col,
        "logrank": logrank,
        "note": "Kaplan-Meier 生存函数(总体 + 分组)+ log-rank 检验,SurvfuncRight",
    })
    state.write("diagnostics", "ph_test", ph_test)
    return state


# ---------------------------------------------------------- survival internals
def _km_curve(sm, dur: np.ndarray, status: np.ndarray) -> dict[str, Any]:
    """Kaplan-Meier survival function via ``statsmodels.SurvfuncRight``.

    Returns the step points plus median survival. Falls back to a hand-rolled
    product-limit estimator if ``statsmodels`` is unavailable.
    """
    # Parity-gated Kaplan-Meier (external/pysurvival) — R survfit-exact, incl.
    # Greenwood cumulative-hazard std.err + log-transform CI; supersedes the
    # statsmodels curve. status is the 1=event indicator pysurvival.km expects.
    try:
        from ..external.pysurvival import km as _km_port
        r = _km_port(np.asarray(dur, float), np.asarray(status, int))
        med = None if not np.isfinite(r.median) else float(r.median)
        return {"times": r.time.tolist(), "surv": r.surv.tolist(),
                "std_err": r.std_err.tolist(), "lower": r.lower.tolist(),
                "upper": r.upper.tolist(), "median": med,
                "n": int(len(dur)), "backend": "pysurvival"}
    except Exception:
        pass
    if sm is not None:
        try:
            sf = sm.SurvfuncRight(dur, status)
            t = np.asarray(sf.surv_times, dtype=float)
            s = np.asarray(sf.surv_prob, dtype=float)
            med = _median_survival(t, s)
            return {"times": t.tolist(), "surv": s.tolist(),
                    "median": med, "n": int(len(dur))}
        except Exception:
            pass
    # product-limit fallback
    order = np.argsort(dur)
    d, e = dur[order], status[order]
    uniq = np.unique(d)
    surv = 1.0
    times, probs = [], []
    at_risk = len(d)
    for tt in uniq:
        di = int(((d == tt) & (e == 1)).sum())
        ni = int((d >= tt).sum())
        if ni > 0:
            surv *= (1.0 - di / ni)
        times.append(float(tt))
        probs.append(float(surv))
    med = _median_survival(np.array(times), np.array(probs))
    return {"times": times, "surv": probs, "median": med, "n": int(len(dur))}


def _median_survival(t: np.ndarray, s: np.ndarray) -> float | None:
    """First time at which the survival function drops to/through 0.5."""
    below = np.where(s <= 0.5)[0]
    if below.size:
        return float(t[below[0]])
    return None


def _cox_newton(dur, status, X, iters: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Newton-Raphson maximizer of the Breslow partial log-likelihood (fallback)."""
    n, k = X.shape
    beta = np.zeros(k)
    order = np.argsort(-dur)  # descending time → cumulative risk sets
    Xo, do, so = X[order], dur[order], status[order]
    for _ in range(iters):
        eta = Xo @ beta
        w = np.exp(eta)
        # cumulative sums over the (descending-time) risk set
        cum_w = np.cumsum(w)
        cum_wx = np.cumsum(w[:, None] * Xo, axis=0)
        grad = np.zeros(k)
        hess = np.zeros((k, k))
        for i in range(n):
            if so[i] != 1:
                continue
            rw = cum_w[i]
            rwx = cum_wx[i]
            mean_x = rwx / rw
            grad += Xo[i] - mean_x
            # second moment over risk set
            wx2 = np.cumsum(
                (w[:, None, None] * (Xo[:, :, None] * Xo[:, None, :])), axis=0
            )[i]
            hess -= wx2 / rw - np.outer(mean_x, mean_x)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _cox_se(dur, status, X, beta) -> np.ndarray:
    """Standard errors from the observed information at ``beta`` (fallback)."""
    n, k = X.shape
    order = np.argsort(-dur)
    Xo, so = X[order], status[order]
    eta = Xo @ beta
    w = np.exp(eta)
    cum_w = np.cumsum(w)
    cum_wx = np.cumsum(w[:, None] * Xo, axis=0)
    info = np.zeros((k, k))
    wx2_cum = np.cumsum(
        (w[:, None, None] * (Xo[:, :, None] * Xo[:, None, :])), axis=0
    )
    for i in range(n):
        if so[i] != 1:
            continue
        rw = cum_w[i]
        mean_x = cum_wx[i] / rw
        info += wx2_cum[i] / rw - np.outer(mean_x, mean_x)
    try:
        cov = np.linalg.inv(info)
        return np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def _ph_schoenfeld_test(
    sm, dur: np.ndarray, status: np.ndarray, X: np.ndarray, names: list[str]
) -> dict[str, Any]:
    """Grambsch-Therneau proportional-hazards test on scaled Schoenfeld residuals.

    This is R's ``survival::cox.zph`` / ``lifelines``'s
    ``check_assumptions``. Under PH, the (scaled) Schoenfeld residual of each
    covariate is uncorrelated with any transform ``g(t)`` of event time; a
    non-zero correlation means the log-hazard ratio drifts with time (PH
    violated). We use the event-time rank as ``g(t)`` (the ``cox.zph`` default,
    ``transform='rank'``) and test, per covariate, the standardized correlation
    ``z = corr · sqrt(m - 1)`` against N(0,1); the global test sums the squared
    per-covariate statistics into a χ²(k). Non-significant (``global_p > alpha``)
    ⇒ PH holds.

    The residuals come straight from ``PHReg.schoenfeld_residuals`` (NaN for
    censored rows). If unavailable, we fall back to a covariate × log(time)
    interaction LR test.
    """
    alpha = 0.05
    if sm is None:
        return {"global_p": None, "per_covariate": {}, "verdict": "unknown",
                "note": "statsmodels 不可用,无法做 PH 检验"}

    try:
        res = sm.PHReg(dur, X, status=status, ties="breslow").fit()
        sr = np.asarray(res.schoenfeld_residuals, dtype=float)  # (n, k), NaN=censored
        mask = ~np.isnan(sr).any(axis=1)
        sr = sr[mask]
        t_event = dur[mask]
        m = sr.shape[0]
        if m < 3:
            raise ValueError("事件数不足,无法做 Schoenfeld 检验")

        # g(t) = rank of event time, centered (cox.zph transform='rank')
        gt = pd.Series(t_event).rank(method="average").to_numpy()
        gt = gt - gt.mean()
        denom = float(gt @ gt)

        from scipy import stats as _st
        per_cov: dict[str, dict[str, float]] = {}
        chi2_stats: list[float] = []
        for j, nm in enumerate(names):
            rj = sr[:, j]
            sd = float(np.std(rj))
            if sd == 0 or denom == 0:
                corr = 0.0
            else:
                corr = float((gt @ rj) / np.sqrt(denom * (rj @ rj)))
            corr = float(np.clip(corr, -0.999999, 0.999999))
            z = corr * np.sqrt(max(m - 1, 1))
            p = float(2.0 * _st.norm.sf(abs(z)))
            chi2_stats.append(z * z)
            per_cov[nm] = {"rho": corr, "chi2": float(z * z), "p": p}

        global_chi2 = float(np.sum(chi2_stats))
        global_p = float(_st.chi2.sf(global_chi2, df=len(names)))
    except Exception:
        return _ph_time_interaction_fallback(sm, dur, status, X, names, alpha)

    verdict = "pass" if global_p > alpha else "fail"
    note = ("PH 假设成立(Schoenfeld 残差与时间无关, p>{:.2g})".format(alpha)
            if verdict == "pass"
            else "PH 假设被拒(存在时间依赖的风险比)")
    return {
        "global_p": global_p,
        "global_chi2": global_chi2,
        "per_covariate": per_cov,
        "verdict": verdict,
        "alpha": alpha,
        "method": "Grambsch-Therneau on scaled Schoenfeld residuals (cox.zph, transform=rank)",
        "note": note,
    }


def _ph_time_interaction_fallback(
    sm, dur, status, X, names: list[str], alpha: float
) -> dict[str, Any]:
    """Fallback PH test: covariate × log(time) interaction LR test (per-covariate).

    Used only when ``schoenfeld_residuals`` is unavailable. Each covariate is
    tested singly (one interaction at a time) to avoid the collinearity that a
    simultaneous augmentation induces; the global p is a Bonferroni combination.
    """
    logt = np.log(np.clip(dur, 1e-12, None))
    logt = logt - logt.mean()
    per_cov: dict[str, dict[str, float]] = {}
    from scipy import stats as _st
    try:
        base = sm.PHReg(dur, X, status=status, ties="breslow").fit()
        ps: list[float] = []
        for j, nm in enumerate(names):
            X_aug = np.column_stack([X, X[:, j] * logt])
            aug = sm.PHReg(dur, X_aug, status=status, ties="breslow").fit()
            lr = 2.0 * (float(aug.llf) - float(base.llf))
            p = float(_st.chi2.sf(max(lr, 0.0), df=1))
            per_cov[nm] = {"lr_chi2": float(max(lr, 0.0)), "p": p}
            ps.append(p)
        global_p = float(min(1.0, min(ps) * len(ps))) if ps else None
    except Exception as exc:
        return {"global_p": None, "per_covariate": per_cov, "verdict": "unknown",
                "note": f"PH 检验拟合失败: {exc!s}"}
    verdict = "pass" if (global_p is not None and global_p > alpha) else \
              ("fail" if global_p is not None else "unknown")
    return {
        "global_p": global_p,
        "per_covariate": per_cov,
        "verdict": verdict,
        "alpha": alpha,
        "method": "fallback: 单协变量 × log(time) 交互 LR 检验 (Bonferroni 合并)",
        "note": "PH 假设" + ("成立" if verdict == "pass" else "被拒" if verdict == "fail" else "未知"),
    }


# ------------------------------------------------------------- conditional_logit
@register(
    name="conditional_logit",
    aliases=["条件Logit", "clogit", "配对Logit", "固定效应Logit"],
    category="longitudinal",
    tier="plus",
    skill="(条件Logit 缺口)",
    languages=["Python"],
    key_tools=["pysurvival"],
    description="条件(固定效应)Logistic 回归(clogit):分层匹配集精确条件似然,消除层内固定效应",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["clogit"]},
    auto_fix="escalate",
)
def conditional_logit(state: StudyState, **kwargs: Any) -> StudyState:
    """Conditional (fixed-effects) logistic regression — matched/stratified clogit.

    Fits the exact conditional partial likelihood (``survival::clogit`` parity)
    for stratum-matched binary data: within each matched set (stratum) the
    stratum-level intercept is conditioned out, so only within-stratum contrasts
    of the covariates identify the coefficients. This is the estimator for 1:M
    matched case-control designs and panel binary FE logit.

    The numeric work is delegated to the parity-gated port
    ``external.pysurvival.clogit(y, strata, X)``; we report each log-odds
    coefficient with its SE, z, p and the odds ratio ``exp(β)``.

    kwargs
    ------
    outcome : str
        0/1 case indicator column. Default: ``variables['outcome']``.
    strata : str
        Matched-set / stratum id column. Default ``kwargs['strata']`` →
        ``design['panel_id']`` → ``"strata"``.
    covariates : list[str]
        Covariates. Default: numeric columns other than outcome / strata.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        state.write("models", "clogit", {
            "coef": {}, "odds_ratio": {}, "strata": None,
            "n": 0, "n_events": 0, "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法拟合条件 Logit")

    strata_col = kwargs.get("strata") or state.design.get("panel_id") or "strata"
    if strata_col not in df.columns:
        return _empty(f"缺少分层列(strata='{strata_col}')")

    outcome = _pick_outcome(df, kwargs, state, exclude=[strata_col])
    if outcome is None:
        return _empty("找不到结果变量(outcome)")
    if outcome == strata_col:
        return _empty(f"outcome 与 strata 不能是同一列('{outcome}')")

    covariates = list(kwargs.get("covariates") or [])
    if not covariates:
        covariates = [
            c for c in df.columns
            if c not in (outcome, strata_col) and pd.api.types.is_numeric_dtype(df[c])
        ]
    covariates = [c for c in covariates if c in df.columns and c != outcome
                  and c != strata_col]
    if not covariates:
        return _empty("没有可用协变量")

    work = df[[outcome, strata_col] + covariates].copy().dropna()
    if work.empty:
        return _empty("有效样本为空(缺失值过滤后)")

    try:
        from ..external.pysurvival import clogit as _clogit

        y = work[outcome].to_numpy(dtype=float)
        strata = work[strata_col].to_numpy()
        X = work[covariates].to_numpy(dtype=float)
        rc = _clogit(y, strata, X)

        coef: dict[str, tuple[float, float, float, float]] = {}
        for i, name in enumerate(covariates):
            coef[name] = (float(rc.coef[i]), float(rc.se[i]),
                          float(rc.z[i]), float(rc.pval[i]))
        odds_ratio = {k: float(np.exp(v[0])) for k, v in coef.items()}
        ll0, ll_fit = rc.loglik
    except Exception as exc:  # pragma: no cover
        return _empty(f"条件 Logit 拟合失败: {exc!s}")

    state.write("models", "clogit", {
        "coef": coef,
        "odds_ratio": odds_ratio,
        "covariates": covariates,
        "outcome": outcome,
        "strata": strata_col,
        "n": int(rc.n),
        "n_events": int(rc.n_event),
        "n_strata": int(work[strata_col].nunique()),
        "loglik_null": float(ll0),
        "loglik_fitted": float(ll_fit),
        "iterations": int(rc.iter),
        "estimator": "pysurvival clogit(精确条件似然, survival::clogit parity)",
        "note": "coef = 条件对数优势比(层内固定效应已消除);OR = exp(coef)",
    })
    return state


# ------------------------------------------------------------------- aft_survreg
@register(
    name="aft_survreg",
    aliases=["参数生存", "AFT", "加速失效时间", "survreg"],
    category="longitudinal",
    tier="plus",
    skill="(参数AFT 缺口)",
    languages=["Python"],
    key_tools=["pysurvival"],
    description="参数加速失效时间(AFT)生存回归:Weibull/指数/对数正态 MLE(survreg),对数时间尺度系数",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["survreg"]},
    auto_fix="escalate",
)
def aft_survreg(state: StudyState, **kwargs: Any) -> StudyState:
    """Parametric accelerated-failure-time (AFT) survival regression — ``survreg``.

    Fits a fully parametric AFT model of ``log(time)`` on the covariates with a
    parametric error distribution (extreme-value → Weibull/exponential, Gaussian
    → lognormal), by exact-likelihood Newton-Raphson. Coefficients are on the
    log-time scale: a positive β lengthens survival (time-ratio ``exp(β)``),
    the opposite sign convention from a Cox log-HR.

    The numeric work is delegated to the parity-gated port
    ``external.pysurvival.survreg(time, status, X, dist)``; an intercept column
    is prepended to the design (matching R's ``~ covariates`` model matrix).

    kwargs
    ------
    time : str
        Duration column. Default ``kwargs['time']`` → ``kwargs['duration']`` →
        ``design['duration']`` → ``"time"``.
    event : str
        Event indicator (1 = observed, 0 = censored). Default
        ``variables['outcome']`` → ``"event"``.
    covariates : list[str]
        Covariates. Default: numeric columns other than time / event.
    dist : str
        ``"weibull"`` (default), ``"exponential"`` or ``"lognormal"``.
    """
    df = _get_datasets(state, kwargs)
    dist = str(kwargs.get("dist") or "weibull").lower()

    def _empty(note: str) -> StudyState:
        state.write("models", "survreg", {
            "coef": {}, "time_ratio": {}, "scale": None, "dist": dist,
            "n": 0, "n_events": 0, "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法拟合参数 AFT 模型")
    if dist not in ("weibull", "exponential", "lognormal"):
        return _empty(f"不支持的分布 dist='{dist}'(可选 weibull/exponential/lognormal)")

    time_col = kwargs.get("time") or kwargs.get("duration") or "time"
    event_col = kwargs.get("event") or state.variables.get("outcome") or "event"
    if time_col not in df.columns and state.design.get("duration") in df.columns:
        time_col = state.design.get("duration")
    if time_col == event_col:
        return _empty(
            f"time 与 event 不能是同一列('{time_col}')— 用 time=/duration= 指定生存时间、"
            f"event= 或 variables['outcome'] 指定事件指示"
        )
    if time_col not in df.columns or event_col not in df.columns:
        return _empty(f"缺少生存列(time='{time_col}' / event='{event_col}')")

    covariates = list(kwargs.get("covariates") or [])
    if not covariates:
        covariates = [
            c for c in df.columns
            if c not in (time_col, event_col) and pd.api.types.is_numeric_dtype(df[c])
        ]
    covariates = [c for c in covariates if c in df.columns
                  and c not in (time_col, event_col)]

    work = df[[time_col, event_col] + covariates].copy().dropna()
    work = work[work[time_col] > 0]
    if work.empty:
        return _empty("有效生存时间为空(time <= 0 或缺失值过滤后)")

    try:
        from ..external.pysurvival import survreg as _survreg

        dur = work[time_col].to_numpy(dtype=float)
        status = work[event_col].to_numpy(dtype=float)
        # prepend an intercept column (R's ~ covariates design matrix)
        cols = [np.ones(len(work))] + [work[c].to_numpy(dtype=float) for c in covariates]
        X = np.column_stack(cols)
        names = ["Intercept"] + list(covariates)
        rs = _survreg(dur, status, X, dist=dist)

        coef: dict[str, tuple[float, float]] = {}
        se = np.asarray(rs.se, dtype=float)
        for i, name in enumerate(names):
            s = float(se[i]) if i < len(se) else float("nan")
            coef[name] = (float(rs.coef[i]), s)
        # time ratio exp(β) for the non-intercept covariates
        time_ratio = {name: float(np.exp(coef[name][0])) for name in covariates}
        log_scale_se = float(se[len(names)]) if len(se) > len(names) else None
    except Exception as exc:  # pragma: no cover
        return _empty(f"参数 AFT 拟合失败: {exc!s}")

    state.write("models", "survreg", {
        "coef": coef,
        "time_ratio": time_ratio,
        "scale": float(rs.scale),
        "log_scale_se": log_scale_se,
        "dist": rs.dist,
        "covariates": covariates,
        "time": time_col,
        "event": event_col,
        "n": int(len(work)),
        "n_events": int(status.sum()),
        "loglik": float(rs.loglik),
        "iterations": int(rs.iter),
        "estimator": f"pysurvival survreg({rs.dist}, AFT MLE, survival::survreg parity)",
        "note": "coef 为对数时间尺度回归系数(正号=生存延长);time_ratio = exp(coef);scale = 尺度参数",
    })
    return state
