"""``sv.tl._moderndid`` — the modern heterogeneity-robust DiD family (pyfixest /
original-paper estimators), implemented natively alongside ``did``/``event_study``/
``fect``.

Two-way-FE event studies are contaminated across periods when effects are dynamic
(Sun-Abraham 2021, Goodman-Bacon 2021). These three estimators fix it differently:

- ``sun_abraham`` — **interaction-weighted (IW)**: estimate cohort-specific
  ``CATT(e, l)`` in a saturated regression (clean never/last-treated controls), then
  aggregate to each relative period ``l`` with cohort-share weights. The
  heterogeneity-robust event study.
- ``did2s`` — **Gardner (2021) two-stage**: fit unit+time FE on untreated cells, then
  regress the residual on treatment (static ATT + dynamic path). Same imputation idea
  as ``fect``, in two-stage-regression form.
- ``local_projection`` — **LP-DiD (Jordà; Dube-Girardi-Jordà-Taylor 2023)**: at each
  horizon ``h`` regress the outcome change since the pre period on the treatment
  switch against clean controls → an impulse-response path.

Inference reuses the absorbing-FE cluster-robust helper ``_within_fit`` and a block
bootstrap over units.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _cols, _get_datasets, _pick_outcome, _within_fit
from ._fect import _build_matrices, _fe_levels


def _panel(df, cols, y_col):
    """Long panel with unit/time indices, outcome, treat_post, and per-unit onset."""
    Y, D, E, units, times, onset = _build_matrices(df, cols, y_col)
    return Y, D, E, units, times, onset


# ======================================================================= sun_abraham
@register(
    name="sun_abraham",
    aliases=["交互加权", "sunab", "interaction_weighted", "sun_abraham_did"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="Sun-Abraham 交互加权事件研究:cohort×相对期 CATT 饱和回归+cohort份额聚合,异质稳健动态效应",
    requires={"design": ["panel_id", "time", "treatment", "first_treated"],
              "variables": ["outcome"]},
    produces={"models": ["sun_abraham"]},
    auto_fix="escalate",
)
def sun_abraham(state: StudyState, **kwargs: Any) -> StudyState:
    """Sun & Abraham (2021) interaction-weighted event study.

    Saturates the event study with cohort×relative-time indicators (never-treated as
    the clean control), estimating each ``CATT(e, l)`` with unit+time fixed effects
    absorbed, then aggregates to a relative-period path weighted by cohort shares.
    Standard errors are cluster-robust (delta method on the clustered covariance).
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)

    def _empty(note):
        state.write("models", "sun_abraham", {"coefs": {}, "note": note})
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time")):
        return _empty("缺少面板数据或设计列(panel_id/time)")
    y_col = _pick_outcome(df, cols, exclude=[c for c in cols.values() if c])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    Y, D, E, units, times, onset = _panel(df, cols, y_col)
    N, T = Y.shape
    base = int(kwargs.get("base", -1))
    lo = int(kwargs.get("min_rel", -(T - 1)))
    hi = int(kwargs.get("max_rel", T - 1))

    # long observation list (only present cells)
    ii, tt = np.where(E)
    y = Y[ii, tt]
    unit_ids = units[ii]
    time_ids = times[tt]
    rel = np.where(onset[ii] >= 0, tt - onset[ii], np.iinfo(np.int32).min)
    cohort = np.where(onset[ii] >= 0, onset[ii], -1)

    # design columns: one per (cohort e, rel l), l in window, l != base, treated cohorts
    keys = sorted({(int(c), int(r)) for c, r in zip(cohort, rel)
                   if c >= 0 and lo <= r <= hi and r != base})
    if not keys:
        return _empty("无可用的 cohort×相对期(可能无 never-treated 对照或无处理前后期)")
    kidx = {k: j for j, k in enumerate(keys)}
    Dmat = np.zeros((len(y), len(keys)))
    for row, (c, r) in enumerate(zip(cohort, rel)):
        k = (int(c), int(r))
        if k in kidx:
            Dmat[row, kidx[k]] = 1.0

    fit = _within_fit(y, Dmat, unit_ids, time_ids, unit_ids)
    if fit is None:
        return _empty("CATT 设计共线或无变异,无法估计")
    beta, V = fit["beta"], fit["V_cluster"]

    # cohort shares: among treated obs at rel l, fraction from cohort e
    catt = {f"{e}|{l}": float(beta[kidx[(e, l)]]) for (e, l) in keys}
    rels = sorted({l for (_e, l) in keys})
    coefs = {str(base): (0.0, 0.0)}
    n_at_rel = {}
    for l in rels:
        cols_l = [(e, ll) for (e, ll) in keys if ll == l]
        n_el = np.array([int(((cohort == e) & (rel == l)).sum()) for (e, ll) in cols_l], float)
        if n_el.sum() <= 0:
            continue
        w = n_el / n_el.sum()
        idxs = [kidx[k] for k in cols_l]
        iw = float(w @ beta[idxs])
        var = float(w @ V[np.ix_(idxs, idxs)] @ w)
        coefs[str(l)] = (iw, float(np.sqrt(max(var, 0.0))))
        n_at_rel[l] = float(n_el.sum())

    ordered = {k: coefs[k] for k in sorted(coefs, key=lambda s: int(s))}
    # overall ATT = post-period IW effects weighted by each period's treated-obs count
    post = [(l, coefs[str(l)][0], n_at_rel[l]) for l in rels if l >= 0 and str(l) in coefs]
    att = (float(sum(v * n for _l, v, n in post) / sum(n for _l, _v, n in post))
           if post else None)
    state.write("models", "sun_abraham", {
        "coefs": ordered, "att_post_mean": att, "catt": catt, "base": base,
        "outcome": y_col, "n": int(len(y)), "n_clusters": fit["n_clusters"],
        "estimator": "sun_abraham_iw",
        "note": "Sun-Abraham 交互加权:cohort×相对期 CATT 聚合;前导期应≈0(异质稳健事件研究)",
    })
    return state


# ============================================================================= did2s
@register(
    name="did2s",
    aliases=["两步DiD", "gardner", "two_stage_did", "did_two_stage"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="Gardner(2021)两步 DiD:未处理估 unit+time FE→残差对处理回归得 ATT(+动态);块 bootstrap SE",
    requires={"design": ["panel_id", "time", "treatment"], "variables": ["outcome"]},
    produces={"models": ["did2s"]},
    auto_fix="escalate",
)
def did2s(state: StudyState, **kwargs: Any) -> StudyState:
    """Gardner (2021) two-stage DiD.

    Stage 1 fits unit + time fixed effects on **untreated** observations; stage 2
    regresses the stage-1 residual on the treatment (static ATT) and on relative-time
    dummies (dynamic path). Heterogeneity-robust, same imputation logic as ``fect`` in
    two-stage-regression form. SE via block bootstrap over units.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)

    def _empty(note):
        state.write("models", "did2s", {"att": None, "note": note})
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time", "treatment")):
        return _empty("缺少面板数据或设计列")
    y_col = _pick_outcome(df, cols, exclude=[c for c in cols.values() if c])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    Y, D, E, units, times, onset = _panel(df, cols, y_col)
    N, T = Y.shape

    def _fit(Y, D, E, onset):
        O = E & (D < 0.5)
        # keep units with an untreated cell
        keep = O.any(1)
        Y, D, E, O, on = Y[keep], D[keep], E[keep], O[keep], onset[keep]
        a, g = _fe_levels(Y, O)
        resid = Y - a[:, None] - g[None, :]
        M = E & (D > 0.5)
        att = float(resid[M].mean()) if M.any() else float("nan")
        # dynamic
        byp = {}
        rel = np.full(Y.shape, np.iinfo(np.int32).min)
        for i in range(Y.shape[0]):
            if on[i] >= 0:
                rel[i] = np.arange(Y.shape[1]) - on[i]
        for s in np.unique(rel[M]):
            c = M & (rel == s)
            if c.any():
                byp[int(s)] = float(resid[c].mean())
        return att, byp, int(M.sum()), int(keep.sum())

    att, byp, n_treat, n_units = _fit(Y, D, E, onset)
    if not np.isfinite(att):
        return _empty("无处理观测或无处理前期")

    nb = int(kwargs.get("nboots", 200))
    seed = int(kwargs.get("seed", 42))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(nb):
        idx = rng.integers(0, N, N)
        try:
            a_b, _, m_b, _ = _fit(Y[idx], D[idx], E[idx], onset[idx])
            if np.isfinite(a_b):
                boots.append(a_b)
        except Exception:
            continue
    boots = np.array(boots)
    se = float(np.std(boots, ddof=1)) if boots.size >= max(30, nb // 4) else None
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if se else None
    p = None
    if se and se > 0:
        p = float(2 * (1 - __import__("scipy.stats", fromlist=["stats"]).norm.cdf(abs(att / se))))
    state.write("models", "did2s", {
        "att": att, "se": se, "ci": ci, "p": p, "att_by_period": byp,
        "n_treated_obs": n_treat, "n_units": n_units, "outcome": y_col,
        "estimator": "gardner_two_stage",
        "note": "Gardner 两步:未处理拟合 FE→残差回归;与 fect 同源,异质稳健",
    })
    return state


# =================================================================== local_projection
@register(
    name="local_projection",
    aliases=["局部投影", "lp_did", "jorda_lp", "impulse_response"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="局部投影 DiD(Jordà;LP-DiD):逐 horizon 回归结果变化于处理切换(清洁对照)→脉冲响应",
    requires={"design": ["panel_id", "time", "treatment", "first_treated"],
              "variables": ["outcome"]},
    produces={"models": ["local_projection"]},
    auto_fix="escalate",
)
def local_projection(state: StudyState, **kwargs: Any) -> StudyState:
    """LP-DiD (Jordà local projections; Dube et al. 2023).

    At each horizon ``h`` regresses the outcome change from the pre period
    (``y_{t0+h} - y_{t0-1}``) on the treatment switch, using clean controls
    (never-treated + not-yet-treated), with time fixed effects and unit-clustered SEs.
    Traces an impulse-response path ``beta_h``.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)

    def _empty(note):
        state.write("models", "local_projection", {"coefs": {}, "note": note})
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time")):
        return _empty("缺少面板数据或设计列")
    y_col = _pick_outcome(df, cols, exclude=[c for c in cols.values() if c])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    Y, D, E, units, times, onset = _panel(df, cols, y_col)
    N, T = Y.shape
    hmax = int(kwargs.get("max_horizon", min(T - 1, 10)))
    hmin = int(kwargs.get("min_horizon", -min(T - 1, 5)))
    treated = onset >= 0

    coefs = {}
    for h in range(hmin, hmax + 1):
        if h == -1:
            coefs["-1"] = (0.0, 0.0)  # normalized base
            continue
        # Build LP sample grouped by cohort (onset), NOT per treated unit: each cohort
        # o contributes its treated units (Δy=y_{o+h}-y_{o-1}, once) and the clean
        # never-treated controls over the SAME calendar span (once per cohort, so a
        # control is not duplicated once-per-treated-unit).
        rows_y, rows_treat, rows_clus, rows_time = [], [], [], []
        for o in sorted({int(v) for v in onset if v >= 0}):
            base_t, tgt = o - 1, o + h
            if base_t < 0 or tgt < 0 or tgt >= T:
                continue
            for i in range(N):
                if onset[i] == o and E[i, base_t] and E[i, tgt]:
                    rows_y.append(Y[i, tgt] - Y[i, base_t]); rows_treat.append(1.0)
                    rows_clus.append(i); rows_time.append(o)
                elif onset[i] == -1 and E[i, base_t] and E[i, tgt]:  # clean control, once/cohort
                    rows_y.append(Y[i, tgt] - Y[i, base_t]); rows_treat.append(0.0)
                    rows_clus.append(i); rows_time.append(o)
        if len(rows_y) < 10 or sum(rows_treat) < 2 or (len(rows_treat) - sum(rows_treat)) < 2:
            continue
        yv = np.array(rows_y); dv = np.array(rows_treat)
        clus = np.array(rows_clus); tm = np.array(rows_time)
        # regress dy ~ treat + cohort(onset) FE ; cluster by unit
        cohd = pd.get_dummies(tm, drop_first=True, dtype=float).to_numpy()
        X = np.column_stack([np.ones(len(yv)), dv, cohd]) if cohd.size else np.column_stack([np.ones(len(yv)), dv])
        XtX = X.T @ X
        beta = np.linalg.solve(XtX + 1e-10 * np.eye(X.shape[1]), X.T @ yv)
        e = yv - X @ beta
        # cluster-robust SE on the treat coef (index 1)
        XtX_inv = np.linalg.pinv(XtX)
        meat = np.zeros_like(XtX)
        for c in np.unique(clus):
            m = clus == c
            s = X[m].T @ e[m]
            meat += np.outer(s, s)
        V = XtX_inv @ meat @ XtX_inv
        coefs[str(h)] = (float(beta[1]), float(np.sqrt(max(V[1, 1], 0.0))))

    if not coefs:
        return _empty("无足够清洁对照构建局部投影(需 never-treated 对照)")
    ordered = {k: coefs[k] for k in sorted(coefs, key=lambda s: int(s))}
    post = [v[0] for k, v in ordered.items() if int(k) >= 0]
    state.write("models", "local_projection", {
        "coefs": ordered, "att_post_mean": float(np.mean(post)) if post else None,
        "outcome": y_col, "estimator": "lp_did",
        "note": "LP-DiD 脉冲响应:逐 horizon DiD(清洁对照+cohort FE+单位聚类 SE)",
    })
    return state


__all__ = ["sun_abraham", "did2s", "local_projection"]
