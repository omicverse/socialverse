"""``sv.tl._meta_ipd`` — individual participant data (IPD) meta-analysis (Tier-3).

Two-stage (per-study effect → pool) and one-stage (stacked mixed model with
study strata + random treatment effect). Continuous outcomes use a within-study
OLS / mixed model; the one-stage engine reuses statsmodels MixedLM when present
and degrades to a stratified fixed-effects estimate otherwise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ..pp._meta_es import _resolve_df
from ._meta import _estimate_tau2


def _ipd_cols(state, kwargs):
    df = _resolve_df(state, kwargs)
    if df is None:
        return None, None
    sc = kwargs.get("study"); yc = kwargs.get("outcome"); tc = kwargs.get("treatment")
    if any(c not in (df.columns if df is not None else []) for c in (sc, yc, tc)):
        return None, None
    return df, (sc, yc, tc)


@register(
    name="ipd_twostage", aliases=["IPD两阶段", "ipd_two_stage"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="IPD 两阶段:每研究个体数据估处理效应(OLS,可调协变量)→ 随机效应合并;等价于聚合数据 meta",
    requires={"sources": ["datasets"]}, produces={"models": ["meta", "ipd"]},
)
def ipd_twostage(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-stage IPD: per-study treatment effect (adjusted OLS) → random-effects pool.

    Long IPD: ``study``,``outcome``,``treatment`` (+ ``covariates=[...]``)."""
    df, cols = _ipd_cols(state, kwargs)
    if df is None:
        state.write("models", "ipd", {"note": "need study/outcome/treatment columns"})
        return state
    from scipy import stats
    sc, yc, tc = cols
    covs = kwargs.get("covariates") or []
    yis, vis, labs = [], [], []
    for s, g in df.groupby(sc, sort=False):
        y = pd.to_numeric(g[yc], errors="coerce").to_numpy(float)
        t = pd.to_numeric(g[tc], errors="coerce").to_numpy(float)
        Xcols = [np.ones(len(g)), t] + [pd.to_numeric(g[c], errors="coerce").to_numpy(float) for c in covs]
        X = np.column_stack(Xcols)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        if len(y) <= X.shape[1] or np.ptp(X[:, 1]) == 0:
            continue
        A = np.linalg.pinv(X.T @ X); beta = A @ (X.T @ y)
        resid = y - X @ beta; s2 = float(resid @ resid) / (len(y) - X.shape[1])
        yis.append(float(beta[1])); vis.append(float(s2 * A[1, 1])); labs.append(str(s))
    if not yis:
        state.write("models", "ipd", {"note": "no estimable studies"})
        return state
    y = np.array(yis); v = np.array(vis)
    tau2 = _estimate_tau2(y, v, "REML"); w = 1 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w)); se = float(np.sqrt(1 / np.sum(w)))
    out = {"model": "ipd_two_stage", "estimate": mu, "se": se, "tau2": tau2,
           "ci_lb": mu - 1.96 * se, "ci_ub": mu + 1.96 * se,
           "zval": mu / se, "pval": float(2 * stats.norm.sf(abs(mu / se))), "k": len(yis)}
    state.write("models", "meta", out)
    state.write("models", "ipd", {"per_study_effect": dict(zip(labs, yis)), "stage": "two"})
    # also expose as effects for downstream forest/heterogeneity
    eff = pd.DataFrame({"yi": y, "vi": v, "sei": np.sqrt(v), "measure": "MD", "study": labs})
    state.write("models", "meta_effects", eff)
    return state


@register(
    name="ipd_onestage", aliases=["IPD一阶段", "ipd_one_stage"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "statsmodels"],
    description="IPD 一阶段:堆叠个体数据的混合模型(研究分层 + 处理随机斜率);连续结局用 MixedLM,缺库降级为分层固定效应",
    requires={"sources": ["datasets"]}, produces={"models": ["ipd"]},
)
def ipd_onestage(state: StudyState, **kwargs: Any) -> StudyState:
    """One-stage IPD: stacked mixed model (study strata + random treatment effect).

    Reuses statsmodels MixedLM (continuous) when available; falls back to a
    study-stratified (within-trial) fixed-effects treatment estimate otherwise.
    Keeps the treatment effect a within-trial contrast to avoid ecological bias."""
    df, cols = _ipd_cols(state, kwargs)
    if df is None:
        state.write("models", "ipd", {"note": "need study/outcome/treatment columns"})
        return state
    sc, yc, tc = cols
    covs = kwargs.get("covariates") or []
    d = df[[sc, yc, tc] + covs].copy()
    d[yc] = pd.to_numeric(d[yc], errors="coerce"); d[tc] = pd.to_numeric(d[tc], errors="coerce")
    d = d.dropna()
    try:
        import statsmodels.formula.api as smf
        d = d.rename(columns={yc: "_y", tc: "_t", sc: "_s"})
        fixed = "_y ~ _t + C(_s)" + ("".join(f" + {c}" for c in covs))   # study fixed effects
        md = smf.mixedlm(fixed, d, groups=d["_s"], re_formula="~_t")     # random treatment slope
        fit = md.fit(reml=True, method="lbfgs", disp=False)
        beta = float(fit.params["_t"]); se = float(fit.bse["_t"])
        from scipy import stats
        out = {"model": "ipd_one_stage(MixedLM)", "estimate": beta, "se": se,
               "ci_lb": beta - 1.96 * se, "ci_ub": beta + 1.96 * se,
               "pval": float(2 * stats.norm.sf(abs(beta / se))),
               "n": int(len(d)), "k": int(d["_s"].nunique())}
    except Exception as exc:
        # fallback: study-stratified within-trial OLS treatment effect
        num, den = 0.0, 0.0
        for s, g in d.groupby(sc):
            t = g[tc].to_numpy(float); y = g[yc].to_numpy(float)
            if np.ptp(t) == 0:
                continue
            tc_ = t - t.mean(); num += float(np.sum(tc_ * (y - y.mean()))); den += float(np.sum(tc_ ** 2))
        beta = num / den if den else float("nan")
        out = {"model": "ipd_one_stage(stratified-FE fallback)", "estimate": beta,
               "note": f"MixedLM unavailable ({type(exc).__name__}); within-trial stratified estimate",
               "n": int(len(d)), "k": int(d[sc].nunique())}
    state.write("models", "ipd", out)
    return state
