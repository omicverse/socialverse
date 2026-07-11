"""``sv.tl._meta_dose`` — dose-response meta-analysis (Tier-3).

Two-stage dose-response: per-study GLS dose slope (Greenland-Longnecker,
accounting for the shared-reference correlation among log-RRs) then random-
effects pooling. Linear (``dosresmeta``) and restricted-cubic-spline nonlinear
(``dosresmeta_spline``, with a Wald nonlinearity test). Native, faithful.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ..pp._meta_es import _resolve_df
from ._meta import _estimate_tau2


def _study_iter(state, kwargs):
    df = _resolve_df(state, kwargs)
    if df is None:
        return None, None
    sc = kwargs.get("study"); dc = kwargs.get("dose"); yc = kwargs.get("logrr"); ec = kwargs.get("se")
    if any(c not in (df.columns if df is not None else []) for c in (sc, dc, yc, ec)):
        return None, None
    return df, (sc, dc, yc, ec)


def _rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell), returns (x, spline term(s))."""
    k = np.asarray(knots, float); nk = len(k)
    terms = [x]
    for j in range(nk - 2):
        def cube(u): return np.clip(u, 0, None) ** 3
        denom = (k[-1] - k[0])
        t = (cube(x - k[j]) - cube(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
             + cube(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / denom ** 2
        terms.append(t)
    return np.column_stack(terms)


@register(
    name="dosresmeta", aliases=["剂量反应meta", "dose_response", "glst"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="两阶段线性剂量反应 meta:各研究 GLS 剂量斜率(过参照点)+ 随机效应合并(每单位剂量的 log-RR)",
    requires={"sources": ["datasets"]}, produces={"models": ["dosres"]},
)
def dosresmeta(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-stage linear dose-response. Long data: ``study``,``dose``,``logrr``,``se``
    (the reference dose row has logrr=0). Reports the pooled log-RR per unit dose."""
    df, cols = _study_iter(state, kwargs)
    if df is None:
        state.write("models", "dosres", {"note": "need study/dose/logrr/se columns"})
        return state
    sc, dc, yc, ec = cols
    rho = float(kwargs.get("rho", 0.5))
    slopes, svars = [], []
    for s, g in df.groupby(sc, sort=False):
        g = g.reset_index(drop=True)
        nz = g[yc].to_numpy(float) != 0
        x = g[dc].to_numpy(float) - g[dc].to_numpy(float)[~nz][0] if (~nz).any() else g[dc].to_numpy(float)
        y = g[yc].to_numpy(float); se = g[ec].to_numpy(float)
        mask = nz if (~nz).any() else np.ones(len(g), bool)
        xs, ys, ses = x[mask], y[mask], se[mask]
        if len(xs) < 1:
            continue
        # shared-reference GL covariance: off-diag = rho·se_i·se_j
        C = rho * np.outer(ses, ses); np.fill_diagonal(C, ses ** 2)
        Ci = np.linalg.pinv(C)
        denom = float(xs @ Ci @ xs)
        if denom <= 0:
            continue
        b = float(xs @ Ci @ ys) / denom
        slopes.append(b); svars.append(1.0 / denom)
    if not slopes:
        state.write("models", "dosres", {"note": "no usable studies"})
        return state
    b = np.array(slopes); v = np.array(svars)
    tau2 = _estimate_tau2(b, v, "REML")
    w = 1 / (v + tau2); mu = float(np.sum(w * b) / np.sum(w)); se = float(np.sqrt(1 / np.sum(w)))
    from scipy import stats
    state.write("models", "dosres", {
        "model": "linear", "slope_per_unit": mu, "se": se,
        "ci_lb": mu - 1.96 * se, "ci_ub": mu + 1.96 * se,
        "rr_per_unit": float(np.exp(mu)), "tau2": tau2, "k": len(slopes),
        "pval": float(2 * stats.norm.sf(abs(mu / se))),
        "per_study_slopes": slopes,
    })
    return state


@register(
    name="dosresmeta_spline", aliases=["样条剂量反应", "nonlinear_dose_response", "rcs_dose"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="非线性剂量反应:限制性立方样条(RCS)基 + 两阶段多元合并 + 非线性 Wald 检验",
    requires={"sources": ["datasets"]}, produces={"models": ["dosres"]},
)
def dosresmeta_spline(state: StudyState, **kwargs: Any) -> StudyState:
    """Nonlinear dose-response via restricted cubic splines (2-stage multivariate pool)."""
    df, cols = _study_iter(state, kwargs)
    if df is None:
        state.write("models", "dosres", {"note": "need study/dose/logrr/se columns"})
        return state
    from scipy import stats
    sc, dc, yc, ec = cols
    allx = df[dc].to_numpy(float)
    knots = kwargs.get("knots") or np.quantile(allx, [0.1, 0.5, 0.9])
    rho = float(kwargs.get("rho", 0.5))
    betas, covs = [], []
    for s, g in df.groupby(sc, sort=False):
        g = g.reset_index(drop=True)
        nz = g[yc].to_numpy(float) != 0
        ref = g[dc].to_numpy(float)[~nz][0] if (~nz).any() else 0.0
        mask = nz if (~nz).any() else np.ones(len(g), bool)
        B = _rcs_basis(g[dc].to_numpy(float) - ref, np.asarray(knots) - ref)[mask]
        y = g[yc].to_numpy(float)[mask]; se = g[ec].to_numpy(float)[mask]
        if len(y) <= B.shape[1]:
            continue
        C = rho * np.outer(se, se); np.fill_diagonal(C, se ** 2); Ci = np.linalg.pinv(C)
        A = np.linalg.pinv(B.T @ Ci @ B)
        betas.append(A @ (B.T @ Ci @ y)); covs.append(A)
    if not betas:
        state.write("models", "dosres", {"note": "no usable studies for spline"})
        return state
    P = betas[0].shape[0]
    # multivariate fixed-effects pool
    Ainv = np.zeros((P, P)); b = np.zeros(P)
    for be, co in zip(betas, covs):
        Wi = np.linalg.pinv(co); Ainv += Wi; b += Wi @ be
    V = np.linalg.pinv(Ainv); beta = V @ b
    # nonlinearity test = Wald on the spline (non-linear) terms
    nl = beta[1:]; Vnl = V[1:, 1:]
    wald = float(nl @ np.linalg.pinv(Vnl) @ nl) if len(nl) else 0.0
    state.write("models", "dosres", {
        "model": "spline_rcs", "coefficients": beta.tolist(), "knots": list(np.asarray(knots, float)),
        "nonlinearity_wald": wald, "nonlinearity_df": len(nl),
        "nonlinearity_pval": float(stats.chi2.sf(wald, len(nl))) if len(nl) else float("nan"),
        "k": len(betas),
    })
    return state
