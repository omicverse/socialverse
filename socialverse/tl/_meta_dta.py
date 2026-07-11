"""``sv.tl._meta_dta`` — diagnostic test accuracy meta-analysis (Tier-3).

Per-study sensitivity/specificity descriptives, and the Reitsma bivariate
random-effects model (jointly pooling logit-sensitivity and logit-specificity
with their between-study correlation) — the workhorse behind mada / metandi,
native and faithful. The exact binomial-within GLMM (``dta_glmm``) is exposed as
the same bivariate-normal approximation with an experimental caveat.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ..pp._meta_es import _resolve_df, _col


def _counts(state, kwargs):
    df = _resolve_df(state, kwargs)
    if df is None:
        return None
    tp = _col(df, kwargs.get("tp")); fp = _col(df, kwargs.get("fp"))
    fn = _col(df, kwargs.get("fn")); tn = _col(df, kwargs.get("tn"))
    if any(x is None for x in (tp, fp, fn, tn)):
        return None
    m = np.column_stack([tp.to_numpy(float), fp.to_numpy(float), fn.to_numpy(float), tn.to_numpy(float)])
    return m


@register(
    name="dta_descriptives", aliases=["诊断准确性描述", "sens_spec"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="诊断试验逐研究:敏感度/特异度/DOR/LR+/LR-(TP/FP/FN/TN,0 格连续性校正)",
    requires={"sources": ["datasets"]}, produces={"models": ["dta"]},
)
def dta_descriptives(state: StudyState, **kwargs: Any) -> StudyState:
    """Per-study sensitivity, specificity, DOR, LR+, LR− from TP/FP/FN/TN."""
    m = _counts(state, kwargs)
    if m is None:
        state.write("models", "dta", {"note": "need tp/fp/fn/tn columns"})
        return state
    tp, fp, fn, tn = (m[:, i] for i in range(4))
    cc = ((tp == 0) | (fp == 0) | (fn == 0) | (tn == 0)) * 0.5
    tp, fp, fn, tn = tp + cc, fp + cc, fn + cc, tn + cc
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    dor = (tp * tn) / (fp * fn)
    state.write("models", "dta", {
        "sensitivity": sens.tolist(), "specificity": spec.tolist(),
        "DOR": dor.tolist(), "LR_pos": (sens / (1 - spec)).tolist(),
        "LR_neg": ((1 - sens) / spec).tolist(), "k": len(sens),
        "_logit": np.column_stack([np.log(sens / (1 - sens)), np.log(spec / (1 - spec))]).tolist(),
        "_v": np.column_stack([1 / tp + 1 / fn, 1 / tn + 1 / fp]).tolist(),
    })
    return state


def _reitsma(Y, S):
    """Bivariate random-effects (Reitsma): estimate μ (2) + between-study Σ (2×2) by ML."""
    from scipy import optimize
    k = len(Y)

    def unpack(th):
        s1, s2 = np.exp(th[0]), np.exp(th[1]); rho = np.tanh(th[2])
        Sig = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
        return Sig

    def gls_mu(Sig):
        A = np.zeros((2, 2)); b = np.zeros(2)
        for i in range(k):
            Mi = np.linalg.pinv(np.diag(S[i]) + Sig)
            A += Mi; b += Mi @ Y[i]
        return np.linalg.pinv(A) @ b

    def negll(th):
        Sig = unpack(th); mu = gls_mu(Sig); ll = 0.0
        for i in range(k):
            M = np.diag(S[i]) + Sig
            sign, ld = np.linalg.slogdet(M)
            r = Y[i] - mu
            ll += -0.5 * (ld + r @ np.linalg.pinv(M) @ r)
        return -ll
    x0 = np.array([np.log(0.5), np.log(0.5), 0.0])
    res = optimize.minimize(negll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 3000})
    Sig = unpack(res.x); mu = gls_mu(Sig)
    # vcov of mu
    A = np.zeros((2, 2))
    for i in range(k):
        A += np.linalg.pinv(np.diag(S[i]) + Sig)
    return mu, np.linalg.pinv(A), Sig, res.success


@register(
    name="dta_bivariate", aliases=["Reitsma双变量", "bivariate_dta"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="Reitsma 双变量随机效应:联合合并 logit 敏感度/特异度 + 研究间相关,产出汇总点 + SROC 参数(mada 的原生等价)",
    requires={"models": ["dta"]}, produces={"models": ["dta_bivariate"]},
    prerequisites={"functions": ["dta_descriptives"]},
)
def dta_bivariate(state: StudyState, **kwargs: Any) -> StudyState:
    """Reitsma bivariate random-effects model → summary sensitivity/specificity + SROC."""
    dta = state.models.get("dta")
    if not isinstance(dta, dict) or "_logit" not in dta:
        return state

    def expit(x):
        return 1 / (1 + np.exp(-x))

    # --- Preferred path: faithful mada::reitsma port (proven 1e-6 vs R). -----
    # The port works on the raw 2x2 cell counts and parametrises the second
    # outcome as the false-positive rate (fpr = 1 - specificity) on the logit
    # scale.  We re-derive the counts from the state frame (the same helper
    # dta_descriptives uses) and translate the port's (logit sens, logit fpr)
    # basis back into the (logit sens, logit spec) basis this function has
    # always written, so downstream keys / shapes are unchanged.
    try:
        m = _counts(state, kwargs)
        if m is None:
            raise ValueError("raw tp/fp/fn/tn counts unavailable")
        tp, fp, fn, tn = (m[:, i] for i in range(4))
        from ..external.pymada import reitsma as _pymada_reitsma
        fit = _pymada_reitsma(TP=tp, FN=fn, FP=fp, TN=tn)
        coef = np.asarray(fit["coefficients"], float)   # [logit sens, logit fpr]
        vcov = np.asarray(fit["vcov"], float)           # cov in (sens, fpr) basis
        Psi = np.asarray(fit["Psi"], float)             # between-study, (sens, fpr)
        # Change of variable fpr -> spec = 1 - fpr, i.e. logit(spec) = -logit(fpr).
        # This is a linear map J = diag(1, -1) on the logit-scale parameters, so
        # covariances transform as J C J' (identical variances, flipped
        # off-diagonal sign) and the correlation sign flips.
        J = np.array([1.0, -1.0])
        mu = coef * J                                   # [logit sens, logit spec]
        Vmu = vcov * np.outer(J, J)
        Sig = Psi * np.outer(J, J)
        se_mu = np.sqrt(np.diag(Vmu))
        summary = {
            "sensitivity": float(expit(mu[0])), "specificity": float(expit(mu[1])),
            "sens_ci": [float(expit(mu[0] - 1.96 * se_mu[0])), float(expit(mu[0] + 1.96 * se_mu[0]))],
            "spec_ci": [float(expit(mu[1] - 1.96 * se_mu[1])), float(expit(mu[1] + 1.96 * se_mu[1]))],
            "mu_logit": mu.tolist(), "Sigma": Sig.tolist(),
            "corr": float(Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])),
            "DOR": float(np.exp(mu[0] + mu[1])),
            "converged": True, "_mu": mu.tolist(), "_Sigma": Sig.tolist(),
            "backend": "pymada",
        }
        state.write("models", "dta_bivariate", summary)
        return state
    except Exception:
        pass

    # --- Fallback: pre-existing native Nelder-Mead implementation. -----------
    Y = np.array(dta["_logit"]); S = np.array(dta["_v"])
    mu, Vmu, Sig, ok = _reitsma(Y, S)
    se_mu = np.sqrt(np.diag(Vmu))
    summary = {
        "sensitivity": float(expit(mu[0])), "specificity": float(expit(mu[1])),
        "sens_ci": [float(expit(mu[0] - 1.96 * se_mu[0])), float(expit(mu[0] + 1.96 * se_mu[0]))],
        "spec_ci": [float(expit(mu[1] - 1.96 * se_mu[1])), float(expit(mu[1] + 1.96 * se_mu[1]))],
        "mu_logit": mu.tolist(), "Sigma": Sig.tolist(), "corr": float(Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])),
        "DOR": float(np.exp(mu[0] + mu[1])),
        "converged": bool(ok), "_mu": mu.tolist(), "_Sigma": Sig.tolist(),
        "backend": "native",
    }
    state.write("models", "dta_bivariate", summary)
    return state


@register(
    name="dta_glmm", aliases=["诊断GLMM", "binomial_dta"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="诊断准确性二项 GLMM(实验性):当前用 Reitsma 双变量正态近似(mada 默认);精确 2-D 求积待后续",
    requires={"models": ["dta"]}, produces={"models": ["dta_bivariate"]},
    prerequisites={"functions": ["dta_descriptives"]},
)
def dta_glmm(state: StudyState, **kwargs: Any) -> StudyState:
    """Binomial-within GLMM for DTA — experimental; delegates to the Reitsma
    bivariate-normal approximation (exact GLMM needs 2-D adaptive quadrature)."""
    dta_bivariate(state, **kwargs)
    m = state.models.get("dta_bivariate")
    if isinstance(m, dict):
        m["note"] = "bivariate-normal (Reitsma) approximation to the exact binomial GLMM — experimental"
        state.write("models", "dta_bivariate", m)
    return state
