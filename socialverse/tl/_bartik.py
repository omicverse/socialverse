"""``sv.tl._bartik`` — shift-share (Bartik) instrumental-variables estimation.

A pervasive identification strategy in labor / urban / trade / political economy: an
endogenous local variable (e.g. employment growth, immigration) is instrumented by a
**shift-share** constructed from local exposure **shares** ``s_{ik}`` (industry k's
weight in unit i) times national **shocks** ``g_k`` (industry k's national growth):
``z_i = Σ_k s_{ik} · g_k``. Under the Goldsmith-Pinkham-Sorkin-Swift (2020) view the
identifying variation is the shares; under Borusyak-Hull-Jaravel it is the shocks.

``bartik_iv`` builds the instrument and runs 2SLS (with a first-stage F for
instrument strength), reusing the design-based causal contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _get_datasets


@register(
    name="bartik_iv",
    aliases=["shift_share", "移份额工具变量", "bartik", "shift_share_iv"],
    category="causal",
    tier="pro",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="Shift-share/Bartik 工具变量:本地份额×全国冲击构造 IV → 2SLS(含一阶段弱工具 F)",
    requires={"variables": ["outcome"]},
    produces={"models": ["bartik_iv"]},
    auto_fix="escalate",
)
def bartik_iv(state: StudyState, **kwargs: Any) -> StudyState:
    """Shift-share (Bartik) IV.

    Keyword arguments: ``shares=`` list of exposure-share columns (unit × sector),
    ``shocks=`` sector-level national shocks (a list/array aligned to ``shares``, or a
    column name whose values are per-sector), ``endog=`` the endogenous regressor
    column, ``outcome=`` (or from variables), ``controls=`` optional exogenous
    controls. Constructs ``z_i = Σ_k s_{ik}·g_k`` and 2SLS-estimates the effect of the
    endogenous regressor on the outcome, reporting the first-stage F.
    """
    df = _get_datasets(state, kwargs)
    sm = __import__("statsmodels.api", fromlist=["api"])
    Y = kwargs.get("outcome") or state.variables.get("outcome")
    endog = kwargs.get("endog") or kwargs.get("treatment") or state.design.get("treatment")
    shares = kwargs.get("shares")
    shocks = kwargs.get("shocks")
    controls = kwargs.get("controls") or []
    if isinstance(controls, str):
        controls = [controls]

    def _empty(note):
        state.write("models", "bartik_iv", {"beta": None, "note": note})
        return state

    if df is None or Y is None or endog is None:
        return _empty("缺少 data / outcome / endog(内生回归变量)")
    if not shares:
        return _empty("缺少 shares=(本地份额列,unit×sector)")

    S = df[list(shares)].apply(pd.to_numeric, errors="coerce").to_numpy(float)  # n × K
    if shocks is None:
        return _empty("缺少 shocks=(各 sector 全国冲击,与 shares 对齐)")
    if isinstance(shocks, str):
        g = pd.to_numeric(df[shocks], errors="coerce").to_numpy(float)
        if g.shape[0] == S.shape[0]:  # given per-row; collapse to per-sector by mean
            return _empty("shocks 应为按 sector 的向量(长度=份额列数),而非按行列")
    else:
        g = np.asarray(shocks, float)
    if g.shape[0] != S.shape[1]:
        return _empty(f"shocks 长度({g.shape[0]})必须等于 shares 列数({S.shape[1]})")

    z = S @ g  # the Bartik instrument
    x = pd.to_numeric(df[endog], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[Y], errors="coerce").to_numpy(float)
    W = (df[list(controls)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
         if controls else np.empty((len(y), 0)))
    ok = np.isfinite(z) & np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(W), axis=1)
    z, x, y, W = z[ok], x[ok], y[ok], W[ok]
    if len(y) < 10:
        return _empty("有效样本过小")

    Wc = np.column_stack([np.ones(len(y)), W]) if W.shape[1] else np.ones((len(y), 1))
    # first stage: x ~ z + controls
    Z1 = np.column_stack([Wc, z])
    fs = sm.OLS(x, Z1).fit(cov_type="HC1")
    xhat = fs.predict(Z1)
    # first-stage F on the excluded instrument z (last coef)
    jz = Z1.shape[1] - 1
    f_stat = float((fs.params[jz] / fs.bse[jz]) ** 2)
    # second stage: y ~ xhat + controls (2SLS point est), IV-correct SE via statsmodels
    X2 = np.column_stack([Wc, x])
    Zinst = np.column_stack([Wc, z])
    estimator = "shift_share_2sls"
    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS
        res = IV2SLS(y, X2, Zinst).fit()
        beta = float(res.params[-1])
        se = float(res.bse[-1])
        p = float(res.pvalues[-1])
    except Exception:  # manual 2SLS with the CORRECT structural-residual variance
        from scipy import stats
        X2h = np.column_stack([Wc, xhat])
        b = np.linalg.lstsq(X2h, y, rcond=None)[0]
        resid = y - X2 @ b                       # residual uses ACTUAL x, not xhat (2SLS)
        dof = max(1, len(y) - X2h.shape[1])
        sigma2 = float(resid @ resid) / dof
        V = sigma2 * np.linalg.pinv(X2h.T @ X2h)
        beta = float(b[-1])
        se = float(np.sqrt(max(V[-1, -1], 0.0)))
        p = float(2 * (1 - stats.t.cdf(abs(beta / se), dof))) if se > 0 else None
        estimator = "shift_share_2sls_fallback"

    ols = sm.OLS(y, X2).fit(cov_type="HC1")
    state.write("models", "bartik_iv", {
        "beta": beta, "se": se, "ci": [beta - 1.96 * se, beta + 1.96 * se], "p": p,
        "first_stage_F": f_stat, "weak_instrument": f_stat < 10,
        "ols_beta": float(ols.params[-1]), "n": int(len(y)), "n_sectors": int(S.shape[1]),
        "endog": endog, "outcome": Y, "estimator": estimator,
        "note": "Shift-share/Bartik IV:z=Σ份额×冲击 的 2SLS;"
                + ("⚠️弱工具(一阶段 F<10);" if f_stat < 10 else "一阶段 F≥10;")
                + ("⚠️IV2SLS 不可用,回退手工 2SLS(SE 已按结构残差修正)"
                   if estimator.endswith("fallback") else ""),
    })
    return state


__all__ = ["bartik_iv"]
