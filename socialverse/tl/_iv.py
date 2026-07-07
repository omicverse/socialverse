"""``sv.tl._iv`` — registered implementation for the endogeneity gap:
**instrumental variables / two-stage least squares (2SLS)**.

The workhorse of applied-microeconomics causal inference when a regressor is
correlated with the error (unobserved confounding, simultaneity, measurement
error). Ported to the ``StudyState`` / ``registry`` spine with *real* estimation
that recovers the known DGP effect, no placeholders.

Champion package this file mirrors
----------------------------------
* ``iv_regress`` — Stata's ``ivregress 2sls`` / R's ``AER::ivreg`` / Python
  ``linearmodels.IV2SLS``. Two-stage least squares: regress the endogenous
  regressor on the instruments and exogenous controls (first stage), then
  regress the outcome on the *fitted* endogenous value plus the exogenous
  controls (second stage). The point estimate equals the fitted-value 2SLS
  slope, **but the naive second-stage OLS standard errors are wrong** — they use
  the fitted-value residuals instead of the true structural residuals. We compute
  the correct 2SLS covariance (residuals formed with the *observed* endogenous
  regressor, not its projection), with either classical or heteroskedasticity-
  robust (White/HC0) variance. ``linearmodels.IV2SLS`` is used when installed as
  a cross-check; otherwise every reported number comes from the hand-rolled
  projection estimator, which is algebraically identical.

  First-stage strength is reported as the first-stage F-statistic on the excluded
  instruments (the Stock-Yogo weak-instrument diagnostic; ``F < 10`` flags a weak
  instrument), alongside the biased OLS endogenous coefficient for contrast.

Registry contract: ``requires`` a ``sources['datasets']`` frame and a declared
``variables['outcome']``; ``produces`` the fitted ``models['iv']`` and the
``diagnostics['first_stage']`` weak-instrument check — so a resolver can refuse
to trust an IV coefficient until the first stage has actually been shown strong.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["iv_regress"]


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


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


# ------------------------------------------------------------------- iv_regress
@register(
    name="iv_regress",
    aliases=["工具变量", "IV", "2SLS", "ivregress"],
    category="causal",
    tier="plus",
    skill="(工具变量 缺口)",
    languages=["Python"],
    key_tools=["statsmodels", "linearmodels"],
    description="工具变量两阶段最小二乘(2SLS):用工具 z 恢复内生 x 的因果效应 + 一阶段 F 弱工具检验",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["iv"], "diagnostics": ["first_stage"]},
    auto_fix="escalate",
)
def iv_regress(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-stage least squares (2SLS) instrumental-variables regression.

    Estimates the causal effect of an endogenous regressor on the outcome using
    excluded instruments. Stage 1 projects each endogenous regressor on
    ``[instruments + exog + const]`` (fitted values + first-stage F on the
    excluded instruments). Stage 2 regresses the outcome on ``[endog_hat + exog +
    const]`` for the point estimates, but the reported standard errors use the
    **correct 2SLS covariance** — residuals are formed with the *observed*
    endogenous regressor (not its first-stage projection), classical or HC0-robust.

    kwargs
    ------
    outcome : str
        Outcome (dependent) column. Default: ``variables['outcome']`` or ``"y"``.
    endogenous : str | list[str]
        Endogenous regressor(s). Default: ``"x"`` if present.
    instruments : list[str]
        Excluded instrument(s) for the endogenous regressor(s). Default: ``["z"]``.
    exog : list[str]
        Exogenous (included) control(s). Default: ``["w"]`` if present.
    cov : str
        ``"robust"`` (HC0, default) or ``"classical"`` for the 2SLS SEs.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        state.write("models", "iv", {
            "coef": {}, "se": {}, "ci": {}, "p": {},
            "n": 0, "estimator": "2SLS", "note": note,
        })
        state.write("diagnostics", "first_stage", {
            "F": None, "weak_instrument": None,
            "ols_endog_coef": None, "iv_endog_coef": None, "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法进行工具变量回归")

    outcome = kwargs.get("outcome") or state.variables.get("outcome") or "y"
    if outcome not in df.columns:
        return _empty(f"找不到结果变量 outcome='{outcome}'")

    endog = _as_list(kwargs.get("endogenous")) or [c for c in ["x"] if c in df.columns]
    endog = [c for c in endog if c in df.columns]
    if not endog:
        return _empty("找不到内生变量(endogenous)")

    instruments = _as_list(kwargs.get("instruments")) or [c for c in ["z"] if c in df.columns]
    instruments = [c for c in instruments if c in df.columns and c not in endog]
    if not instruments:
        return _empty("找不到工具变量(instruments)")
    if len(instruments) < len(endog):
        return _empty(
            f"欠识别:工具数({len(instruments)}) < 内生变量数({len(endog)})"
        )

    exog = _as_list(kwargs.get("exog"))
    if not exog and "w" in df.columns and "w" not in endog + instruments + [outcome]:
        exog = ["w"]
    exog = [c for c in exog if c in df.columns and c not in endog + instruments + [outcome]]

    cov = str(kwargs.get("cov", "robust")).lower()
    robust = cov not in ("classical", "nonrobust", "homoskedastic")

    cols = [outcome] + endog + instruments + exog
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(work))
    if n <= len(endog) + len(exog) + 2:
        return _empty("有效样本量不足,无法拟合 2SLS")

    y = work[outcome].to_numpy(dtype=float)
    const = np.ones((n, 1))
    En = work[endog].to_numpy(dtype=float)        # (n, p_endog) observed endogenous
    Z = work[instruments].to_numpy(dtype=float)   # (n, p_inst) excluded instruments
    Wc = work[exog].to_numpy(dtype=float) if exog else np.empty((n, 0))

    # ---- naming for coefficient vectors (structural equation order) ----------
    # structural: y = const + endog... + exog...
    struct_names = ["const"] + endog + exog

    # ================= first stage: F on the excluded instruments =============
    # endog_j ~ const + exog + instruments ; F tests the joint sig of instruments
    X_full = np.column_stack([const, Wc, Z]) if Wc.size else np.column_stack([const, Z])
    X_rest = np.column_stack([const, Wc]) if Wc.size else const  # restricted (no instruments)
    fs_F_list: list[float] = []
    endog_hat = np.empty_like(En)
    for j in range(En.shape[1]):
        ej = En[:, j]
        beta_full, _, _, _ = np.linalg.lstsq(X_full, ej, rcond=None)
        fitted = X_full @ beta_full
        endog_hat[:, j] = fitted
        rss_full = float((ej - fitted) @ (ej - fitted))
        beta_rest, _, _, _ = np.linalg.lstsq(X_rest, ej, rcond=None)
        rss_rest = float((ej - X_rest @ beta_rest) @ (ej - X_rest @ beta_rest))
        q = Z.shape[1]                       # # excluded instruments
        k_full = X_full.shape[1]
        dof = max(n - k_full, 1)
        # F = ((RSS_r - RSS_f)/q) / (RSS_f/(n-k_full))
        num = (rss_rest - rss_full) / q
        den = rss_full / dof
        fs_F_list.append(float(num / den) if den > 0 else float("inf"))
    first_stage_F = float(min(fs_F_list))    # weakest first stage governs

    # ================= second stage: point estimates via projection ===========
    # X2 = [const, endog_hat, exog] ; beta_2sls = (X2'X2)^-1 X2'y
    X2 = np.column_stack([const, endog_hat, Wc]) if Wc.size else np.column_stack([const, endog_hat])
    XtX = X2.T @ X2
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X2.T @ y)

    # ---- CORRECT 2SLS residuals: use OBSERVED endogenous, not the projection --
    Xstruct = np.column_stack([const, En, Wc]) if Wc.size else np.column_stack([const, En])
    resid = y - Xstruct @ beta               # structural residuals (true SEs)
    k = X2.shape[1]
    dof = max(n - k, 1)

    if robust:
        # HC0 sandwich: (X2'X2)^-1 (Σ u_i^2 x2_i x2_i') (X2'X2)^-1
        meat = (X2 * (resid ** 2)[:, None]).T @ X2
        vcov = XtX_inv @ meat @ XtX_inv
        se_kind = "robust(HC0)"
    else:
        sigma2 = float(resid @ resid) / dof
        vcov = XtX_inv * sigma2
        se_kind = "classical"

    se = np.sqrt(np.clip(np.diag(vcov), 0.0, np.inf))

    # z-based inference (large-sample, matches ivregress default)
    from scipy import stats as _st
    zstat = np.where(se > 0, beta / se, 0.0)
    pvals = 2.0 * _st.norm.sf(np.abs(zstat))
    crit = float(_st.norm.ppf(0.975))
    ci_lo = beta - crit * se
    ci_hi = beta + crit * se

    coef = {nm: float(b) for nm, b in zip(struct_names, beta)}
    se_d = {nm: float(s) for nm, s in zip(struct_names, se)}
    ci_d = {nm: (float(lo), float(hi)) for nm, lo, hi in zip(struct_names, ci_lo, ci_hi)}
    p_d = {nm: float(p) for nm, p in zip(struct_names, pvals)}

    # ---- biased OLS endogenous coefficient for contrast ----------------------
    X_ols = np.column_stack([const, En, Wc]) if Wc.size else np.column_stack([const, En])
    beta_ols, _, _, _ = np.linalg.lstsq(X_ols, y, rcond=None)
    # endogenous coef is the first after the const
    ols_endog = float(beta_ols[1])
    iv_endog = float(beta[1])

    # ---- optional linearmodels cross-check (algebraically identical) ---------
    lm = _try_import("linearmodels.iv")
    lm_endog = None
    if lm is not None:
        try:
            from linearmodels.iv import IV2SLS  # noqa: WPS433
            exog_df = pd.DataFrame({"const": np.ones(n)}, index=work.index)
            for c in exog:
                exog_df[c] = work[c].to_numpy(dtype=float)
            res_lm = IV2SLS(
                dependent=work[outcome],
                exog=exog_df,
                endog=work[endog],
                instruments=work[instruments],
            ).fit(cov_type="robust" if robust else "unadjusted")
            lm_endog = float(res_lm.params[endog[0]])
        except Exception:
            lm_endog = None

    estimator = "2SLS (projection; linearmodels 交叉验证)" if lm_endog is not None else "2SLS (hand-rolled projection)"

    state.write("models", "iv", {
        "coef": coef,
        "se": se_d,
        "ci": ci_d,
        "p": p_d,
        "outcome": outcome,
        "endogenous": endog,
        "instruments": instruments,
        "exog": exog,
        "n": n,
        "estimator": estimator,
        "cov_type": se_kind,
        "linearmodels_endog_coef": lm_endog,
        "note": (
            "2SLS 系数用一阶段拟合值估计;标准误用正确的 2SLS 协方差"
            "(残差用真实内生变量而非拟合值,{})".format(se_kind)
        ),
    })

    weak = bool(first_stage_F < 10.0)
    state.write("diagnostics", "first_stage", {
        "F": first_stage_F,
        "weak_instrument": weak,
        "ols_endog_coef": ols_endog,
        "iv_endog_coef": iv_endog,
        "endogenous": endog,
        "instruments": instruments,
        "note": (
            "一阶段 F(排除性工具联合显著)= {:.2f};F<10 判为弱工具。"
            "OLS 内生系数 {:.3f}(受混杂偏误)vs 2SLS {:.3f}。".format(
                first_stage_F, ols_endog, iv_endog
            )
        ),
    })
    return state


# ------------------------------------------------------------------- self-test
if __name__ == "__main__":
    from socialverse.datasets import load_iv
    from socialverse._state import StudyState as _S

    df = load_iv()
    st = _S()
    st.sources["datasets"] = df
    st.variables["outcome"] = "y"

    st = iv_regress(
        st, endogenous="x", instruments=["z"], exog=["w"], cov="robust"
    )
    m = st.models["iv"]
    fs = st.diagnostics["first_stage"]

    truth = 1.5
    iv_x = m["coef"]["x"]
    ols_x = fs["ols_endog_coef"]
    F = fs["F"]

    print("=== IV 2SLS self-test (truth: x effect = 1.5) ===")
    print(f"2SLS x coef      : {iv_x:.4f}   (truth 1.5, tol ±0.2)")
    print(f"  SE / 95% CI    : {m['se']['x']:.4f} / "
          f"({m['ci']['x'][0]:.3f}, {m['ci']['x'][1]:.3f})   p={m['p']['x']:.2e}")
    print(f"OLS x coef       : {ols_x:.4f}   (biased UP, should be > 1.5)")
    print(f"first-stage F    : {F:.2f}   (should be >> 10; weak={fs['weak_instrument']})")
    print(f"const / w coefs  : const={m['coef']['const']:.3f}, w={m['coef']['w']:.3f}")
    print(f"cov_type / n     : {m['cov_type']} / n={m['n']}")

    ok = (abs(iv_x - truth) < 0.2) and (ols_x > 1.5) and (F > 10)
    print(f"\nRECOVERED TRUTH: {ok}")
    assert abs(iv_x - truth) < 0.2, f"2SLS x={iv_x} not within 0.2 of 1.5"
    assert ols_x > 1.5, f"OLS x={ols_x} should overstate (>1.5)"
    assert F > 10, f"first-stage F={F} should be >> 10"
    print("ALL CHECKS PASSED")
