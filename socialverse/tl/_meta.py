"""``sv.tl._meta`` — native meta-analysis / evidence-synthesis engine.

Pure Python (numpy / scipy), **no R / metafor dependency**. Reproduces the
core of a standard multilevel prevalence/severity meta-analysis (the pattern of
Dreisoerner et al. 2026, *Nat. Hum. Behav.*: 3-level ``rma.mv`` on arcsine/logit
prevalence, heterogeneity, meta-regression + FDR, Egger, forest).

Effect sizes come from ``state.models['meta_effects']`` (a ``(yi, vi, …)``
frame produced by ``sv.pp.escalc`` / ``sv.pp.es_*``). Functions here fit and
diagnose, writing to ``models`` / ``diagnostics``.

Faithfulness (honest): fixed/random inverse-variance pooling, DL τ², Q/I²/H²,
prediction intervals, Egger's OLS test, and the 3-level variance decomposition
are **exact closed-form**. REML/ML τ² and the ``rma_mv`` variance components are
**iterative (scipy)** — statistically exact, faithful to metafor, but not
bit-identical (different optimizer path). No MCMC anywhere.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState


# ------------------------------------------------------------------ helpers
def _effects(state: StudyState) -> pd.DataFrame | None:
    eff = state.models.get("meta_effects")
    if isinstance(eff, pd.DataFrame) and {"yi", "vi"}.issubset(eff.columns) and len(eff):
        return eff.reset_index(drop=True)
    return None


def _weighted_mean(y, w):
    return float(np.sum(w * y) / np.sum(w))


def _tau2_DL(y, v):
    """DerSimonian-Laird τ² (closed form, exact)."""
    w = 1.0 / v
    mu = _weighted_mean(y, w)
    Q = float(np.sum(w * (y - mu) ** 2))
    k = len(y)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    return max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0, Q


def _reml_2level(y, v, X, tau2_method="REML"):
    """Random-effects / mixed model with a single between-study variance τ².

    Returns dict with beta, vcov (of beta), tau2, se, k. ``X`` is the design
    matrix (n×p); intercept-only ⇒ classic random-effects pooling.
    """
    from scipy import optimize, linalg
    y = np.asarray(y, float); v = np.asarray(v, float); X = np.asarray(X, float)
    n, p = X.shape

    def fit_beta(tau2):
        w = 1.0 / (v + tau2)
        XtWX = X.T @ (w[:, None] * X)
        XtWy = X.T @ (w * y)
        vcov = linalg.pinv(XtWX)
        beta = vcov @ XtWy
        return beta, vcov, w

    def neg_reml(logtau2):
        tau2 = np.exp(logtau2)
        beta, vcov, w = fit_beta(tau2)
        resid = y - X @ beta
        # REML restricted log-likelihood (up to constants)
        ll = -0.5 * (np.sum(np.log(v + tau2)) + np.log(max(linalg.det(X.T @ (w[:, None] * X)), 1e-300))
                     + np.sum(w * resid ** 2))
        return -ll

    if tau2_method.upper() == "DL" and p == 1:
        tau2, _ = _tau2_DL(y, v)
    else:
        # bounded search over log τ² (add a tiny floor so log is finite)
        lo, hi = np.log(1e-8), np.log(max(np.var(y), 1e-6) * 100 + 1e-6)
        res = optimize.minimize_scalar(neg_reml, bounds=(lo, hi), method="bounded")
        tau2 = float(np.exp(res.x))
        if tau2 < 1e-7:
            tau2 = 0.0
    beta, vcov, w = fit_beta(tau2)
    return {
        "beta": beta, "vcov": vcov, "tau2": float(tau2),
        "se": np.sqrt(np.diag(vcov)), "k": n, "p": p, "y": y, "v": v, "X": X,
    }


def _reml_3level(y, V, X, study, ml=False):
    """3-level mixed model with KNOWN sampling covariance V and two random
    intercepts: level-3 between-study (σ²₃) + level-2 within-study/between-ES (σ²₂).

    M = V + σ²₂·I + σ²₃·S,  S[i,j]=1 iff same study.
    REML profile over (log σ²₂, log σ²₃); GLS for β. Returns dict.
    """
    from scipy import optimize, linalg
    y = np.asarray(y, float); X = np.asarray(X, float)
    V = np.asarray(V, float)
    n, p = X.shape
    study = np.asarray(study)
    S = (study[:, None] == study[None, :]).astype(float)
    I = np.eye(n)

    def M_of(s2, s3):
        return V + s2 * I + s3 * S

    def fit(s2, s3):
        M = M_of(s2, s3)
        Mi = linalg.pinv(M)
        XtMiX = X.T @ Mi @ X
        vcov = linalg.pinv(XtMiX)
        beta = vcov @ (X.T @ Mi @ y)
        return beta, vcov, M, Mi, XtMiX

    def neg_ll(theta):
        s2, s3 = np.exp(theta[0]), np.exp(theta[1])
        beta, vcov, M, Mi, XtMiX = fit(s2, s3)
        resid = y - X @ beta
        sign, logdetM = np.linalg.slogdet(M)
        sign2, logdetXMX = np.linalg.slogdet(XtMiX)
        quad = float(resid @ Mi @ resid)
        reml = -0.5 * (logdetM + logdetXMX + quad)     # REML
        ml_ll = -0.5 * (logdetM + quad)                # ML
        return -(ml_ll if ml else reml)

    # start from method-of-moments-ish scale
    v_typ = float(np.median(np.diag(V)))
    x0 = np.log([max(v_typ, 1e-4), max(v_typ, 1e-4)])
    res = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    s2, s3 = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    s2 = 0.0 if s2 < 1e-7 else s2
    s3 = 0.0 if s3 < 1e-7 else s3
    beta, vcov, M, Mi, XtMiX = fit(s2, s3)
    return {
        "beta": beta, "vcov": vcov, "se": np.sqrt(np.diag(vcov)),
        "sigma2_2": s2, "sigma2_3": s3, "sigma2_total": s2 + s3,
        "k": n, "p": p, "n_studies": int(len(np.unique(study))),
        "y": y, "V": V, "X": X, "study": study, "converged": bool(res.success),
    }


def _estimate_tau2(y, v, method="REML"):
    """Between-study variance τ² by any of the standard estimators.

    Closed-form: DL, HE (Hedges), HS (Hunter-Schmidt), SJ (Sidik-Jonkman).
    Iterative (scipy): ML, REML (profile), PM/EB (Paule-Mandel generalised-Q root).
    All clamp at 0. Faithful to metafor's estimating equations.
    """
    from scipy import optimize
    y = np.asarray(y, float); v = np.asarray(v, float); k = len(y)
    method = str(method).upper()
    w = 1.0 / v
    mu_fe = _weighted_mean(y, w)
    Q = float(np.sum(w * (y - mu_fe) ** 2))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    if method == "DL":
        return max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    if method == "HE":                       # Hedges (unweighted)
        return max(0.0, float(np.sum((y - np.mean(y)) ** 2) / (k - 1) - np.mean(v)))
    if method == "HS":                       # Hunter-Schmidt (negatively biased)
        return max(0.0, (Q - k) / float(np.sum(w)))
    if method == "SJ":                       # Sidik-Jonkman
        tau2_0 = max(float(np.sum((y - np.mean(y)) ** 2) / k), 1e-8)
        u = 1.0 / (v / tau2_0 + 1.0)
        mu_u = np.sum(u * y) / np.sum(u)
        return float(tau2_0 * np.sum(u * (y - mu_u) ** 2) / (k - 1))
    if method in ("PM", "EB"):               # Paule-Mandel / empirical Bayes
        def gen_q(t2):
            wt = 1.0 / (v + t2); mu = np.sum(wt * y) / np.sum(wt)
            return float(np.sum(wt * (y - mu) ** 2) - (k - 1))
        if gen_q(0.0) <= 0:
            return 0.0
        hi = max(Q, 1.0)
        while gen_q(hi) > 0 and hi < 1e8:
            hi *= 2
        return float(optimize.brentq(gen_q, 0.0, hi))

    def neg_ll(logt2):                        # ML / REML profile
        t2 = np.exp(logt2); wt = 1.0 / (v + t2); mu = np.sum(wt * y) / np.sum(wt)
        base = np.sum(np.log(v + t2)) + np.sum(wt * (y - mu) ** 2)
        if method == "REML":
            base += np.log(np.sum(wt))
        return 0.5 * base
    lo, hi = np.log(1e-8), np.log(max(np.var(y), 1e-6) * 100 + 1e-6)
    res = optimize.minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
    t2 = float(np.exp(res.x))
    return 0.0 if t2 < 1e-7 else t2


def _typical_v(v):
    """Higgins-Thompson 'typical' within-study sampling variance."""
    w = 1.0 / v
    k = len(v)
    denom = np.sum(w) ** 2 - np.sum(w ** 2)
    return float((k - 1) * np.sum(w) / denom) if denom > 0 else float(np.mean(v))


def _design(state, eff, kwargs):
    """Build X (intercept + optional moderators) from the effects frame."""
    n = len(eff)
    mods = kwargs.get("moderators") or kwargs.get("mods")
    if isinstance(mods, str):
        mods = [mods]
    Xcols = ["(intercept)"]
    X = [np.ones(n)]
    names = []
    for m in mods or []:
        if m in eff.columns:
            col = eff[m]
            if col.dtype == object or str(col.dtype).startswith("category"):
                dummies = pd.get_dummies(col, prefix=m, drop_first=True)
                for c in dummies.columns:
                    X.append(dummies[c].to_numpy(float)); Xcols.append(c); names.append(c)
            else:
                X.append(pd.to_numeric(col, errors="coerce").to_numpy(float))
                Xcols.append(m); names.append(m)
    return np.column_stack(X), Xcols, names


# ==================================================================== vcalc
@register(
    name="vcalc",
    aliases=["构造抽样协方差", "V矩阵", "impute_cov"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="按簇内相关 rho 插补构造块对角抽样协方差矩阵 V(相依效应量的多层/稳健估计必需输入)",
    requires={"models": ["meta_effects"]},
    produces={"models": ["meta_V"]},
)
def vcalc(state: StudyState, **kwargs: Any) -> StudyState:
    """Impute a within-cluster correlation ρ → block-diagonal sampling covariance V.

    kwargs: ``cluster=`` column on the effects frame (default ``'study'``/``'cluster'``),
    ``rho=0.6`` (assumed within-cluster correlation of sampling errors). Off-diagonals
    for effects in the same cluster = ρ·√(vᵢvⱼ); diagonal = vᵢ. Feeds ``rma_mv``.
    """
    eff = _effects(state)
    if eff is None:
        return state
    v = eff["vi"].to_numpy(float)
    rho = float(kwargs.get("rho", 0.6))
    clus_col = kwargs.get("cluster") or ("cluster" if "cluster" in eff.columns else "study")
    if clus_col not in eff.columns:
        V = np.diag(v)  # no clustering info → independent
    else:
        g = eff[clus_col].to_numpy()
        same = (g[:, None] == g[None, :])
        se = np.sqrt(v)
        V = rho * (se[:, None] * se[None, :]) * same
        np.fill_diagonal(V, v)
    state.write("models", "meta_V", V)
    return state


# ==================================================================== meta_fixed / meta_random
@register(
    name="meta_fixed",
    aliases=["固定效应meta", "common_effect", "fe_meta"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="固定/共同效应逆方差合并(所有研究同一真值假设);随机效应用 meta_random",
    requires={"models": ["meta_effects"]},
    produces={"models": ["meta"]},
)
def meta_fixed(state: StudyState, **kwargs: Any) -> StudyState:
    """Common-effect (fixed-effect) inverse-variance pooling."""
    eff = _effects(state)
    if eff is None:
        state.write("models", "meta", {"estimate": None, "note": "no meta_effects; run sv.pp.escalc first"})
        return state
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    w = 1.0 / v
    mu = _weighted_mean(y, w)
    se = float(np.sqrt(1.0 / np.sum(w)))
    from scipy import stats
    z = mu / se
    state.write("models", "meta", {
        "model": "fixed", "estimate": mu, "se": se,
        "ci_lb": mu - 1.96 * se, "ci_ub": mu + 1.96 * se,
        "zval": z, "pval": float(2 * stats.norm.sf(abs(z))), "tau2": 0.0, "k": len(y),
    })
    return state


@register(
    name="meta_random",
    aliases=["随机效应meta", "re_meta", "random_effects"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="随机效应逆方差合并;τ² 估计量 method='REML'(默认)/'DL'/'ML'/'PM'/'SJ'/'HS'/'HE';可选 HKSJ(Knapp-Hartung)t 校正",
    requires={"models": ["meta_effects"]},
    produces={"models": ["meta"]},
)
def meta_random(state: StudyState, **kwargs: Any) -> StudyState:
    """Random-effects inverse-variance pooling.

    kwargs: ``method='REML'`` (default) | ``'DL'`` | ``'ML'`` | ``'PM'`` (Paule-Mandel)
    | ``'SJ'`` | ``'HS'`` | ``'HE'`` — the full τ² estimator roster; ``knapp_hartung=False``
    (t-distribution CI with a HKSJ scale correction — materially widens CIs, use it).
    """
    eff = _effects(state)
    if eff is None:
        state.write("models", "meta", {"estimate": None, "note": "no meta_effects; run sv.pp.escalc first"})
        return state
    from scipy import stats
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    method = str(kwargs.get("method", "REML")).upper()
    hk = bool(kwargs.get("knapp_hartung", kwargs.get("hksj", False)))

    # Parity-gated backend: metafor::rma reconstruction (external/pymetafor).
    # Covers the metafor τ² estimators REML/ML/DL/EE with 1e-6 parity; also
    # yields I²/H²/Q_E/SE(τ²)/prediction-interval. PM/SJ/HS/HE fall through to
    # the legacy estimator roster below.
    if method in ("REML", "ML", "DL", "EE", "FE", "CE"):
        from ..external.pymetafor import rma as _rma_port
        r = _rma_port(y, v, method=method, test=("knha" if hk else "z"))
        pi = r.predict()
        out = {
            "model": "random", "method": method, "estimate": float(r.beta[0]),
            "se": float(r.se[0]), "ci_lb": float(r.ci_lb[0]), "ci_ub": float(r.ci_ub[0]),
            "zval": float(r.zval[0]), "pval": float(r.pval[0]),
            "tau2": r.tau2, "tau": float(np.sqrt(r.tau2)), "se_tau2": r.se_tau2,
            "I2": r.I2, "H2": r.H2, "QE": r.QE, "QEp": r.QEp,
            "pi_lb": pi["pi_lb"], "pi_ub": pi["pi_ub"],
            "k": r.k, "knapp_hartung": hk, "backend": "pymetafor",
        }
        # Per-study BLUP (empirical-Bayes shrinkage): 把每个观测效应向合并值收缩,
        # 给出收缩后的预测、SE 与预测区间(metafor::blup.rma.uni 等价)。
        try:
            from ..external.pymetafor import blup as _blup_port
            bl = _blup_port(r)
            out["blup"] = [
                {"pred": float(bl.pred[i]), "se": float(bl.se[i]),
                 "pi_lb": float(bl.pi_lb[i]), "pi_ub": float(bl.pi_ub[i])}
                for i in range(len(bl.pred))
            ]
        except Exception as exc:  # never let BLUP break the pooled result
            out["blup"] = None
            out["blup_note"] = f"BLUP unavailable: {exc}"
        state.write("models", "meta", out)
        return state

    tau2 = _estimate_tau2(y, v, method)
    k = len(y)
    w0 = 1.0 / (v + tau2)
    mu = float(np.sum(w0 * y) / np.sum(w0))
    se = float(np.sqrt(1.0 / np.sum(w0)))
    hk = bool(kwargs.get("knapp_hartung", kwargs.get("hksj", False)))
    if hk:
        w = 1.0 / (v + tau2)
        resid = y - mu
        s2 = float(np.sum(w * resid ** 2) / (k - 1))          # HKSJ scale
        se = float(np.sqrt(s2 / np.sum(w)))
        crit = stats.t.ppf(0.975, k - 1); tstat = mu / se
        pval = float(2 * stats.t.sf(abs(tstat), k - 1))
    else:
        crit = 1.959963985; tstat = mu / se
        pval = float(2 * stats.norm.sf(abs(tstat)))
    state.write("models", "meta", {
        "model": "random", "method": method, "estimate": mu, "se": se,
        "ci_lb": mu - crit * se, "ci_ub": mu + crit * se,
        "zval": tstat, "pval": pval, "tau2": tau2, "tau": float(np.sqrt(tau2)),
        "k": k, "knapp_hartung": hk,
    })
    return state


# ==================================================================== rma_mv (3-level)
@register(
    name="rma_mv",
    aliases=["多层meta", "三层meta", "multilevel_meta", "rma.mv"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="多层/三层随机效应 meta(已知抽样协方差 V):被试/结局/研究三级方差分量 + 可选调节变量;metafor::rma.mv 的原生等价(REML)",
    requires={"models": ["meta_effects"]},
    produces={"models": ["meta"]},
    prerequisites={"optional_functions": ["vcalc"]},
)
def rma_mv(state: StudyState, **kwargs: Any) -> StudyState:
    """Three-level random-effects meta-analysis (metafor::rma.mv equivalent).

    Uses ``models['meta_V']`` (from ``sv.tl.vcalc``) if present, else diag(vi).
    Random structure: level-3 between-**study** (σ²₃) + level-2 within-study /
    between-**effect** (σ²₂). kwargs: ``study=`` column (grouping; default
    ``'study'``), ``moderators=`` for a multilevel meta-regression, ``method='REML'``,
    ``knapp_hartung=True`` (t-reference with df = k−p for CIs/p, metafor ``test="t"``).
    The engine behind the ECR-style prevalence/severity meta.
    """
    eff = _effects(state)
    if eff is None:
        state.write("models", "meta", {"estimate": None, "note": "no meta_effects; run sv.pp.escalc first"})
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    V = state.models.get("meta_V")
    if not isinstance(V, np.ndarray) or V.shape != (len(y), len(y)):
        V = np.diag(v)
    study_col = kwargs.get("study") or ("study" if "study" in eff.columns else None)
    study = eff[study_col].to_numpy() if study_col else np.arange(len(y))
    X, Xcols, mod_names = _design(state, eff, kwargs)
    fit = _reml_3level(y, V, X, study, ml=(str(kwargs.get("method", "REML")).upper() == "ML"))
    beta, se = fit["beta"], fit["se"]
    tstat = beta / se
    # Knapp-Hartung: reference the coefficient tests to a t-distribution with
    # df = k - p (metafor rma.mv test="t"), instead of the normal — materially
    # affects CI/p when k is small. This is the multilevel analog of HKSJ.
    hk = bool(kwargs.get("knapp_hartung", kwargs.get("test", "") in ("t", "knha")))
    dfree = max(fit["k"] - X.shape[1], 1)
    crit = float(stats.t.ppf(0.975, dfree)) if hk else 1.959963985

    def _p(t):
        return float(2 * stats.t.sf(abs(t), dfree)) if hk else float(2 * stats.norm.sf(abs(t)))
    coefs = {name: {"estimate": float(beta[i]), "se": float(se[i]),
                    "zval": float(tstat[i]), "pval": _p(tstat[i]),
                    "ci_lb": float(beta[i] - crit * se[i]), "ci_ub": float(beta[i] + crit * se[i])}
             for i, name in enumerate(Xcols)}
    out = {
        "model": "multilevel(3-level)", "method": kwargs.get("method", "REML"),
        "coefs": coefs, "terms": Xcols, "moderators": mod_names, "p": X.shape[1],
        "sigma2_2": fit["sigma2_2"], "sigma2_3": fit["sigma2_3"],
        "sigma2_total": fit["sigma2_total"], "tau2": fit["sigma2_total"],
        "k": fit["k"], "n_studies": fit["n_studies"], "converged": fit["converged"],
        "knapp_hartung": hk, "df": dfree if hk else None,
    }
    # convenience top-level for the intercept (the pooled estimate)
    ic = coefs.get("(intercept)", {})
    out.update({"estimate": ic.get("estimate"), "se": ic.get("se"),
                "ci_lb": ic.get("ci_lb"), "ci_ub": ic.get("ci_ub"),
                "zval": ic.get("zval"), "pval": ic.get("pval")})
    state.write("models", "meta", out)
    return state


# ==================================================================== heterogeneity
@register(
    name="meta_heterogeneity",
    aliases=["异质性", "Q检验", "I2", "heterogeneity"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="异质性报告:Cochran's Q(χ²检验)、I²、H²、τ、τ²;估计量一致的 I²(典型方差路径)",
    requires={"models": ["meta"]},
    produces={"diagnostics": ["heterogeneity"]},
)
def meta_heterogeneity(state: StudyState, **kwargs: Any) -> StudyState:
    """Cochran's Q, I², H², τ, τ² from the fitted model + effects."""
    eff = _effects(state); m = state.models.get("meta")
    if eff is None or not isinstance(m, dict):
        return state
    from scipy import stats
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    w = 1.0 / v
    mu_fe = _weighted_mean(y, w)
    Q = float(np.sum(w * (y - mu_fe) ** 2))
    p = int(m.get("p", len(m.get("terms", ["(intercept)"]))))
    k = len(y); df = k - p
    tau2 = float(m.get("tau2", 0.0))
    s2_typ = _typical_v(v)
    I2 = 100.0 * tau2 / (tau2 + s2_typ) if (tau2 + s2_typ) > 0 else 0.0
    H2 = (tau2 + s2_typ) / s2_typ if s2_typ > 0 else float("nan")
    het = {
        "Q": Q, "df": df, "Q_pval": float(stats.chi2.sf(Q, df)) if df > 0 else float("nan"),
        "I2": I2, "H2": H2, "tau2": tau2, "tau": float(np.sqrt(tau2)),
        "substantial": I2 > 50.0,
        "note": "I² is the estimator-consistent (typical-variance) form",
    }
    state.write("diagnostics", "heterogeneity", het)
    return state


@register(
    name="ma_i2_multilevel",
    aliases=["多层I2", "分层异质性", "i2_levels"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="三层 meta 的分层 I²(Cheung 2014):level-2(结局内)/level-3(研究间)方差占比 + 抽样误差占比",
    requires={"models": ["meta"]},
    produces={"diagnostics": ["i2_multilevel"]},
)
def ma_i2_multilevel(state: StudyState, **kwargs: Any) -> StudyState:
    """Cheung (2014) variance decomposition: I²_level2 / I²_level3 / sampling share."""
    eff = _effects(state); m = state.models.get("meta")
    if eff is None or not isinstance(m, dict) or "sigma2_2" not in m:
        return state
    v = eff["vi"].to_numpy(float)
    s2_typ = _typical_v(v)
    s2, s3 = float(m["sigma2_2"]), float(m["sigma2_3"])
    total = s2 + s3 + s2_typ
    state.write("diagnostics", "i2_multilevel", {
        "sampling_share": 100.0 * s2_typ / total if total else float("nan"),
        "I2_level2_within_study": 100.0 * s2 / total if total else 0.0,
        "I2_level3_between_study": 100.0 * s3 / total if total else 0.0,
        "sigma2_2": s2, "sigma2_3": s3, "typical_sampling_var": s2_typ,
    })
    return state


@register(
    name="meta_prediction_interval",
    aliases=["预测区间", "prediction_interval", "PI"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="Higgins-Thompson-Spiegelhalter 预测区间 θ ± t·√(τ²+SE²)(未来研究真值落入范围;k 小时偏窄,会告警)",
    requires={"models": ["meta"]},
    produces={"diagnostics": ["prediction_interval"]},
)
def meta_prediction_interval(state: StudyState, **kwargs: Any) -> StudyState:
    """95% prediction interval (HTS). Under-covers at small k — flags it."""
    m = state.models.get("meta")
    if not isinstance(m, dict) or m.get("estimate") is None:
        return state
    from scipy import stats
    mu, se, tau2, k = m["estimate"], m["se"], float(m.get("tau2", 0.0)), int(m.get("k", 0))
    p = int(m.get("p", 1))
    df = max(k - p - 1, 1)
    level = float(kwargs.get("level", 0.95))
    crit = stats.t.ppf(1 - (1 - level) / 2, df)
    spread = crit * np.sqrt(tau2 + se ** 2)
    state.write("diagnostics", "prediction_interval", {
        "pi_lb": mu - spread, "pi_ub": mu + spread, "level": level, "df": df,
        "warning": "PI under-covers when k is small (<~10)" if k < 10 else "",
    })
    return state


# ==================================================================== meta-regression
@register(
    name="metareg",
    aliases=["元回归", "meta_regression", "moderator_analysis"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="混合效应元回归:用调节变量解释效应量异质性(τ² 加权 REML);产出各系数 + 残余 τ²",
    requires={"models": ["meta_effects"]},
    produces={"models": ["metareg"]},
)
def metareg(state: StudyState, **kwargs: Any) -> StudyState:
    """Mixed-effects meta-regression on moderators.

    kwargs: ``moderators=[...]`` (numeric or categorical columns on the effects
    frame). Fits β + residual τ² by REML; reports per-coefficient tests. For a
    multilevel meta-regression pass ``moderators`` to ``sv.tl.rma_mv`` instead.
    """
    eff = _effects(state)
    if eff is None:
        state.write("models", "metareg", {"note": "no meta_effects"})
        return state
    from scipy import stats
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    X, Xcols, mod_names = _design(state, eff, kwargs)
    if len(mod_names) == 0:
        state.write("models", "metareg", {"note": "no moderators given"})
        return state
    fit = _reml_2level(y, v, X)
    beta, se = fit["beta"], fit["se"]
    tstat = beta / se
    coefs = {name: {"estimate": float(beta[i]), "se": float(se[i]),
                    "zval": float(tstat[i]), "pval": float(2 * stats.norm.sf(abs(tstat[i])))}
             for i, name in enumerate(Xcols)}
    # pseudo-R²: reduction in τ² vs intercept-only
    tau2_full = fit["tau2"]
    tau2_null = _reml_2level(y, v, np.ones((len(y), 1)))["tau2"]
    r2 = max(0.0, 100.0 * (tau2_null - tau2_full) / tau2_null) if tau2_null > 0 else 0.0
    state.write("models", "metareg", {
        "coefs": coefs, "terms": Xcols, "moderators": mod_names,
        "tau2_residual": tau2_full, "tau2_null": tau2_null, "R2": r2, "k": fit["k"],
    })
    return state


@register(
    name="metareg_fdr",
    aliases=["元回归FDR", "moderator_fdr", "QM检验"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="元回归综合检验 QM + 各调节变量检验,并对多重调节变量做 Benjamini-Hochberg FDR 校正",
    requires={"models": ["metareg"]},
    produces={"diagnostics": ["metareg_fdr"]},
)
def metareg_fdr(state: StudyState, **kwargs: Any) -> StudyState:
    """Omnibus QM test + Benjamini-Hochberg FDR across moderator coefficients."""
    mr = state.models.get("metareg")
    if not isinstance(mr, dict) or "coefs" not in mr:
        return state
    from scipy import stats
    mods = mr.get("moderators", [])
    coefs = mr["coefs"]
    # omnibus QM = sum of (z²) over moderator terms (χ² with m df, Wald)
    zs = [coefs[t]["zval"] for t in coefs if t != "(intercept)"]
    QM = float(np.sum(np.square(zs))); m = len(zs)
    QM_p = float(stats.chi2.sf(QM, m)) if m > 0 else float("nan")
    # per-moderator raw p, then BH FDR
    raw = {t: coefs[t]["pval"] for t in coefs if t != "(intercept)"}
    names = list(raw); pvals = np.array([raw[t] for t in names], float)
    order = np.argsort(pvals); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pvals) + 1)
    adj = pvals * len(pvals) / ranks
    # enforce monotonicity
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    adj_final = np.empty_like(adj); adj_final[order] = np.clip(adj_sorted, 0, 1)
    fdr = {names[i]: {"pval": float(pvals[i]), "pval_fdr": float(adj_final[i]),
                      "significant_fdr": bool(adj_final[i] < float(kwargs.get("alpha", 0.05)))}
           for i in range(len(names))}
    state.write("diagnostics", "metareg_fdr", {
        "QM": QM, "QM_df": m, "QM_pval": QM_p, "per_moderator": fdr,
        "alpha": float(kwargs.get("alpha", 0.05)),
    })
    return state


# ==================================================================== egger
@register(
    name="egger_test",
    aliases=["Egger检验", "小研究效应", "egger", "asymmetry_test"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="Egger 回归检验小研究效应/漏斗图不对称(标准差正态离差对精度回归,检验截距≠0)",
    requires={"models": ["meta_effects"]},
    produces={"diagnostics": ["egger"]},
)
def egger_test(state: StudyState, **kwargs: Any) -> StudyState:
    """Egger's regression test for small-study effects / funnel asymmetry.

    Classic form: regress the standard normal deviate (yᵢ/seᵢ) on precision
    (1/seᵢ) by OLS; the intercept ≠ 0 indicates asymmetry. (For clustered data a
    3-level Egger = rma_mv with sei as a moderator.)
    """
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    sei = np.sqrt(v)
    snd = y / sei; prec = 1.0 / sei
    Xd = np.column_stack([np.ones_like(prec), prec])
    beta, *_ = np.linalg.lstsq(Xd, snd, rcond=None)
    resid = snd - Xd @ beta
    n = len(y); dof = n - 2
    mse = float(resid @ resid) / dof
    cov = mse * np.linalg.pinv(Xd.T @ Xd)
    b0, se0 = float(beta[0]), float(np.sqrt(cov[0, 0]))
    t = b0 / se0
    state.write("diagnostics", "egger", {
        "intercept": b0, "se": se0, "tval": t, "df": dof,
        "pval": float(2 * stats.t.sf(abs(t), dof)),
        "slope": float(beta[1]),
        "asymmetry": float(2 * stats.t.sf(abs(t), dof)) < 0.10,
        "note": "classic Egger OLS; clustered data → 3-level Egger via rma_mv(moderators=['sei'])",
    })
    return state
