"""``sv.tl._meta_adv`` — advanced meta-analytic diagnostics & structures (Tier-3).

Random-forest meta-regression (metaforest), likelihood-ratio test, profile-
likelihood CI for τ², cluster wild bootstrap, leave-one-cluster-out influence,
all-subsets multimodel inference (AICc + Akaike weights), and within-study-ρ
sensitivity.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _design, _estimate_tau2


def _ml_fit(y, v, X):
    """ML fit of a meta-regression; returns beta, tau2, loglik, vcov."""
    from scipy import optimize
    def negll(logt2):
        t2 = np.exp(logt2); w = 1.0 / (v + t2)
        A = np.linalg.pinv(X.T @ (w[:, None] * X)); beta = A @ (X.T @ (w * y))
        r = y - X @ beta
        return 0.5 * (np.sum(np.log(v + t2)) + np.sum(w * r ** 2))
    res = optimize.minimize_scalar(negll, bounds=(np.log(1e-8), np.log(np.var(y) * 100 + 1e-6)),
                                   method="bounded")
    t2 = float(np.exp(res.x)); w = 1.0 / (v + t2)
    A = np.linalg.pinv(X.T @ (w[:, None] * X)); beta = A @ (X.T @ (w * y))
    ll = -negll(res.x) - 0.5 * len(y) * np.log(2 * np.pi)
    return beta, t2, float(ll), A


@register(
    name="metaforest", aliases=["随机森林元回归", "meta_forest_rf"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["scikit-learn", "numpy"],
    description="随机森林元回归(metaforest):meta 权重加权 RF 探索非线性/交互调节效应 + 变量重要性(方向性,非逐比特同 ranger)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["metaforest"]},
)
def metaforest(state: StudyState, **kwargs: Any) -> StudyState:
    """Weighted random-forest meta-regression → moderator importance (exploratory)."""
    eff = _effects(state)
    if eff is None:
        return state
    mods = kwargs.get("moderators") or kwargs.get("mods") or []
    if isinstance(mods, str):
        mods = [mods]
    mods = [m for m in mods if m in eff.columns]
    if not mods:
        state.write("diagnostics", "metaforest", {"note": "no moderators given"})
        return state
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        state.write("diagnostics", "metaforest", {"note": "scikit-learn unavailable"})
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    Xm = pd.get_dummies(eff[mods], drop_first=False).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tau2 = _estimate_tau2(y, v, "REML"); w = 1.0 / (v + tau2)
    rf = RandomForestRegressor(n_estimators=int(kwargs.get("n_estimators", 500)),
                               min_samples_leaf=int(kwargs.get("min_samples_leaf", 3)),
                               random_state=int(kwargs.get("seed", 42)), oob_score=True)
    rf.fit(Xm.to_numpy(float), y, sample_weight=w)
    imp = dict(sorted(zip(Xm.columns.tolist(), rf.feature_importances_.tolist()),
                      key=lambda kv: -kv[1]))
    state.write("diagnostics", "metaforest", {
        "importance": {k: float(x) for k, x in imp.items()},
        "oob_r2": float(getattr(rf, "oob_score_", float("nan"))),
        "note": "sklearn RF (directional, not bit-identical to ranger); exploratory moderator screening",
    })
    return state


@register(
    name="ma_lrt", aliases=["似然比检验", "likelihood_ratio"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="嵌套 meta 模型的似然比检验(ML):比较含/不含某组调节变量,或 τ²=0 边界检验(半 χ²)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["ma_lrt"]},
)
def ma_lrt(state: StudyState, **kwargs: Any) -> StudyState:
    """LRT between nested models (ML). kwargs: ``moderators=[...]`` for the full model
    (reduced = intercept-only), or ``test='tau2'`` for the τ²=0 boundary test."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    if str(kwargs.get("test", "")) == "tau2":
        _, _, ll_full, _ = _ml_fit(y, v, np.ones((len(y), 1)))
        w = 1 / v; mu = np.sum(w * y) / np.sum(w)
        ll_null = float(-0.5 * (np.sum(np.log(v)) + np.sum(w * (y - mu) ** 2) + len(y) * np.log(2 * np.pi)))
        lr = 2 * (ll_full - ll_null)
        p = 0.5 * stats.chi2.sf(lr, 1)   # boundary: half chi2_1
        state.write("diagnostics", "ma_lrt", {"test": "tau2=0", "LR": float(lr), "pval": float(p),
                                              "note": "boundary test — half χ²₁ mixture"})
        return state
    X, Xcols, mods = _design(state, eff, kwargs)
    _, _, ll_full, _ = _ml_fit(y, v, X)
    _, _, ll_red, _ = _ml_fit(y, v, np.ones((len(y), 1)))
    lr = 2 * (ll_full - ll_red); dfree = len(mods)
    state.write("diagnostics", "ma_lrt", {
        "moderators": mods, "LR": float(lr), "df": dfree,
        "pval": float(stats.chi2.sf(lr, dfree)) if dfree > 0 else float("nan")})
    return state


@register(
    name="ma_profile", aliases=["剖面似然", "profile_likelihood_ci"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="τ² 的剖面似然置信区间(似然下降 χ²/2 处的界),比 Wald 区间在小 k 更可靠",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["ma_profile"]},
)
def ma_profile(state: StudyState, **kwargs: Any) -> StudyState:
    """Profile-likelihood CI for τ²."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    def prof_ll(t2):
        w = 1 / (v + t2); mu = np.sum(w * y) / np.sum(w)
        return float(-0.5 * (np.sum(np.log(v + t2)) + np.sum(w * (y - mu) ** 2)))
    t2_hat = _estimate_tau2(y, v, "ML")
    ll_max = prof_ll(t2_hat)
    cut = ll_max - stats.chi2.ppf(0.95, 1) / 2
    grid = np.linspace(0, max(t2_hat * 8 + np.var(y), 1e-3), 2000)
    lls = np.array([prof_ll(t) for t in grid])
    inside = grid[lls >= cut]
    lb = float(inside.min()) if len(inside) else 0.0
    ub = float(inside.max()) if len(inside) else float(t2_hat)
    state.write("diagnostics", "ma_profile", {"tau2": float(t2_hat), "tau2_ci": [lb, ub], "method": "profile-ML"})
    return state


@register(
    name="ma_cwb_test", aliases=["簇wild自助", "cluster_wild_bootstrap"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="元回归的簇 wild 自助检验(小簇数更可靠):Rademacher 权重按簇施于限制残差,自助 t 分布",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["ma_cwb_test"]},
)
def ma_cwb_test(state: StudyState, **kwargs: Any) -> StudyState:
    """Cluster wild bootstrap test for a moderator (small number of clusters)."""
    eff = _effects(state)
    if eff is None:
        return state
    target = kwargs.get("target") or (kwargs.get("moderators") or [None])[0]
    clus_col = kwargs.get("cluster") or ("study" if "study" in eff.columns else None)
    if target is None or target not in eff.columns or clus_col is None:
        state.write("diagnostics", "ma_cwb_test", {"note": "need target moderator + cluster column"})
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    X, Xcols, mods = _design(state, eff, {"moderators": [target]})
    ti = Xcols.index(target)
    tau2 = _estimate_tau2(y, v, "REML"); w = 1 / (v + tau2)

    def fit(yy):
        A = np.linalg.pinv(X.T @ (w[:, None] * X)); b = A @ (X.T @ (w * yy))
        se = np.sqrt(A[ti, ti]); return b, b[ti] / se
    b_obs, t_obs = fit(y)
    # restricted (null: target coef = 0) fit + residuals
    Xr = np.delete(X, ti, axis=1)
    Ar = np.linalg.pinv(Xr.T @ (w[:, None] * Xr)); br = Ar @ (Xr.T @ (w * y))
    resid = y - Xr @ br
    clusters = eff[clus_col].to_numpy()
    uniq = pd.unique(clusters)
    rng = np.random.default_rng(int(kwargs.get("seed", 42)))
    B = int(kwargs.get("nboot", 999)); count = 0
    for _ in range(B):
        signs = {g: rng.choice([-1.0, 1.0]) for g in uniq}
        s = np.array([signs[c] for c in clusters])
        ystar = Xr @ br + s * resid
        _, tstar = fit(ystar)
        if abs(tstar) >= abs(t_obs):
            count += 1
    state.write("diagnostics", "ma_cwb_test", {
        "target": target, "coef": float(b_obs[ti]), "t_obs": float(t_obs),
        "nboot": B, "pval_cwb": (count + 1) / (B + 1), "n_clusters": int(len(uniq))})
    return state


@register(
    name="ma_cluster_influence", aliases=["簇影响", "leave_one_cluster_out"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="留一簇(研究)影响诊断:剔除每个簇后重估合并效应/τ²,定位驱动结论的簇",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["ma_cluster_influence"]},
)
def ma_cluster_influence(state: StudyState, **kwargs: Any) -> StudyState:
    """Leave-one-cluster-out influence."""
    eff = _effects(state)
    clus_col = kwargs.get("cluster") or ("study" if "study" in (eff.columns if eff is not None else []) else None)
    if eff is None or clus_col is None:
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); clus = eff[clus_col].to_numpy()
    def pool(mask):
        t2 = _estimate_tau2(y[mask], v[mask], "REML") if mask.sum() > 1 else 0.0
        w = 1 / (v[mask] + t2); return float(np.sum(w * y[mask]) / np.sum(w)), t2
    mu0, t20 = pool(np.ones(len(y), bool))
    rows = []
    for g in pd.unique(clus):
        m = clus != g
        if m.sum() < 2:
            continue
        mu, t2 = pool(m)
        rows.append({"omitted_cluster": str(g), "estimate": mu, "delta": mu - mu0, "tau2": t2})
    state.write("diagnostics", "ma_cluster_influence", {"full_estimate": mu0, "full_tau2": t20, "rows": rows})
    return state


@register(
    name="metareg_multimodel", aliases=["多模型平均", "multimodel_inference", "aicc_averaging"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="全子集多模型推断:所有调节变量组合的 AICc + Akaike 权重 + 模型平均系数与变量重要性(须用 ML)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["metareg_multimodel"]},
)
def metareg_multimodel(state: StudyState, **kwargs: Any) -> StudyState:
    """All-subsets AICc + Akaike weights + model-averaged coefficients (ML)."""
    eff = _effects(state)
    mods = kwargs.get("moderators") or kwargs.get("mods") or []
    if isinstance(mods, str):
        mods = [mods]
    mods = [m for m in mods if eff is not None and m in eff.columns]
    if eff is None or not mods:
        state.write("diagnostics", "metareg_multimodel", {"note": "need ≥1 moderator"})
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); k = len(y)
    results = []
    for r in range(0, len(mods) + 1):
        for sub in combinations(mods, r):
            X, Xcols, _ = _design(state, eff, {"moderators": list(sub)})
            beta, t2, ll, A = _ml_fit(y, v, X)
            npar = X.shape[1] + 1
            aicc = -2 * ll + 2 * npar + (2 * npar * (npar + 1)) / max(k - npar - 1, 1)
            results.append({"model": list(sub), "aicc": aicc, "coefs": dict(zip(Xcols, beta.tolist()))})
    amin = min(r["aicc"] for r in results)
    for r in results:
        r["delta"] = r["aicc"] - amin; r["weight"] = np.exp(-0.5 * r["delta"])
    Z = sum(r["weight"] for r in results)
    for r in results:
        r["weight"] /= Z
    # model-averaged coefficients + importance
    avg = {m: 0.0 for m in mods}; imp = {m: 0.0 for m in mods}
    for r in results:
        for m in mods:
            if m in r["coefs"]:
                avg[m] += r["weight"] * r["coefs"][m]; imp[m] += r["weight"]
    top = sorted(results, key=lambda r: r["aicc"])[:5]
    state.write("diagnostics", "metareg_multimodel", {
        "n_models": len(results),
        "top_models": [{"model": r["model"], "aicc": round(r["aicc"], 2), "weight": round(r["weight"], 3)} for r in top],
        "averaged_coefs": {m: float(avg[m]) for m in mods},
        "importance": dict(sorted(({m: float(imp[m]) for m in mods}).items(), key=lambda kv: -kv[1])),
    })
    return state


@register(
    name="ma_rho_sensitivity", aliases=["rho敏感性", "rho_sensitivity"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="簇内相关 ρ 敏感性:在 ρ 网格上重建 V + 重估相依效应量的合并/τ²,看结论对 ρ 假设的稳健性",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["ma_rho_sensitivity"]},
)
def ma_rho_sensitivity(state: StudyState, **kwargs: Any) -> StudyState:
    """Sensitivity of the multilevel pooled estimate to the assumed within-study ρ."""
    from .._registry import registry
    eff = _effects(state)
    clus = kwargs.get("cluster") or ("study" if "study" in (eff.columns if eff is not None else []) else None)
    if eff is None or clus is None:
        state.write("diagnostics", "ma_rho_sensitivity", {"note": "need cluster column"})
        return state
    grid = kwargs.get("rho_grid") or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for rho in grid:
        s2 = StudyState(); s2.write("models", "meta_effects", eff.copy())
        registry.get("vcalc").func(s2, cluster=clus, rho=float(rho))
        registry.get("rma_mv").func(s2, study=clus)
        m = s2.models.get("meta") or {}
        rows.append({"rho": float(rho), "estimate": m.get("estimate"),
                     "se": m.get("se"), "tau2": m.get("sigma2_total")})
    ests = [r["estimate"] for r in rows if r["estimate"] is not None]
    state.write("diagnostics", "ma_rho_sensitivity", {
        "rows": rows, "estimate_range": [min(ests), max(ests)] if ests else None,
        "robust": (max(ests) - min(ests) < 0.1) if ests else None})
    return state
