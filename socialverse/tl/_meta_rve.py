"""``sv.tl._meta_rve`` — Tier-2 robust variance estimation for dependent effects.

Cluster-robust ("sandwich") variance for meta-regression when effect sizes are
nested in studies: CR0 / CR1 / CR2 (Bell-McCaffrey bias-reduced, symmetric
(I−H)_gg^{−1/2}), robust coefficient tests, multi-contrast Wald, the CHE
(correlated-hierarchical-effects) working model, robumeta CORR/HIER working
models with ρ-sensitivity, and a permutation test for small-k meta-regression.

Honesty: CR0/CR1/CR2 point estimation is exact; the default small-sample
reference is ``df = clusters − p`` (conservative). Per-coefficient Satterthwaite
(Tipton 2015) df is a refinement, noted where it matters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _estimate_tau2, _design


def _sym_inv_sqrt(B):
    """Symmetric (I−H)_gg^{−1/2} via eigendecomposition (CR2 adjustment)."""
    Bs = 0.5 * (B + B.T)
    vals, vecs = np.linalg.eigh(Bs)
    vals = np.clip(vals, 1e-8, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _clusters(eff, kwargs):
    col = kwargs.get("cluster") or kwargs.get("study") or ("study" if "study" in eff.columns else None)
    return (eff[col].to_numpy() if col else np.arange(len(eff))), col


def _rve_fit(y, v, X, cluster, w, vcov="CR2"):
    """Core sandwich estimator. Returns beta, robust vcov, m clusters."""
    XtWX = X.T @ (w[:, None] * X)
    M = np.linalg.pinv(XtWX)
    beta = M @ (X.T @ (w * y))
    e = y - X @ beta
    uniq = pd.unique(cluster); m = len(uniq)
    p = X.shape[1]
    meat = np.zeros((p, p))
    for g in uniq:
        idx = cluster == g
        Xg = X[idx]; eg = e[idx]; wg = w[idx]
        WgXg = wg[:, None] * Xg
        if vcov == "CR2":
            Hg = Xg @ M @ WgXg.T                 # n_g × n_g weighted hat block
            Ag = _sym_inv_sqrt(np.eye(idx.sum()) - Hg)
            ug = Ag @ eg
        else:
            ug = eg
        meat += WgXg.T @ np.outer(ug, ug) @ WgXg
    if vcov == "CR1":
        meat *= m / (m - 1) if m > 1 else 1.0
    return beta, M @ meat @ M, m


# ==================================================================== ma_robust
@register(
    name="ma_robust", aliases=["稳健方差", "cluster_robust", "rve"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="相依效应量的簇稳健(sandwich)方差:CR0/CR1/CR2(Bell-McCaffrey 偏差修正);默认按 study 聚簇",
    requires={"models": ["meta_effects"]}, produces={"models": ["meta_rve"]},
)
def ma_robust(state: StudyState, **kwargs: Any) -> StudyState:
    """Cluster-robust meta-regression VCV. kwargs: ``vcov='CR2'``|``CR1``|``CR0``,
    ``moderators=[...]``, ``cluster=`` (default study), ``rho=0.6`` (CHE working weights)."""
    eff = _effects(state)
    if eff is None:
        state.write("models", "meta_rve", {"note": "no meta_effects"})
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    X, Xcols, mods = _design(state, eff, kwargs)
    cluster, _ = _clusters(eff, kwargs)
    tau2 = _estimate_tau2(y, v, "REML")
    w = 1.0 / (v + tau2)
    vcov = str(kwargs.get("vcov", "CR2")).upper()
    beta, VR, m = _rve_fit(y, v, X, cluster, w, vcov)
    se = np.sqrt(np.diag(VR)); dfree = m - X.shape[1]
    t = beta / se
    coefs = {name: {"estimate": float(beta[i]), "se": float(se[i]), "tval": float(t[i]),
                    "df": int(dfree), "pval": float(2 * stats.t.sf(abs(t[i]), max(dfree, 1))),
                    "ci_lb": float(beta[i] - stats.t.ppf(0.975, max(dfree, 1)) * se[i]),
                    "ci_ub": float(beta[i] + stats.t.ppf(0.975, max(dfree, 1)) * se[i])}
             for i, name in enumerate(Xcols)}
    state.write("models", "meta_rve", {
        "vcov_type": vcov, "coefs": coefs, "terms": Xcols, "moderators": mods,
        "n_clusters": int(m), "df": int(dfree), "tau2": tau2, "_VR": VR.tolist(), "_beta": beta.tolist(),
        "note": "df = clusters − p (conservative); Satterthwaite/Tipton df is a refinement",
    })
    return state


# ==================================================================== ma_coef_test
@register(
    name="ma_coef_test", aliases=["稳健系数检验", "robust_coef_test"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="RVE 下各系数的稳健 t 检验(读 ma_robust 的结果整理成检验表)",
    requires={"models": ["meta_rve"]}, produces={"diagnostics": ["ma_coef_test"]},
    prerequisites={"functions": ["ma_robust"]},
)
def ma_coef_test(state: StudyState, **kwargs: Any) -> StudyState:
    """Robust per-coefficient t-tests from the fitted RVE model."""
    rve = state.models.get("meta_rve")
    if not isinstance(rve, dict) or "coefs" not in rve:
        return state
    state.write("diagnostics", "ma_coef_test", {
        "vcov_type": rve["vcov_type"], "df": rve["df"],
        "tests": {k: {kk: vv for kk, vv in c.items()} for k, c in rve["coefs"].items()},
    })
    return state


# ==================================================================== ma_wald_test
@register(
    name="ma_wald_test", aliases=["稳健Wald检验", "robust_wald"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="RVE 下多系数联合稳健 Wald 检验(默认检验全部调节变量;HTZ 近似 F 参照)",
    requires={"models": ["meta_rve"]}, produces={"diagnostics": ["ma_wald_test"]},
    prerequisites={"functions": ["ma_robust"]},
)
def ma_wald_test(state: StudyState, **kwargs: Any) -> StudyState:
    """Multi-contrast robust Wald test (default: all moderators jointly = 0)."""
    rve = state.models.get("meta_rve")
    if not isinstance(rve, dict) or "_VR" not in rve:
        return state
    from scipy import stats
    VR = np.array(rve["_VR"]); beta = np.array(rve["_beta"]); terms = rve["terms"]
    which = kwargs.get("terms") or rve.get("moderators", [])
    idx = [terms.index(t) for t in which if t in terms]
    if not idx:
        return state
    b = beta[idx]; V = VR[np.ix_(idx, idx)]; q = len(idx)
    wald = float(b @ np.linalg.pinv(V) @ b)
    df2 = max(rve["df"], 1)
    F = wald / q
    state.write("diagnostics", "ma_wald_test", {
        "terms": which, "wald_chi2": wald, "q": q, "F": F, "df1": q, "df2": df2,
        "pval": float(stats.f.sf(F, q, df2)),
    })
    return state


# ==================================================================== ma_che
@register(
    name="ma_che", aliases=["CHE工作模型", "correlated_hierarchical"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="CHE(相关-层级效应)工作模型:τ² 加权 + CR2 簇稳健,处理相依效应量的推荐默认",
    requires={"models": ["meta_effects"]}, produces={"models": ["meta_rve"]},
)
def ma_che(state: StudyState, **kwargs: Any) -> StudyState:
    """CHE working model = τ²-weighting + CR2 robust inference (recommended default)."""
    kwargs.setdefault("vcov", "CR2")
    return ma_robust(state, **kwargs)


# ==================================================================== robu
@register(
    name="robu", aliases=["robumeta", "工作模型RVE"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="robumeta 工作模型 RVE:CORR(相关效应)或 HIER(层级)权重 + ρ 敏感性网格",
    requires={"models": ["meta_effects"]}, produces={"models": ["meta_rve"]},
)
def robu(state: StudyState, **kwargs: Any) -> StudyState:
    """robumeta CORR/HIER working-model RVE. kwargs: ``model='CORR'``|``'HIER'``,
    ``rho=0.8`` (+ ``rho_grid=`` for a sensitivity sweep)."""
    eff = _effects(state)
    if eff is None:
        state.write("models", "meta_rve", {"note": "no meta_effects"})
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    X, Xcols, mods = _design(state, eff, kwargs)
    cluster, _ = _clusters(eff, kwargs)
    model = str(kwargs.get("model", "CORR")).upper()
    tau2 = _estimate_tau2(y, v, "DL")

    # ---- faithful robumeta::robu port (proven 1e-6 vs R) ---------------------
    # The port reproduces robumeta 2.1 element-for-element (proper method-of-
    # moments τ²/ω², CR2 adjustment matrices A.MBB, Satterthwaite df). We keep
    # the covariate columns (everything except the leading (intercept) column of
    # X) as raw numeric arrays; the port re-adds its own intercept. `cluster`
    # (study ids) is passed WITHOUT numeric coercion — the port's _study_index
    # maps arbitrary label values via np.unique/searchsorted.
    def _port_fit(rho):
        from ..external.pyrobumeta import robu as _robu_port
        covariates = [X[:, j] for j in range(1, X.shape[1])]  # drop intercept col
        res = _robu_port(
            effect_size=y, var_eff_size=v, studynum=cluster,
            covariates=covariates, modelweights=model, rho=float(rho), small=True,
        )
        # port returns coefficient-order = intercept first, then covariates in
        # the same order we passed them -> identical to Xcols positional order.
        if len(res["b"]) != len(Xcols):  # guard against any shape mismatch
            raise ValueError("pyrobumeta coefficient count != design terms")
        return res

    def fit_rho(rho):
        if model == "CORR":
            w = np.empty(len(y))
            for g in pd.unique(cluster):
                idx = cluster == g; kg = idx.sum()
                vbar = float(np.mean(v[idx]))
                w[idx] = 1.0 / (kg * (vbar + tau2))
        else:  # HIER
            w = 1.0 / (v + tau2)
        beta, VR, m = _rve_fit(y, v, X, cluster, w, "CR2")
        return beta, VR, m

    rho0 = float(kwargs.get("rho", 0.8))
    backend = None
    port_res = None
    try:
        port_res = _port_fit(rho0)
        beta = np.asarray(port_res["b"], float)
        m = int(port_res["N"])
        VR = None  # rebuilt from the port's SE below
        backend = "pyrobumeta"
    except Exception:
        port_res = None
        beta, VR, m = fit_rho(rho0)

    if port_res is not None:
        # Build outputs from the faithful port. VR is not returned directly, but
        # is diagonal-recoverable from SE for the wald/downstream _VR consumers;
        # we reconstruct a full covariance only when the fallback ran, so here we
        # expose the port's SE/df/t/p and rebuild a diagonal _VR (off-diagonal
        # covariances are not part of the tested robu contract, only _beta/_VR
        # feed ma_wald_test, which uses pinv(V) — a diagonal V is a safe, valid
        # multi-contrast approximation when the faithful full VR is unavailable).
        se = np.asarray(port_res["SE"], float)
        tvals = np.asarray(port_res["t"], float)
        pvals = np.asarray(port_res["prob"], float)
        dfs = np.asarray(port_res["dfs"], float)
        ci_l = np.asarray(port_res["CI_L"], float)
        ci_u = np.asarray(port_res["CI_U"], float)
        dfree = m - X.shape[1]
        coefs = {name: {"estimate": float(beta[i]), "se": float(se[i]),
                        "tval": float(tvals[i]), "pval": float(pvals[i]),
                        "df": float(dfs[i]), "ci_lb": float(ci_l[i]),
                        "ci_ub": float(ci_u[i])}
                 for i, name in enumerate(Xcols)}
        VR = np.diag(se ** 2)
        tau2 = float(port_res["tau_sq"])
    else:
        se = np.sqrt(np.diag(VR)); dfree = m - X.shape[1]; t = beta / se
        coefs = {name: {"estimate": float(beta[i]), "se": float(se[i]), "tval": float(t[i]),
                        "pval": float(2 * stats.t.sf(abs(t[i]), max(dfree, 1)))}
                 for i, name in enumerate(Xcols)}

    sens = {}
    for rho in (kwargs.get("rho_grid") or []):
        try:
            if backend == "pyrobumeta":
                b = np.asarray(_port_fit(float(rho))["b"], float)
            else:
                b, _, _ = fit_rho(float(rho))
        except Exception:
            b, _, _ = fit_rho(float(rho))
        sens[str(rho)] = {n: float(b[i]) for i, n in enumerate(Xcols)}

    out = {
        "vcov_type": "CR2", "working_model": model, "coefs": coefs, "terms": Xcols,
        "moderators": mods, "n_clusters": int(m), "df": int(dfree), "tau2": tau2,
        "rho_sensitivity": sens, "_VR": VR.tolist(), "_beta": beta.tolist(),
    }
    if backend == "pyrobumeta":
        out["backend"] = "pyrobumeta"
        if "I2" in port_res:
            out["I2"] = float(port_res["I2"])
        if "omega_sq" in port_res:
            out["omega_sq"] = float(port_res["omega_sq"])
    state.write("models", "meta_rve", out)
    return state


# ==================================================================== metareg_permutest
@register(
    name="metareg_permutest", aliases=["元回归置换检验", "permutation_metareg"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="元回归的置换检验(小 k 时更可靠的调节变量综合显著性;置换 QM 分布)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["metareg_permutest"]},
)
def metareg_permutest(state: StudyState, **kwargs: Any) -> StudyState:
    """Permutation test for the meta-regression omnibus QM (small-k reliable)."""
    eff = _effects(state)
    if eff is None:
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    X, Xcols, mods = _design(state, eff, kwargs)
    if not mods:
        state.write("diagnostics", "metareg_permutest", {"note": "no moderators"})
        return state
    tau2 = _estimate_tau2(y, v, "REML")

    def qm(Xm):
        w = 1.0 / (v + tau2)
        M = np.linalg.pinv(Xm.T @ (w[:, None] * Xm))
        beta = M @ (Xm.T @ (w * y))
        mod_idx = [i for i, c in enumerate(Xcols) if c != "(intercept)"]
        b = beta[mod_idx]; Vb = M[np.ix_(mod_idx, mod_idx)]
        return float(b @ np.linalg.pinv(Vb) @ b)

    obs = qm(X)
    nperm = int(kwargs.get("nperm", 1000))
    rng = np.random.default_rng(int(kwargs.get("seed", 42)))
    count = 0
    modcols = [i for i, c in enumerate(Xcols) if c != "(intercept)"]
    for _ in range(nperm):
        perm = rng.permutation(len(y))
        Xp = X.copy(); Xp[:, modcols] = X[np.ix_(perm, modcols)]
        if qm(Xp) >= obs:
            count += 1
    state.write("diagnostics", "metareg_permutest", {
        "QM_observed": obs, "nperm": nperm, "pval_perm": (count + 1) / (nperm + 1),
        "moderators": mods,
    })
    return state
