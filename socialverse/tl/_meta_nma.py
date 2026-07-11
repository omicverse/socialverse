"""``sv.tl._meta_nma`` — frequentist network meta-analysis (Tier-3).

Contrast-based GLS network meta-analysis — the same estimator as the graph-
theoretical netmeta package, **exactly reproducible, no R, no MCMC**. Pools
direct + indirect evidence across >2 treatments; multi-arm trials handled with
the exact within-study covariance. Plus P-score ranking, rankogram/SUCRA,
design-by-treatment inconsistency, back-calculation node-splitting, and
component (additive) NMA.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState


def _contrasts(state):
    c = state.models.get("nma_contrasts")
    return c if isinstance(c, pd.DataFrame) and len(c) and {"treat1", "treat2", "TE", "seTE"}.issubset(c.columns) else None


def _build(df, treatments, ref):
    """Design matrix X (contrasts × basic params) and covariance V."""
    basic = [t for t in treatments if t != ref]
    pos = {t: i for i, t in enumerate(basic)}
    n = len(df); p = len(basic)
    X = np.zeros((n, p)); y = df["TE"].to_numpy(float)
    V = np.diag(df["seTE"].to_numpy(float) ** 2)
    for i, (_, r) in enumerate(df.iterrows()):
        if r["treat1"] in pos:
            X[i, pos[r["treat1"]]] += 1
        if r["treat2"] in pos:
            X[i, pos[r["treat2"]]] -= 1
    # multi-arm within-study covariance = shared baseline-arm variance
    if "studlab" in df.columns and "vbase" in df.columns:
        stud = df["studlab"].to_numpy(); vb = df["vbase"].to_numpy(float)
        t2 = df["treat2"].to_numpy()
        for i in range(n):
            for j in range(i + 1, n):
                if stud[i] == stud[j] and t2[i] == t2[j]:   # share the baseline arm
                    V[i, j] = V[j, i] = vb[i]
    return X, y, V, basic


def _gls(X, y, V):
    Vi = np.linalg.pinv(V)
    A = np.linalg.pinv(X.T @ Vi @ X)
    beta = A @ (X.T @ Vi @ y)
    resid = y - X @ beta
    Q = float(resid @ Vi @ resid)
    return beta, A, Q


# ==================================================================== netmeta
@register(
    name="netmeta", aliases=["网络meta", "network_meta", "nma"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="频率学派图论网络 meta(对比法 GLS,多臂精确协方差):合并直接+间接证据,产出全配对联赛表 + Q;netmeta 的原生等价(无 R/MCMC)",
    requires={"models": ["nma_contrasts"]}, produces={"models": ["nma"]},
    prerequisites={"functions": ["nma_pairwise"]},
)
def netmeta(state: StudyState, **kwargs: Any) -> StudyState:
    """Frequentist network meta-analysis. kwargs: ``reference=`` (default first
    treatment), ``comb='random'``|``'fixed'``. Returns per-treatment effects vs the
    reference + the full pairwise league table + heterogeneity/inconsistency Q."""
    df = _contrasts(state)
    if df is None:
        state.write("models", "nma", {"note": "no nma_contrasts; run sv.pp.nma_pairwise"})
        return state
    from scipy import stats, optimize
    treatments = sorted(set(df["treat1"]) | set(df["treat2"]), key=str)
    ref = kwargs.get("reference") or treatments[0]
    comb = str(kwargs.get("comb", "random")).lower()
    X, y, V, basic = _build(df, treatments, ref)
    beta_fe, A_fe, Q = _gls(X, y, V)
    dof = len(y) - len(basic)
    tau2 = 0.0
    if comb.startswith("rand") and dof > 0:
        def genQ(t2):
            Vt = V + t2 * np.eye(len(y))
            _, _, q = _gls(X, y, Vt)
            return q - dof
        if genQ(0.0) > 0:
            hi = 1.0
            while genQ(hi) > 0 and hi < 1e6:
                hi *= 2
            tau2 = float(optimize.brentq(genQ, 0.0, hi))
    Vt = V + tau2 * np.eye(len(y))
    beta, A, _ = _gls(X, y, Vt)
    # full β incl. reference (0) + vcov
    idx = {t: i for i, t in enumerate(basic)}
    T = len(treatments)
    full_beta = np.array([0.0 if t == ref else beta[idx[t]] for t in treatments])
    full_cov = np.zeros((T, T))
    for i, ti in enumerate(treatments):
        for j, tj in enumerate(treatments):
            if ti == ref or tj == ref:
                continue
            full_cov[i, j] = A[idx[ti], idx[tj]]
    effects = {t: {"vs_ref": float(full_beta[i]), "se": float(np.sqrt(max(full_cov[i, i], 0))),
                   "reference": ref} for i, t in enumerate(treatments)}
    league = {}
    for i, ti in enumerate(treatments):
        for j, tj in enumerate(treatments):
            if i == j:
                continue
            d = full_beta[i] - full_beta[j]
            se = np.sqrt(max(full_cov[i, i] + full_cov[j, j] - 2 * full_cov[i, j], 0))
            league[f"{ti} vs {tj}"] = {"estimate": float(d), "se": float(se),
                                       "ci_lb": float(d - 1.96 * se), "ci_ub": float(d + 1.96 * se),
                                       "pval": float(2 * stats.norm.sf(abs(d / se))) if se > 0 else float("nan")}
    I2 = max(0.0, 100 * (Q - dof) / Q) if Q > 0 and dof > 0 else 0.0
    state.write("models", "nma", {
        "treatments": treatments, "reference": ref, "sm": df["measure"].iloc[0] if "measure" in df else "",
        "model": comb, "effects": effects, "league": league,
        "Q": Q, "df": dof, "Q_pval": float(stats.chi2.sf(Q, dof)) if dof > 0 else float("nan"),
        "tau2": tau2, "I2": I2, "n_studies": int(df["studlab"].nunique()) if "studlab" in df else len(df),
        "_beta": full_beta.tolist(), "_cov": full_cov.tolist(),
    })
    return state


# ==================================================================== netrank
@register(
    name="netrank", aliases=["网络排名", "pscore", "sucra_pscore"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="网络 meta 的 P-score(频率学派 SUCRA,闭式无重采样):各处理优于其它处理的平均概率",
    requires={"models": ["nma"]}, produces={"diagnostics": ["netrank"]},
    prerequisites={"functions": ["netmeta"]},
)
def netrank(state: StudyState, **kwargs: Any) -> StudyState:
    """P-score = frequentist SUCRA (closed form). kwargs: ``small_values='desirable'``|``'undesirable'``."""
    nma = state.models.get("nma")
    if not isinstance(nma, dict) or "_beta" not in nma:
        return state
    from scipy import stats
    beta = np.array(nma["_beta"]); cov = np.array(nma["_cov"]); tr = nma["treatments"]
    T = len(tr); desirable_low = str(kwargs.get("small_values", "desirable")).startswith("desir")
    scores = {}
    for i in range(T):
        probs = []
        for k in range(T):
            if k == i:
                continue
            d = beta[i] - beta[k]; se = np.sqrt(max(cov[i, i] + cov[k, k] - 2 * cov[i, k], 1e-12))
            # P(i better than k): lower better ⇒ P(θ_i < θ_k)
            probs.append(stats.norm.cdf(-d / se) if desirable_low else stats.norm.cdf(d / se))
        scores[tr[i]] = float(np.mean(probs))
    ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
    state.write("diagnostics", "netrank", {"pscore": ranked, "small_values": "desirable" if desirable_low else "undesirable"})
    return state


# ==================================================================== nma_rankogram
@register(
    name="nma_rankogram", aliases=["排序图", "rankogram", "sucra"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="排序概率图 + SUCRA:从网络估计的多元正态抽样得各处理占各名次的概率",
    requires={"models": ["nma"]}, produces={"diagnostics": ["rankogram"]},
    prerequisites={"functions": ["netmeta"]},
)
def nma_rankogram(state: StudyState, **kwargs: Any) -> StudyState:
    """Rankogram + SUCRA via MVN simulation from the network estimates."""
    nma = state.models.get("nma")
    if not isinstance(nma, dict) or "_beta" not in nma:
        return state
    beta = np.array(nma["_beta"]); cov = np.array(nma["_cov"]); tr = nma["treatments"]; T = len(tr)
    nsim = int(kwargs.get("nsim", 5000)); desirable_low = str(kwargs.get("small_values", "desirable")).startswith("desir")
    rng = np.random.default_rng(int(kwargs.get("seed", 42)))
    cov_pd = cov + 1e-10 * np.eye(T)
    draws = rng.multivariate_normal(beta, cov_pd, size=nsim)
    order = draws if desirable_low else -draws
    ranks = np.argsort(np.argsort(order, axis=1), axis=1) + 1   # rank 1 = best
    rankmat = {tr[i]: [float(np.mean(ranks[:, i] == r)) for r in range(1, T + 1)] for i in range(T)}
    sucra = {tr[i]: float(np.sum([np.mean(ranks[:, i] <= r) for r in range(1, T)]) / (T - 1)) for i in range(T)}
    state.write("diagnostics", "rankogram", {
        "rank_probabilities": rankmat, "SUCRA": dict(sorted(sucra.items(), key=lambda kv: -kv[1])), "nsim": nsim})
    return state


# ==================================================================== nma_inconsistency
@register(
    name="nma_inconsistency", aliases=["网络不一致性", "design_by_treatment"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="设计×处理全局不一致性 Q 分解:总 Q = 设计内异质性 + 设计间不一致性(直接/间接冲突)",
    requires={"models": ["nma_contrasts", "nma"]}, produces={"diagnostics": ["nma_inconsistency"]},
    prerequisites={"functions": ["netmeta"]},
)
def nma_inconsistency(state: StudyState, **kwargs: Any) -> StudyState:
    """Design-by-treatment global Q decomposition (heterogeneity vs inconsistency)."""
    df = _contrasts(state); nma = state.models.get("nma")
    if df is None or not isinstance(nma, dict):
        return state
    from scipy import stats
    Q_total = nma["Q"]; df_total = nma["df"]
    # within-design heterogeneity: pool per design (unique treatment set per study)
    Q_het = 0.0; df_het = 0
    if "studlab" in df.columns:
        df = df.copy()
        # design = frozenset of treatments compared in the study
        dmap = {s: frozenset(set(g["treat1"]) | set(g["treat2"]))
                for s, g in df.groupby("studlab")}
        df["_design"] = df["studlab"].map(lambda s: dmap[s])
        for dsg, g in df.groupby(df["_design"].astype(str)):
            if len(g) < 2:
                continue
            y = g["TE"].to_numpy(float); w = 1 / g["seTE"].to_numpy(float) ** 2
            mu = np.sum(w * y) / np.sum(w)
            Q_het += float(np.sum(w * (y - mu) ** 2)); df_het += len(g) - 1
    Q_inc = max(0.0, Q_total - Q_het); df_inc = max(0, df_total - df_het)
    state.write("diagnostics", "nma_inconsistency", {
        "Q_total": Q_total, "Q_heterogeneity": Q_het, "Q_inconsistency": Q_inc,
        "df_inconsistency": df_inc,
        "inconsistency_pval": float(stats.chi2.sf(Q_inc, df_inc)) if df_inc > 0 else float("nan"),
        "inconsistent": (df_inc > 0 and float(stats.chi2.sf(Q_inc, df_inc)) < 0.05),
    })
    return state


# ==================================================================== netsplit
@register(
    name="netsplit", aliases=["节点劈分", "node_splitting", "side"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="节点劈分(SIDE):对每个有直接证据的对比,比较直接 vs 间接估计(反算法),检验局部不一致性",
    requires={"models": ["nma_contrasts", "nma"]}, produces={"diagnostics": ["netsplit"]},
    prerequisites={"functions": ["netmeta"]},
)
def netsplit(state: StudyState, **kwargs: Any) -> StudyState:
    """Node-splitting (SIDE) via back-calculation: direct vs indirect per comparison."""
    df = _contrasts(state); nma = state.models.get("nma")
    if df is None or not isinstance(nma, dict):
        return state
    from scipy import stats
    tr = nma["treatments"]; idx = {t: i for i, t in enumerate(tr)}
    beta = np.array(nma["_beta"]); cov = np.array(nma["_cov"]); tau2 = float(nma.get("tau2", 0.0))
    rows = []
    for (t1, t2), g in df.groupby(["treat1", "treat2"]):
        # random-effects direct pooling (same τ² as the network, so SEs are comparable)
        w = 1 / (g["seTE"].to_numpy(float) ** 2 + tau2); y = g["TE"].to_numpy(float)
        dir_est = float(np.sum(w * y) / np.sum(w)); dir_se = float(np.sqrt(1 / np.sum(w)))
        i, j = idx[t1], idx[t2]
        net_est = beta[i] - beta[j]
        net_se = float(np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 1e-12)))
        # back-calculate indirect: 1/se_net² = 1/se_dir² + 1/se_ind²
        prec_ind = 1 / net_se ** 2 - 1 / dir_se ** 2
        if prec_ind <= 1e-6:
            continue
        ind_se = float(np.sqrt(1 / prec_ind))
        ind_est = float((net_est / net_se ** 2 - dir_est / dir_se ** 2) / prec_ind)
        z = (dir_est - ind_est) / np.sqrt(dir_se ** 2 + ind_se ** 2)
        rows.append({"comparison": f"{t1} vs {t2}", "direct": dir_est, "direct_se": dir_se,
                     "indirect": ind_est, "indirect_se": ind_se, "network": float(net_est),
                     "difference": float(dir_est - ind_est), "z": float(z),
                     "pval": float(2 * stats.norm.sf(abs(z)))})
    state.write("diagnostics", "netsplit", {"comparisons": rows,
                "n_inconsistent": int(sum(r["pval"] < 0.05 for r in rows))})
    return state


# ==================================================================== netcomb
@register(
    name="netcomb", aliases=["成分网络meta", "component_nma", "additive_nma"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="成分/加性网络 meta:把复合干预拆成成分,估计各成分的加性效应(处理名如 'A+B' 自动解析)",
    requires={"models": ["nma_contrasts"]}, produces={"models": ["nma_components"]},
    prerequisites={"functions": ["nma_pairwise"]},
)
def netcomb(state: StudyState, **kwargs: Any) -> StudyState:
    """Component (additive) NMA. Treatments split on ``sep='+'`` into components;
    each treatment's effect = sum of its component effects."""
    df = _contrasts(state)
    if df is None:
        return state
    from scipy import stats
    sep = kwargs.get("sep", "+")
    treatments = sorted(set(df["treat1"]) | set(df["treat2"]), key=str)
    comps = sorted({c.strip() for t in treatments for c in str(t).split(sep)})
    cpos = {c: i for i, c in enumerate(comps)}

    def comp_vec(t):
        v = np.zeros(len(comps))
        for c in str(t).split(sep):
            v[cpos[c.strip()]] += 1
        return v
    n = len(df); X = np.zeros((n, len(comps))); y = df["TE"].to_numpy(float)
    V = np.diag(df["seTE"].to_numpy(float) ** 2)
    for i, (_, r) in enumerate(df.iterrows()):
        X[i] = comp_vec(r["treat1"]) - comp_vec(r["treat2"])
    Vi = np.linalg.pinv(V); A = np.linalg.pinv(X.T @ Vi @ X)
    beta = A @ (X.T @ Vi @ y); se = np.sqrt(np.diag(A))
    effects = {comps[i]: {"estimate": float(beta[i]), "se": float(se[i]),
                          "pval": float(2 * stats.norm.sf(abs(beta[i] / se[i]))) if se[i] > 0 else float("nan")}
               for i in range(len(comps))}
    state.write("models", "nma_components", {"components": comps, "effects": effects})
    return state
