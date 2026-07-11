"""``sv.tl._meta_selection`` — selection models & advanced publication bias (Tier-3).

Vevea-Hedges step-function selection model (the normalizing constant is a
closed-form sum of normal CDFs — no quadrature), p-curve, p-uniform &
p-uniform*, the Mathur-VanderWeele selection sensitivity (S-value), and a
one-look publication-bias report orchestrating the whole suite.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _estimate_tau2


@register(
    name="selection_model_stepfun", aliases=["阶梯选择模型", "vevea_hedges", "weightfunction"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="Vevea-Hedges 阶梯函数选择模型:按 p 值区间估计发表权重 + 偏倚校正后的 μ + 选择性 LRT(归一化常数为正态 CDF 闭式和,无求积)",
    requires={"models": ["meta_effects"]}, produces={"models": ["selection_model"]},
)
def selection_model_stepfun(state: StudyState, **kwargs: Any) -> StudyState:
    """Vevea-Hedges step-function weight model. kwargs: ``steps=[.025,.05,.5,1]``
    (one-sided p cutpoints). Reports selection-adjusted μ, weights, and a
    likelihood-ratio test for selection."""
    eff = _effects(state)
    if eff is None:
        state.write("models", "selection_model", {"note": "no meta_effects"})
        return state
    from scipy import optimize, stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); sei = np.sqrt(v)
    steps = sorted(set(list(kwargs.get("steps", [0.025, 0.05, 0.5])) + [1.0]))
    bounds_p = np.array([0.0] + list(steps))          # p-interval edges
    z_edges = stats.norm.isf(bounds_p)                # one-sided z thresholds (desc)
    L = len(steps)
    p_obs = stats.norm.sf(y / sei)                     # one-sided p per study
    interval = np.clip(np.searchsorted(bounds_p, p_obs, side="right") - 1, 0, L - 1)

    def negll(theta, selection=True):
        mu, logt2 = theta[0], theta[1]; t2 = np.exp(logt2)
        omega = np.concatenate([[1.0], np.exp(theta[2:])]) if selection else np.ones(L)
        s2 = v + t2; s = np.sqrt(s2)
        ll = 0.0
        for i in range(len(y)):
            dens = stats.norm.pdf(y[i], mu, s[i])
            # normalizing constant A_i = Σ_l ω_l · P(y in interval l) under N(mu, s_i²)
            yl = sei[i] * z_edges                       # yi thresholds for each p edge (desc)
            probs = stats.norm.cdf(yl[:-1], mu, s[i]) - stats.norm.cdf(yl[1:], mu, s[i])
            A = np.sum(omega * probs)
            w_i = omega[interval[i]]
            ll += np.log(max(w_i * dens, 1e-300)) - np.log(max(A, 1e-300))
        return -ll
    mu0 = float(np.sum(y / v) / np.sum(1 / v)); t20 = _estimate_tau2(y, v, "REML")
    x0 = np.concatenate([[mu0, np.log(max(t20, 1e-4))], np.zeros(L - 1)])
    res = optimize.minimize(negll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 5000})
    ll_sel = -res.fun
    # unadjusted (no selection) fit for the LRT
    res0 = optimize.minimize(lambda th: negll(np.concatenate([th, np.zeros(L - 1)]), selection=False),
                             x0[:2], method="Nelder-Mead")
    ll_null = -res0.fun
    lr = 2 * (ll_sel - ll_null); dfree = L - 1
    omega = np.concatenate([[1.0], np.exp(res.x[2:])])
    state.write("models", "selection_model", {
        "mu_adjusted": float(res.x[0]), "tau2": float(np.exp(res.x[1])),
        "mu_unadjusted": mu0, "weights": omega.tolist(), "steps": steps,
        "lrt_selection": float(lr), "lrt_df": dfree,
        "lrt_pval": float(stats.chi2.sf(lr, dfree)) if dfree > 0 else float("nan"),
        "converged": bool(res.success),
    })
    return state


@register(
    name="pcurve", aliases=["p曲线", "p_curve"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="p-curve 分析:仅用显著研究,检验 p 值右偏(证据价值)vs 平坦(无效应/p-hacking)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["pcurve"]},
)
def pcurve(state: StudyState, **kwargs: Any) -> StudyState:
    """p-curve: right-skew (evidential value) vs flatness test on significant studies."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    z = np.abs(eff["yi"].to_numpy(float) / np.sqrt(eff["vi"].to_numpy(float)))
    p2 = 2 * stats.norm.sf(z)                         # two-sided p
    sig = p2 < 0.05
    k = int(sig.sum())
    if k < 2:
        state.write("diagnostics", "pcurve", {"note": "need ≥2 significant studies", "k_significant": k})
        return state
    ppr = p2[sig] / 0.05                               # pp-value for right-skew
    Z_right = float(np.sum(stats.norm.ppf(ppr)) / np.sqrt(k))
    ppf = (1 - p2[sig]) / (1 - 0.05)                   # (approx) flatness comparison
    state.write("diagnostics", "pcurve", {
        "k_significant": k, "Z_right_skew": Z_right,
        "right_skew_pval": float(stats.norm.cdf(Z_right)),
        "evidential_value": Z_right < -1.645,
        "note": "right-skew (Z<−1.645) ⇒ evidential value; flat ⇒ no effect / selective reporting",
    })
    return state


@register(
    name="puniform", aliases=["p_uniform", "p均匀"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="p-uniform:仅用显著研究,要求条件 p 值均匀来估计校正后效应 + 发表偏倚检验(van Assen)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["puniform"]},
)
def puniform(state: StudyState, **kwargs: Any) -> StudyState:
    """p-uniform effect estimate from significant studies (conditional-p uniformity)."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats, optimize
    y = eff["yi"].to_numpy(float); sei = np.sqrt(eff["vi"].to_numpy(float))
    alpha = float(kwargs.get("alpha", 0.05))
    zc = stats.norm.isf(alpha / 2)
    sig = (y / sei) > zc                               # significant & positive
    k = int(sig.sum())
    if k < 2:
        state.write("diagnostics", "puniform", {"note": "need ≥2 significant positive studies", "k": k})
        return state
    ys, ss = y[sig], sei[sig]

    def mean_q(mu):
        ycrit = ss * zc
        q = stats.norm.sf((ys - mu) / ss) / stats.norm.sf((ycrit - mu) / ss)
        return np.mean(q) - 0.5
    lo, hi = -5.0, 5.0
    try:
        mu_hat = float(optimize.brentq(mean_q, lo, hi))
    except ValueError:
        mu_hat = float("nan")
    mu_naive = float(np.sum(y / sei ** 2) / np.sum(1 / sei ** 2))
    state.write("diagnostics", "puniform", {
        "effect_puniform": mu_hat, "effect_naive": mu_naive, "k_significant": k,
        "publication_bias_suspected": np.isfinite(mu_hat) and (mu_naive - mu_hat) > 0.1,
    })
    return state


@register(
    name="puniform_star", aliases=["p_uniform_star", "puni_star"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="p-uniform*(实验性):联合估计 μ 与 τ²,用全部(含不显著)研究的条件似然;比 p-uniform 更稳但求解更娇气",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["puniform_star"]},
)
def puniform_star(state: StudyState, **kwargs: Any) -> StudyState:
    """p-uniform* (experimental): jointly estimate μ and τ² via conditional likelihood."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats, optimize
    y = eff["yi"].to_numpy(float); sei = np.sqrt(eff["vi"].to_numpy(float))
    zc = stats.norm.isf(float(kwargs.get("alpha", 0.05)) / 2)

    def negll(theta):
        mu, logt2 = theta; t2 = np.exp(logt2); s = np.sqrt(sei ** 2 + t2)
        ycrit = sei * zc
        sig = (y / sei) > zc
        ll = 0.0
        for i in range(len(y)):
            if sig[i]:
                num = stats.norm.pdf(y[i], mu, s[i]); den = stats.norm.sf((ycrit[i] - mu) / s[i])
            else:
                num = stats.norm.pdf(y[i], mu, s[i]); den = stats.norm.cdf((ycrit[i] - mu) / s[i])
            ll += np.log(max(num, 1e-300)) - np.log(max(den, 1e-300))
        return -ll
    mu0 = float(np.median(y))
    res = optimize.minimize(negll, [mu0, np.log(0.05)], method="Nelder-Mead",
                            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000})
    state.write("diagnostics", "puniform_star", {
        "effect": float(res.x[0]), "tau2": float(np.exp(res.x[1])),
        "converged": bool(res.success), "note": "experimental conditional-likelihood solver",
    })
    return state


@register(
    name="pubbias_sensitivity", aliases=["发表偏倚敏感性", "svalue", "mathur_vanderweele"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="发表偏倚选择敏感性(Mathur-VanderWeele):显著'肯定'研究被发表的倾向比 η 下的校正估计,及使效应降到阈值所需的 η(S 值)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["pubbias_sensitivity"]},
)
def pubbias_sensitivity(state: StudyState, **kwargs: Any) -> StudyState:
    """Selection sensitivity: corrected estimate as affirmative studies are η× over-published."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); sei = np.sqrt(v)
    affirmative = (y / sei > stats.norm.isf(0.025)) & (y > 0)   # significant & positive

    def corrected(eta):
        w = np.where(affirmative, 1.0, eta) / v                 # up-weight nonaffirmative by η
        return float(np.sum(w * y) / np.sum(w))
    grid = [1, 2, 3, 5, 10]
    curve = {str(e): corrected(e) for e in grid}
    q = float(kwargs.get("threshold", 0.0))
    eta_max = float(kwargs.get("eta_max", 50.0))
    # η needed to bring estimate to q (search); None ⇒ robust past eta_max
    s_value = None
    for e in np.linspace(1, eta_max, 500):
        if (corrected(1) > q) == (corrected(e) <= q):
            s_value = float(e); break
    robust = s_value is None
    state.write("diagnostics", "pubbias_sensitivity", {
        "estimate_unadjusted": corrected(1), "correction_curve": curve,
        "threshold": q, "s_value_eta": s_value, "eta_max_searched": eta_max,
        "robust_to_selection": robust, "estimate_at_eta_max": corrected(eta_max),
        "note": (f"effect robust: not reduced to {q} even at η={eta_max:g}" if robust
                 else "η = affirmative-study over-publication ratio; S-value = η to reach threshold"),
    })
    return state


@register(
    name="pubbias_report", aliases=["发表偏倚报告", "pubbias_summary"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="发表偏倚一览报告:汇总 Egger/剪补/PET-PEESE/阶梯选择模型/敏感性到一张表,给整体判断",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["pubbias_report"]},
)
def pubbias_report(state: StudyState, **kwargs: Any) -> StudyState:
    """One-look publication-bias report orchestrating the full suite."""
    from .._registry import registry
    eff = _effects(state)
    if eff is None:
        return state
    for fn in ["egger_test", "trim_and_fill", "pet_peese", "selection_model_stepfun", "pubbias_sensitivity"]:
        try:
            registry.get(fn).func(state)
        except Exception:
            pass
    d = state.diagnostics
    report = {
        "egger_pval": (d.get("egger") or {}).get("pval"),
        "trimfill_k0": (d.get("trim_and_fill") or {}).get("k0_missing"),
        "trimfill_adjusted": (d.get("trim_and_fill") or {}).get("estimate_adjusted"),
        "petpeese_corrected": (d.get("pet_peese") or {}).get("corrected_estimate"),
        "selection_lrt_pval": (state.models.get("selection_model") or {}).get("lrt_pval"),
        "selection_adjusted_mu": (state.models.get("selection_model") or {}).get("mu_adjusted"),
        "sensitivity_svalue": (d.get("pubbias_sensitivity") or {}).get("s_value_eta"),
    }
    flags = sum([
        (report["egger_pval"] is not None and report["egger_pval"] < 0.10),
        (report["trimfill_k0"] or 0) > 0,
        (report["selection_lrt_pval"] is not None and report["selection_lrt_pval"] < 0.10),
    ])
    report["concern_level"] = ["low", "some", "moderate", "high"][min(flags, 3)]
    state.write("diagnostics", "pubbias_report", report)
    return state
