"""``sv.tl._meta2`` — Tier-2 estimators, CIs, conversions & 2×2 poolers.

τ² confidence interval (Q-profile), between-scale effect-size conversion,
proportion back-transform, subgroup analysis with Q_between, and the two
exact 2×2 poolers (Mantel-Haenszel, Peto) for rare-event binary outcomes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _weighted_mean, _estimate_tau2, _typical_v
from ..pp._meta_es import _resolve_df, _col, _passthrough


# ==================================================================== tau2_ci
@register(
    name="tau2_ci", aliases=["τ²置信区间", "tau2_confint", "q_profile"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="τ²/I²/H² 的 Q-profile 置信区间(广义 Q 反解);量化异质性本身的不确定性",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["tau2_ci"]},
)
def tau2_ci(state: StudyState, **kwargs: Any) -> StudyState:
    """Q-profile CI for τ² (and derived I² / H²). Exact via generalized-Q inversion."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats, optimize
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    k = len(y); level = float(kwargs.get("level", 0.95))

    def genQ(t2):
        w = 1.0 / (v + t2); mu = np.sum(w * y) / np.sum(w)
        return float(np.sum(w * (y - mu) ** 2))

    q_hi = stats.chi2.ppf(1 - (1 - level) / 2, k - 1)   # bounds the lower τ²
    q_lo = stats.chi2.ppf((1 - level) / 2, k - 1)

    def solve(target):
        if genQ(0.0) <= target:
            return 0.0
        hi = 1.0
        while genQ(hi) > target and hi < 1e8:
            hi *= 2
        try:
            return float(optimize.brentq(lambda t: genQ(t) - target, 0.0, hi))
        except ValueError:
            return float("nan")

    lb = solve(q_hi); ub = solve(q_lo)
    s2 = _typical_v(v)
    def i2(t2): return 100.0 * t2 / (t2 + s2) if (t2 + s2) > 0 else 0.0
    state.write("diagnostics", "tau2_ci", {
        "tau2_lb": lb, "tau2_ub": ub, "level": level,
        "I2_lb": i2(lb), "I2_ub": i2(ub),
        "H2_lb": (lb + s2) / s2 if s2 else float("nan"),
        "H2_ub": (ub + s2) / s2 if s2 else float("nan"),
    })
    return state


# ==================================================================== es_convert
@register(
    name="es_convert", aliases=["效应量换算", "convert_es", "d_to_r"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="在效应量尺度间换算存好的 meta_effects:SMD↔ZCOR(r)、SMD↔logOR、ZCOR↔COR(含 delta 方差)",
    requires={"models": ["meta_effects"]}, produces={"models": ["meta_effects"]},
)
def es_convert(state: StudyState, **kwargs: Any) -> StudyState:
    """Convert the stored effects to another scale. kwargs: ``to=`` one of
    ``ZCOR``/``COR``/``SMD``/``OR``. Delta-method variance conversion."""
    eff = state.models.get("meta_effects")
    if not isinstance(eff, pd.DataFrame) or not len(eff):
        return state
    eff = eff.copy()
    src = str(eff["measure"].iloc[0]).upper() if "measure" in eff else "GEN"
    to = str(kwargs.get("to", "ZCOR")).upper()
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    a = 4.0  # d↔r bridge constant (equal groups assumption)
    if src in ("SMD", "SMDH") and to in ("ZCOR", "COR"):
        r = y / np.sqrt(y ** 2 + a); vr = a ** 2 * v / (y ** 2 + a) ** 3
        if to == "ZCOR":
            y2 = np.arctanh(r); v2 = vr / (1 - r ** 2) ** 2
        else:
            y2, v2 = r, vr
    elif src in ("SMD", "SMDH") and to == "OR":
        y2 = y * np.pi / np.sqrt(3); v2 = v * np.pi ** 2 / 3
    elif src == "ZCOR" and to == "COR":
        r = np.tanh(y); y2 = r; v2 = v * (1 - r ** 2) ** 2
    elif src == "OR" and to in ("SMD", "SMDH"):
        y2 = y * np.sqrt(3) / np.pi; v2 = v * 3 / np.pi ** 2
    else:
        return state  # unsupported pair: no-op
    eff["yi"], eff["vi"], eff["sei"], eff["measure"] = y2, v2, np.sqrt(v2), to
    state.write("models", "meta_effects", eff)
    return state


# ==================================================================== backtransform_proportion
@register(
    name="backtransform_proportion", aliases=["比例回变换", "backtransform_prop"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="把合并后的 logit/arcsine/PFT 比例估计+CI 回变换成 0–1 比例(患病率 meta 的报告步骤;PFT 回变换会告警)",
    requires={"models": ["meta"]}, produces={"diagnostics": ["pooled_proportion"]},
)
def backtransform_proportion(state: StudyState, **kwargs: Any) -> StudyState:
    """Invert the pooled logit / arcsine / Freeman-Tukey proportion + CI to 0–1."""
    m = state.models.get("meta"); eff = _effects(state)
    if not isinstance(m, dict) or m.get("estimate") is None or eff is None:
        return state
    measure = str(kwargs.get("measure") or (eff["measure"].iloc[0] if "measure" in eff else "PLO")).upper()
    est, lb, ub = m["estimate"], m["ci_lb"], m["ci_ub"]

    def bt(x):
        if measure == "PLO":
            return 1.0 / (1.0 + np.exp(-x))
        if measure == "PAS":
            return np.sin(np.clip(x, 0, np.pi / 2)) ** 2
        if measure == "PFT":
            n_bar = float(kwargs.get("n_harmonic", 1.0 / np.mean(1.0 / eff.get("n", pd.Series([1])))) if "n" in eff else 1.0)
            return 0.5 * (1 - np.sign(np.cos(2 * x)) * np.sqrt(1 - (np.sin(2 * x) + (np.sin(2 * x) - 1 / np.sin(2 * x)) / n_bar) ** 2))
        if measure == "PR":
            return x
        return x
    out = {"measure": measure, "proportion": float(bt(est)),
           "ci_lb": float(bt(lb)), "ci_ub": float(bt(ub))}
    if measure == "PFT":
        out["warning"] = "Freeman-Tukey back-transform is unstable (Schwarzer 2019); prefer PLO"
    state.write("diagnostics", "pooled_proportion", out)
    return state


# ==================================================================== subgroup
@register(
    name="subgroup", aliases=["亚组分析", "subgroup_analysis", "q_between"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="按分类调节变量做亚组随机效应合并 + 组间异质性 Q_between 检验(离散调节变量的差异检验)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["subgroup"]},
)
def subgroup(state: StudyState, **kwargs: Any) -> StudyState:
    """Categorical moderator → per-subgroup random-effects pool + Q_between test."""
    eff = _effects(state)
    mod = kwargs.get("moderator") or kwargs.get("by")
    if eff is None or mod is None or mod not in eff.columns:
        return state
    from scipy import stats
    method = str(kwargs.get("method", "REML")).upper()
    groups = {}
    q_within = 0.0
    for g, sub in eff.groupby(mod, sort=False):
        y, v = sub["yi"].to_numpy(float), sub["vi"].to_numpy(float)
        t2 = _estimate_tau2(y, v, method) if len(y) > 1 else 0.0
        w = 1.0 / (v + t2); mu = float(np.sum(w * y) / np.sum(w)); se = float(np.sqrt(1 / np.sum(w)))
        wf = 1 / v; muf = np.sum(wf * y) / np.sum(wf)
        q_within += float(np.sum(wf * (y - muf) ** 2))
        groups[str(g)] = {"k": len(y), "estimate": mu, "se": se, "tau2": t2,
                          "ci_lb": mu - 1.96 * se, "ci_ub": mu + 1.96 * se}
    y, v = eff["yi"].to_numpy(float), eff["vi"].to_numpy(float)
    wf = 1 / v; muf = np.sum(wf * y) / np.sum(wf)
    q_total = float(np.sum(wf * (y - muf) ** 2))
    q_between = max(0.0, q_total - q_within); dfb = len(groups) - 1
    state.write("diagnostics", "subgroup", {
        "moderator": mod, "groups": groups,
        "Q_between": q_between, "df": dfb,
        "Q_between_pval": float(stats.chi2.sf(q_between, dfb)) if dfb > 0 else float("nan"),
    })
    return state


# ==================================================================== meta_mh
@register(
    name="meta_mh", aliases=["MH合并", "mantel_haenszel"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="Mantel-Haenszel 2×2 合并 OR + Robins-Breslow-Greenland 方差(稀有事件二分类,无需连续性校正)",
    requires={"sources": ["datasets"]}, produces={"models": ["meta"]},
)
def meta_mh(state: StudyState, **kwargs: Any) -> StudyState:
    """Mantel-Haenszel OR pooling from 2×2 counts (ai,bi,ci,di) + RBG variance."""
    df = _resolve_df(state, kwargs)
    a = _col(df, kwargs.get("ai")); b = _col(df, kwargs.get("bi"))
    c = _col(df, kwargs.get("ci")); d = _col(df, kwargs.get("di"))
    if any(x is None for x in (a, b, c, d)):
        state.write("models", "meta", {"estimate": None, "note": "need ai/bi/ci/di columns"})
        return state
    from scipy import stats
    a, b, c, d = (x.to_numpy(float) for x in (a, b, c, d))
    n = a + b + c + d
    R = a * d / n; S = b * c / n
    or_mh = R.sum() / S.sum()
    P = (a + d) / n; Q = (b + c) / n
    var_ln = (np.sum(P * R) / (2 * R.sum() ** 2)
              + np.sum(P * S + Q * R) / (2 * R.sum() * S.sum())
              + np.sum(Q * S) / (2 * S.sum() ** 2))
    ln = np.log(or_mh); se = np.sqrt(var_ln)
    state.write("models", "meta", {
        "model": "mantel_haenszel", "measure": "OR", "estimate": float(ln), "se": float(se),
        "or": float(or_mh), "ci_lb": float(ln - 1.96 * se), "ci_ub": float(ln + 1.96 * se),
        "or_ci_lb": float(np.exp(ln - 1.96 * se)), "or_ci_ub": float(np.exp(ln + 1.96 * se)),
        "zval": float(ln / se), "pval": float(2 * stats.norm.sf(abs(ln / se))),
        "tau2": 0.0, "k": len(a),
    })
    return state


# ==================================================================== meta_peto
@register(
    name="meta_peto", aliases=["Peto合并", "peto_or"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="Peto 一步法合并 OR(极稀有事件、平衡设计;不加连续性校正)",
    requires={"sources": ["datasets"]}, produces={"models": ["meta"]},
)
def meta_peto(state: StudyState, **kwargs: Any) -> StudyState:
    """Peto one-step OR (rare events). logOR = Σ(O−E)/ΣV, var = 1/ΣV."""
    df = _resolve_df(state, kwargs)
    a = _col(df, kwargs.get("ai")); b = _col(df, kwargs.get("bi"))
    c = _col(df, kwargs.get("ci")); d = _col(df, kwargs.get("di"))
    if any(x is None for x in (a, b, c, d)):
        state.write("models", "meta", {"estimate": None, "note": "need ai/bi/ci/di columns"})
        return state
    from scipy import stats
    a, b, c, d = (x.to_numpy(float) for x in (a, b, c, d))
    n = a + b + c + d; n1 = a + b; n2 = c + d; m1 = a + c; m2 = b + d
    E = n1 * m1 / n
    V = n1 * n2 * m1 * m2 / (n ** 2 * (n - 1))
    ln = float(np.sum(a - E) / np.sum(V)); se = float(np.sqrt(1 / np.sum(V)))
    state.write("models", "meta", {
        "model": "peto", "measure": "OR", "estimate": ln, "se": se, "or": float(np.exp(ln)),
        "ci_lb": ln - 1.96 * se, "ci_ub": ln + 1.96 * se,
        "or_ci_lb": float(np.exp(ln - 1.96 * se)), "or_ci_ub": float(np.exp(ln + 1.96 * se)),
        "zval": ln / se, "pval": float(2 * stats.norm.sf(abs(ln / se))), "tau2": 0.0, "k": len(a),
    })
    return state
