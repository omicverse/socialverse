"""``sv.tl._meta_bias`` — Tier-2 publication-bias / small-study-effect tools.

Trim-and-fill (Duval-Tweedie L0), PET / PEESE / PET-PEESE precision-effect
regressions, Begg rank-correlation test, fail-safe N (Rosenthal / Orwin), and
the test of excess significance (Ioannidis-Trikalinos). Egger lives in ``_meta``.

Honesty: trim-and-fill over/under-corrects under heavy heterogeneity; PET-PEESE
is biased downward under p-hacking; fail-safe N is deprecated as a primary
measure — all flagged. These are *diagnostics*, not corrections to trust blindly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _estimate_tau2


def _reg(name, aliases, desc, produces="diagnostics"):
    return register(name=name, aliases=aliases, category="social_science_quant",
                    tier="pro", skill="meta-analysis", languages=["Python"],
                    key_tools=["numpy", "scipy"], description=desc,
                    requires={"models": ["meta_effects"]}, produces={produces: [name]})


# ==================================================================== trim_and_fill
@_reg("trim_and_fill", ["剪补法", "duval_tweedie"],
      "Duval-Tweedie 剪补法:估计缺失研究数 k0 + 镜像填补 + 校正后合并估计(重异质性下会过/欠校正,仅诊断)")
def trim_and_fill(state: StudyState, **kwargs: Any) -> StudyState:
    """Duval-Tweedie trim-and-fill (L0). Estimates k0 missing studies + adjusted effect."""
    eff = _effects(state)
    if eff is None:
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); n0 = len(y)
    side = kwargs.get("side")

    def pooled(mask):
        w = 1.0 / v[mask]; return float(np.sum(w * y[mask]) / np.sum(w))

    keep = np.ones(n0, bool); est = pooled(keep); k0 = 0
    for _ in range(100):
        idx = np.where(keep)[0]
        yc = y[idx]; res = yc - est; n = len(yc)
        order = np.argsort(np.abs(res)); rank = np.empty(n); rank[order] = np.arange(1, n + 1)
        sgn = np.sign(res)
        if side is None:
            side = "right" if np.sum(sgn * rank) > 0 else "left"
        s = 1 if side == "right" else -1
        Tn = float(np.sum(rank[sgn == s]))
        L0 = (4 * Tn - n * (n + 1)) / (2 * n - 1)
        k0_new = int(max(0, round(L0)))
        if k0_new == k0:
            break
        k0 = k0_new
        order_all = np.argsort(y)
        trim = set((order_all[-k0:] if side == "right" else order_all[:k0]).tolist())
        keep = np.array([i not in trim for i in range(n0)])
        if keep.sum() < 2:
            keep = np.ones(n0, bool); break
        est = pooled(keep)
    if k0 > 0:
        order_all = np.argsort(y)
        extreme = order_all[-k0:] if side == "right" else order_all[:k0]
        yf = 2 * est - y[extreme]; vf = v[extreme]
        yfull = np.concatenate([y, yf]); vfull = np.concatenate([v, vf])
    else:
        yfull, vfull = y, v
    w = 1.0 / vfull; adj = float(np.sum(w * yfull) / np.sum(w))
    w0 = 1.0 / v; orig = float(np.sum(w0 * y) / np.sum(w0))
    state.write("diagnostics", "trim_and_fill", {
        "k0_missing": k0, "side": side, "estimate_observed": orig,
        "estimate_adjusted": adj, "k_filled": n0 + k0,
        "note": "L0 estimator; over/under-corrects under heavy heterogeneity — diagnostic only",
    })
    return state


# ==================================================================== PET / PEESE
@_reg("pet", ["精度效应检验", "precision_effect_test"],
      "PET:yi 对 sei 的 WLS 回归,截距=校正后效应(检验小研究效应下真效应是否≠0)")
def pet(state: StudyState, **kwargs: Any) -> StudyState:
    """Precision-Effect Test: WLS yᵢ ~ seᵢ (weights 1/vᵢ); intercept = corrected effect."""
    eff = _effects(state)
    if eff is None:
        return state
    state.write("diagnostics", "pet", _pet_peese(eff, "sei"))
    return state


@_reg("peese", ["精度效应估计", "precision_effect_estimate"],
      "PEESE:yi 对 vi(方差)的 WLS 回归,截距=校正后效应(PET 拒绝后的效应量估计)")
def peese(state: StudyState, **kwargs: Any) -> StudyState:
    """PEESE: WLS yᵢ ~ vᵢ (weights 1/vᵢ); intercept = corrected effect (use when PET rejects)."""
    eff = _effects(state)
    if eff is None:
        return state
    state.write("diagnostics", "peese", _pet_peese(eff, "vi"))
    return state


@_reg("pet_peese", ["条件PET-PEESE", "petpeese"],
      "PET-PEESE 条件切换:PET 截距显著(单侧)→ 用 PEESE 估计,否则用 PET(发表偏倚下的校正效应)")
def pet_peese(state: StudyState, **kwargs: Any) -> StudyState:
    """Conditional PET-PEESE: if PET intercept is significant (one-sided) report PEESE, else PET."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    p = _pet_peese(eff, "sei"); pe = _pet_peese(eff, "vi")
    t = p["intercept"] / p["intercept_se"] if p["intercept_se"] else 0.0
    use_peese = stats.t.sf(abs(t), p["df"]) < 0.05   # one-sided
    chosen = pe if use_peese else p
    state.write("diagnostics", "pet_peese", {
        "model": "PEESE" if use_peese else "PET",
        "corrected_estimate": chosen["intercept"], "se": chosen["intercept_se"],
        "pet": p, "peese": pe,
        "note": "biased downward under p-hacking; report alongside uncorrected estimate",
    })
    return state


def _pet_peese(eff, xcol):
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    x = np.sqrt(v) if xcol == "sei" else v
    X = np.column_stack([np.ones_like(x), x]); w = 1.0 / v
    XtW = X.T * w
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
    resid = y - X @ beta; n = len(y); dof = n - 2
    mse = float(np.sum(w * resid ** 2) / dof)
    cov = mse * np.linalg.pinv(XtW @ X)
    return {"intercept": float(beta[0]), "intercept_se": float(np.sqrt(cov[0, 0])),
            "slope": float(beta[1]), "df": dof}


# ==================================================================== begg
@_reg("begg_test", ["Begg秩相关检验", "begg_mazumdar"],
      "Begg-Mazumdar 秩相关检验:标准化效应与方差的 Kendall τ 关联(漏斗不对称的非参检验)")
def begg_test(state: StudyState, **kwargs: Any) -> StudyState:
    """Begg-Mazumdar rank-correlation test (Kendall τ of standardized effect vs variance)."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    w = 1.0 / v; mu = np.sum(w * y) / np.sum(w)
    vs = v - 1.0 / np.sum(w)
    star = (y - mu) / np.sqrt(np.maximum(vs, 1e-12))
    tau, p = stats.kendalltau(star, v)
    state.write("diagnostics", "begg", {"kendall_tau": float(tau), "pval": float(p),
                                        "asymmetry": float(p) < 0.10})
    return state


# ==================================================================== failsafe_n
@_reg("failsafe_n", ["失安全数", "file_drawer"],
      "失安全数 file-drawer N:Rosenthal(需多少零效应研究翻盘)+ Orwin(降到平凡阈值)。已不建议作主要证据")
def failsafe_n(state: StudyState, **kwargs: Any) -> StudyState:
    """Fail-safe N — Rosenthal (to nullify) + Orwin (to trivial). Deprecated as primary."""
    eff = _effects(state)
    if eff is None:
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); k = len(y)
    z = y / np.sqrt(v)
    z_a = stats.norm.ppf(1 - float(kwargs.get("alpha", 0.05)))
    rosenthal = float((np.sum(z)) ** 2 / z_a ** 2 - k)
    w = 1.0 / v; d_bar = float(np.sum(w * y) / np.sum(w))
    d_c = float(kwargs.get("trivial", 0.1))
    orwin = float(k * (abs(d_bar) - d_c) / d_c) if d_c > 0 else float("nan")
    state.write("diagnostics", "failsafe_n", {
        "rosenthal": max(0.0, rosenthal), "orwin": max(0.0, orwin),
        "note": "deprecated as a primary publication-bias measure (Becker 2005)",
    })
    return state


# ==================================================================== excess_significance
@_reg("excess_significance", ["过度显著检验", "TES", "ioannidis_trikalinos"],
      "过度显著性检验 TES:观测显著研究数 vs 按合并效应算的期望数(Ioannidis-Trikalinos)")
def excess_significance(state: StudyState, **kwargs: Any) -> StudyState:
    """Test of excess significance: observed vs expected count of significant studies."""
    eff = _effects(state); m = state.models.get("meta")
    if eff is None:
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); sei = np.sqrt(v); k = len(y)
    theta = float(m["estimate"]) if isinstance(m, dict) and m.get("estimate") is not None \
        else float(np.sum(y / v) / np.sum(1 / v))
    zc = stats.norm.ppf(1 - float(kwargs.get("alpha", 0.05)) / 2)
    O = int(np.sum(np.abs(y / sei) > zc))
    power = 1 - stats.norm.cdf(zc - theta / sei) + stats.norm.cdf(-zc - theta / sei)
    E = float(np.sum(power))
    chi2 = (O - E) ** 2 / E + (O - E) ** 2 / max(k - E, 1e-9)
    state.write("diagnostics", "excess_significance", {
        "observed_sig": O, "expected_sig": E, "chi2": float(chi2),
        "pval": float(stats.chi2.sf(chi2, 1)), "excess": O > E,
        "note": "inflated Type-I under between-study heterogeneity",
    })
    return state
