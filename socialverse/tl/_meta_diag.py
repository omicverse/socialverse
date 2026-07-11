"""``sv.tl._meta_diag`` — Tier-2 influence / sensitivity diagnostics.

Leave-one-out, cumulative meta-analysis, per-study influence (studentized
deleted residual, DFFITS, Cook's D, leverage, weight, τ²-deleted), and an
outlier-excluded refit. All refit-based (works for any pooler, including
``rma_mv``), applying metafor's influence cutoffs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _estimate_tau2


def _pool(y, v, method="REML"):
    t2 = _estimate_tau2(y, v, method) if len(y) > 1 else 0.0
    w = 1.0 / (v + t2); mu = float(np.sum(w * y) / np.sum(w)); se = float(np.sqrt(1 / np.sum(w)))
    s2 = _typical(v)
    i2 = 100.0 * t2 / (t2 + s2) if (t2 + s2) > 0 else 0.0
    return mu, se, t2, i2


def _typical(v):
    w = 1.0 / v; k = len(v); d = np.sum(w) ** 2 - np.sum(w ** 2)
    return float((k - 1) * np.sum(w) / d) if d > 0 else float(np.mean(v))


def _labels(eff):
    for c in ("slab", "study"):
        if c in eff.columns:
            return eff[c].astype(str).tolist()
    return [f"Study {i+1}" for i in range(len(eff))]


# ==================================================================== leave_one_out
@register(
    name="leave_one_out", aliases=["逐一剔除", "loo_meta", "sensitivity_loo"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="逐一剔除每篇研究后重估合并效应/CI/τ²/I²(敏感性分析:哪篇在驱动结论)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["leave_one_out"]},
)
def leave_one_out(state: StudyState, **kwargs: Any) -> StudyState:
    """Drop each study in turn, refit random-effects; report the trajectory."""
    eff = _effects(state)
    if eff is None:
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    method = str(kwargs.get("method", "REML")).upper(); lab = _labels(eff)
    rows = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        mu, se, t2, i2 = _pool(y[m], v[m], method)
        rows.append({"omitted": lab[i], "estimate": mu, "ci_lb": mu - 1.96 * se,
                     "ci_ub": mu + 1.96 * se, "tau2": t2, "I2": i2})
    state.write("diagnostics", "leave_one_out", {"rows": rows, "method": method})
    return state


# ==================================================================== cumulative_ma
@register(
    name="cumulative_ma", aliases=["累积meta", "cumulative_meta"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="按年份/精度排序的累积 meta:每加入一篇后的合并估计轨迹(证据随时间如何积累/稳定)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["cumulative_ma"]},
)
def cumulative_ma(state: StudyState, **kwargs: Any) -> StudyState:
    """Ordered running pool (by ``order=`` column, e.g. year; default by precision)."""
    eff = _effects(state)
    if eff is None:
        return state
    order = kwargs.get("order")
    if order and order in eff.columns:
        idx = np.argsort(pd.to_numeric(eff[order], errors="coerce").to_numpy(float))
    else:
        idx = np.argsort(-1.0 / eff["vi"].to_numpy(float))  # most precise first
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); lab = _labels(eff)
    method = str(kwargs.get("method", "REML")).upper()
    rows = []
    for j in range(1, len(idx) + 1):
        sel = idx[:j]
        mu, se, t2, i2 = _pool(y[sel], v[sel], method)
        rows.append({"added": lab[idx[j - 1]], "k": j, "estimate": mu,
                     "ci_lb": mu - 1.96 * se, "ci_ub": mu + 1.96 * se, "tau2": t2, "I2": i2})
    state.write("diagnostics", "cumulative_ma", {"rows": rows, "ordered_by": order or "precision"})
    return state


# ==================================================================== influence
@register(
    name="influence", aliases=["影响诊断", "influence_diagnostics", "cooks_distance"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="逐研究影响诊断:学生化删除残差、DFFITS、Cook's D、杠杆值、权重、τ²-deleted,并按 metafor 阈值标注离群/影响点",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["influence"]},
)
def influence(state: StudyState, **kwargs: Any) -> StudyState:
    """Per-study influence (refit-based, works for any pooler). metafor cutoffs applied."""
    eff = _effects(state)
    if eff is None:
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); k = len(y); lab = _labels(eff)
    method = str(kwargs.get("method", "REML")).upper()
    mu_f, se_f, t2_f, _ = _pool(y, v, method)
    w_f = 1.0 / (v + t2_f); hat = w_f / np.sum(w_f)
    dffits_cut = 3 * np.sqrt(1.0 / k)
    rows = []
    for i in range(k):
        m = np.arange(k) != i
        mu_i, se_i, t2_i, _ = _pool(y[m], v[m], method)
        rstud = (y[i] - mu_i) / np.sqrt(v[i] + t2_i)
        dffits = (mu_f - mu_i) / se_i if se_i > 0 else 0.0
        cook = (mu_f - mu_i) ** 2 / se_f ** 2
        covr = (se_i ** 2) / (se_f ** 2)
        influential = (abs(rstud) > 1.96) or (abs(dffits) > dffits_cut) or (hat[i] > 3.0 / k)
        rows.append({"study": lab[i], "rstudent": float(rstud), "dffits": float(dffits),
                     "cooks_d": float(cook), "hat": float(hat[i]), "weight_pct": float(100 * hat[i]),
                     "tau2_del": float(t2_i), "cov_ratio": float(covr),
                     "influential": bool(influential)})
    state.write("diagnostics", "influence", {
        "rows": rows, "dffits_cutoff": float(dffits_cut), "hat_cutoff": float(3.0 / k),
        "n_influential": int(sum(r["influential"] for r in rows)),
    })
    return state


# ==================================================================== outlier_refit
@register(
    name="outlier_refit", aliases=["剔除离群重估", "refit_without_outliers"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="剔除被 influence 标注的离群研究后重估合并,报告与原估计的差异(稳健性)",
    requires={"models": ["meta_effects"]}, produces={"diagnostics": ["outlier_refit"]},
    prerequisites={"optional_functions": ["influence"]},
)
def outlier_refit(state: StudyState, **kwargs: Any) -> StudyState:
    """Refit excluding studentized-residual outliers; report the shift."""
    eff = _effects(state)
    if eff is None:
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); lab = _labels(eff)
    method = str(kwargs.get("method", "REML")).upper(); cut = float(kwargs.get("cutoff", 1.96))
    inf = state.diagnostics.get("influence")
    if isinstance(inf, dict) and inf.get("rows"):
        flags = np.array([abs(r["rstudent"]) > cut for r in inf["rows"]])
    else:
        mu_f, *_ = _pool(y, v, method)
        flags = np.array([abs((y[i] - mu_f) / np.sqrt(v[i])) > cut for i in range(len(y))])
    keep = ~flags
    mu0, se0, t20, i20 = _pool(y, v, method)
    if keep.sum() >= 2 and flags.any():
        mu1, se1, t21, i21 = _pool(y[keep], v[keep], method)
    else:
        mu1, se1, t21, i21 = mu0, se0, t20, i20
    state.write("diagnostics", "outlier_refit", {
        "removed": [lab[i] for i in np.where(flags)[0]], "n_removed": int(flags.sum()),
        "estimate_full": mu0, "estimate_trimmed": mu1, "delta": mu1 - mu0,
        "I2_full": i20, "I2_trimmed": i21, "tau2_full": t20, "tau2_trimmed": t21,
    })
    return state
