"""``sv.pl._meta_adv_plot`` — Tier-3 figures.

Network geometry (netgraph), net-heat league matrix, summary ROC (SROC) for
diagnostic accuracy, GOSH subset cloud, and the dose-response curve. Pure
matplotlib (circular network layout — no networkx dependency).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._figure import _save, _out_path, _register_figure


def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _empty(state, plt, key, msg, kwargs):
    fig, ax = plt.subplots(); ax.set_axis_off(); ax.text(.5, .5, msg, ha="center", color="0.4")
    _register_figure(state, key, _save(fig, _out_path(kwargs, key)), note="空")


# ==================================================================== netgraph
@register(
    name="netgraph", aliases=["网络图", "network_graph"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="网络 meta 的网络几何图:处理为节点、直接对比为边(边宽∝研究数),环形布局",
    requires={"models": ["nma_contrasts"]}, produces={"artifacts": ["figures"]},
    prerequisites={"functions": ["nma_pairwise"]},
)
def netgraph(state: StudyState, **kwargs: Any) -> StudyState:
    """Network geometry: treatments = nodes, direct comparisons = edges (width ∝ #studies)."""
    plt = _mpl()
    df = state.models.get("nma_contrasts")
    if not isinstance(df, pd.DataFrame) or not len(df):
        _empty(state, plt, "netgraph", "no network", kwargs); return state
    tr = sorted(set(df["treat1"]) | set(df["treat2"]), key=str); T = len(tr)
    ang = {t: 2 * np.pi * i / T for i, t in enumerate(tr)}
    pos = {t: (np.cos(a), np.sin(a)) for t, a in ang.items()}
    counts = df.groupby(["treat1", "treat2"]).size()
    fig, ax = plt.subplots(figsize=(6.5, 6.5)); ax.set_axis_off()
    for (t1, t2), n in counts.items():
        (x1, y1), (x2, y2) = pos[t1], pos[t2]
        ax.plot([x1, x2], [y1, y2], color="#1f4e79", lw=0.8 + 1.6 * np.log1p(n), alpha=0.5, zorder=1)
    for t, (x, yy) in pos.items():
        ax.scatter([x], [yy], s=700, color="#eef2f6", edgecolor="#1f4e79", lw=1.5, zorder=2)
        ax.text(x, yy, str(t), ha="center", va="center", fontsize=9, zorder=3)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_title(str(kwargs.get("title", "Network of treatments")), fontsize=11)
    _register_figure(state, "netgraph", _save(fig, _out_path(kwargs, "netgraph")), note="网络几何图")
    return state


# ==================================================================== netheat
@register(
    name="netheat", aliases=["网络热图", "league_heatmap"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="网络 meta 联赛表热图:全配对效应量矩阵(颜色=效应方向与大小)",
    requires={"models": ["nma"]}, produces={"artifacts": ["figures"]},
    prerequisites={"functions": ["netmeta"]},
)
def netheat(state: StudyState, **kwargs: Any) -> StudyState:
    """League-table heatmap of all pairwise network estimates."""
    plt = _mpl()
    nma = state.models.get("nma")
    if not isinstance(nma, dict) or "_beta" not in nma:
        _empty(state, plt, "netheat", "no NMA", kwargs); return state
    tr = nma["treatments"]; beta = np.array(nma["_beta"]); T = len(tr)
    M = beta[:, None] - beta[None, :]
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * T, 1.0 + 0.7 * T))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-np.max(np.abs(M)) or -1, vmax=np.max(np.abs(M)) or 1)
    ax.set_xticks(range(T)); ax.set_xticklabels(tr, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels(tr, fontsize=8)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > np.max(np.abs(M)) * 0.5 else "0.2")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row vs column")
    ax.set_title(str(kwargs.get("title", "Network league (row vs column)")), fontsize=11)
    _register_figure(state, "netheat", _save(fig, _out_path(kwargs, "netheat")), note="联赛表热图")
    return state


# ==================================================================== sroc
@register(
    name="sroc", aliases=["SROC曲线", "summary_roc"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="诊断准确性汇总 ROC(SROC):研究点(1-特异度, 敏感度)+ 汇总点 + 置信椭圆(读 dta_bivariate)",
    requires={"models": ["dta", "dta_bivariate"]}, produces={"artifacts": ["figures"]},
    prerequisites={"functions": ["dta_bivariate"]},
)
def sroc(state: StudyState, **kwargs: Any) -> StudyState:
    """Summary ROC: study points + bivariate summary point + confidence ellipse."""
    plt = _mpl()
    dta = state.models.get("dta"); biv = state.models.get("dta_bivariate")
    if not isinstance(dta, dict) or "sensitivity" not in dta or not isinstance(biv, dict):
        _empty(state, plt, "sroc", "no DTA fit", kwargs); return state
    sens = np.array(dta["sensitivity"]); spec = np.array(dta["specificity"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(1 - spec, sens, s=30, color="#1f4e79", alpha=0.7, zorder=3, label="studies")
    ax.scatter([1 - biv["specificity"]], [biv["sensitivity"]], s=140, marker="D",
               color="#b0281a", zorder=4, label="summary")
    # confidence ellipse from Sigma (in logit space, mapped approximately)
    mu = np.array(biv["_mu"]); Sig = np.array(biv["_Sigma"])
    th = np.linspace(0, 2 * np.pi, 200)
    vals, vecs = np.linalg.eigh(Sig)
    ell = (vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * np.array([np.cos(th), np.sin(th)]))) * 2.45
    def expit(x): return 1 / (1 + np.exp(-x))
    ex = 1 - expit(mu[1] + ell[1]); ey = expit(mu[0] + ell[0])
    ax.plot(ex, ey, color="#b0281a", lw=1, ls="--", alpha=0.7, label="95% region")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.7, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(str(kwargs.get("title", "Summary ROC")), fontsize=11)
    _register_figure(state, "sroc", _save(fig, _out_path(kwargs, "sroc")), note="SROC 曲线")
    return state


# ==================================================================== gosh
@register(
    name="gosh", aliases=["GOSH图", "gosh_plot"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="GOSH 图:大量随机子集的合并估计 vs I² 云图,揭示多峰/离群驱动的异质性结构",
    requires={"models": ["meta_effects"]}, produces={"artifacts": ["figures"]},
)
def gosh(state: StudyState, **kwargs: Any) -> StudyState:
    """GOSH: pooled estimate vs I² over many random study subsets."""
    plt = _mpl()
    eff = state.models.get("meta_effects")
    if not isinstance(eff, pd.DataFrame) or len(eff) < 4:
        _empty(state, plt, "gosh", "need ≥4 studies", kwargs); return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); k = len(y)
    rng = np.random.default_rng(int(kwargs.get("seed", 42)))
    nsub = int(kwargs.get("nsub", 2000)); ests, i2s = [], []
    for _ in range(nsub):
        m = rng.random(k) < rng.uniform(0.4, 0.9)
        if m.sum() < 2:
            continue
        w = 1 / v[m]; mu = np.sum(w * y[m]) / np.sum(w)
        Q = np.sum(w * (y[m] - mu) ** 2); dfree = m.sum() - 1
        ests.append(mu); i2s.append(max(0, 100 * (Q - dfree) / Q) if Q > 0 else 0)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(ests, i2s, s=4, color="#1f4e79", alpha=0.25)
    ax.set_xlabel("Pooled effect (subset)"); ax.set_ylabel("I² (%)")
    ax.set_title(str(kwargs.get("title", "GOSH plot")), fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    _register_figure(state, "gosh", _save(fig, _out_path(kwargs, "gosh")), note="GOSH 子集云图")
    return state


# ==================================================================== dose_response_plot
@register(
    name="dose_response_plot", aliases=["剂量反应曲线", "dose_curve"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="剂量反应曲线:合并后的线性/样条剂量反应 + CI 带(读 models['dosres'])",
    requires={"models": ["dosres"]}, produces={"artifacts": ["figures"]},
    prerequisites={"functions": ["dosresmeta"]},
)
def dose_response_plot(state: StudyState, **kwargs: Any) -> StudyState:
    """Dose-response curve (linear slope or spline) with a CI band."""
    plt = _mpl()
    dr = state.models.get("dosres")
    if not isinstance(dr, dict) or dr.get("model") is None:
        _empty(state, plt, "dose_response_plot", "no dose-response fit", kwargs); return state
    xmax = float(kwargs.get("dose_max", 10.0)); x = np.linspace(0, xmax, 100)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if dr["model"] == "linear":
        b, se = dr["slope_per_unit"], dr["se"]
        yv = b * x
        ax.fill_between(x, (b - 1.96 * se) * x, (b + 1.96 * se) * x, color="#1f4e79", alpha=0.15)
        ax.plot(x, yv, color="#1f4e79", lw=1.6)
    else:
        from ..tl._meta_dose import _rcs_basis
        knots = np.asarray(dr.get("knots", [0, xmax / 2, xmax]), float)
        B = _rcs_basis(x, knots); beta = np.array(dr["coefficients"])
        ax.plot(x, B @ beta, color="#1f4e79", lw=1.6)
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("Dose"); ax.set_ylabel("log relative risk")
    ax.set_title(str(kwargs.get("title", "Dose-response")), fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    _register_figure(state, "dose_response_plot", _save(fig, _out_path(kwargs, "dose_response_plot")),
                     note="剂量反应曲线")
    return state
