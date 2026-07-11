"""``sv.pl._meta_plot`` — meta-analysis figures (forest + funnel).

Pure matplotlib. Reads the fitted model (``models['meta']``), the per-study
effects (``models['meta_effects']``), and heterogeneity / prediction-interval
diagnostics, and renders the two canonical meta-analysis plots.

Registered as ``meta_forest`` (the existing ``sv.pl.forest`` draws a *coefficient*
forest for DID/regression — this one is the study-level meta forest with a pooled
diamond + prediction interval) and ``funnel``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._figure import _save, _out_path, _register_figure


def _eff(state):
    eff = state.models.get("meta_effects")
    return eff if isinstance(eff, pd.DataFrame) and {"yi", "vi"}.issubset(eff.columns) and len(eff) else None


# ==================================================================== meta_forest
@register(
    name="meta_forest",
    aliases=["meta森林图", "森林图meta", "forest_meta"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="meta 分析森林图:各研究效应量+CI(权重框)+ 合并菱形 + 预测区间条;导出 PNG",
    requires={"models": ["meta_effects", "meta"]},
    produces={"artifacts": ["figures"]},
)
def meta_forest(state: StudyState, **kwargs: Any) -> StudyState:
    """Study-level forest plot: per-study CI, weight boxes, pooled diamond, PI bar.

    kwargs: ``slab=`` label column (default a ``slab``/``study`` column or index),
    ``title=``, ``xlab=``, ``out=``. Reads ``models['meta']`` for the pooled
    diamond and ``diagnostics['prediction_interval']`` for the PI bar if present.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eff = _eff(state)
    m = state.models.get("meta")
    if eff is None or not isinstance(m, dict) or m.get("estimate") is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.text(0.5, 0.5, "no fitted meta-analysis to plot\n(run sv.pp.escalc → sv.tl.meta_random/rma_mv)",
                ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        path = _save(fig, _out_path(kwargs, "meta_forest"))
        _register_figure(state, "meta_forest", path, note="空:无拟合模型")
        return state

    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    sei = np.sqrt(v)
    k = len(y)
    if "slab" in eff:
        labels = eff["slab"].astype(str).tolist()
    elif "study" in eff:
        labels = eff["study"].astype(str).tolist()
    else:
        labels = [f"Study {i+1}" for i in range(k)]

    lo, hi = y - 1.96 * sei, y + 1.96 * sei
    w = 1.0 / (v + float(m.get("tau2", 0.0)))
    wsz = 30 + 300 * (w / w.max())  # marker size ∝ weight

    fig_h = max(3.5, 0.34 * k + 2.0)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    ys = np.arange(k)[::-1]
    ax.axvline(m["estimate"], color="#b0281a", linestyle="--", linewidth=1, zorder=0, alpha=0.6)
    for i in range(k):
        ax.plot([lo[i], hi[i]], [ys[i], ys[i]], color="#33475b", linewidth=1.3, zorder=2)
    ax.scatter(y, ys, s=wsz, marker="s", color="#1f4e79", zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)

    # pooled diamond just below the studies
    dy = -1.4
    est, clb, cub = m["estimate"], m["ci_lb"], m["ci_ub"]
    ax.add_patch(plt.Polygon([[clb, dy], [est, dy + 0.32], [cub, dy], [est, dy - 0.32]],
                             closed=True, color="#b0281a", zorder=4))
    ax.text(ax.get_xlim()[0], dy, "Pooled", fontsize=8, va="center", ha="left", fontweight="bold")

    # prediction interval bar under the diamond
    pi = state.diagnostics.get("prediction_interval")
    if isinstance(pi, dict) and pi.get("pi_lb") is not None:
        ax.plot([pi["pi_lb"], pi["pi_ub"]], [dy - 0.7, dy - 0.7],
                color="#b0281a", linewidth=2.2, zorder=4, alpha=0.55)
        ax.text(est, dy - 1.15, "95% prediction interval", fontsize=7,
                color="#b0281a", ha="center", va="center")

    ax.set_ylim(dy - 1.6, k - 0.4)
    ax.set_xlabel(str(kwargs.get("xlab", f"Effect size ({eff['measure'].iloc[0] if 'measure' in eff else 'yi'})")))
    het = state.diagnostics.get("heterogeneity") or {}
    sub = ""
    if het:
        sub = f"  ·  I² = {het.get('I2', float('nan')):.1f}%,  τ² = {het.get('tau2', float('nan')):.3f}"
    ax.set_title(str(kwargs.get("title", "Meta-analysis forest plot")) + sub, fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    path = _save(fig, _out_path(kwargs, "meta_forest"))
    _register_figure(state, "meta_forest", path,
                     note=f"{k} 研究森林图 + 合并菱形 + 预测区间")
    return state


# ==================================================================== funnel
@register(
    name="funnel",
    aliases=["漏斗图", "funnel_plot"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="漏斗图:效应量对标准误散点 + 合并线 + 伪 95% 置信漏斗;目测小研究效应/发表偏倚",
    requires={"models": ["meta_effects", "meta"]},
    produces={"artifacts": ["figures"]},
)
def funnel(state: StudyState, **kwargs: Any) -> StudyState:
    """Funnel plot: effect vs. standard error, pooled line + pseudo-95% CI funnel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eff = _eff(state)
    m = state.models.get("meta")
    if eff is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.text(0.5, 0.5, "no effects to plot", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        path = _save(fig, _out_path(kwargs, "funnel"))
        _register_figure(state, "funnel", path, note="空:无效应量")
        return state

    y = eff["yi"].to_numpy(float); sei = np.sqrt(eff["vi"].to_numpy(float))
    est = float(m["estimate"]) if isinstance(m, dict) and m.get("estimate") is not None \
        else float(np.average(y, weights=1 / eff["vi"].to_numpy(float)))

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    se_max = float(sei.max()) * 1.05
    se_grid = np.linspace(0, se_max, 100)
    ax.plot(est + 1.96 * se_grid, se_grid, color="0.55", linewidth=1, linestyle="--", zorder=1)
    ax.plot(est - 1.96 * se_grid, se_grid, color="0.55", linewidth=1, linestyle="--", zorder=1)
    ax.fill_betweenx(se_grid, est - 1.96 * se_grid, est + 1.96 * se_grid,
                     color="#1f4e79", alpha=0.06, zorder=0)
    ax.axvline(est, color="#b0281a", linewidth=1, zorder=1)
    ax.scatter(y, sei, s=28, color="#1f4e79", edgecolor="white", linewidth=0.5, zorder=3)

    ax.invert_yaxis()  # precise studies (small SE) on top
    ax.set_xlabel(str(kwargs.get("xlab", "Effect size")))
    ax.set_ylabel("Standard error")
    eg = state.diagnostics.get("egger") or {}
    sub = f"  ·  Egger p = {eg.get('pval', float('nan')):.3f}" if eg else ""
    ax.set_title(str(kwargs.get("title", "Funnel plot")) + sub, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    path = _save(fig, _out_path(kwargs, "funnel"))
    _register_figure(state, "funnel", path, note="漏斗图 + 伪95%CI + 合并线")
    return state
