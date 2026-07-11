"""``sv.pl._meta_plot2`` — Tier-2 meta-analysis figures.

Contour-enhanced funnel (significance shading), Baujat (heterogeneity vs
influence), PRISMA 2020 flow diagram, and the risk-of-bias traffic-light matrix.
All pure matplotlib, reusing the figure helpers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._figure import _save, _out_path, _register_figure


def _eff(state):
    e = state.models.get("meta_effects")
    return e if isinstance(e, pd.DataFrame) and {"yi", "vi"}.issubset(e.columns) and len(e) else None


# ==================================================================== funnel_contour
@register(
    name="funnel_contour", aliases=["等高线漏斗图", "contour_funnel"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="等高线增强漏斗图:叠加 p<.01/.05/.10 显著性阴影带,区分发表偏倚与真实小研究效应",
    requires={"models": ["meta_effects", "meta"]}, produces={"artifacts": ["figures"]},
)
def funnel_contour(state: StudyState, **kwargs: Any) -> StudyState:
    """Contour-enhanced funnel: significance regions (p<.01/.05/.10) shaded behind the points."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    eff = _eff(state); m = state.models.get("meta")
    if eff is None:
        fig, ax = plt.subplots(); ax.set_axis_off(); ax.text(.5, .5, "no effects", ha="center")
        _register_figure(state, "funnel_contour", _save(fig, _out_path(kwargs, "funnel_contour")), note="空")
        return state
    y = eff["yi"].to_numpy(float); sei = np.sqrt(eff["vi"].to_numpy(float))
    est = float(m["estimate"]) if isinstance(m, dict) and m.get("estimate") is not None else \
        float(np.average(y, weights=1 / eff["vi"].to_numpy(float)))
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    se_max = float(sei.max()) * 1.05
    se_g = np.linspace(1e-4, se_max, 100)
    shades = [(0.10, "#dfe6ec"), (0.05, "#eef2f6"), (0.01, "#f7f9fb")]
    for alpha, col in shades:
        z = stats.norm.ppf(1 - alpha / 2)
        ax.fill_betweenx(se_g, est - z * se_g, est + z * se_g, color=col, zorder=0)
    for alpha in (0.10, 0.05, 0.01):
        z = stats.norm.ppf(1 - alpha / 2)
        ax.plot(est + z * se_g, se_g, color="0.6", lw=0.7, ls=":")
        ax.plot(est - z * se_g, se_g, color="0.6", lw=0.7, ls=":")
    ax.axvline(est, color="#b0281a", lw=1)
    ax.scatter(y, sei, s=30, color="#1f4e79", edgecolor="white", lw=0.5, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel(str(kwargs.get("xlab", "Effect size"))); ax.set_ylabel("Standard error")
    ax.set_title(str(kwargs.get("title", "Contour-enhanced funnel")), fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    _register_figure(state, "funnel_contour", _save(fig, _out_path(kwargs, "funnel_contour")),
                     note="等高线漏斗图 p<.01/.05/.10")
    return state


# ==================================================================== baujat
@register(
    name="baujat", aliases=["Baujat图", "baujat_plot"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib", "numpy"],
    description="Baujat 图:各研究对异质性 Q 的贡献(x)vs 对合并估计的影响(y),定位异质性来源",
    requires={"models": ["meta_effects"]}, produces={"artifacts": ["figures"]},
)
def baujat(state: StudyState, **kwargs: Any) -> StudyState:
    """Baujat: per-study Q-contribution (x) vs influence on the pooled estimate (y)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eff = _eff(state)
    if eff is None:
        fig, ax = plt.subplots(); ax.set_axis_off(); ax.text(.5, .5, "no effects", ha="center")
        _register_figure(state, "baujat", _save(fig, _out_path(kwargs, "baujat")), note="空")
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float); k = len(y)
    w = 1.0 / v; mu = np.sum(w * y) / np.sum(w)
    qcontrib = w * (y - mu) ** 2
    infl = np.empty(k)
    for i in range(k):
        m = np.arange(k) != i
        mu_i = np.sum(w[m] * y[m]) / np.sum(w[m])
        infl[i] = w[i] * (mu - mu_i) ** 2 * np.sum(w[m])  # squared standardized change
    lab = (eff["study"].astype(str).tolist() if "study" in eff.columns
           else [str(i + 1) for i in range(k)])
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(qcontrib, infl, s=28, color="#1f4e79", zorder=3)
    for i in range(k):
        ax.annotate(lab[i], (qcontrib[i], infl[i]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points", color="0.35")
    ax.set_xlabel("Contribution to overall heterogeneity (Q)")
    ax.set_ylabel("Influence on pooled estimate")
    ax.set_title(str(kwargs.get("title", "Baujat plot")), fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    _register_figure(state, "baujat", _save(fig, _out_path(kwargs, "baujat")),
                     note="Baujat 异质性贡献 vs 影响")
    return state


# ==================================================================== prisma_diagram
@register(
    name="prisma_diagram", aliases=["PRISMA流程图", "prisma_flowchart"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib"],
    description="PRISMA 2020 四阶段筛选流程图(identification→screening→eligibility→included);读 governance['prisma']",
    requires={"governance": ["prisma"]}, produces={"artifacts": ["figures"]},
    prerequisites={"functions": ["prisma_flow"]},
)
def prisma_diagram(state: StudyState, **kwargs: Any) -> StudyState:
    """PRISMA 2020 four-stage box-arrow diagram from the recorded record counts."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = state.governance.get("prisma") or {}
    fig, ax = plt.subplots(figsize=(7.2, 8.4)); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    boxes = [
        (5, 11, f"Records identified\n(n = {p.get('identified', '—')})"),
        (5, 9, f"After duplicates removed\n(n = {p.get('after_dedup', '—')})"),
        (5, 7, f"Records screened\n(n = {p.get('screened', '—')})"),
        (5, 5, f"Full-text assessed\n(n = {p.get('full_text', '—')})"),
        (5, 3, f"Studies included\n(n = {p.get('included', '—')})"),
    ]
    excl = [
        (8.2, 9, f"Duplicates\n(n = {p.get('duplicates', '—')})"),
        (8.2, 7, f"Excluded on title/abstract\n(n = {p.get('excluded_screen', '—')})"),
        (8.2, 5, f"Excluded full-text\n(n = {p.get('excluded_fulltext', '—')})"),
    ]
    for x, yy, txt in boxes:
        ax.add_patch(plt.Rectangle((x - 2, yy - 0.6), 4, 1.2, fill=True, facecolor="#eef2f6",
                                   edgecolor="#1f4e79", lw=1.2))
        ax.text(x, yy, txt, ha="center", va="center", fontsize=8.5)
    for x, yy, txt in excl:
        ax.add_patch(plt.Rectangle((x - 1.3, yy - 0.5), 3, 1.0, fill=True, facecolor="#f7ece9",
                                   edgecolor="#b0281a", lw=1))
        ax.text(x + 0.2, yy, txt, ha="center", va="center", fontsize=7.5)
    for yy in (11, 9, 7, 5):
        ax.annotate("", xy=(5, yy - 1.4), xytext=(5, yy - 0.6),
                    arrowprops=dict(arrowstyle="-|>", color="#1f4e79"))
    for yy in (9, 7, 5):
        ax.annotate("", xy=(6.9, yy), xytext=(5, yy),
                    arrowprops=dict(arrowstyle="-|>", color="#b0281a"))
    ax.set_title(str(kwargs.get("title", "PRISMA 2020 flow")), fontsize=11)
    _register_figure(state, "prisma_diagram", _save(fig, _out_path(kwargs, "prisma_diagram")),
                     note="PRISMA 2020 流程图")
    return state


# ==================================================================== rob_traffic_light
@register(
    name="rob_traffic_light", aliases=["偏倚风险交通灯", "rob_matrix"],
    category="figure", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["matplotlib"],
    description="偏倚风险交通灯矩阵(研究×domain,低/中/高/未知配色);读 governance['risk_of_bias']",
    requires={"governance": ["risk_of_bias"]}, produces={"artifacts": ["figures"]},
    prerequisites={"functions": ["risk_of_bias"]},
)
def rob_traffic_light(state: StudyState, **kwargs: Any) -> StudyState:
    """Risk-of-bias traffic-light matrix (study × domain) from recorded judgements."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rob = state.governance.get("risk_of_bias") or {}
    judged = rob.get("studies") or {}
    if not judged:
        fig, ax = plt.subplots(); ax.set_axis_off(); ax.text(.5, .5, "no RoB judgements", ha="center")
        _register_figure(state, "rob_traffic_light", _save(fig, _out_path(kwargs, "rob_traffic_light")), note="空")
        return state
    studies = list(judged); domains = rob.get("domains") or list(next(iter(judged.values())).keys())
    palette = {"low": "#2e7d32", "some": "#f9a825", "moderate": "#f9a825",
               "high": "#c62828", "critical": "#8e0000", "unclear": "#9e9e9e", "no_info": "#9e9e9e"}
    fig, ax = plt.subplots(figsize=(1.4 + 0.9 * len(domains), 1.0 + 0.42 * len(studies)))
    for r, sname in enumerate(studies):
        for c, dom in enumerate(domains):
            j = str(judged[sname].get(dom, "unclear")).lower()
            ax.scatter(c, len(studies) - 1 - r, s=260, color=palette.get(j, "#9e9e9e"),
                       edgecolor="white", lw=1, zorder=3)
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(studies))); ax.set_yticklabels(studies[::-1], fontsize=8)
    ax.set_xlim(-0.5, len(domains) - 0.5); ax.set_ylim(-0.5, len(studies) - 0.5)
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_title(str(kwargs.get("title", f"Risk of bias ({rob.get('tool', 'RoB')})")), fontsize=11)
    _register_figure(state, "rob_traffic_light", _save(fig, _out_path(kwargs, "rob_traffic_light")),
                     note="偏倚风险交通灯矩阵")
    return state
