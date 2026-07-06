"""``sv.pl._figure`` — registered implementations for the ``social-science-figure``
and ``manuscript-docx-review`` skills (the *plot / render* phase of the pipeline).

This is the terminal namespace of the ``StudyState`` / ``registry`` spine: it turns
the fitted models and coded qualitative artifacts into **deliverables** — forest
plots, event-study dynamic-effect curves, design-weighted coefficient charts,
theme co-occurrence network figures, and a conservatively-typeset manuscript DOCX.

The registry contract keeps rendering honest: a forest plot *requires*
``models.did`` (you cannot draw a coefficient you never estimated), an event-study
figure *requires* ``models.event_study``, etc. Every figure function *produces*
``artifacts.figures`` so a resolver can chain analysis → figure without guessing.

Real rendering only — every figure is a genuine ``matplotlib`` (Agg backend, no
display) draw saved to a real PNG file whose path is recorded back into
``artifacts.figures``. The theme map lays out a real ``networkx`` spring layout.
``manuscript_docx`` builds a real ``python-docx`` document when the (optional)
dependency is present and degrades to a plain-text/Markdown sibling otherwise,
always emitting a structural coverage checklist into ``diagnostics.coverage``.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any

import numpy as np

# Force a non-interactive backend *before* pyplot is imported anywhere, so nothing
# ever tries to open a window (headless CI / kernel safe).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .._registry import register  # noqa: E402
from .._state import StudyState  # noqa: E402


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


#: shared visual defaults — one place so every figure looks like one product.
_DPI = 200
_FONT = 10
_FIGSIZE = (7.0, 4.5)

# CJK-capable fonts to prefer (labels here are Chinese). Best-effort only: we keep
# whichever the host actually has and always fall back to DejaVu Sans, so a machine
# without a CJK font still renders the figure (with tofu glyphs) instead of failing.
_CJK_CANDIDATES = (
    "PingFang SC", "PingFang HK", "Heiti SC", "Hiragino Sans GB", "STHeiti",
    "Songti SC", "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
    "SimHei", "Arial Unicode MS",
)


def _cjk_fonts() -> list[str]:
    """Return installed CJK-capable font family names, in preference order (cached)."""
    cached = getattr(_cjk_fonts, "_cache", None)
    if cached is not None:
        return cached
    fonts: list[str] = []
    try:
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        fonts = [name for name in _CJK_CANDIDATES if name in have]
    except Exception:
        fonts = []
    _cjk_fonts._cache = fonts  # type: ignore[attr-defined]
    return fonts


def _apply_style() -> None:
    """Apply the shared, deterministic rcParams for a uniform figure look."""
    plt.rcParams.update({
        "figure.dpi": _DPI,
        "savefig.dpi": _DPI,
        "font.size": _FONT,
        "font.family": "sans-serif",
        # Prefer a CJK font if present, then DejaVu Sans as the guaranteed fallback.
        "font.sans-serif": _cjk_fonts() + ["DejaVu Sans"],
        "axes.unicode_minus": False,  # keep the ASCII hyphen so negatives render
        "axes.titlesize": _FONT + 2,
        "axes.labelsize": _FONT,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })


def _out_path(kwargs: dict[str, Any], stem: str) -> str:
    """Resolve the output PNG path: explicit ``out=`` kwarg, else a scratch temp file.

    A directory-only ``out=`` is honored by joining ``<stem>.png`` onto it. The
    parent directory is created if missing. PNG is used throughout, so ``savefig``
    keeps ``bbox_inches='tight'`` (raster crops cleanly).
    """
    out = kwargs.get("out")
    if out:
        out = os.path.expanduser(str(out))
        if os.path.isdir(out) or out.endswith(os.sep):
            out = os.path.join(out, f"{stem}.png")
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return out
    fd, path = tempfile.mkstemp(prefix=f"sv_{stem}_", suffix=".png")
    os.close(fd)
    return path


def _save(fig: "plt.Figure", path: str) -> str:
    """Save ``fig`` to ``path`` (PNG, tight bbox) and close it. Returns the path."""
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _as_ci(point: float, lo: Any, hi: Any) -> tuple[float, float]:
    """Coerce a (possibly missing) CI pair into finite half-width offsets around ``point``."""
    try:
        lo_f, hi_f = float(lo), float(hi)
        if np.isfinite(lo_f) and np.isfinite(hi_f):
            return point - lo_f, hi_f - point
    except (TypeError, ValueError):
        pass
    return 0.0, 0.0


# --------------------------------------------------------------------------- forest
@register(
    name="forest",
    aliases=["森林图", "forest_plot"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib", "numpy"],
    description="把 DID/回归系数画成森林图(点估计 + 置信区间),导出 PNG",
    requires={"models": ["did"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def forest(state: StudyState, **kwargs: Any) -> StudyState:
    """Draw a forest plot of coefficient point estimates with confidence intervals.

    Reads ``models['did']`` (the DID/TWFE ATT and its CI) by default, or an explicit
    ``coefs=`` kwarg — a mapping ``{label: {'coef'|'att': v, 'ci': [lo, hi]}}`` or
    ``{label: (v, se)}`` — so the same primitive renders any set of coefficients.
    Each row gets a point marker and a horizontal CI whisker; a dashed zero line
    marks the null. The PNG path is stored at ``artifacts.figures['forest']``.
    """
    _apply_style()

    rows = _forest_rows(state, kwargs)
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    if not rows:
        ax.text(0.5, 0.5, "no coefficients to plot", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        path = _save(fig, _out_path(kwargs, "forest"))
        _register_figure(state, "forest", path, note="空:models.did 无可绘制系数")
        return state

    labels = [r[0] for r in rows]
    points = np.array([r[1] for r in rows], dtype=float)
    lo_err = np.array([r[2] for r in rows], dtype=float)
    hi_err = np.array([r[3] for r in rows], dtype=float)
    ys = np.arange(len(rows))[::-1]  # first row on top

    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1, zorder=0)
    ax.errorbar(
        points, ys, xerr=[lo_err, hi_err], fmt="o", color="#1f4e79",
        ecolor="#1f4e79", elinewidth=1.4, capsize=3, markersize=5, zorder=3,
    )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("系数估计 (点估计 ± 95% CI)")
    ax.set_title(str(kwargs.get("title", "森林图 · 系数点估计")))

    path = _save(fig, _out_path(kwargs, "forest"))
    _register_figure(state, "forest", path,
                     note=f"{len(rows)} 个系数的点估计+CI 森林图")
    return state


def _forest_rows(state: StudyState, kwargs: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    """Normalize the coefficient source into ``(label, point, lo_err, hi_err)`` rows."""
    coefs = kwargs.get("coefs")
    rows: list[tuple[str, float, float, float]] = []

    if isinstance(coefs, dict) and coefs:
        for label, spec in coefs.items():
            point, lo, hi = _coef_point_ci(spec)
            if point is None:
                continue
            lerr, herr = _as_ci(point, lo, hi)
            rows.append((str(label), point, lerr, herr))
        return rows

    model = state.models.get("did") or {}
    if isinstance(model, dict):
        att = model.get("att", model.get("coef"))
        if att is not None and np.isfinite(_safe_float(att)):
            point = float(att)
            ci = model.get("ci") or [None, None]
            lo, hi = (ci[0], ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else (None, None)
            lerr, herr = _as_ci(point, lo, hi)
            label = str(model.get("outcome") or kwargs.get("label") or "ATT")
            rows.append((label, point, lerr, herr))
    return rows


def _coef_point_ci(spec: Any) -> tuple[float | None, Any, Any]:
    """Pull ``(point, lo, hi)`` from a heterogeneous coefficient spec."""
    if isinstance(spec, dict):
        point = spec.get("coef", spec.get("att", spec.get("estimate")))
        ci = spec.get("ci")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            return (_safe_float(point) if point is not None else None, ci[0], ci[1])
        se = spec.get("se")
        if point is not None and se is not None:
            p = _safe_float(point)
            s = _safe_float(se)
            return (p, p - 1.96 * s, p + 1.96 * s)
        return (_safe_float(point) if point is not None else None, None, None)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        p, s = _safe_float(spec[0]), _safe_float(spec[1])
        return (p, p - 1.96 * s, p + 1.96 * s)
    p = _safe_float(spec)
    return (p if np.isfinite(p) else None, None, None)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ event_study_plot
@register(
    name="event_study_plot",
    aliases=["事件研究图"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib"],
    description="事件研究动态效应图(相对处理时点,含 0 参考线)",
    requires={"models": ["event_study"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def event_study_plot(state: StudyState, **kwargs: Any) -> StudyState:
    """Plot the event-study dynamic effects across relative treatment time.

    Reads ``models['event_study']['coefs']`` — ``{rel_period: (coef, se)}`` — and
    draws one point per relative period with a 95% CI whisker (``coef ± 1.96·se``),
    connected in order. A vertical line at the treatment onset (``rel = 0``, or the
    normalized ``base`` boundary) and a horizontal zero line frame the pre/post
    contrast. The PNG path is stored at ``artifacts.figures['event_study']``.
    """
    _apply_style()

    model = state.models.get("event_study") or {}
    coefs = model.get("coefs") if isinstance(model, dict) else None
    base = int(model.get("base", -1)) if isinstance(model, dict) else -1

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    periods, points, errs = _event_series(coefs)
    if not periods:
        ax.text(0.5, 0.5, "no event-study coefficients", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        path = _save(fig, _out_path(kwargs, "event_study"))
        _register_figure(state, "event_study", path,
                         note="空:models.event_study 无 coefs")
        return state

    x = np.array(periods, dtype=float)
    y = np.array(points, dtype=float)
    e = np.array(errs, dtype=float)

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1, zorder=0)
    # Treatment onset reference: the boundary between the last lead and first lag.
    onset = 0.0 if 0 in periods else base + 0.5
    ax.axvline(onset, color="#b03a2e", linestyle=":", linewidth=1.3, zorder=1,
               label="处理时点")
    ax.plot(x, y, "-", color="#1f4e79", linewidth=1.2, alpha=0.8, zorder=2)
    ax.errorbar(x, y, yerr=e, fmt="o", color="#1f4e79", ecolor="#1f4e79",
                elinewidth=1.3, capsize=3, markersize=5, zorder=3)

    ax.set_xticks(x)
    ax.set_xlabel("相对处理时点 (event time)")
    ax.set_ylabel("动态效应 (相对 base 期)")
    ax.set_title(str(kwargs.get("title", "事件研究 · 动态处理效应")))
    ax.legend(loc="best", frameon=False)

    path = _save(fig, _out_path(kwargs, "event_study"))
    _register_figure(state, "event_study", path,
                     note=f"{len(periods)} 个相对期动态效应,onset={onset:g}")
    return state


def _event_series(coefs: Any) -> tuple[list[int], list[float], list[float]]:
    """Normalize ``{period: (coef, se)}`` (or ``{period: coef}``) into sorted series."""
    if not isinstance(coefs, dict) or not coefs:
        return [], [], []
    parsed: list[tuple[int, float, float]] = []
    for k, v in coefs.items():
        try:
            period = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            coef = _safe_float(v[0])
            se = _safe_float(v[1]) if len(v) >= 2 else 0.0
        else:
            coef, se = _safe_float(v), 0.0
        if not np.isfinite(coef):
            continue
        if not np.isfinite(se):
            se = 0.0
        parsed.append((period, coef, 1.96 * se))
    parsed.sort(key=lambda t: t[0])
    periods = [p for p, _, _ in parsed]
    points = [c for _, c, _ in parsed]
    errs = [e for _, _, e in parsed]
    return periods, points, errs


# --------------------------------------------------------------------- survey_dist
@register(
    name="survey_dist",
    aliases=["调查分布图"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib", "numpy"],
    description="加权估计的系数/分布可视化(设计加权 vs 朴素对比)",
    requires={"models": ["weighted_reg"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def survey_dist(state: StudyState, **kwargs: Any) -> StudyState:
    """Visualize design-weighted regression coefficients as a horizontal bar chart.

    Reads ``models['weighted_reg']`` — ``coef`` / ``se`` / ``ci`` keyed by term. Each
    non-intercept term becomes a horizontal bar (design-weighted point estimate) with
    a 95% CI error whisker; where an unweighted contrast is available it is overlaid
    as a light marker so the reader can see how much the survey weights moved the
    estimate. The PNG path is stored at ``artifacts.figures['survey']``.
    """
    _apply_style()

    model = state.models.get("weighted_reg") or {}
    coef = model.get("coef") if isinstance(model, dict) else None
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    if not isinstance(coef, dict) or not coef:
        ax.text(0.5, 0.5, "no weighted coefficients", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        path = _save(fig, _out_path(kwargs, "survey"))
        _register_figure(state, "survey", path,
                         note="空:models.weighted_reg 无 coef")
        return state

    se = model.get("se") if isinstance(model.get("se"), dict) else {}
    ci = model.get("ci") if isinstance(model.get("ci"), dict) else {}
    unweighted = {}
    sens = state.diagnostics.get("sensitivity")
    if isinstance(sens, dict) and isinstance(sens.get("unweighted"), dict):
        unweighted = sens["unweighted"]

    # drop the intercept for readability unless explicitly requested
    show_const = bool(kwargs.get("show_const", False))
    terms = [t for t in coef if show_const or t not in ("const", "Intercept")]
    if not terms:
        terms = list(coef)

    points = np.array([_safe_float(coef[t]) for t in terms], dtype=float)
    lo_err = np.zeros(len(terms))
    hi_err = np.zeros(len(terms))
    for i, t in enumerate(terms):
        pair = ci.get(t) if isinstance(ci, dict) else None
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            lo_err[i], hi_err[i] = _as_ci(points[i], pair[0], pair[1])
        elif t in se:
            s = _safe_float(se[t])
            if np.isfinite(s):
                lo_err[i] = hi_err[i] = 1.96 * s

    ys = np.arange(len(terms))[::-1]
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1, zorder=0)
    ax.barh(ys, points, height=0.55, color="#2e86c1", alpha=0.85, zorder=2,
            label="设计加权")
    ax.errorbar(points, ys, xerr=[lo_err, hi_err], fmt="none", ecolor="#154360",
                elinewidth=1.4, capsize=3, zorder=3)

    if unweighted:
        uw = np.array([_safe_float(unweighted.get(t, np.nan)) for t in terms], dtype=float)
        mask = np.isfinite(uw)
        if mask.any():
            ax.scatter(uw[mask], ys[mask], marker="D", color="#7f8c8d",
                       s=28, zorder=4, label="朴素(未加权)")

    ax.set_yticks(ys)
    ax.set_yticklabels(terms)
    ax.set_ylim(-0.6, len(terms) - 0.4)
    ax.set_xlabel("系数估计 (点估计 ± 95% CI)")
    ax.set_title(str(kwargs.get("title", "调查加权估计 · 系数分布")))
    ax.legend(loc="best", frameon=False)

    path = _save(fig, _out_path(kwargs, "survey"))
    _register_figure(state, "survey", path,
                     note=f"{len(terms)} 个加权系数条形图" +
                          ("(含朴素对比)" if unweighted else ""))
    return state


# ----------------------------------------------------------------------- theme_map
@register(
    name="theme_map",
    aliases=["主题地图"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib", "networkx"],
    description="质性主题共现网络图(spring layout)",
    requires={"codes": ["theme_map"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def theme_map(state: StudyState, **kwargs: Any) -> StudyState:
    """Draw the qualitative code co-occurrence network as a spring-layout graph.

    Reads ``codes['theme_map']`` — ``{'nodes': [...], 'adjacency': {n: {nbr: w}}}`` —
    builds a ``networkx`` graph, lays it out with a deterministic (``seed=0``) spring
    layout, and draws nodes sized by degree/count and edges weighted by co-occurrence.
    Node fill encodes the assigned theme when the map carries one. Falls back to a
    circular layout if ``networkx`` is unavailable. Stored at
    ``artifacts.figures['theme_map']``.
    """
    _apply_style()

    tmap = state.codes.get("theme_map")
    if kwargs.get("adjacency") is not None or kwargs.get("nodes") is not None:
        tmap = {"nodes": kwargs.get("nodes"), "adjacency": kwargs.get("adjacency", {})}
    nodes, edges, node_theme = _theme_graph_data(tmap)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_axis_off()

    if not nodes:
        ax.text(0.5, 0.5, "no theme-map nodes", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
        path = _save(fig, _out_path(kwargs, "theme_map"))
        _register_figure(state, "theme_map", path, note="空:codes.theme_map 无节点")
        return state

    nx = _try_import("networkx")
    pos = _layout(nx, nodes, edges)

    # degree-weighted node sizes
    deg: dict[str, float] = {n: 0.0 for n in nodes}
    for a, b, w in edges:
        deg[a] += w
        deg[b] += w
    max_deg = max(deg.values()) if deg and max(deg.values()) > 0 else 1.0

    themes = sorted({t for t in node_theme.values() if t})
    cmap = plt.get_cmap("tab10")
    theme_color = {t: cmap(i % 10) for i, t in enumerate(themes)}
    node_colors = [theme_color.get(node_theme.get(n), "#4c72b0") for n in nodes]

    # edges first (under nodes)
    max_w = max((w for _, _, w in edges), default=1.0) or 1.0
    for a, b, w in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], "-", color="0.6",
                linewidth=0.6 + 2.4 * (w / max_w), alpha=0.5, zorder=1)

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    sizes = [120 + 680 * (deg[n] / max_deg) for n in nodes]
    ax.scatter(xs, ys, s=sizes, c=node_colors, edgecolors="white",
               linewidths=1.0, zorder=2)
    for n in nodes:
        ax.annotate(str(n), pos[n], fontsize=_FONT - 2, ha="center", va="center",
                    zorder=3)

    if themes:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=theme_color[t], markeredgecolor="white",
                          label=str(t)) for t in themes]
        ax.legend(handles=handles, loc="best", frameon=False, title="主题")

    ax.set_title(str(kwargs.get("title", "主题共现网络")))
    path = _save(fig, _out_path(kwargs, "theme_map"))
    _register_figure(state, "theme_map", path,
                     note=f"{len(nodes)} 节点 / {len(edges)} 边 主题共现图")
    return state


def _theme_graph_data(
    tmap: Any,
) -> tuple[list[str], list[tuple[str, str, float]], dict[str, str]]:
    """Extract ``(nodes, weighted_edges, node->theme)`` from a theme_map dict."""
    if not isinstance(tmap, dict):
        return [], [], {}
    adjacency = tmap.get("adjacency")
    if not isinstance(adjacency, dict):
        adjacency = {}
    nodes = list(tmap.get("nodes") or adjacency.keys())
    node_theme: dict[str, str] = {}
    themes_field = tmap.get("themes")
    if isinstance(themes_field, dict):
        # {theme: {codes: [...]}} inverse mapping, if present
        for theme, info in themes_field.items():
            for c in (info.get("codes", []) if isinstance(info, dict) else []):
                node_theme[str(c)] = str(theme)

    seen: set[frozenset] = set()
    edges: list[tuple[str, str, float]] = []
    for a, nbrs in adjacency.items():
        if not isinstance(nbrs, dict):
            continue
        for b, w in nbrs.items():
            key = frozenset((str(a), str(b)))
            if str(a) == str(b) or key in seen:
                continue
            seen.add(key)
            edges.append((str(a), str(b), _safe_float(w) if np.isfinite(_safe_float(w)) else 1.0))
    return [str(n) for n in nodes], edges, node_theme


def _layout(nx: Any, nodes: list[str], edges: list[tuple[str, str, float]]) -> dict[str, tuple[float, float]]:
    """Deterministic node positions — spring layout via networkx, else a circle."""
    if nx is not None:
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for a, b, w in edges:
            g.add_edge(a, b, weight=w)
        try:
            return nx.spring_layout(g, seed=0, k=None)
        except Exception:
            pass
    n = max(len(nodes), 1)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return {node: (float(np.cos(a)), float(np.sin(a)))
            for node, a in zip(nodes, angles)}


# ------------------------------------------------------------------ manuscript_docx
@register(
    name="manuscript_docx",
    aliases=["稿件生成", "manuscript_docx"],
    category="figure",
    tier="community",
    skill="manuscript-docx-review",
    languages=["Python"],
    key_tools=["python-docx", "matplotlib"],
    description="手稿保守审校 + 公式安全 DOCX 生成 + 结构/渲染质检",
    requires={"sources": ["datasets"]},
    produces={"artifacts": ["docx", "pdf"], "diagnostics": ["coverage"]},
    auto_fix="auto",
)
def manuscript_docx(state: StudyState, **kwargs: Any) -> StudyState:
    """Render a manuscript to DOCX (conservative typesetting) and run a structural QC.

    The manuscript is taken from ``kwargs['manuscript']`` — either a plain string
    (Markdown-ish, ``#``-prefixed headings) or a dict
    ``{'title':..., 'sections': [{'heading':..., 'body':...}, ...]}``. A real
    ``python-docx`` document is written when the (optional) dependency is present;
    otherwise the same content is emitted to a Markdown sibling and the fallback is
    recorded. Either way a **coverage checklist** — section / heading / figure /
    table counts and math-safety flags — is written to ``diagnostics.coverage``, and
    the produced file path is stored at ``artifacts.docx``.

    "Conservative" = the writer never rewrites prose; it only structures what it is
    given and flags (rather than silently drops) anything it cannot render safely
    (e.g. raw ``$...$`` math, which DOCX cannot typeset without an equation engine).
    """
    manuscript = kwargs.get("manuscript")
    title, sections = _parse_manuscript(manuscript, state)

    coverage = _coverage_checklist(title, sections, state)
    stem = str(kwargs.get("stem", "manuscript"))
    out = kwargs.get("out")

    docx = _try_import("docx")
    if docx is not None:
        path = _write_docx(docx, title, sections, out, stem)
        coverage["renderer"] = "python-docx"
        coverage["fallback"] = False
    else:
        path = _write_markdown_fallback(title, sections, out, stem)
        coverage["renderer"] = "markdown-fallback"
        coverage["fallback"] = True
        coverage["fallback_reason"] = "python-docx 未安装 — 已降级为 .md 保留全部内容"

    coverage["output_path"] = path
    state.write("artifacts", "docx", path)
    state.write("diagnostics", "coverage", coverage)
    return state


def _parse_manuscript(
    manuscript: Any, state: StudyState,
) -> tuple[str, list[dict[str, str]]]:
    """Normalize any manuscript input into ``(title, [{'heading', 'body'}, ...])``."""
    if isinstance(manuscript, dict):
        title = str(manuscript.get("title") or "Untitled manuscript")
        raw_sections = manuscript.get("sections") or []
        sections: list[dict[str, str]] = []
        for s in raw_sections:
            if isinstance(s, dict):
                sections.append({
                    "heading": str(s.get("heading", s.get("title", ""))),
                    "body": str(s.get("body", s.get("text", ""))),
                })
            else:
                sections.append({"heading": "", "body": str(s)})
        if not sections and manuscript.get("body"):
            sections.append({"heading": "", "body": str(manuscript["body"])})
        return title, sections

    if isinstance(manuscript, str) and manuscript.strip():
        return _parse_markdown(manuscript)

    # No manuscript supplied: assemble a minimal skeleton from the study provenance
    # so the function still produces a real, non-placeholder document.
    title = "Study manuscript (auto-skeleton)"
    steps = getattr(state, "provenance", []) or []
    body_lines = [f"{i + 1}. {r.get('function', '?')}" for i, r in enumerate(steps)]
    body = "本文档由 socialverse 依据研究出处(provenance)自动生成骨架。\n" + \
           ("已执行步骤:\n" + "\n".join(body_lines) if body_lines else "(无出处记录)")
    return title, [{"heading": "方法与出处", "body": body}]


def _parse_markdown(text: str) -> tuple[str, list[dict[str, str]]]:
    """Split ``#``-headed Markdown into a title + section list (no prose rewriting)."""
    lines = text.splitlines()
    title = ""
    sections: list[dict[str, str]] = []
    cur_heading = ""
    cur_body: list[str] = []

    def flush() -> None:
        if cur_heading or cur_body:
            sections.append({"heading": cur_heading,
                             "body": "\n".join(cur_body).strip()})

    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
        elif stripped.startswith("#"):
            flush()
            cur_heading = stripped.lstrip("#").strip()
            cur_body = []
        else:
            cur_body.append(ln)
    flush()
    if not title:
        title = "Untitled manuscript"
    return title, sections


def _has_math(s: str) -> bool:
    """Heuristic: does the text contain inline/display TeX math DOCX cannot typeset?"""
    if "$$" in s:
        return True
    # a $...$ pair on one line
    parts = s.split("$")
    return len(parts) >= 3


def _coverage_checklist(
    title: str, sections: list[dict[str, str]], state: StudyState,
) -> dict[str, Any]:
    """Structural QC of the manuscript — the deliverable's self-audit."""
    headings = [s["heading"] for s in sections if s.get("heading")]
    full_text = "\n".join([title] + [s.get("body", "") for s in sections])
    n_words = len(full_text.split())
    # figures already produced upstream in this study
    figs = state.artifacts.get("figures")
    n_figures = len(figs) if isinstance(figs, dict) else (1 if figs else 0)
    tables = state.artifacts.get("tables")
    n_tables = len(tables) if isinstance(tables, (dict, list)) else (
        1 if tables is not None else 0)
    math_sections = [s["heading"] or f"§{i + 1}"
                     for i, s in enumerate(sections) if _has_math(s.get("body", ""))]

    required = ["引言", "方法", "结果", "讨论"]
    present = [r for r in required if any(r in h for h in headings)]

    return {
        "title": title,
        "n_sections": len(sections),
        "headings": headings,
        "n_words": n_words,
        "n_figures": int(n_figures),
        "n_tables": int(n_tables),
        "has_math": bool(math_sections),
        "math_sections": math_sections,
        "math_note": ("检测到 TeX 数学($…$/$$…$$)——DOCX 无公式引擎,已保留原文"
                      "并标记待人工核对" if math_sections else "未检测到需转换的公式"),
        "required_sections": required,
        "present_required": present,
        "missing_required": [r for r in required if r not in present],
        "structure_ok": len(sections) > 0 and bool(title),
    }


def _write_docx(docx: Any, title: str, sections: list[dict[str, str]],
                out: Any, stem: str) -> str:
    """Write a real .docx with conservative styling (title + heading/body sections)."""
    document = docx.Document()
    document.add_heading(title, level=0)
    for s in sections:
        if s.get("heading"):
            document.add_heading(s["heading"], level=1)
        body = s.get("body", "")
        for para in body.split("\n\n"):
            para = para.strip()
            if para:
                document.add_paragraph(para)
    path = _doc_out_path(out, stem, ".docx")
    document.save(path)
    return path


def _write_markdown_fallback(title: str, sections: list[dict[str, str]],
                             out: Any, stem: str) -> str:
    """Emit the manuscript as Markdown when python-docx is unavailable (no content lost)."""
    parts = [f"# {title}", ""]
    for s in sections:
        if s.get("heading"):
            parts.append(f"## {s['heading']}")
            parts.append("")
        if s.get("body"):
            parts.append(s["body"])
            parts.append("")
    path = _doc_out_path(out, stem, ".md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path


def _doc_out_path(out: Any, stem: str, ext: str) -> str:
    """Resolve a document output path with the given extension (mirrors ``_out_path``)."""
    if out:
        out = os.path.expanduser(str(out))
        if os.path.isdir(out) or out.endswith(os.sep):
            out = os.path.join(out, f"{stem}{ext}")
        elif not out.lower().endswith(ext):
            out = os.path.splitext(out)[0] + ext
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return out
    fd, path = tempfile.mkstemp(prefix=f"sv_{stem}_", suffix=ext)
    os.close(fd)
    return path


# --------------------------------------------------------------------- provenance
def _register_figure(state: StudyState, key: str, path: str, *, note: str = "") -> None:
    """Attach a produced figure path (and metadata) to ``artifacts.figures[key]``.

    ``artifacts['figures']`` is a ``{key: {'path', 'dpi', 'note'}}`` registry so a
    study can carry many figures without clobbering; the raw path is also kept at
    the top level (``figures[key + '_path']``) for the simplest consumers.
    """
    figures = state.artifacts.get("figures")
    if not isinstance(figures, dict):
        figures = {}
    figures[key] = {"path": path, "dpi": _DPI, "note": note}
    figures[f"{key}_path"] = path
    state.write("artifacts", "figures", figures)


__all__ = [
    "forest",
    "event_study_plot",
    "survey_dist",
    "theme_map",
    "manuscript_docx",
]
