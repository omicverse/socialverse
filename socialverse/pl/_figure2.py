"""``sv.pl._figure2`` — the *gap-method* figure primitives (plot / render phase).

This is the visual terminus for the quasi-experimental, survival, spatial,
synthetic-control and stylometric methods that the ``sv.tl`` gap modules fit. It
is the sibling of :mod:`socialverse.pl._figure`: same headless ``matplotlib``
(Agg) discipline, same ``artifacts.figures[key] = path`` provenance, same registry
contracts that keep rendering honest — you cannot draw an RD jump you never
estimated (``rdd_plot`` *requires* ``models.rdd``), a survival curve without a
Kaplan-Meier fit (``km_curve`` *requires* ``models.km``), a Moran scatter without
the autocorrelation diagnostic (``moran_scatter`` *requires* ``diagnostics.moran``),
etc.

Each of the five champion references is a real, widely-used package; here they are
matched with genuine ``matplotlib`` draws (no placeholders):

* ``rdd_plot``      — the ``rdrobust`` / ``rdplot`` (Calonico–Cattaneo–Titiunik)
  RD scatter: binned means of the outcome against the running variable, with two
  local-linear fits meeting at the cutoff and a vertical threshold line.
* ``km_curve``      — the ``lifelines`` / R ``survival::plot.survfit`` Kaplan–Meier
  step curve, one step function per group.
* ``moran_scatter`` — the ``esda.moran.Moran_Local`` / R ``moran.plot`` Moran
  scatterplot: standardized value ``z`` against its spatial lag ``Wz``, whose OLS
  slope *is* Moran's ``I``.
* ``synth_path``    — the ``SparseSC`` / R ``Synth::path.plot`` treated-vs-synthetic
  outcome paths with a treatment-onset line.
* ``dendrogram``    — the R ``stylo`` / ``scipy.cluster.hierarchy.dendrogram``
  Burrows's-Delta hierarchical clustering tree.

Every figure recomputes only what the ``matplotlib`` draw needs from the model /
diagnostic the contract guarantees (and, where the raw scatter is not carried in
the model, from ``sources.datasets``), and degrades gracefully — an empty-but-valid
PNG with an explanatory note — rather than raising, when a model is degenerate.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any

import numpy as np

# matplotlib is an optional backend — import it lazily so these figure functions
# still REGISTER on a bare install (numpy+pandas only). A missing backend surfaces
# only if a figure is actually drawn, with a clear install hint. The Agg backend is
# forced *before* pyplot is imported so nothing ever opens a window (headless safe).
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:  # pragma: no cover
    class _MissingMpl:
        def __getattr__(self, _name):
            raise ImportError(
                "matplotlib is required for socialverse figures — "
                "install it with: pip install 'socialverse[figure]'"
            )

    plt = _MissingMpl()  # type: ignore[assignment]

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
# whichever the host actually has and always fall back to DejaVu Sans.
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
        "font.sans-serif": _cjk_fonts() + ["DejaVu Sans"],
        "axes.unicode_minus": False,
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


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _empty_fig(state: StudyState, kwargs: dict[str, Any], key: str,
               msg: str, note: str) -> StudyState:
    """Render a valid-but-empty PNG carrying an explanatory message (never raise)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color="0.4")
    ax.set_axis_off()
    path = _save(fig, _out_path(kwargs, key))
    _register_figure(state, key, path, note=note)
    return state


def _resolve_frame(state: StudyState, kwargs: dict[str, Any]):
    """Resolve the working DataFrame from kwargs or ``sources.datasets``.

    Mirrors the ``sv.tl`` coercion: ``data=`` / ``df=`` kwarg wins; else
    ``sources['datasets']`` — which may be a bare DataFrame, a ``{name: df}``
    mapping, or a ``(df, W)`` tuple (spatial) — is unpacked to its first frame.
    Returns ``None`` if pandas is missing or no frame is available.
    """
    pd = _try_import("pandas")
    if pd is None:
        return None
    df = kwargs.get("data")
    if df is None:
        df = kwargs.get("df")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, tuple) and df and isinstance(df[0], pd.DataFrame):
        df = df[0]
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _resolve_W(state: StudyState, kwargs: dict[str, Any], n: int) -> np.ndarray | None:
    """Pull a row-normalized ``n x n`` weights matrix from kwargs or the datasets tuple."""
    W = kwargs.get("W")
    if W is None:
        ds = state.sources.get("datasets")
        if isinstance(ds, tuple) and len(ds) >= 2:
            W = ds[1]
    if W is None:
        return None
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] != n:
        return None
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


# ------------------------------------------------------------------------ rdd_plot
@register(
    name="rdd_plot",
    aliases=["断点回归图", "rd_plot"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib"],
    description="断点回归图:分箱散点 + 两侧局部线性拟合 + 断点竖线",
    requires={"models": ["rdd"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def rdd_plot(state: StudyState, **kwargs: Any) -> StudyState:
    """Draw the sharp-RDD figure: binned scatter + two-sided local-linear fits.

    Reads ``models['rdd']`` (the estimated ``jump``, side ``intercept``s and the
    ``cutoff`` / ``running`` / ``outcome`` names). The raw ``rdplot`` scatter is
    rebuilt from ``sources['datasets']`` — the running variable is partitioned into
    equal-count bins on each side of the cutoff and each bin's mean outcome is a
    marker — and the two local-linear fit lines are drawn to meet at the cutoff,
    where the gap between the right and left boundary intercepts is exactly the
    estimated RD jump. A dashed vertical line marks the threshold. The PNG path is
    stored at ``artifacts.figures['rdd']``.

    Champion reference: ``rdrobust::rdplot`` (Calonico–Cattaneo–Titiunik).
    """
    model = state.models.get("rdd") or {}
    if not isinstance(model, dict) or model.get("jump") is None:
        return _empty_fig(state, kwargs, "rdd",
                          "no RDD estimate to plot", "空:models.rdd 无断点估计")

    _apply_style()
    cutoff = _safe_float(model.get("cutoff", 0.0))
    running = str(kwargs.get("running") or model.get("running") or "running")
    outcome = str(kwargs.get("outcome") or model.get("outcome") or "y")

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.axvline(cutoff, color="#b03a2e", linestyle="--", linewidth=1.4, zorder=1,
               label="断点 (cutoff)")

    df = _resolve_frame(state, kwargs)
    drew_data = False
    if df is not None and running in df.columns and outcome in df.columns:
        pd = _try_import("pandas")
        r = pd.to_numeric(df[running], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[outcome], errors="coerce").to_numpy(float)
        ok = np.isfinite(r) & np.isfinite(y)
        r, y = r[ok], y[ok]
        if r.size >= 4:
            n_bins = int(kwargs.get("bins", 20))
            for side, color, mk in (("left", "#1f4e79", "o"),
                                     ("right", "#2e8b57", "s")):
                mask = r < cutoff if side == "left" else r >= cutoff
                rs, ys = r[mask], y[mask]
                if rs.size == 0:
                    continue
                bx, by = _bin_means(rs, ys, max(2, n_bins // 2))
                ax.scatter(bx, by, s=26, color=color, alpha=0.75, zorder=2,
                           edgecolors="white", linewidths=0.5,
                           label=f"分箱均值 ({'左' if side == 'left' else '右'})")
                # local-linear fit line on this side (least squares on raw points)
                slope, intercept = _ls_line(rs - cutoff, ys)
                xs = np.linspace(rs.min(), rs.max(), 50)
                ax.plot(xs, intercept + slope * (xs - cutoff), "-",
                        color=color, linewidth=1.8, zorder=3)
            drew_data = True

    if not drew_data:
        # No raw data reachable: draw the two boundary intercepts + jump annotation
        # straight from the model so the figure is still faithful to the estimate.
        li = _safe_float(model.get("left_intercept"))
        ri = _safe_float(model.get("right_intercept"))
        if np.isfinite(li):
            ax.plot([cutoff - 1, cutoff], [li, li], "-", color="#1f4e79",
                    linewidth=2.0, label="左侧边界截距")
            ax.scatter([cutoff], [li], color="#1f4e79", zorder=3)
        if np.isfinite(ri):
            ax.plot([cutoff, cutoff + 1], [ri, ri], "-", color="#2e8b57",
                    linewidth=2.0, label="右侧边界截距")
            ax.scatter([cutoff], [ri], color="#2e8b57", zorder=3)

    jump = _safe_float(model.get("jump"))
    ax.set_xlabel(f"running variable ({running})")
    ax.set_ylabel(f"outcome ({outcome})")
    ax.set_title(str(kwargs.get("title",
                                f"断点回归 · 跳跃 τ = {jump:.3g}")))
    ax.legend(loc="best", frameon=False, fontsize=_FONT - 2)

    path = _save(fig, _out_path(kwargs, "rdd"))
    _register_figure(state, "rdd", path,
                     note=f"RDD 断点图,τ={jump:.3g},cutoff={cutoff:g}")
    return state


def _bin_means(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count binning: return per-bin mean ``x`` and mean ``y`` (rdplot style)."""
    n = x.size
    n_bins = int(min(max(1, n_bins), n))
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    edges = np.array_split(np.arange(n), n_bins)
    bx = np.array([xs[idx].mean() for idx in edges if len(idx)])
    by = np.array([ys[idx].mean() for idx in edges if len(idx)])
    return bx, by


def _ls_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least-squares ``slope, intercept`` of ``y ~ x`` (finite-safe)."""
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.mean(y)) if y.size else 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


# ------------------------------------------------------------------------ km_curve
@register(
    name="km_curve",
    aliases=["生存曲线", "kaplan_meier_plot", "KM图"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib"],
    description="Kaplan-Meier 生存曲线(按组阶梯,含总体)",
    requires={"models": ["km"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def km_curve(state: StudyState, **kwargs: Any) -> StudyState:
    """Plot Kaplan–Meier survival curves as step functions, one per group.

    Reads ``models['km']`` — ``{'overall': {'times','surv'}, 'by_group':
    {g: {'times','surv'}}}`` as produced by ``sv.tl.survival``. Each group's
    survival function is drawn as a right-continuous step curve
    (``drawstyle='steps-post'``, the standard Kaplan–Meier presentation); the
    pooled overall curve is overlaid as a light dashed reference. The y-axis is
    fixed to ``[0, 1]``. The PNG path is stored at ``artifacts.figures['km']``.

    Champion reference: ``lifelines.KaplanMeierFitter.plot`` /
    R ``survival::plot.survfit``.
    """
    model = state.models.get("km") or {}
    by_group = model.get("by_group") if isinstance(model, dict) else None
    overall = model.get("overall") if isinstance(model, dict) else None

    curves = _km_series(by_group, overall)
    if not curves:
        return _empty_fig(state, kwargs, "km",
                          "no Kaplan-Meier curve to plot", "空:models.km 无生存曲线")

    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    cmap = plt.get_cmap("tab10")

    for i, (label, t, s) in enumerate(curves):
        # begin each curve at S(0)=1 so the step starts from the origin
        t_plot = np.concatenate([[0.0], t])
        s_plot = np.concatenate([[1.0], s])
        ax.step(t_plot, s_plot, where="post", linewidth=1.8, zorder=3,
                color=cmap(i % 10), label=f"组 {label}")

    if isinstance(overall, dict) and overall.get("times") and len(curves) > 1:
        ot = np.asarray(overall["times"], dtype=float)
        os_ = np.asarray(overall["surv"], dtype=float)
        ok = np.isfinite(ot) & np.isfinite(os_)
        if ok.any():
            ot, os_ = ot[ok], os_[ok]
            ax.step(np.concatenate([[0.0], ot]), np.concatenate([[1.0], os_]),
                    where="post", linewidth=1.2, linestyle="--", color="0.4",
                    alpha=0.8, zorder=2, label="总体")

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("时间 (time)")
    ax.set_ylabel("生存概率 S(t)")
    ax.set_title(str(kwargs.get("title", "Kaplan-Meier 生存曲线")))
    ax.legend(loc="best", frameon=False)

    path = _save(fig, _out_path(kwargs, "km"))
    _register_figure(state, "km", path,
                     note=f"{len(curves)} 条 KM 生存阶梯曲线")
    return state


def _km_series(by_group: Any, overall: Any) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Normalize KM curves into ``[(label, times, surv), ...]`` (finite-filtered)."""
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _one(label: str, d: Any) -> None:
        if not isinstance(d, dict):
            return
        t = np.asarray(d.get("times") or [], dtype=float)
        s = np.asarray(d.get("surv") or [], dtype=float)
        if t.size == 0 or s.size == 0 or t.size != s.size:
            return
        ok = np.isfinite(t) & np.isfinite(s)
        if ok.any():
            curves.append((label, t[ok], s[ok]))

    if isinstance(by_group, dict) and by_group:
        for g in sorted(by_group, key=lambda k: str(k)):
            _one(str(g), by_group[g])
    if not curves:
        _one("overall", overall)
    return curves


# --------------------------------------------------------------------- moran_scatter
@register(
    name="moran_scatter",
    aliases=["莫兰散点图", "moran_plot"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib"],
    description="Moran 散点图(z vs Wz),回归斜率 = Moran's I",
    requires={"diagnostics": ["moran"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def moran_scatter(state: StudyState, **kwargs: Any) -> StudyState:
    """Draw the Moran scatterplot: standardized value ``z`` vs its spatial lag ``Wz``.

    Reads ``diagnostics['moran']`` (the estimated global ``I`` and the variable
    name), rebuilds ``z`` (mean-centred outcome) and its row-standardized spatial
    lag ``Wz`` from ``sources['datasets']`` (the ``(df, W)`` tuple), and plots one
    point per spatial unit split into the four quadrants (HH / LL positive
    association, HL / LH negative). The OLS regression line through the cloud has
    slope equal to Moran's ``I`` — that line, and the reported ``I`` from the
    diagnostic, are both drawn. Reference lines at ``z = 0`` and ``Wz = 0`` mark
    the quadrant boundaries. The PNG path is stored at
    ``artifacts.figures['moran']``.

    Champion reference: ``esda`` + ``splot.esda.plot_moran`` / R ``moran.plot``.
    """
    diag = state.diagnostics.get("moran") or {}
    if not isinstance(diag, dict) or diag.get("I") is None:
        return _empty_fig(state, kwargs, "moran",
                          "no Moran's I diagnostic", "空:diagnostics.moran 无 I")

    I_reported = _safe_float(diag.get("I"))
    variable = str(kwargs.get("variable") or diag.get("variable") or "y")

    z, Wz = _moran_zwz(state, kwargs, variable)
    if z is None:
        # Data not reachable — draw a minimal reference figure from the reported I
        # so the artifact still reflects the estimate honestly.
        _apply_style()
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        xs = np.linspace(-2.5, 2.5, 50)
        ax.axhline(0, color="0.6", lw=1, zorder=0)
        ax.axvline(0, color="0.6", lw=1, zorder=0)
        ax.plot(xs, I_reported * xs, "-", color="#b03a2e", linewidth=1.8,
                label=f"斜率 = I = {I_reported:.3f}")
        ax.set_xlabel(f"z ({variable})")
        ax.set_ylabel("空间滞后 Wz")
        ax.set_title(str(kwargs.get("title", f"Moran 散点图 · I = {I_reported:.3f}")))
        ax.legend(loc="best", frameon=False)
        path = _save(fig, _out_path(kwargs, "moran"))
        _register_figure(state, "moran", path,
                         note=f"Moran 参考线(无原始数据),I={I_reported:.3f}")
        return state

    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.axhline(0.0, color="0.6", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="0.6", linewidth=1.0, zorder=0)

    # quadrant colouring: HH/LL (positive assoc) vs HL/LH (negative assoc)
    pos = (z > 0) == (Wz > 0)
    ax.scatter(z[pos], Wz[pos], s=22, color="#c0392b", alpha=0.7, zorder=2,
               edgecolors="white", linewidths=0.4, label="HH / LL")
    if (~pos).any():
        ax.scatter(z[~pos], Wz[~pos], s=22, color="#2874a6", alpha=0.7, zorder=2,
                   edgecolors="white", linewidths=0.4, label="HL / LH")

    # OLS fit line through (z, Wz): its slope is Moran's I
    slope, intercept = _ls_line(z, Wz)
    xs = np.linspace(float(z.min()), float(z.max()), 50)
    ax.plot(xs, intercept + slope * xs, "-", color="#1f4e79", linewidth=1.8,
            zorder=3, label=f"拟合斜率 = {slope:.3f} (≈ I)")

    ax.set_xlabel(f"标准化值 z ({variable})")
    ax.set_ylabel("空间滞后 Wz")
    ax.set_title(str(kwargs.get("title", f"Moran 散点图 · I = {I_reported:.3f}")))
    ax.legend(loc="best", frameon=False, fontsize=_FONT - 2)

    path = _save(fig, _out_path(kwargs, "moran"))
    _register_figure(state, "moran", path,
                     note=f"Moran 散点图,n={z.size},拟合斜率={slope:.3f},报告 I={I_reported:.3f}")
    return state


def _moran_zwz(state: StudyState, kwargs: dict[str, Any], variable: str):
    """Rebuild ``(z, Wz)`` from the spatial data + weights, or ``(None, None)``.

    ``z`` is the mean-centred outcome; ``Wz`` is the row-standardized spatial lag
    ``W z``. The variable / weights come from ``sources['datasets']`` (the
    ``(df, W)`` tuple), matching how ``sv.tl.spatial_autocorr`` computed ``I``.
    """
    df = _resolve_frame(state, kwargs)
    if df is None or variable not in df.columns:
        # try a numeric column fallback
        if df is not None:
            num = [c for c in df.columns
                   if df[c].dtype.kind in "if" and c not in ("id", "row", "col")]
            if num:
                variable = num[0]
            else:
                return None, None
        else:
            return None, None
    pd = _try_import("pandas")
    y = pd.to_numeric(df[variable], errors="coerce").to_numpy(float)
    ok = np.isfinite(y)
    y = y[ok]
    n = y.size
    if n < 3:
        return None, None
    W = _resolve_W(state, kwargs, n)
    if W is None:
        return None, None
    z = y - y.mean()
    sd = z.std(ddof=0)
    if sd > 0:
        z = z / sd  # standardize so the axes are comparable (moran.plot convention)
    Wz = W @ z
    return z, Wz


# ---------------------------------------------------------------------- synth_path
@register(
    name="synth_path",
    aliases=["合成控制路径图", "synthetic_path"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib"],
    description="合成控制路径图:treated vs synthetic + 处理时点竖线",
    requires={"models": ["synth"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def synth_path(state: StudyState, **kwargs: Any) -> StudyState:
    """Plot the treated vs synthetic outcome paths with a treatment-onset line.

    Reads ``models['synth']['path']`` — ``{'time','treated','synthetic','gap'}`` as
    produced by ``sv.tl.synthetic_control`` — and draws the treated unit's observed
    trajectory against its synthetic counterpart. A vertical line at ``treat_time``
    marks the treatment onset; the pre-period paths coincide (small pre-RMSE) and
    the post-period divergence is the estimated effect. The PNG path is stored at
    ``artifacts.figures['synth']``.

    Champion reference: R ``Synth::path.plot`` / ``SparseSC``.
    """
    model = state.models.get("synth") or {}
    path_d = model.get("path") if isinstance(model, dict) else None
    if not isinstance(path_d, dict) or not path_d.get("time"):
        return _empty_fig(state, kwargs, "synth",
                          "no synthetic-control path", "空:models.synth 无反事实路径")

    t = np.asarray(path_d.get("time"), dtype=float)
    treated = np.asarray(path_d.get("treated"), dtype=float)
    synthetic = np.asarray(path_d.get("synthetic"), dtype=float)
    ok = np.isfinite(t) & np.isfinite(treated) & np.isfinite(synthetic)
    if ok.sum() < 2:
        return _empty_fig(state, kwargs, "synth",
                          "degenerate synthetic-control path", "空:合成控制路径退化")
    t, treated, synthetic = t[ok], treated[ok], synthetic[ok]

    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    ax.plot(t, treated, "-o", color="#1f4e79", linewidth=1.8, markersize=4,
            zorder=3, label="处理单元 (treated)")
    ax.plot(t, synthetic, "--s", color="#b03a2e", linewidth=1.6, markersize=4,
            zorder=3, label="合成对照 (synthetic)")

    treat_time = model.get("treat_time")
    tt = _safe_float(treat_time)
    if np.isfinite(tt):
        ax.axvline(tt, color="0.4", linestyle=":", linewidth=1.4, zorder=1,
                   label="处理时点")

    att = _safe_float(model.get("att"))
    ax.set_xlabel("时间")
    ax.set_ylabel(f"结果 ({model.get('outcome', 'y')})")
    title = str(kwargs.get("title", "合成控制 · treated vs synthetic"))
    if np.isfinite(att):
        title += f"  (ATT = {att:.3g})"
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    path = _save(fig, _out_path(kwargs, "synth"))
    _register_figure(state, "synth", path,
                     note=f"合成控制路径图,ATT={att:.3g},T={t.size}")
    return state


# ----------------------------------------------------------------------- dendrogram
@register(
    name="dendrogram",
    aliases=["树状图", "文体聚类树", "cluster_tree"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=["matplotlib"],
    description="文体计量层次聚类树状图(Burrows's Delta average linkage)",
    requires={"models": ["stylometry"]},
    produces={"artifacts": ["figures"]},
    auto_fix="escalate",
)
def dendrogram(state: StudyState, **kwargs: Any) -> StudyState:
    """Draw the Burrows's-Delta hierarchical clustering tree as a dendrogram.

    Reads ``models['stylometry']`` — the scipy-format ``linkage`` matrix
    (``[idx_a, idx_b, dist, size]`` rows) and the ordered ``documents`` labels as
    produced by ``sv.tl.stylometry``. Uses
    ``scipy.cluster.hierarchy.dendrogram`` for the drawing coordinates when SciPy
    is present, and otherwise walks the linkage matrix with a self-contained
    U-join plotter (same tree, no dependency). The join heights are Delta
    distances, so documents by the same author fuse low and different authors fuse
    high. The PNG path is stored at ``artifacts.figures['dendrogram']``.

    Champion reference: R ``stylo`` dendrogram / ``scipy.cluster.hierarchy``.
    """
    model = state.models.get("stylometry") or {}
    Z = model.get("linkage") if isinstance(model, dict) else None
    labels = list(model.get("documents") or []) if isinstance(model, dict) else []

    Z = np.asarray(Z, dtype=float) if Z is not None else np.zeros((0, 4))
    if Z.ndim != 2 or Z.shape[0] == 0:
        return _empty_fig(state, kwargs, "dendrogram",
                          "no linkage to plot", "空:models.stylometry 无 linkage")

    n = Z.shape[0] + 1
    if not labels or len(labels) != n:
        labels = [str(i) for i in range(n)]

    _apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    sch = _try_import("scipy.cluster.hierarchy")
    backend = "numpy"
    if sch is not None:
        try:
            thr = 0.7 * float(Z[:, 2].max()) if Z[:, 2].max() > 0 else 0.0
            sch.dendrogram(Z, labels=labels, ax=ax, color_threshold=thr,
                           leaf_rotation=90)
            backend = "scipy"
        except Exception:
            _plot_dendrogram_numpy(ax, Z, labels)
    else:
        _plot_dendrogram_numpy(ax, Z, labels)

    ax.set_ylabel("Delta 距离")
    acc = model.get("accuracy")
    title = str(kwargs.get("title", "文体计量层次聚类 (Burrows's Delta)"))
    if isinstance(acc, (int, float)):
        title += f"  (归属准确率 = {float(acc):.0%})"
    ax.set_title(title)

    path = _save(fig, _out_path(kwargs, "dendrogram"))
    _register_figure(state, "dendrogram", path,
                     note=f"文体聚类树状图,{n} 文档,backend={backend}")
    return state


def _plot_dendrogram_numpy(ax: "plt.Axes", Z: np.ndarray, labels: list[str]) -> None:
    """Minimal dendrogram plotter for the no-scipy path (U-shaped joins, in-order).

    Mirrors ``sv.tl._stylometry._plot_dendrogram_numpy``: assign each leaf an
    in-order x-position, then draw the classic U join for every linkage row at its
    Delta-distance height.
    """
    n = len(labels)
    x_pos: dict[int, float] = {}
    height: dict[int, float] = {i: 0.0 for i in range(n)}

    def leaves(node: int) -> list[int]:
        if node < n:
            return [node]
        a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
        return leaves(a) + leaves(b)

    root = n + Z.shape[0] - 1
    order = leaves(root)
    for k, leaf in enumerate(order):
        x_pos[leaf] = float(k)

    def xcoord(node: int) -> float:
        if node in x_pos:
            return x_pos[node]
        a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
        x = (xcoord(a) + xcoord(b)) / 2.0
        x_pos[node] = x
        return x

    for m in range(Z.shape[0]):
        a, b = int(Z[m, 0]), int(Z[m, 1])
        h = float(Z[m, 2])
        node = n + m
        height[node] = h
        xa, xb = xcoord(a), xcoord(b)
        ha, hb = height[a], height[b]
        ax.plot([xa, xa, xb, xb], [ha, h, h, hb], color="#1f4e79", lw=1.4)

    ax.set_xticks(range(n))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=8)
    ax.set_xlim(-0.5, n - 0.5)


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
    "rdd_plot",
    "km_curve",
    "moran_scatter",
    "synth_path",
    "dendrogram",
]
