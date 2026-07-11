"""``sv.style`` — one-line publication plotting setup for socialverse.

The social-science analog of ``ov.style`` / ``ov.plot_set``: instead of the long
``matplotlib.rcParams.update({...})`` ritual (and the manual CJK-font dance every
notebook repeats), call ``sv.style()`` once at the top of an analysis and get a
consistent, publication-ready, **CJK-aware, vector-friendly** matplotlib style.

No OmicVerse, no scanpy — pure matplotlib. ``resolve_publication_font()`` picks a
real LOCAL font (Arial → Liberation/Nimbus Sans → DejaVu) so nothing tries to
download a font at runtime and silently degrade on offline / CN / server kernels.
"""
from __future__ import annotations

import os
from typing import Any

from ._registry import register

# Preferred sans-serif faces (metric-compatible order) and CJK fallbacks.
_SANS = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "Arimo", "DejaVu Sans")
_CJK = ("PingFang SC", "Hiragino Sans GB", "Songti SC", "STHeiti", "Noto Sans CJK SC",
        "Source Han Sans SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS")


def _installed_font_names():
    try:
        from matplotlib import font_manager as fm
        return {f.name for f in fm.fontManager.ttflist}
    except Exception:
        return set()


def resolve_publication_font(cjk: bool = True) -> str:
    """Return a real, locally-installed sans-serif family name for figures.

    Prefers Arial (or a metric-compatible open substitute like Liberation/Nimbus
    Sans) and falls back to DejaVu Sans. Never triggers a network font download.
    """
    have = _installed_font_names()
    for cand in _SANS:
        if cand in have:
            return cand
    return "DejaVu Sans"


@register(
    name="style",
    aliases=["plot_set", "绘图设置", "设置绘图", "样式", "ov_plot_set", "plotset", "sv_plot_set"],
    category="figure", tier="community", skill="social-science-figure",
    languages=["Python"], key_tools=["matplotlib"],
    description="一行式出版级绘图设置(ov.style 的社科版):设 matplotlib rcParams —— 本地字体解析 + CJK 中文标签 + 干净留白 + 矢量友好(SVG/PDF 可编辑文字);替代冗长的 rcParams 仪式",
    examples=["sv.style()", "sv.style(font_path=resolve_publication_font())", "sv.style(dpi=120, fontsize=13, cjk=True)"],
    related=["pl.meta_forest", "pl.funnel"],
)
def style(font_path: str | None = None, dpi: int = 100, dpi_save: int = 300,
          fontsize: int = 12, figsize: Any = None, facecolor: str = "white",
          cjk: bool = True, vector_friendly: bool = True, grid: bool = False,
          verbose: bool = True) -> None:
    """Apply socialverse's publication matplotlib style (global rcParams).

    Parameters
    ----------
    font_path : str or None
        A TTF/OTF file path, or a font family name. If ``None``, a good local
        sans-serif is resolved automatically (no network download).
    dpi, dpi_save : int
        On-screen and saved-figure DPI.
    fontsize : int
        Base font size (ticks/legend derive from it).
    figsize : int | (w, h) | None
        Default figure size; a scalar becomes a square.
    facecolor : str
        Figure/axes background.
    cjk : bool
        Prepend CJK fonts so Chinese/Japanese/Korean labels render.
    vector_friendly : bool
        Keep text as editable text in SVG/PDF (``svg.fonttype='none'``,
        ``pdf.fonttype=42``) so figures stay editable in Illustrator/Inkscape.
    grid : bool
        Show a light axes grid.
    """
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm
    except Exception:
        if verbose:
            print("sv.style: matplotlib 未安装(pip install 'socialverse[figure]');跳过。")
        return

    have = {f.name for f in fm.fontManager.ttflist}
    # resolve the primary sans-serif face
    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            primary = fm.FontProperties(fname=font_path).get_name()
        except Exception:
            primary = resolve_publication_font()
    elif font_path and font_path in have:
        primary = font_path
    else:
        primary = resolve_publication_font()

    sans = [primary]
    if cjk:
        sans += [c for c in _CJK if c in have]
    sans += ["DejaVu Sans"]

    rc = {
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": fontsize,
        "axes.unicode_minus": False,                       # keep ASCII minus + CJK safe
        "figure.dpi": dpi, "savefig.dpi": dpi_save, "savefig.bbox": "tight",
        "figure.facecolor": facecolor, "axes.facecolor": facecolor, "savefig.facecolor": facecolor,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": grid, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
        "axes.titlesize": fontsize + 1, "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1, "ytick.labelsize": fontsize - 1,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": fontsize - 1,
        "figure.titlesize": fontsize + 2, "figure.titleweight": "bold",
    }
    if figsize is not None:
        rc["figure.figsize"] = (figsize, figsize) if isinstance(figsize, (int, float)) else tuple(figsize)
    if vector_friendly:
        rc.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    mpl.rcParams.update(rc)

    if verbose:
        cjk_on = cjk and any(c in have for c in _CJK)
        print(f"socialverse style · font={primary} · dpi={dpi}/{dpi_save} · "
              f"CJK={'on' if cjk_on else 'off'} · vector-friendly={vector_friendly}")


# ov.plot_set-style alias so `sv.plot_set(...)` also works.
plot_set = style
