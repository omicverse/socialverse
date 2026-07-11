"""``sv.pp._meta_es2`` — Tier-2 effect-size converters & aggregation.

Extends the ingest layer: recover standardized effects from reported test
statistics (t / F / χ² / p), single-arm and ratio-of-means continuous effects,
incidence rates, Cohen's h / point-biserial, and collapse dependent effects to
one synthetic effect per study. All exact closed-form (compute.es / esc / MAd
family). Feeds the same ``models['meta_effects']`` frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._meta_es import _resolve_df, _col, _passthrough, _store


def _reg(name, aliases, desc):
    return register(name=name, aliases=aliases, category="social_science_quant",
                    tier="pro", skill="meta-analysis", languages=["Python"],
                    key_tools=["numpy", "scipy"], description=desc,
                    requires={"sources": ["datasets"]}, produces={"models": ["meta_effects"]})


# ---------------------------------------------------------- from test statistics
@_reg("es_from_t", ["从t值取效应量", "t_to_d"],
      "两组 t 值 + n1/n2 → Cohen's d + 抽样方差(实验/试验只报了 t 时)")
def es_from_t(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-group t statistic + n1,n2 → Cohen's d. d = t·√(1/n1+1/n2)."""
    df = _resolve_df(state, kwargs)
    t = _col(df, kwargs.get("t")); n1 = _col(df, kwargs.get("n1")); n2 = _col(df, kwargs.get("n2"))
    if t is None or n1 is None or n2 is None:
        return state
    t, n1, n2 = t.to_numpy(float), n1.to_numpy(float), n2.to_numpy(float)
    d = t * np.sqrt(1 / n1 + 1 / n2)
    vi = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    return _store(state, _passthrough(df, kwargs, d, vi, "SMD"), kwargs.get("append", True))


@_reg("es_from_f", ["从F值取效应量", "f_to_d"],
      "单自由度 F(1,·) + n1/n2 → Cohen's d(F=t²;方向须由 sign= 提供)")
def es_from_f(state: StudyState, **kwargs: Any) -> StudyState:
    """One-df F + n1,n2 → Cohen's d (F = t²; sign column/kwarg gives direction)."""
    df = _resolve_df(state, kwargs)
    F = _col(df, kwargs.get("f") or kwargs.get("F")); n1 = _col(df, kwargs.get("n1")); n2 = _col(df, kwargs.get("n2"))
    if F is None or n1 is None or n2 is None:
        return state
    F, n1, n2 = F.to_numpy(float), n1.to_numpy(float), n2.to_numpy(float)
    sc = _col(df, kwargs.get("sign"))
    sign = sc.to_numpy(float) if sc is not None else float(kwargs.get("sign", 1))
    d = np.sign(sign) * np.sqrt(F) * np.sqrt(1 / n1 + 1 / n2)
    vi = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    return _store(state, _passthrough(df, kwargs, d, vi, "SMD"), kwargs.get("append", True))


@_reg("es_from_chisq", ["从卡方取效应量", "chisq_to_r"],
      "单自由度 χ²(1) + N → 相关 r(Fisher z);2×2 关联强度")
def es_from_chisq(state: StudyState, **kwargs: Any) -> StudyState:
    """One-df χ² + total N → r = √(χ²/N) then Fisher z."""
    df = _resolve_df(state, kwargs)
    x2 = _col(df, kwargs.get("chisq") or kwargs.get("x2")); N = _col(df, kwargs.get("n"))
    if x2 is None or N is None:
        return state
    x2, N = x2.to_numpy(float), N.to_numpy(float)
    r = np.clip(np.sqrt(x2 / N), -0.999, 0.999)
    sc = _col(df, kwargs.get("sign"))
    if sc is not None:
        r = np.sign(sc.to_numpy(float)) * r
    yi = np.arctanh(r); vi = 1.0 / (N - 3)
    return _store(state, _passthrough(df, kwargs, yi, vi, "ZCOR"), kwargs.get("append", True))


@_reg("es_from_p", ["从p值取效应量", "p_to_z"],
      "双侧 p 值 + N → 相关 r(Fisher z)(经 z=Φ⁻¹(1-p/2);方向须由 sign= 提供)")
def es_from_p(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-sided p + N → r via z = Φ⁻¹(1−p/2), r = z/√N, then Fisher z."""
    from scipy import stats
    df = _resolve_df(state, kwargs)
    p = _col(df, kwargs.get("p")); N = _col(df, kwargs.get("n"))
    if p is None or N is None:
        return state
    p, N = p.to_numpy(float), N.to_numpy(float)
    z = stats.norm.isf(np.clip(p, 1e-12, 1) / 2)
    sc = _col(df, kwargs.get("sign"))
    if sc is not None:
        z = np.sign(sc.to_numpy(float)) * z
    r = np.clip(z / np.sqrt(N), -0.999, 0.999)
    yi = np.arctanh(r); vi = 1.0 / (N - 3)
    return _store(state, _passthrough(df, kwargs, yi, vi, "ZCOR"), kwargs.get("append", True))


# ---------------------------------------------------------- continuous families
@_reg("es_single_mean", ["单臂均值效应量", "single_mean"],
      "单臂连续量→均值 MN(vi=sd²/n)或对数均值 MNLN;单组结局合并")
def es_single_mean(state: StudyState, **kwargs: Any) -> StudyState:
    """Single-arm mean → MN (raw, vi=sd²/n) or MNLN (log mean, delta-method)."""
    df = _resolve_df(state, kwargs)
    measure = str(kwargs.get("measure", "MN")).upper()
    m = _col(df, kwargs.get("mean") or kwargs.get("m")); sd = _col(df, kwargs.get("sd")); n = _col(df, kwargs.get("n"))
    if m is None or sd is None or n is None:
        return state
    m, sd, n = m.to_numpy(float), sd.to_numpy(float), n.to_numpy(float)
    if measure == "MNLN":
        yi = np.log(m); vi = sd ** 2 / (n * m ** 2)
    else:
        yi = m; vi = sd ** 2 / n; measure = "MN"
    return _store(state, _passthrough(df, kwargs, yi, vi, measure), kwargs.get("append", True))


@_reg("es_ratio_of_means", ["均值比", "rom", "lnRR"],
      "两组均值比 ROM/lnRR = ln(m1/m2) + Nakagawa 抽样方差(连续结局的比值度量)")
def es_ratio_of_means(state: StudyState, **kwargs: Any) -> StudyState:
    """Log ratio-of-means ROM = ln(m1/m2), vi = sd1²/(n1·m1²) + sd2²/(n2·m2²)."""
    df = _resolve_df(state, kwargs)
    cols = {k: _col(df, kwargs.get(k)) for k in ("m1", "sd1", "n1", "m2", "sd2", "n2")}
    if any(v is None for v in cols.values()):
        return state
    m1, sd1, n1, m2, sd2, n2 = (cols[k].to_numpy(float) for k in ("m1", "sd1", "n1", "m2", "sd2", "n2"))
    yi = np.log(m1 / m2)
    vi = sd1 ** 2 / (n1 * m1 ** 2) + sd2 ** 2 / (n2 * m2 ** 2)
    return _store(state, _passthrough(df, kwargs, yi, vi, "ROM"), kwargs.get("append", True))


@_reg("es_from_ir", ["发病率效应量", "incidence_rate"],
      "事件数/人时→对数发病率 IRLN(vi=1/events)或两组对数率比 IRR;计数结局")
def es_from_ir(state: StudyState, **kwargs: Any) -> StudyState:
    """Incidence rate → IRLN (log rate, vi=1/events) or IRR (log rate ratio, 2-group)."""
    df = _resolve_df(state, kwargs)
    measure = str(kwargs.get("measure", "IRLN")).upper()
    x1 = _col(df, kwargs.get("events1") or kwargs.get("events") or kwargs.get("x1"))
    t1 = _col(df, kwargs.get("time1") or kwargs.get("time") or kwargs.get("t1"))
    if x1 is None or t1 is None:
        return state
    x1, t1 = x1.to_numpy(float), t1.to_numpy(float)
    if measure == "IRR":
        x2 = _col(df, kwargs.get("events2") or kwargs.get("x2")); t2 = _col(df, kwargs.get("time2") or kwargs.get("t2"))
        if x2 is None or t2 is None:
            return state
        x2, t2 = x2.to_numpy(float), t2.to_numpy(float)
        zero = (x1 == 0) | (x2 == 0)
        x1c, x2c = x1 + 0.5 * zero, x2 + 0.5 * zero
        yi = np.log((x1c / t1) / (x2c / t2)); vi = 1 / x1c + 1 / x2c
    else:
        xc = np.where(x1 == 0, 0.5, x1)
        yi = np.log(xc / t1); vi = 1.0 / xc; measure = "IRLN"
    return _store(state, _passthrough(df, kwargs, yi, vi, measure), kwargs.get("append", True))


# ---------------------------------------------------------- binary / correlation
@_reg("cohens_h", ["cohen_h", "比例反正弦差"],
      "两比例→Cohen's h = 2·asin√p1 − 2·asin√p2(方差稳定的比例差效应量)")
def cohens_h(state: StudyState, **kwargs: Any) -> StudyState:
    """Two proportions → Cohen's h (arcsine difference). vi ≈ 1/n1 + 1/n2."""
    df = _resolve_df(state, kwargs)
    p1 = _col(df, kwargs.get("p1")); p2 = _col(df, kwargs.get("p2"))
    n1 = _col(df, kwargs.get("n1")); n2 = _col(df, kwargs.get("n2"))
    if p1 is None or p2 is None:
        return state
    p1, p2 = p1.to_numpy(float), p2.to_numpy(float)
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    if n1 is not None and n2 is not None:
        vi = 1 / n1.to_numpy(float) + 1 / n2.to_numpy(float)
    else:
        vi = np.full_like(h, np.nan)
    return _store(state, _passthrough(df, kwargs, h, vi, "COHENH"), kwargs.get("append", True))


@_reg("pointbiserial_to_d", ["点二列转d", "rpb_to_d"],
      "点二列相关 r_pb + n1/n2 → Cohen's d(二分-连续关联转标准化均差)")
def pointbiserial_to_d(state: StudyState, **kwargs: Any) -> StudyState:
    """Point-biserial r + n1,n2 → Cohen's d.  d = r·√(1/(h·(1−r²))), h = n1n2/(n1+n2)²·(N)."""
    df = _resolve_df(state, kwargs)
    r = _col(df, kwargs.get("r") or kwargs.get("rpb")); n1 = _col(df, kwargs.get("n1")); n2 = _col(df, kwargs.get("n2"))
    if r is None or n1 is None or n2 is None:
        return state
    r, n1, n2 = r.to_numpy(float), n1.to_numpy(float), n2.to_numpy(float)
    # d = r·√a / √(1−r²),  a = (n1+n2)²/(n1·n2)  (=4 for equal groups → d = 2r/√(1−r²))
    a = (n1 + n2) ** 2 / (n1 * n2)
    d = np.sqrt(a) * r / np.sqrt(1 - r ** 2)
    vi = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    return _store(state, _passthrough(df, kwargs, d, vi, "SMD"), kwargs.get("append", True))


# ---------------------------------------------------------- POMP severity
@_reg("es_pomp", ["POMP分数", "pomp", "percent_of_max"],
      "POMP 分数(percent of maximum possible):把任意量表的均值±SD 线性重标定到 0–100,供严重程度 meta 跨量表合并((mean−min)/(max−min)×100)")
def es_pomp(state: StudyState, **kwargs: Any) -> StudyState:
    """Percent-of-maximum-possible severity effect size (Dreisoerner et al. 2026).

    kwargs: ``mean``,``sd``,``n`` columns + scale bounds — either per-row columns
    ``smin=``/``smax=`` or scalars ``min_val=``/``max_val=``. Produces yi = POMP mean
    = (mean−min)/(max−min)·100, vi = sd_pomp²/n. ``reverse=True`` for positive-
    wellbeing scales (higher = better → reverse-coded so higher POMP = worse)."""
    df = _resolve_df(state, kwargs)
    m = _col(df, kwargs.get("mean") or kwargs.get("m")); sd = _col(df, kwargs.get("sd")); n = _col(df, kwargs.get("n"))
    if m is None or sd is None or n is None:
        return state
    m, sd, n = m.to_numpy(float), sd.to_numpy(float), n.to_numpy(float)
    lo_c = _col(df, kwargs.get("smin")); hi_c = _col(df, kwargs.get("smax"))
    lo = lo_c.to_numpy(float) if lo_c is not None else float(kwargs.get("min_val", 0.0))
    hi = hi_c.to_numpy(float) if hi_c is not None else float(kwargs.get("max_val", 100.0))
    rng = np.asarray(hi, float) - np.asarray(lo, float)
    pomp = (m - lo) / rng * 100.0
    sd_pomp = sd / rng * 100.0
    vi = sd_pomp ** 2 / n
    if kwargs.get("reverse", False):
        pomp = 100.0 - pomp
    return _store(state, _passthrough(df, kwargs, pomp, vi, "POMP"), kwargs.get("append", True))


# ---------------------------------------------------------- dependency collapse
@register(
    name="ma_aggregate",
    aliases=["聚合相依效应量", "aggregate_effects", "composite_effect"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="把每研究的多个相依效应量按假定相关 rho 合成为一个复合效应量(Borenstein 复合;多层模型的简化替代)",
    requires={"models": ["meta_effects"]},
    produces={"models": ["meta_effects"]},
)
def ma_aggregate(state: StudyState, **kwargs: Any) -> StudyState:
    """Collapse each study's dependent effects into one composite (Borenstein).

    kwargs: ``cluster=`` study id column (default ``'study'``), ``rho=0.6`` assumed
    within-study correlation. Composite mean = mean(yᵢ); composite var =
    (1/m²)(Σvᵢ + Σ_{i≠j} ρ√(vᵢvⱼ)). A simpler alternative to full ``rma_mv``.
    """
    eff = state.models.get("meta_effects")
    if not isinstance(eff, pd.DataFrame) or not len(eff):
        return state
    clus = kwargs.get("cluster") or ("study" if "study" in eff.columns else None)
    rho = float(kwargs.get("rho", 0.6))
    if clus is None:
        return state
    out = []
    for g, sub in eff.groupby(clus, sort=False):
        yi = sub["yi"].to_numpy(float); vi = sub["vi"].to_numpy(float); m = len(sub)
        se = np.sqrt(vi)
        cross = rho * (se[:, None] * se[None, :]); np.fill_diagonal(cross, vi)
        comp_v = cross.sum() / m ** 2
        row = {"yi": float(yi.mean()), "vi": float(comp_v), "sei": float(np.sqrt(comp_v)),
               "measure": sub["measure"].iloc[0] if "measure" in sub else "GEN", "study": g}
        out.append(row)
    agg = pd.DataFrame(out)
    state.write("models", "meta_effects", agg)
    return state
