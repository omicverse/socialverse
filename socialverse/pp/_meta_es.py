"""``sv.pp._meta_es`` — effect-size preparation for meta-analysis.

The ingest layer of the meta-analysis module: turn each study's raw summary
statistics into a tidy ``(yi, vi)`` effect estimate + sampling variance, the
universal input every pooler consumes. Mirrors ``metafor::escalc`` / the
``compute.es`` / ``esc`` family — all **exact closed-form** transforms (no
iteration, no R).

Effect-size table is stored in ``state.models['meta_effects']`` — a DataFrame
with columns ``yi, vi, sei, measure`` plus any passed-through study id /
cluster id / moderators. Downstream ``sv.tl.meta_*`` read it from there.

Honesty note: proportion transforms are exact; the **Freeman-Tukey double
arcsine (PFT)** back-transform is numerically fragile (Schwarzer 2019), so the
default proportion measure is **PLO (logit)**.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState


# ------------------------------------------------------------------ helpers
def _resolve_df(state: StudyState, kwargs: dict) -> pd.DataFrame | None:
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    return df if isinstance(df, pd.DataFrame) else None


def _col(df, name):
    """A numeric Series for column `name`, or None."""
    if name is None or df is None or name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce")


def _passthrough(df, kwargs, yi, vi, measure):
    """Assemble the tidy effects frame, carrying study id / cluster / moderators."""
    out = pd.DataFrame({"yi": np.asarray(yi, float), "vi": np.asarray(vi, float)})
    out["sei"] = np.sqrt(out["vi"])
    out["measure"] = measure
    for role in ("study", "cluster", "slab"):
        c = kwargs.get(role)
        if c and df is not None and c in df.columns:
            out[role] = df[c].to_numpy()
    mods = kwargs.get("moderators") or kwargs.get("mods")
    if isinstance(mods, str):
        mods = [mods]
    for m in mods or []:
        if df is not None and m in df.columns:
            out[m] = df[m].to_numpy()
    return out


def _store(state: StudyState, eff: pd.DataFrame, append: bool):
    prev = state.models.get("meta_effects")
    if append and isinstance(prev, pd.DataFrame) and len(prev):
        eff = pd.concat([prev, eff], ignore_index=True)
    # drop rows that failed to compute
    eff = eff[np.isfinite(eff["yi"]) & np.isfinite(eff["vi"]) & (eff["vi"] > 0)].reset_index(drop=True)
    state.write("models", "meta_effects", eff)
    return state


# ==================================================================== es_proportion
@register(
    name="es_proportion",
    aliases=["患病率效应量", "prop_es", "escalc_prop", "prevalence_es"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="单组比例→效应量:PR(原始)/PLN(对数)/PLO(logit,默认)/PAS(反正弦)/PFT(双反正弦)+ 抽样方差,患病率 meta 分析的 ingest",
    requires={"sources": ["datasets"]},
    produces={"models": ["meta_effects"]},
)
def es_proportion(state: StudyState, **kwargs: Any) -> StudyState:
    """Single-group proportion → transformed effect size + sampling variance.

    Columns via kwargs: ``cases=`` (events) and ``n=`` (total), OR ``proportion=``
    + ``n=``. ``measure=`` one of ``PLO`` (logit, default), ``PAS`` (arcsine),
    ``PR`` (raw), ``PLN`` (log), ``PFT`` (Freeman-Tukey double arcsine).
    ``append=True`` (default) stacks onto any existing effects.
    """
    df = _resolve_df(state, kwargs)
    measure = str(kwargs.get("measure", "PLO")).upper()
    n = _col(df, kwargs.get("n"))
    xi = _col(df, kwargs.get("cases") or kwargs.get("events") or kwargs.get("xi"))
    p = _col(df, kwargs.get("proportion") or kwargs.get("p"))
    if n is None or (xi is None and p is None):
        state.write("models", "meta_effects", state.models.get("meta_effects"))
        return state
    n = n.to_numpy(float)
    if xi is not None:
        xi = xi.to_numpy(float)
        p = xi / n
    else:
        p = p.to_numpy(float)
        xi = p * n
    with np.errstate(divide="ignore", invalid="ignore"):
        if measure == "PR":
            yi, vi = p, p * (1 - p) / n
        elif measure == "PLN":
            yi, vi = np.log(p), (1 - p) / (p * n)
        elif measure == "PLO":  # logit — default, most stable
            yi = np.log(p / (1 - p))
            vi = 1.0 / (n * p) + 1.0 / (n * (1 - p))
        elif measure == "PAS":  # arcsine
            yi, vi = np.arcsin(np.sqrt(p)), 1.0 / (4 * n)
        elif measure == "PFT":  # Freeman-Tukey double arcsine
            yi = 0.5 * (np.arcsin(np.sqrt(xi / (n + 1))) + np.arcsin(np.sqrt((xi + 1) / (n + 1))))
            vi = 1.0 / (4 * n + 2)
        else:
            raise ValueError(f"unknown proportion measure {measure!r}")
    eff = _passthrough(df, kwargs, yi, vi, measure)
    return _store(state, eff, kwargs.get("append", True))


# ==================================================================== es_from_means
@register(
    name="es_from_means",
    aliases=["标准化均差", "smd", "cohen_d", "escalc_smd"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="两组均值/标准差→Cohen's d(SMD)+ 抽样方差;连续结局试验/实验的 ingest",
    requires={"sources": ["datasets"]},
    produces={"models": ["meta_effects"]},
)
def es_from_means(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-group means + SDs → Cohen's d (SMD) + sampling variance.

    kwargs columns: ``m1,sd1,n1,m2,sd2,n2``. Set ``hedges=True`` to apply the
    small-sample J correction (Hedges g). Uses the pooled within-group SD.
    """
    df = _resolve_df(state, kwargs)
    cols = {k: _col(df, kwargs.get(k)) for k in ("m1", "sd1", "n1", "m2", "sd2", "n2")}
    if any(v is None for v in cols.values()):
        return state
    m1, sd1, n1, m2, sd2, n2 = (cols[k].to_numpy(float) for k in ("m1", "sd1", "n1", "m2", "sd2", "n2"))
    sp = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp
    vi = (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))
    measure = "SMD"
    if kwargs.get("hedges", False):
        df_ = n1 + n2 - 2
        J = 1.0 - 3.0 / (4 * df_ - 1)
        d, vi, measure = J * d, J**2 * vi, "SMDH"
    eff = _passthrough(df, kwargs, d, vi, measure)
    return _store(state, eff, kwargs.get("append", True))


# ==================================================================== hedges_correct
@register(
    name="hedges_correct",
    aliases=["hedges_g", "小样本校正", "es_hedges"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="对已算的 SMD 施加 Hedges J 小样本校正→Hedges g(需 df 或 n1/n2)",
    requires={"models": ["meta_effects"]},
    produces={"models": ["meta_effects"]},
)
def hedges_correct(state: StudyState, **kwargs: Any) -> StudyState:
    """Apply the Hedges J small-sample correction to the stored SMD effects.

    Uses ``df`` from a ``df`` column if present, else ``n1+n2-2`` if those
    columns exist on the effects frame; otherwise a no-op with a note.
    """
    eff = state.models.get("meta_effects")
    if not isinstance(eff, pd.DataFrame) or "yi" not in eff:
        return state
    eff = eff.copy()
    if "df" in eff:
        dof = pd.to_numeric(eff["df"], errors="coerce").to_numpy(float)
    elif {"n1", "n2"}.issubset(eff.columns):
        dof = pd.to_numeric(eff["n1"], errors="coerce").to_numpy(float) + \
              pd.to_numeric(eff["n2"], errors="coerce").to_numpy(float) - 2
    else:
        return state
    J = 1.0 - 3.0 / (4 * dof - 1)
    eff["yi"] = eff["yi"].to_numpy(float) * J
    eff["vi"] = eff["vi"].to_numpy(float) * J**2
    eff["sei"] = np.sqrt(eff["vi"])
    eff["measure"] = "SMDH"
    state.write("models", "meta_effects", eff)
    return state


# ==================================================================== es_from_2x2
@register(
    name="es_from_2x2",
    aliases=["列联表效应量", "or_es", "escalc_2x2", "log_or"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="2×2 列联表→logOR/logRR/RD + 抽样方差(0 格自动连续性校正);二分类结局的 ingest",
    requires={"sources": ["datasets"]},
    produces={"models": ["meta_effects"]},
)
def es_from_2x2(state: StudyState, **kwargs: Any) -> StudyState:
    """2×2 table → log odds ratio / log risk ratio / risk difference + variance.

    kwargs columns: ``ai,bi,ci,di`` (or ``ai,n1i,ci,n2i``). ``measure=`` one of
    ``OR`` (log, default), ``RR`` (log), ``RD``. Adds 0.5 to cells of any row
    with a zero cell (continuity correction; log measures only).
    """
    df = _resolve_df(state, kwargs)
    measure = str(kwargs.get("measure", "OR")).upper()
    ai = _col(df, kwargs.get("ai")); bi = _col(df, kwargs.get("bi"))
    ci = _col(df, kwargs.get("ci")); di = _col(df, kwargs.get("di"))
    if bi is None and kwargs.get("n1i"):
        n1i = _col(df, kwargs.get("n1i")); bi = n1i - ai if (n1i is not None and ai is not None) else None
    if di is None and kwargs.get("n2i"):
        n2i = _col(df, kwargs.get("n2i")); di = n2i - ci if (n2i is not None and ci is not None) else None
    if any(v is None for v in (ai, bi, ci, di)):
        return state
    a, b, c, d = (v.to_numpy(float) for v in (ai, bi, ci, di))
    if measure in ("OR", "RR"):
        zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        a, b, c, d = (x + 0.5 * zero for x in (a, b, c, d))
    with np.errstate(divide="ignore", invalid="ignore"):
        if measure == "OR":
            yi = np.log((a * d) / (b * c)); vi = 1/a + 1/b + 1/c + 1/d
        elif measure == "RR":
            yi = np.log((a / (a + b)) / (c / (c + d)))
            vi = 1/a - 1/(a + b) + 1/c - 1/(c + d)
        elif measure == "RD":
            p1, p2 = a / (a + b), c / (c + d)
            yi = p1 - p2
            vi = p1 * (1 - p1) / (a + b) + p2 * (1 - p2) / (c + d)
        else:
            raise ValueError(f"unknown 2x2 measure {measure!r}")
    eff = _passthrough(df, kwargs, yi, vi, measure)
    return _store(state, eff, kwargs.get("append", True))


# ==================================================================== es_from_r
@register(
    name="es_from_r",
    aliases=["相关系数效应量", "cor_es", "fisher_z", "escalc_r"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="相关系数 r→Fisher z(ZCOR,默认,方差稳定)或原始 COR + 抽样方差;相关研究的 ingest",
    requires={"sources": ["datasets"]},
    produces={"models": ["meta_effects"]},
)
def es_from_r(state: StudyState, **kwargs: Any) -> StudyState:
    """Correlation r → Fisher z (ZCOR, default) or raw COR + sampling variance.

    kwargs columns: ``r`` and ``n``. ``measure=`` ``ZCOR`` (default) or ``COR``.
    Pool on ZCOR then back-transform for interpretation (r = tanh(z)).
    """
    df = _resolve_df(state, kwargs)
    measure = str(kwargs.get("measure", "ZCOR")).upper()
    r = _col(df, kwargs.get("r")); n = _col(df, kwargs.get("n"))
    if r is None or n is None:
        return state
    r, n = r.to_numpy(float), n.to_numpy(float)
    if measure == "ZCOR":
        yi = np.arctanh(r); vi = 1.0 / (n - 3)
    elif measure == "COR":
        yi = r; vi = (1 - r**2) ** 2 / (n - 1)
    else:
        raise ValueError(f"unknown r measure {measure!r}")
    eff = _passthrough(df, kwargs, yi, vi, measure)
    return _store(state, eff, kwargs.get("append", True))


# ==================================================================== es_from_ci
@register(
    name="es_from_ci",
    aliases=["从置信区间取效应量", "es_generic", "escalc_ci"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description="已发表 估计值 + 置信区间/标准误→yi/vi(泛用 ingest;比值型须传对数尺度)",
    requires={"sources": ["datasets"]},
    produces={"models": ["meta_effects"]},
)
def es_from_ci(state: StudyState, **kwargs: Any) -> StudyState:
    """Published point estimate + CI (or SE) → generic ``(yi, vi)``.

    kwargs columns: ``est`` and either (``lower``,``upper``) or ``sei``.
    ``ci_level=0.95``. For ratio measures pass the estimate + bounds already on
    the **log scale** (``log_scale=True`` logs est/lower/upper for you).
    """
    from scipy import stats
    df = _resolve_df(state, kwargs)
    est = _col(df, kwargs.get("est") or kwargs.get("yi"))
    if est is None:
        return state
    est = est.to_numpy(float)
    sei_c = _col(df, kwargs.get("sei") or kwargs.get("se"))
    if sei_c is not None:
        sei = sei_c.to_numpy(float)
    else:
        lo = _col(df, kwargs.get("lower")); hi = _col(df, kwargs.get("upper"))
        if lo is None or hi is None:
            return state
        lo, hi = lo.to_numpy(float), hi.to_numpy(float)
        if kwargs.get("log_scale", False):
            est, lo, hi = np.log(est), np.log(lo), np.log(hi)
        z = stats.norm.ppf(1 - (1 - float(kwargs.get("ci_level", 0.95))) / 2)
        sei = (hi - lo) / (2 * z)
    eff = _passthrough(df, kwargs, est, sei**2, str(kwargs.get("measure", "GEN")))
    return _store(state, eff, kwargs.get("append", True))


# ==================================================================== escalc (dispatcher)
_ESCALC_DISPATCH = {
    "proportion": es_proportion, "PLO": es_proportion, "PAS": es_proportion, "PR": es_proportion,
    "PLN": es_proportion, "PFT": es_proportion,
    "means": es_from_means, "SMD": es_from_means, "smd": es_from_means,
    "2x2": es_from_2x2, "OR": es_from_2x2, "RR": es_from_2x2, "RD": es_from_2x2,
    "r": es_from_r, "ZCOR": es_from_r, "COR": es_from_r,
    "ci": es_from_ci, "generic": es_from_ci, "GEN": es_from_ci,
}


@register(
    name="escalc",
    aliases=["效应量计算器", "effect_size", "es_dispatch"],
    category="social_science_quant",
    tier="pro",
    skill="meta-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="效应量计算总入口:按 measure 路由到对应转换器,产出统一 (yi,vi,measure) 供 sv.tl.meta_* / rma_mv 使用",
    requires={"sources": ["datasets"]},
    produces={"models": ["meta_effects"]},
)
def escalc(state: StudyState, **kwargs: Any) -> StudyState:
    """Effect-size front door — route to the right converter by ``measure=``.

    ``measure=`` selects the family: ``PLO``/``PAS``/``PR``/``PLN``/``PFT``
    (proportion), ``SMD`` (means), ``OR``/``RR``/``RD`` (2×2), ``ZCOR``/``COR``
    (correlation), ``GEN`` (estimate+CI). All other kwargs pass through to the
    chosen converter. This is the ergonomic entry point; the family functions
    can also be called directly.
    """
    measure = str(kwargs.get("measure", "PLO"))
    fn = _ESCALC_DISPATCH.get(measure) or _ESCALC_DISPATCH.get(measure.upper())
    if fn is None:
        raise ValueError(
            f"unknown measure {measure!r}. one of: PLO/PAS/PR/PLN/PFT (proportion), "
            f"SMD (means), OR/RR/RD (2x2), ZCOR/COR (r), GEN (estimate+CI)"
        )
    return fn(state, **kwargs)
