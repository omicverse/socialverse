"""``sv.tl._survey`` — registered implementations for the *survey* domain.

Two registry entries back the survey skills:

- :func:`design_survey` (``survey-design``) — the pre-collection step: turn
  constructs into scale items, estimate internal-consistency reliability
  (Cronbach's α) and the required sample size (normal-approximation power with a
  design-effect / DEFF inflation).
- :func:`survey_estimate` (``complex-survey-analysis``) — the post-collection
  step: a *design-based* weighted regression with cluster-robust (PSU) standard
  errors, plus an unweighted sensitivity contrast.

Both speak the 12-slot :class:`~socialverse._state.StudyState` vocabulary through
the ``@register`` contract, so the resolver can chain
``design_survey → (collect) → survey_estimate``: ``design_survey`` produces the
``design.sampling_frame`` / ``variables.scales`` that a study needs before
``survey_estimate`` consumes ``design.weights`` + ``variables.outcome``.

Heavy/optional deps (``pingouin``, ``factor_analyzer``) are imported lazily and
degraded gracefully — the reliability and power maths are done in plain NumPy so
the functions always return a real, deterministic result.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["design_survey", "survey_estimate",
           "survey_by", "survey_ratio", "survey_ciprop", "survey_crosstab"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import — returns the module or ``None`` if unavailable."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _as_frame(items: Any) -> pd.DataFrame | None:
    """Coerce a response matrix (DataFrame / ndarray / list) to a numeric frame."""
    if items is None:
        return None
    if isinstance(items, pd.DataFrame):
        df = items
    else:
        df = pd.DataFrame(np.asarray(items))
    return df.apply(pd.to_numeric, errors="coerce")


def _cronbach_alpha(matrix: pd.DataFrame) -> tuple[float, int]:
    """Cronbach's α = (k/(k-1)) · (1 − Σ item_var / total_var).

    Uses complete cases and sample variance (ddof=1). Returns ``(alpha, k)``.
    """
    data = matrix.dropna(axis=0, how="any")
    k = int(data.shape[1])
    if k < 2 or data.shape[0] < 2:
        return float("nan"), k
    item_var = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan"), k
    alpha = (k / (k - 1.0)) * (1.0 - float(item_var.sum()) / float(total_var))
    return float(alpha), k


def _z(p: float) -> float:
    """Standard-normal quantile via scipy if present, else a rational fallback."""
    stats = _try_import("scipy.stats")
    if stats is not None:
        return float(stats.norm.ppf(p))
    # Acklam-style rational approximation (deterministic, no deps).
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return float((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                     ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(-(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                     ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return float((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def _resolve_dataset(state: StudyState, kwargs: dict) -> pd.DataFrame:
    """Resolve the modelling frame like :func:`survey_estimate` does.

    Accepts a ``data`` override, then ``state.sources['datasets']`` which may be a
    DataFrame, a dict of frames, or any DataFrame-coercible mapping.
    """
    data = kwargs.get("data")
    if data is None:
        data = state.sources.get("datasets")
    if isinstance(data, dict):
        data = next(iter(data.values()), None)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data) if data is not None else pd.DataFrame()
    return data


def _design_from_state(state: StudyState, data: pd.DataFrame, used_cols: list):
    """Build an ``external.pysurvey`` ``SurveyDesign`` from the ``design`` slots.

    Reads ``design['weights'/'strata'|'stratum'/'psu'/'fpc']`` and slices grouping
    ids from the ORIGINAL (un-coerced) frame by row index so string stratum/PSU
    labels are preserved. Returns ``(design, weights_array, model_index)``.
    """
    from ..external.pysurvey import svydesign  # parity-gated port

    weight_col = state.design.get("weights")
    psu_col = state.design.get("psu")
    stratum_col = state.design.get("strata") or state.design.get("stratum")
    fpc_col = state.design.get("fpc")

    keep = [c for c in used_cols if c and c in data.columns]
    num = data[keep].apply(pd.to_numeric, errors="coerce") if keep else data.copy()
    subset = [c for c in used_cols if c and c in num.columns]
    num = num.dropna(axis=0, subset=subset) if subset else num
    idx = num.index

    if weight_col and weight_col in data.columns:
        w = pd.to_numeric(data.loc[idx, weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    else:
        w = np.ones(len(idx), dtype=float)

    strata = data.loc[idx, stratum_col].to_numpy() if (stratum_col and stratum_col in data.columns) else None
    psu = (data.loc[idx, psu_col].to_numpy()
           if (psu_col and psu_col in data.columns and data.loc[idx, psu_col].notna().any()) else None)
    fpc = (pd.to_numeric(data.loc[idx, fpc_col], errors="coerce").to_numpy(float)
           if (fpc_col and fpc_col in data.columns) else None)

    des = svydesign({}, weights=w, ids=psu, strata=strata, fpc=fpc)
    return des, w, idx


def _sample_size(effect_size: float, deff: float, alpha: float, power: float) -> int:
    """Normal-approximation n ≈ ((z_α/2 + z_β) / es)² · DEFF, rounded up."""
    es = abs(float(effect_size))
    if es <= 0 or not np.isfinite(es):
        return 0
    z_a = _z(1.0 - alpha / 2.0)
    z_b = _z(power)
    n = ((z_a + z_b) / es) ** 2 * max(float(deff), 1.0)
    return int(np.ceil(n))


# --------------------------------------------------------------------- design
@register(
    name="design_survey",
    aliases=["问卷设计", "survey_design"],
    category="survey",
    tier="plus",
    skill="survey-design",
    languages=["Python"],
    key_tools=["statsmodels", "pingouin", "factor_analyzer", "numpy"],
    description="采集前设计测量工具与抽样:构念→题项、信度(Cronbach α)、功效/样本量(含 DEFF)",
    requires={"estimand": ["target"]},
    produces={
        "design": ["sampling_frame"],
        "variables": ["scales", "constructs"],
        "diagnostics": ["reliability", "power"],
    },
    auto_fix="none",
)
def design_survey(state: StudyState, **kwargs: Any) -> StudyState:
    """Design the measurement instrument and sampling before data collection.

    Parameters (via ``kwargs``)
    ---------------------------
    items : DataFrame | ndarray, optional
        Pilot response matrix (rows = respondents, cols = scale items) for the
        Cronbach's α reliability estimate.
    constructs : list | dict, optional
        Latent constructs the instrument measures. Defaults to the estimand target.
    scales : dict, optional
        ``{construct: [item, ...]}`` mapping. Inferred from ``items`` columns if absent.
    sampling_frame : Any, optional
        The population frame descriptor to record on ``design``.
    effect_size, deff, alpha, power : float, optional
        Power inputs (standardized effect, design effect, α, 1−β). Defaults
        ``0.2 / 1.5 / 0.05 / 0.8``.
    """
    target = state.estimand.get("target")

    # -- constructs & scales -------------------------------------------------
    matrix = _as_frame(kwargs.get("items"))
    scales = kwargs.get("scales")
    if scales is None and matrix is not None:
        scales = {"scale_1": [str(c) for c in matrix.columns]}
    if scales is None:
        scales = {}

    constructs = kwargs.get("constructs")
    if constructs is None:
        constructs = list(scales.keys()) or ([str(target)] if target is not None else [])

    # -- reliability (Cronbach's α) -----------------------------------------
    if matrix is not None:
        alpha_val, k = _cronbach_alpha(matrix)
        # Prefer pingouin's implementation when available (CI, listwise handling);
        # fall back silently to the NumPy value computed above.
        pg = _try_import("pingouin")
        if pg is not None:
            try:
                res = pg.cronbach_alpha(data=matrix.dropna(axis=0, how="any"))
                alpha_val = float(res[0])
            except Exception:
                pass
        reliability = {"alpha": alpha_val, "k": k, "n_items": k,
                       "n_respondents": int(matrix.dropna(axis=0, how="any").shape[0])}
    else:
        reliability = {"alpha": float("nan"), "k": len(next(iter(scales.values()), [])),
                       "n_items": len(next(iter(scales.values()), [])),
                       "note": "no pilot `items` supplied — α not estimable"}

    # -- power / required sample size ---------------------------------------
    effect_size = float(kwargs.get("effect_size", 0.2))
    deff = float(kwargs.get("deff", 1.5))
    alpha_lvl = float(kwargs.get("alpha", 0.05))
    power_tgt = float(kwargs.get("power", 0.8))
    n_required = _sample_size(effect_size, deff, alpha_lvl, power_tgt)
    power = {
        "n_required": n_required,
        "deff": deff,
        "effect_size": effect_size,
        "alpha": alpha_lvl,
        "power": power_tgt,
        "n_per_group": int(np.ceil(n_required / 2.0)) if n_required else 0,
    }

    # -- write outputs -------------------------------------------------------
    sampling_frame = kwargs.get("sampling_frame", kwargs.get("frame", target))
    state.write("design", "sampling_frame", sampling_frame)
    state.write("variables", "scales", scales)
    state.write("variables", "constructs", constructs)
    state.write("diagnostics", "reliability", reliability)
    state.write("diagnostics", "power", power)
    return state


# --------------------------------------------------------------------- estimate
@register(
    name="survey_estimate",
    aliases=["加权估计", "complex_survey"],
    category="survey",
    tier="plus",
    skill="complex-survey-analysis",
    languages=["Python"],
    key_tools=["statsmodels", "scipy", "numpy"],
    description="复杂抽样设计加权关联/患病率:权重+PSU cluster-robust 的 design-based 回归",
    requires={"sources": ["datasets"], "design": ["weights"], "variables": ["outcome"]},
    produces={
        "models": ["weighted_reg"],
        "diagnostics": ["sensitivity"],
        "artifacts": ["tables"],
    },
    auto_fix="escalate",
)
def survey_estimate(state: StudyState, **kwargs: Any) -> StudyState:
    """Design-based weighted regression with PSU cluster-robust SEs.

    Reads the registered dataset, the survey weight column, the outcome, and any
    exposure / control columns; fits ``y ~ exposure + controls`` by WLS with the
    survey weights, using ``cov_type="cluster"`` on the PSU when a PSU column is
    declared on ``design``. Also fits the unweighted OLS as a sensitivity contrast.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional
        Overrides ``state.sources['datasets']`` (which may itself be a DataFrame,
        a dict of frames, or a mapping name→frame).
    exposure, controls : str | list[str], optional
        Overrides ``state.variables['exposure' / 'controls']``.
    """
    sm = _try_import("statsmodels.api")

    # -- resolve the dataset -------------------------------------------------
    data = kwargs.get("data")
    if data is None:
        data = state.sources.get("datasets")
    if isinstance(data, dict):
        data = next(iter(data.values()), None)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data) if data is not None else pd.DataFrame()

    weight_col = state.design.get("weights")
    outcome = state.variables.get("outcome")
    psu_col = state.design.get("psu")
    stratum_col = state.design.get("strata") or state.design.get("stratum")
    fpc_col = state.design.get("fpc")
    exposure = kwargs.get("exposure", state.variables.get("exposure"))
    controls = kwargs.get("controls", state.variables.get("controls")) or []
    if isinstance(exposure, str):
        exposure = [exposure]
    exposure = list(exposure or [])
    if isinstance(controls, str):
        controls = [controls]
    controls = list(controls or [])

    predictors = [c for c in (exposure + controls) if c in data.columns]

    # Assemble a clean modelling frame (complete cases over used columns).
    used = [c for c in ([outcome] + predictors + [weight_col, psu_col, stratum_col, fpc_col])
            if c and c in data.columns]
    model_df = data[used].apply(pd.to_numeric, errors="coerce") if used else data.copy()
    if outcome in model_df.columns:
        subset = [c for c in ([outcome] + predictors) if c in model_df.columns]
        model_df = model_df.dropna(axis=0, subset=subset)

    y = model_df[outcome] if outcome in model_df.columns else pd.Series(dtype=float)
    if weight_col in model_df.columns:
        w = model_df[weight_col].fillna(0.0).clip(lower=0.0)
    else:
        w = pd.Series(np.ones(len(model_df)), index=model_df.index)

    if predictors:
        X = model_df[predictors].astype(float)
    else:
        X = pd.DataFrame(index=model_df.index)
    n = int(len(y))

    coef: dict[str, float] = {}
    se: dict[str, float] = {}
    ci: dict[str, list[float]] = {}
    unweighted_coef: dict[str, float] = {}
    fitted = False

    backend = None
    design_total = None      # design-based total of the outcome (svytotal)
    if n > len(predictors) + 1 and n > 1:
        # Design-based estimation via the parity-gated survey reconstruction
        # (external/pysurvey) — exact R-survey Taylor-linearization incl. strata
        # + FPC, superseding the statsmodels cluster-robust approximation which
        # ignored both. See external/pysurvey.
        try:
            from ..external.pysurvey import svydesign, svyglm, svymean, svytotal
            # Grouping ids (strata/PSU) are read from the ORIGINAL frame by row
            # index — model_df was numeric-coerced, which would turn string
            # stratum labels ("E"/"H"/"M") into NaN and collapse the variance.
            idx = model_df.index
            strata = data.loc[idx, stratum_col].to_numpy() if (stratum_col and stratum_col in data.columns) else None
            psu = data.loc[idx, psu_col].to_numpy() if (psu_col and psu_col in data.columns and data.loc[idx, psu_col].notna().any()) else None
            fpc = pd.to_numeric(data.loc[idx, fpc_col], errors="coerce").to_numpy(float) if (fpc_col and fpc_col in data.columns) else None
            des = svydesign({outcome: y.to_numpy(float)}, weights=w.to_numpy(float),
                            ids=psu, strata=strata, fpc=fpc)
            names = ["const"] + predictors
            if predictors:
                g = svyglm(y.to_numpy(float), X.to_numpy(float), des)
                coef = {names[i]: float(g["coef"][i]) for i in range(len(names))}
                se = {names[i]: float(g["se"][i]) for i in range(len(names))}
                ci = {names[i]: [float(g["ci_lb"][i]), float(g["ci_ub"][i])] for i in range(len(names))}
            else:
                mr = svymean(y.to_numpy(float), des)
                coef = {"const": mr["estimate"]}; se = {"const": mr["se"]}
                ci = {"const": [mr["ci_lb"], mr["ci_ub"]]}
            # design-based TOTAL of the outcome (svytotal) — a distinct estimand
            # socialverse previously did not surface.
            _dt = svytotal(y.to_numpy(float), des)
            design_total = {"estimate": _dt["estimate"], "se": _dt["se"], "df": _dt["df"]}
            fitted = True
            backend = "pysurvey"
        except Exception:
            fitted = False
        # unweighted OLS sensitivity contrast (statsmodels, optional)
        if fitted and sm is not None:
            try:
                Xc = sm.add_constant(X, has_constant="add")
                ols = sm.OLS(y, Xc).fit()
                unweighted_coef = {k: float(v) for k, v in ols.params.items()}
            except Exception:
                unweighted_coef = {}

    if not fitted:
        # Design-based fallback with no statsmodels / degenerate frame: at least a
        # weighted mean of the outcome (design-weighted prevalence/level).
        if n > 0 and float(w.sum()) > 0:
            wmean = float(np.average(y.to_numpy(dtype=float), weights=w.to_numpy(dtype=float)))
            unw = float(np.mean(y.to_numpy(dtype=float)))
        else:
            wmean, unw = float("nan"), float("nan")
        coef = {"const": wmean}
        se = {"const": float("nan")}
        ci = {"const": [float("nan"), float("nan")]}
        unweighted_coef = {"const": unw}

    weighted_reg = {
        "coef": coef,
        "se": se,
        "ci": ci,
        "design_total": design_total,
        "n": n,
        "outcome": outcome,
        "exposure": exposure,
        "controls": controls,
        "cov_type": "survey.taylor" if backend == "pysurvey" else ("none" if not fitted else "HC1"),
        "backend": backend,
        "psu": psu_col,
        "weights": weight_col,
    }

    # -- sensitivity: weighted vs unweighted coefficients -------------------
    sensitivity = {
        "weighted": coef,
        "unweighted": unweighted_coef,
        "delta": {k: float(coef.get(k, float("nan")) - unweighted_coef.get(k, float("nan")))
                  for k in set(coef) | set(unweighted_coef)},
        "note": "design-based vs naive OLS — large delta ⇒ weights/clustering matter",
    }

    # -- artifact: a tidy coefficient table ----------------------------------
    rows = []
    for term in coef:
        lo, hi = ci.get(term, [float("nan"), float("nan")])
        rows.append({
            "term": term,
            "coef": coef.get(term, float("nan")),
            "se": se.get(term, float("nan")),
            "ci_low": lo,
            "ci_high": hi,
            "coef_unweighted": unweighted_coef.get(term, float("nan")),
        })
    table = pd.DataFrame(rows, columns=["term", "coef", "se", "ci_low", "ci_high",
                                        "coef_unweighted"])

    state.write("models", "weighted_reg", weighted_reg)
    state.write("diagnostics", "sensitivity", sensitivity)
    state.write("artifacts", "tables", table)
    return state


# --------------------------------------------------------------------- svyby
@register(
    name="survey_by",
    aliases=["分组加权估计", "survey_domain", "svyby"],
    category="survey",
    tier="plus",
    skill="complex-survey-analysis",
    languages=["Python"],
    key_tools=["scipy", "numpy"],
    description="复杂抽样分域(子总体)估计:按分组列计算 design-based 域均值/域总计(svyby),方差取自完整设计",
    requires={"sources": ["datasets"], "design": ["weights"], "variables": ["outcome"]},
    produces={"models": ["survey_by"], "artifacts": ["tables"]},
    auto_fix="escalate",
)
def survey_by(state: StudyState, **kwargs: Any) -> StudyState:
    """Design-based domain (subpopulation) means/totals per level of a group column.

    Reads the outcome from ``variables['outcome']`` and the grouping column from
    ``kwargs['by']`` / ``variables['group'|'by']``. Delegates to
    :func:`external.pysurvey.svyby`, whose domain variance is taken over the *full*
    design (strata/PSUs), so the SE differs from a naive subset ``svymean``.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional — overrides ``state.sources['datasets']``.
    outcome : str, optional — overrides ``state.variables['outcome']``.
    by : str, optional — grouping column; defaults to ``variables['group'|'by']``.
    stat : {"svymean", "svytotal"}, optional — domain statistic (default ``svymean``).
    """
    data = _resolve_dataset(state, kwargs)
    outcome = kwargs.get("outcome", state.variables.get("outcome"))
    by = kwargs.get("by", state.variables.get("group") or state.variables.get("by"))
    stat = kwargs.get("stat", "svymean")

    result: dict[str, Any] = {
        "outcome": outcome, "by": by, "stat": stat, "backend": None,
        "levels": [], "estimate": {}, "se": {}, "df": None, "n": 0,
    }
    table = pd.DataFrame(columns=["level", "estimate", "se"])

    try:
        if not outcome or outcome not in data.columns:
            raise ValueError(f"outcome column {outcome!r} not found in dataset")
        if not by or by not in data.columns:
            raise ValueError(f"grouping column {by!r} not found in dataset")

        from ..external.pysurvey import svyby

        des, w, idx = _design_from_state(state, data, [outcome])
        yv = pd.to_numeric(data.loc[idx, outcome], errors="coerce").to_numpy(float)
        gv = data.loc[idx, by].to_numpy()
        out = svyby(yv, gv, des, stat=stat)

        levels = list(out["levels"])
        ests = [float(v) for v in np.asarray(out["estimate"]).ravel()]
        ses = [float(v) for v in np.asarray(out["se"]).ravel()]
        result.update({
            "backend": "pysurvey",
            "levels": levels,
            "estimate": {str(lv): e for lv, e in zip(levels, ests)},
            "se": {str(lv): s for lv, s in zip(levels, ses)},
            "df": int(out["df"]),
            "n": int(len(idx)),
        })
        table = pd.DataFrame(
            {"level": [str(lv) for lv in levels], "estimate": ests, "se": ses},
            columns=["level", "estimate", "se"],
        )
    except Exception as exc:  # graceful degrade — never crash on missing input
        result["error"] = f"survey_by 未能完成:{exc}"

    state.write("models", "survey_by", result)
    state.write("artifacts", "tables", table)
    return state


# --------------------------------------------------------------------- svyratio
@register(
    name="survey_ratio",
    aliases=["加权比率", "survey_ratio_est", "svyratio"],
    category="survey",
    tier="plus",
    skill="complex-survey-analysis",
    languages=["Python"],
    key_tools=["scipy", "numpy"],
    description="复杂抽样比率估计:两列的 design-based 比率 R=Σw·num/Σw·den,Taylor 线性化 SE(svyratio)",
    requires={"sources": ["datasets"], "design": ["weights"]},
    produces={"models": ["survey_ratio"]},
    auto_fix="escalate",
)
def survey_ratio(state: StudyState, **kwargs: Any) -> StudyState:
    """Design-based ratio of two columns with Taylor-linearized SE.

    Numerator/denominator columns come from ``kwargs['num'|'den']`` or, by default,
    ``variables['outcome']`` (numerator) and ``variables['denominator'|'size']``.
    Delegates to :func:`external.pysurvey.svyratio`.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional — overrides ``state.sources['datasets']``.
    num : str, optional — numerator column; defaults to ``variables['outcome']``.
    den : str, optional — denominator column; defaults to
        ``variables['denominator'|'size']``.
    """
    data = _resolve_dataset(state, kwargs)
    num = kwargs.get("num", state.variables.get("outcome"))
    den = kwargs.get("den", state.variables.get("denominator") or state.variables.get("size"))

    result: dict[str, Any] = {
        "num": num, "den": den, "backend": None,
        "estimate": float("nan"), "se": float("nan"),
        "ci": [float("nan"), float("nan")], "df": None, "n": 0,
    }

    try:
        if not num or num not in data.columns:
            raise ValueError(f"numerator column {num!r} not found in dataset")
        if not den or den not in data.columns:
            raise ValueError(f"denominator column {den!r} not found in dataset")

        from ..external.pysurvey import svyratio

        des, w, idx = _design_from_state(state, data, [num, den])
        nv = pd.to_numeric(data.loc[idx, num], errors="coerce").to_numpy(float)
        dv = pd.to_numeric(data.loc[idx, den], errors="coerce").to_numpy(float)
        out = svyratio(nv, dv, des)
        result.update({
            "backend": "pysurvey",
            "estimate": float(out["estimate"]),
            "se": float(out["se"]),
            "ci": [float(out["ci_lb"]), float(out["ci_ub"])],
            "df": int(out["df"]),
            "n": int(len(idx)),
        })
    except Exception as exc:  # graceful degrade — never crash on missing input
        result["error"] = f"survey_ratio 未能完成:{exc}"

    state.write("models", "survey_ratio", result)
    return state


# --------------------------------------------------------------------- svyciprop
@register(
    name="survey_ciprop",
    aliases=["加权比例置信区间", "survey_prop_ci", "svyciprop"],
    category="survey",
    tier="plus",
    skill="complex-survey-analysis",
    languages=["Python"],
    key_tools=["scipy", "numpy"],
    description="复杂抽样比例置信区间:0/1 指标的 design-based 加权比例 + logit 法置信区间(svyciprop)",
    requires={"sources": ["datasets"], "design": ["weights"], "variables": ["outcome"]},
    produces={"models": ["survey_ciprop"]},
    auto_fix="escalate",
)
def survey_ciprop(state: StudyState, **kwargs: Any) -> StudyState:
    """Design-based proportion of a 0/1 indicator with a logit-method CI.

    Reads the binary outcome from ``variables['outcome']`` (override via
    ``kwargs['outcome']``) and delegates to :func:`external.pysurvey.svyciprop`,
    which returns the weighted proportion, its variance/SE, and a logit-scale CI.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional — overrides ``state.sources['datasets']``.
    outcome : str, optional — 0/1 indicator column; defaults to
        ``variables['outcome']``.
    """
    data = _resolve_dataset(state, kwargs)
    outcome = kwargs.get("outcome", state.variables.get("outcome"))

    result: dict[str, Any] = {
        "outcome": outcome, "backend": None,
        "estimate": float("nan"), "var": float("nan"), "se": float("nan"),
        "ci": [float("nan"), float("nan")], "df": None, "n": 0,
    }

    try:
        if not outcome or outcome not in data.columns:
            raise ValueError(f"outcome column {outcome!r} not found in dataset")

        from ..external.pysurvey import svyciprop

        des, w, idx = _design_from_state(state, data, [outcome])
        yv = pd.to_numeric(data.loc[idx, outcome], errors="coerce").to_numpy(float)
        out = svyciprop(yv, des)
        result.update({
            "backend": "pysurvey",
            "estimate": float(out["estimate"]),
            "var": float(out["var"]),
            "se": float(out["se"]),
            "ci": [float(out["ci_lb"]), float(out["ci_ub"])],
            "df": int(out["df"]),
            "n": int(len(idx)),
        })
    except Exception as exc:  # graceful degrade — never crash on missing input
        result["error"] = f"survey_ciprop 未能完成:{exc}"

    state.write("models", "survey_ciprop", result)
    return state


# --------------------------------------------------------------------- crosstab
@register(
    name="survey_crosstab",
    aliases=["加权交叉表", "survey_pivot", "weighted_crosstab", "design_based_crosstab", "透视表推断"],
    category="survey",
    tier="plus",
    skill="complex-survey-analysis",
    languages=["Python"],
    key_tools=["scipy", "numpy", "pandas"],
    description=(
        "设计校正的加权交叉表(透视表工具的推断入口):每格 design-based 加权均值/比例 + "
        "95% 置信区间(svyby 域方差,含分层/PSU)+ 组间差异的 design-based Wald F 检验;"
        "附未加权敏感性对照与 Kish 有效样本量。把描述性透视表升级成真正的调查估计。"
    ),
    requires={},   # the Pivot tool passes data + design via kwargs; validated in-body
    produces={"models": ["survey_crosstab"], "artifacts": ["tables"]},
    auto_fix="escalate",
)
def survey_crosstab(state: StudyState, **kwargs: Any) -> StudyState:
    """Design-based weighted cross-tab — the inference layer behind the Pivot tool.

    Turns a *descriptive* weighted crosstab into a real survey estimate: for every
    ``group`` (× optional ``split``) cell it reports the design-based weighted
    mean/proportion with a 95% CI (domain variance over the FULL design, i.e. strata
    + PSUs via :func:`external.pysurvey.svyby`), plus a design-based **Wald F test**
    for whether the outcome's weighted mean differs across ``group`` (WLS ``y~group``
    with the survey sandwich variance). A binary 0/1 outcome is reported as a
    proportion. Also returns the unweighted means as a sensitivity contrast and the
    Kish effective sample size.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional — the survey frame (overrides ``sources['datasets']``).
    outcome : str — the measure column (0/1 → proportion; else weighted mean).
    group : str — the primary grouping (pivot rows).
    split : str, optional — a secondary grouping (pivot columns).
    weights / strata / psu / fpc : str, optional — design columns (override ``design``).
    level : float, optional — CI level (default 95).
    """
    from scipy import stats

    data = _resolve_dataset(state, kwargs)
    outcome = kwargs.get("outcome", state.variables.get("outcome"))
    group = kwargs.get("group", state.variables.get("group") or state.variables.get("by"))
    split = kwargs.get("split")
    level = float(kwargs.get("level", 95.0))

    # explicit design kwargs override state.design (the Pivot tool passes them directly)
    for k in ("weights", "strata", "psu", "fpc"):
        if kwargs.get(k):
            state.write("design", k, kwargs[k])

    result: dict[str, Any] = {
        "outcome": outcome, "group": group, "split": split, "level": level,
        "binary": False, "cells": [], "test": None, "design": {}, "unweighted": {},
        "backend": None, "error": None,
    }
    table = pd.DataFrame(columns=["group", "split", "estimate", "ci_lo", "ci_hi", "n"])
    SEP = "␟"  # unit separator to join group×split levels

    try:
        if not outcome or outcome not in data.columns:
            raise ValueError(f"测量列(outcome)‘{outcome}’不在数据里")
        if not group or group not in data.columns:
            raise ValueError(f"分组列(group)‘{group}’不在数据里")

        from ..external.pysurvey import svyby
        from ..external.pysurvey.survey import _vcov_total

        des, w, idx = _design_from_state(state, data, [outcome])
        yv = pd.to_numeric(data.loc[idx, outcome], errors="coerce").to_numpy(float)
        gv = data.loc[idx, group].astype(str).to_numpy()
        sv_ = data.loc[idx, split].astype(str).to_numpy() if (split and split in data.columns) else None

        finite = yv[np.isfinite(yv)]
        uniq_y = np.unique(finite)
        is_binary = uniq_y.size <= 2 and set(np.round(uniq_y, 6).tolist()).issubset({0.0, 1.0})
        result["binary"] = bool(is_binary)

        # per-cell domain estimates (group [× split]); full-design (strata/PSU) variance
        cell_key = (np.array([f"{g}{SEP}{s}" for g, s in zip(gv, sv_)]) if sv_ is not None else gv)
        out = svyby(yv, cell_key, des, stat="svymean")
        df_dom = int(out["df"])
        crit = float(stats.t.ppf(1 - (1 - level / 100.0) / 2, max(df_dom, 1)))
        cells = []
        for lvl, est, se in zip(out["levels"], np.asarray(out["estimate"]).ravel(),
                                np.asarray(out["se"]).ravel()):
            lvl = str(lvl)
            g, s = (lvl.split(SEP, 1) if SEP in lvl else (lvl, None))
            n_cell = int(np.sum(cell_key == lvl))
            lo, hi = float(est) - crit * float(se), float(est) + crit * float(se)
            if is_binary:
                lo, hi = max(0.0, lo), min(1.0, hi)
            cells.append({"group": g, "split": s, "estimate": float(est), "se": float(se),
                          "ci_lo": lo, "ci_hi": hi, "n": n_cell})
        result["cells"] = cells
        result["backend"] = "pysurvey"

        # design-based omnibus test: does the weighted mean of outcome differ by group?
        levels_g = sorted(set(gv.tolist()))
        if len(levels_g) >= 2:
            Dg = pd.get_dummies(pd.Series(gv), drop_first=True).to_numpy(float)
            X = np.hstack([np.ones((len(idx), 1)), Dg])
            A = X.T @ (w[:, None] * X)
            beta = np.linalg.solve(A, X.T @ (w * yv))
            resid = yv - X @ beta
            Gm = (w * resid)[:, None] * X
            B = _vcov_total(Gm, des)
            Ainv = np.linalg.inv(A)
            V = Ainv @ B @ Ainv
            q = X.shape[1] - 1
            bD, VD = beta[1:], V[1:, 1:]
            wald = float(bD @ np.linalg.solve(VD, bD))
            dfden = max(int(des.degf) - q + 1, 1)
            Fstat = wald / q
            pval = float(stats.f.sf(Fstat, q, dfden))
            result["test"] = {
                "method": "design-based Wald F (y~group, survey sandwich)",
                "stat": float(Fstat), "df_num": int(q), "df_den": int(dfden),
                "p": pval, "significant": bool(pval < 0.05),
            }

        # unweighted sensitivity (naive means per group)
        dfu = pd.DataFrame({"g": gv, "y": yv})
        result["unweighted"] = {str(k): float(v) for k, v in dfu.groupby("g")["y"].mean().items()}

        # design summary + Kish effective N
        sw, sw2 = float(w.sum()), float((w ** 2).sum())
        n_eff = (sw * sw / sw2) if sw2 > 0 else float(len(idx))
        result["design"] = {
            "n": int(len(idx)), "sum_w": sw, "n_eff": float(n_eff),
            "deff_kish": float(len(idx) / n_eff) if n_eff > 0 else None,
            "weights": state.design.get("weights"),
            "strata": state.design.get("strata") or state.design.get("stratum"),
            "psu": state.design.get("psu"), "df": df_dom,
        }
        table = pd.DataFrame(
            [{"group": c["group"], "split": c["split"], "estimate": c["estimate"],
              "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"], "n": c["n"]} for c in cells],
            columns=["group", "split", "estimate", "ci_lo", "ci_hi", "n"],
        )
    except Exception as exc:  # graceful degrade — never crash on missing/odd input
        result["error"] = f"survey_crosstab 未能完成:{exc}"

    state.write("models", "survey_crosstab", result)
    state.write("artifacts", "tables", table)
    return state
