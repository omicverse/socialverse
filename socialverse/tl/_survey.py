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

__all__ = ["design_survey", "survey_estimate"]


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
    used = [c for c in ([outcome] + predictors + [weight_col, psu_col]) if c in data.columns]
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

    if sm is not None and n > len(predictors) + 1 and n > 1:
        Xc = sm.add_constant(X, has_constant="add")
        try:
            wls = sm.WLS(y, Xc, weights=w)
            if psu_col in model_df.columns and model_df[psu_col].notna().any():
                res = wls.fit(cov_type="cluster",
                              cov_kwds={"groups": model_df[psu_col].values})
            else:
                res = wls.fit(cov_type="HC1")
            conf = res.conf_int()
            coef = {k: float(v) for k, v in res.params.items()}
            se = {k: float(v) for k, v in res.bse.items()}
            ci = {k: [float(conf.loc[k, 0]), float(conf.loc[k, 1])] for k in res.params.index}
            fitted = True
            # unweighted sensitivity contrast
            try:
                ols = sm.OLS(y, Xc).fit()
                unweighted_coef = {k: float(v) for k, v in ols.params.items()}
            except Exception:
                unweighted_coef = {}
        except Exception:
            fitted = False

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
        "n": n,
        "outcome": outcome,
        "exposure": exposure,
        "controls": controls,
        "cov_type": "cluster" if (psu_col in model_df.columns) else ("HC1" if fitted else "none"),
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
