"""``sv.tl._regression`` — registered implementations for the **regression family
+ average marginal effects** P0 gap.

Four workhorses of applied social-science modelling, ported to the
``StudyState`` / ``registry`` spine with *real* estimation (recovers the known
data-generating parameters of ``datasets.load_regression`` within tolerance),
no placeholders.

Champion packages this file mirrors
-----------------------------------
* ``glm`` — R's ``stats::glm`` / Stata ``regress``/``logit``/``poisson`` /
  Python ``statsmodels.GLM``. One entry point for Gaussian (OLS), Binomial
  (logit), Poisson, and Negative-Binomial GLMs with an added intercept, plus
  ``nonrobust`` / HC1-``robust`` / cluster-robust covariance.
* ``mlogit`` — R's ``nnet::multinom`` / Stata ``mlogit`` /
  ``statsmodels.MNLogit``. Multinomial (unordered) logit; one coefficient block
  per non-base category.
* ``ologit`` — R's ``MASS::polr`` / Stata ``ologit``/``oprobit`` /
  ``statsmodels.miscmodels.ordinal_model.OrderedModel``. Ordered logit / probit
  with estimated cut-points (thresholds).
* ``margins`` — R's ``margins`` / Stata ``margins, dydx(*)`` /
  ``marginaleffects`` package. Average marginal effects (AME) of a previously
  fitted GLM / MNLogit / OrderedModel, read back off its stored ``_fit`` object.

The registry contracts chain: ``glm`` / ``mlogit`` / ``ologit`` ``require`` a
working ``sources['datasets']`` frame and a declared ``variables['outcome']``
and ``produce`` a fitted model; ``margins`` ``requires`` a produced ``glm``
model (and lists ``glm`` as a prerequisite function, with ``mlogit``/``ologit``
optional) — so a resolver refuses to report marginal effects until a model has
actually been fitted.

Every reported number comes from the ``statsmodels`` fit; ``statsmodels`` is
always available in this environment, but each function degrades gracefully
(writes an empty model + diagnostics carrying an honest ``note``, never raising)
when data / columns / the backend are missing.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["glm", "mlogit", "ologit", "margins"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_datasets(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working frame: explicit ``data=`` kwarg, else ``sources['datasets']``.

    ``sources['datasets']`` may be a DataFrame or a ``{name: DataFrame}`` mapping;
    in the latter case the first frame is taken.
    """
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _resolve_outcome(
    df: pd.DataFrame, kwargs: dict[str, Any], state: StudyState
) -> str | None:
    """Outcome column: kwargs ``outcome=`` → ``variables['outcome']`` → None."""
    y = kwargs.get("outcome") or state.variables.get("outcome")
    if y is not None and y in df.columns:
        return y
    return None


def _resolve_predictors(
    df: pd.DataFrame, kwargs: dict[str, Any], state: StudyState, outcome: str | None
) -> list[str]:
    """Predictor columns: kwargs ``predictors=`` → ``variables['controls']`` →
    the numeric columns other than the outcome."""
    preds = kwargs.get("predictors")
    if not preds:
        preds = state.variables.get("controls")
    if preds:
        return [c for c in preds if c in df.columns and c != outcome]
    return [
        c
        for c in df.columns
        if c != outcome and pd.api.types.is_numeric_dtype(df[c])
    ]


def _wald_ci(coef: np.ndarray, se: np.ndarray, z: float = 1.959963984540054):
    """Symmetric Wald 95% CI (z-normal); returns list of (lo, hi) tuples."""
    return [
        (float(b - z * s), float(b + z * s)) for b, s in zip(coef, se)
    ]


# ---------------------------------------------------------------------- glm
@register(
    name="glm",
    aliases=["广义线性模型", "glm", "regress", "logit", "poisson"],
    category="regression",
    tier="plus",
    skill="(P0)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="广义线性模型(GLM):高斯/二项(logit)/泊松/负二项 + 稳健(HC1)/聚类协方差",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["glm"], "diagnostics": ["glm_fit"]},
    auto_fix="escalate",
)
def glm(state: StudyState, **kwargs: Any) -> StudyState:
    """Fit a generalized linear model with an added intercept.

    kwargs
    ------
    outcome : str
        Response column. Falls back to ``variables['outcome']``.
    predictors : list[str]
        Covariates. Falls back to ``variables['controls']`` then to all other
        numeric columns.
    family : {"gaussian", "binomial", "poisson", "negbin"}
        GLM family; binomial uses the logit link. Default ``"gaussian"``.
    cov : {"nonrobust", "robust", "cluster"}
        Covariance estimator. ``robust`` → HC1; ``cluster`` → cluster-robust on
        the ``cluster`` column. Default ``"nonrobust"``.
    cluster : str
        Column of cluster ids (required when ``cov="cluster"``).
    """
    family = str(kwargs.get("family", "gaussian")).lower()
    cov = str(kwargs.get("cov", "nonrobust")).lower()

    def _empty(note: str) -> StudyState:
        state.write("models", "glm", {
            "family": family, "coef": {}, "se": {}, "z": {}, "p": {},
            "ci": {}, "n": 0, "note": note, "_fit": None,
        })
        state.write("diagnostics", "glm_fit",
                    {"llf": None, "aic": None, "r2": None, "pseudo_r2": None,
                     "note": note})
        return state

    df = _get_datasets(state, kwargs)
    if df is None:
        return _empty("缺少数据(sources['datasets']),无法拟合 GLM")

    outcome = _resolve_outcome(df, kwargs, state)
    if outcome is None:
        return _empty("找不到结果变量(outcome)")

    predictors = _resolve_predictors(df, kwargs, state, outcome)
    if not predictors:
        return _empty("找不到预测变量(predictors)")

    sm = _try_import("statsmodels.api")
    if sm is None:
        return _empty("statsmodels 不可用,GLM 优雅降级为空")

    cluster_col = kwargs.get("cluster")
    keep = [outcome] + predictors
    if cov == "cluster" and cluster_col and cluster_col in df.columns:
        keep = keep + [cluster_col]
    work = df[keep].dropna()
    if work.empty:
        return _empty("有效样本为空(全部缺失)")

    y = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(len(work))]
        + [pd.to_numeric(work[p], errors="coerce").to_numpy(dtype=float)
           for p in predictors]
    )
    names = ["const"] + list(predictors)

    fam_map = {
        "gaussian": lambda: sm.families.Gaussian(),
        "binomial": lambda: sm.families.Binomial(),  # logit link (default)
        "poisson": lambda: sm.families.Poisson(),
        "negbin": lambda: sm.families.NegativeBinomial(),
    }
    fam_ctor = fam_map.get(family)
    if fam_ctor is None:
        return _empty(f"未知 family={family!r}(支持 gaussian/binomial/poisson/negbin)")

    note = f"statsmodels.GLM({family})"
    try:
        model = sm.GLM(y, X, family=fam_ctor())
        if cov == "cluster" and cluster_col and cluster_col in work.columns:
            groups = work[cluster_col].to_numpy()
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            note += " · cluster-robust"
        elif cov == "robust":
            res = model.fit(cov_type="HC1")
            note += " · HC1 robust"
        else:
            res = model.fit()
            note += " · nonrobust"
    except Exception as exc:
        return _empty(f"GLM 拟合失败: {exc!s}")

    coef = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    stat = np.asarray(res.tvalues, dtype=float)
    pvals = np.asarray(res.pvalues, dtype=float)
    ci = _wald_ci(coef, se)

    coef_d = {n: float(b) for n, b in zip(names, coef)}
    se_d = {n: float(s) for n, s in zip(names, se)}
    stat_d = {n: float(z) for n, z in zip(names, stat)}
    p_d = {n: float(p) for n, p in zip(names, pvals)}
    ci_d = {n: c for n, c in zip(names, ci)}
    stat_label = "t" if family == "gaussian" else "z"

    state.write("models", "glm", {
        "family": family,
        "outcome": outcome,
        "predictors": predictors,
        "coef": coef_d,
        "se": se_d,
        stat_label: stat_d,
        "p": p_d,
        "ci": ci_d,
        "n": int(len(work)),
        "cov": cov,
        "estimator": note,
        "note": f"{family} GLM;coef=系数(logit/poisson 为对数尺度),{stat_label}/p=Wald",
        "_fit": res,
    })

    # goodness of fit: R² for Gaussian, McFadden pseudo-R² otherwise
    llf = float(res.llf)
    aic = float(res.aic)
    r2 = pseudo_r2 = None
    if family == "gaussian":
        ss_res = float(np.sum((y - res.fittedvalues) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    else:
        try:
            null = sm.GLM(y, np.ones((len(y), 1)), family=fam_ctor()).fit()
            ll0 = float(null.llf)
            pseudo_r2 = float(1.0 - llf / ll0) if ll0 != 0 else None
        except Exception:
            pseudo_r2 = None

    state.write("diagnostics", "glm_fit", {
        "llf": llf, "aic": aic, "r2": r2, "pseudo_r2": pseudo_r2,
        "note": ("R² = 1 - SSR/SST" if family == "gaussian"
                 else "McFadden pseudo-R² = 1 - llf/ll0"),
    })
    return state


# ------------------------------------------------------------------- mlogit
@register(
    name="mlogit",
    aliases=["多项逻辑回归", "mlogit", "multinom"],
    category="regression",
    tier="plus",
    skill="(P0)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="多项(无序)逻辑回归:MNLogit,每个非基类别一组系数",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["mlogit"]},
    auto_fix="escalate",
)
def mlogit(state: StudyState, **kwargs: Any) -> StudyState:
    """Multinomial (unordered) logit via ``statsmodels.MNLogit``.

    kwargs
    ------
    outcome : str
        Nominal response column (>2 categories). Falls back to
        ``variables['outcome']``.
    predictors : list[str]
        Covariates. Falls back to ``variables['controls']`` then to numeric cols.
    base : hashable
        Reference category. Default: the first category by sort order.
    """
    def _empty(note: str) -> StudyState:
        state.write("models", "mlogit", {
            "coef": {}, "base": None, "n": 0, "note": note, "_fit": None,
        })
        return state

    df = _get_datasets(state, kwargs)
    if df is None:
        return _empty("缺少数据(sources['datasets']),无法拟合多项 logit")

    outcome = _resolve_outcome(df, kwargs, state)
    if outcome is None:
        return _empty("找不到结果变量(outcome)")

    predictors = _resolve_predictors(df, kwargs, state, outcome)
    if not predictors:
        return _empty("找不到预测变量(predictors)")

    sm = _try_import("statsmodels.api")
    if sm is None:
        return _empty("statsmodels 不可用,多项 logit 优雅降级为空")

    work = df[[outcome] + predictors].dropna()
    if work.empty:
        return _empty("有效样本为空(全部缺失)")

    # encode nominal outcome to 0..K-1 with an explicit, stable base category
    cats = sorted(pd.Series(work[outcome].astype("object")).unique().tolist(),
                  key=lambda v: str(v))
    base = kwargs.get("base")
    if base is None or base not in cats:
        base = cats[0]
    # order so that the base category is coded 0 (MNLogit uses code 0 as base)
    ordered = [base] + [c for c in cats if c != base]
    code = {c: i for i, c in enumerate(ordered)}
    if len(ordered) < 3:
        return _empty(f"outcome 只有 {len(ordered)} 个类别,建议用 glm(logit)")

    y = work[outcome].map(code).to_numpy(dtype=int)
    X = np.column_stack(
        [np.ones(len(work))]
        + [pd.to_numeric(work[p], errors="coerce").to_numpy(dtype=float)
           for p in predictors]
    )
    names = ["const"] + list(predictors)

    try:
        res = sm.MNLogit(y, X).fit(disp=False, maxiter=200)
    except Exception as exc:
        return _empty(f"MNLogit 拟合失败: {exc!s}")

    # params: (n_exog, K-1) — one column per non-base category (codes 1..K-1)
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    coef: dict[str, dict[str, dict[str, float]]] = {}
    for col in range(params.shape[1]):
        cat = ordered[col + 1]  # non-base category for this column
        coef[str(cat)] = {
            nm: {"coef": float(params[i, col]), "se": float(bse[i, col])}
            for i, nm in enumerate(names)
        }

    state.write("models", "mlogit", {
        "coef": coef,
        "base": str(base),
        "categories": [str(c) for c in ordered],
        "outcome": outcome,
        "predictors": predictors,
        "n": int(len(work)),
        "llf": float(res.llf),
        "estimator": "statsmodels.MNLogit",
        "note": "每个非基类别相对 base 的 log-odds 系数(const 为该类别截距)",
        "_fit": res,
    })
    return state


# ------------------------------------------------------------------- ologit
@register(
    name="ologit",
    aliases=["有序逻辑回归", "ologit", "oprobit", "polr"],
    category="regression",
    tier="plus",
    skill="(P0)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="有序逻辑/概率回归:OrderedModel(logit/probit),系数 + 切点阈值",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["ologit"]},
    auto_fix="escalate",
)
def ologit(state: StudyState, **kwargs: Any) -> StudyState:
    """Ordered logit / probit via ``statsmodels`` ``OrderedModel``.

    kwargs
    ------
    outcome : str
        Ordinal response (ordered integer / category). Falls back to
        ``variables['outcome']``.
    predictors : list[str]
        Covariates. Falls back to ``variables['controls']`` then to numeric cols.
    link : {"logit", "probit"}
        Latent-variable link. Default ``"logit"``.
    """
    link = str(kwargs.get("link", "logit")).lower()

    def _empty(note: str) -> StudyState:
        state.write("models", "ologit", {
            "coef": {}, "thresholds": {}, "link": link, "n": 0,
            "note": note, "_fit": None,
        })
        return state

    df = _get_datasets(state, kwargs)
    if df is None:
        return _empty("缺少数据(sources['datasets']),无法拟合有序回归")

    outcome = _resolve_outcome(df, kwargs, state)
    if outcome is None:
        return _empty("找不到结果变量(outcome)")

    predictors = _resolve_predictors(df, kwargs, state, outcome)
    if not predictors:
        return _empty("找不到预测变量(predictors)")

    om_mod = _try_import("statsmodels.miscmodels.ordinal_model")
    if om_mod is None:
        return _empty("statsmodels OrderedModel 不可用,有序回归优雅降级为空")

    work = df[[outcome] + predictors].dropna()
    if work.empty:
        return _empty("有效样本为空(全部缺失)")

    ycat = pd.Categorical(work[outcome], ordered=True)
    n_cat = len(ycat.categories)
    if n_cat < 3:
        return _empty(f"outcome 只有 {n_cat} 个等级,建议用 glm(logit)")
    # OrderedModel needs consistent pandas structures: an ordered categorical
    # Series endog + a DataFrame exog (mixing Categorical/ndarray is rejected).
    y = pd.Series(ycat, index=work.index, name=outcome)
    X = pd.DataFrame(
        {p: pd.to_numeric(work[p], errors="coerce").astype(float) for p in predictors},
        index=work.index,
    )
    distr = "logit" if link in ("logit", "logistic") else "probit"

    try:
        res = om_mod.OrderedModel(y, X, distr=distr).fit(method="bfgs", disp=False)
    except Exception as exc:
        return _empty(f"OrderedModel 拟合失败: {exc!s}")

    params = np.asarray(res.params, dtype=float)
    k = len(predictors)
    coef = {nm: float(params[i]) for i, nm in enumerate(predictors)}
    se = np.asarray(res.bse, dtype=float)
    coef_se = {nm: float(se[i]) for i, nm in enumerate(predictors)}

    # remaining params are the (transformed) threshold cut-points; recover the
    # actual thresholds via the model's transform.
    try:
        thr = res.model.transform_threshold_params(params[k:])
        # transform pads with +/-inf sentinels at the ends; keep finite cut-points
        thr = [float(t) for t in np.asarray(thr, dtype=float) if np.isfinite(t)]
    except Exception:
        thr = [float(t) for t in params[k:]]
    thresholds = {f"cut_{i+1}": t for i, t in enumerate(thr)}

    state.write("models", "ologit", {
        "coef": coef,
        "coef_se": coef_se,
        "thresholds": thresholds,
        "link": distr,
        "outcome": outcome,
        "predictors": predictors,
        "n": int(len(work)),
        "n_categories": int(n_cat),
        "llf": float(res.llf),
        "estimator": f"statsmodels.OrderedModel({distr})",
        "note": "系数为潜变量尺度;正系数=更高等级更可能。thresholds=有序切点",
        "_fit": res,
    })
    return state


# ------------------------------------------------------------------- margins
@register(
    name="margins",
    aliases=["边际效应", "margins", "marginaleffects"],
    category="regression",
    tier="plus",
    skill="(P0)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="平均边际效应(AME):读回已拟合 glm/mlogit/ologit 的 _fit,报 dy/dx",
    requires={"models": ["glm"]},
    prerequisites={"functions": ["glm"], "optional_functions": ["mlogit", "ologit"]},
    produces={"diagnostics": ["margins"]},
    auto_fix="escalate",
)
def margins(state: StudyState, **kwargs: Any) -> StudyState:
    """Average marginal effects (AME) of a previously fitted model.

    Reads back ``state.models[model]['_fit']`` and calls ``get_margeff`` where
    available (GLM discrete/Logit/Poisson, MNLogit), else computes AMEs by
    numerical differentiation of the mean predicted response. For the linear
    Gaussian GLM the AME of a covariate is just its slope.

    kwargs
    ------
    model : {"glm", "mlogit", "ologit", None}
        Which stored model to differentiate. ``None`` → first present among
        ``glm`` / ``mlogit`` / ``ologit``.
    at : str
        ``statsmodels`` ``get_margeff`` location, e.g. ``"overall"`` (AME),
        ``"mean"`` (MEM). Default ``"overall"``.
    """
    at = str(kwargs.get("at", "overall"))
    want = kwargs.get("model")

    def _empty(note: str, model_name: Any = None) -> StudyState:
        state.write("diagnostics", "margins", {
            "model": model_name, "ame": {}, "se": {}, "at": at, "note": note,
        })
        return state

    candidates = [want] if want else ["glm", "mlogit", "ologit"]
    model_name = None
    entry = None
    for c in candidates:
        e = state.models.get(c)
        if isinstance(e, dict) and e.get("_fit") is not None:
            model_name, entry = c, e
            break
    if entry is None:
        return _empty("没有可用的已拟合模型(_fit),请先运行 glm/mlogit/ologit", want)

    res = entry["_fit"]
    predictors = list(entry.get("predictors") or [])

    # exogenous names in the fitted design (const + predictors); AME reported for
    # the non-constant covariates only.
    exog_names = ["const"] + predictors

    ame: dict[str, float] = {}
    se: dict[str, float] = {}
    note = ""

    # linear Gaussian GLM: AME == slope (closed form), report directly.
    if model_name == "glm" and str(entry.get("family", "")).lower() == "gaussian":
        coef = entry.get("coef", {})
        se_map = entry.get("se", {})
        for p in predictors:
            if p in coef:
                ame[p] = float(coef[p])
                se[p] = float(se_map.get(p, float("nan")))
        note = "线性(高斯)GLM:AME = 斜率系数"
        state.write("diagnostics", "margins",
                    {"model": model_name, "ame": ame, "se": se, "at": at,
                     "note": note})
        return state

    # discrete / nonlinear models: use statsmodels get_margeff where available.
    got = False
    if hasattr(res, "get_margeff"):
        try:
            meff = res.get_margeff(at=at)
            m = np.asarray(meff.margeff, dtype=float)
            ms = np.asarray(meff.margeff_se, dtype=float)
            # get_margeff drops the constant → align to predictors
            if m.ndim == 1 and m.shape[0] == len(predictors):
                for p, v, s in zip(predictors, m, ms):
                    ame[p] = float(v)
                    se[p] = float(s)
                got = True
                note = f"statsmodels get_margeff(at={at})"
            elif m.ndim == 2:
                # MNLogit: (n_predictors, K-1) — average |effect| across outcomes,
                # and also expose the per-outcome effects.
                per_outcome = {}
                cats = [c for c in (entry.get("categories") or [])][1:]
                for j in range(m.shape[1]):
                    lbl = cats[j] if j < len(cats) else f"cat{j+1}"
                    per_outcome[str(lbl)] = {
                        p: float(m[i, j]) for i, p in enumerate(predictors)
                    }
                for i, p in enumerate(predictors):
                    ame[p] = float(np.mean(m[i, :]))
                    se[p] = float(np.mean(ms[i, :])) if ms.ndim == 2 else float("nan")
                note = (f"statsmodels get_margeff(at={at});多项:ame=各类别均值,"
                        f"per_outcome 给每个类别")
                state.write("diagnostics", "margins", {
                    "model": model_name, "ame": ame, "se": se,
                    "per_outcome": per_outcome, "at": at, "note": note,
                })
                return state
        except Exception:
            got = False

    if not got:
        # numerical AME: average of finite-difference derivatives of the mean
        # predicted response w.r.t. each predictor.
        try:
            df = _get_datasets(state, kwargs)
            model = getattr(res, "model", None)
            exog = getattr(model, "exog", None)
            if exog is None:
                raise ValueError("拟合对象没有 exog,无法做数值边际")
            exog = np.asarray(exog, dtype=float)
            # locate predictor columns in the design: for glm/mlogit const is col 0,
            # for ologit there is no const column.
            has_const = not (model_name == "ologit")
            base_pred = res.predict(exog)
            base_pred = np.asarray(base_pred, dtype=float)
            for idx, p in enumerate(predictors):
                col = idx + (1 if has_const else 0)
                h = np.std(exog[:, col]) * 1e-3
                if h == 0:
                    h = 1e-4
                bumped = exog.copy()
                bumped[:, col] = bumped[:, col] + h
                up = np.asarray(res.predict(bumped), dtype=float)
                deriv = (up - base_pred) / h
                if deriv.ndim == 2:
                    # ordered/multinomial predict → per-category probabilities;
                    # report the mean over rows of the mean |effect| across cats.
                    ame[p] = float(np.mean(np.mean(np.abs(deriv), axis=1)))
                else:
                    ame[p] = float(np.mean(deriv))
                se[p] = float("nan")
            note = "数值差分平均边际效应(get_margeff 不可用时的回退)"
        except Exception as exc:
            return _empty(f"边际效应计算失败: {exc!s}", model_name)

    state.write("diagnostics", "margins",
                {"model": model_name, "ame": ame, "se": se, "at": at, "note": note})
    return state


# --------------------------------------------------------------------- self-test
if __name__ == "__main__":
    from socialverse import datasets as ds

    df = ds.load_regression()
    # a large-sample draw for provable parameter recovery of the nonlinear GLMs,
    # where the n=600/seed=0 default sample carries genuine sampling noise
    # (the estimator itself is exact — it matches statsmodels to 4 dp — but the
    # MLE for logit/poisson at n=600 can sit ~0.2 off the DGP truth).
    df_big = ds.load_regression(n=40000, seed=1)

    def _fresh(outcome: str, frame: pd.DataFrame = df) -> StudyState:
        return StudyState(
            sources={"datasets": frame},
            variables={"outcome": outcome},
        )

    print("=" * 68)
    print("glm — parameter recovery (truth vs recovered, tol ±0.15)")
    print("=" * 68)

    # OLS: truth const 1.0, x1 0.5, x2 -0.4  (recovered at default n=600)
    s = glm(_fresh("y"), predictors=["x1", "x2"], family="gaussian")
    c = s.models["glm"]["coef"]
    print(f"OLS[n=600]     const  truth  1.00  recovered {c['const']:+.3f}")
    print(f"OLS[n=600]     x1     truth +0.50  recovered {c['x1']:+.3f}")
    print(f"OLS[n=600]     x2     truth -0.40  recovered {c['x2']:+.3f}")
    print(f"               r2={s.diagnostics['glm_fit']['r2']:.3f}")
    assert abs(c["x1"] - 0.5) < 0.15 and abs(c["x2"] + 0.4) < 0.15

    # Logit: truth x1 0.8, x2 -0.5.  Show n=600 (honest, ~0.18 off), assert on
    # the large sample where the MLE provably converges to the DGP truth.
    s600 = glm(_fresh("y_bin"), predictors=["x1", "x2"], family="binomial")
    c600 = s600.models["glm"]["coef"]
    print(f"logit[n=600]   x1     truth +0.80  recovered {c600['x1']:+.3f} "
          f"(sampling noise)")
    s = glm(_fresh("y_bin", df_big), predictors=["x1", "x2"], family="binomial")
    c = s.models["glm"]["coef"]
    print(f"logit[n=40k]   x1     truth +0.80  recovered {c['x1']:+.3f}")
    print(f"logit[n=40k]   x2     truth -0.50  recovered {c['x2']:+.3f}")
    print(f"               pseudo_r2={s.diagnostics['glm_fit']['pseudo_r2']:.3f}")
    assert abs(c["x1"] - 0.8) < 0.15 and abs(c["x2"] + 0.5) < 0.15

    # Poisson: truth x1 0.4, x2 -0.1
    s600 = glm(_fresh("y_count"), predictors=["x1", "x2"], family="poisson")
    print(f"poisson[n=600] x1     truth +0.40  recovered "
          f"{s600.models['glm']['coef']['x1']:+.3f}")
    s = glm(_fresh("y_count", df_big), predictors=["x1", "x2"], family="poisson")
    c = s.models["glm"]["coef"]
    print(f"poisson[n=40k] x1     truth +0.40  recovered {c['x1']:+.3f}")
    print(f"poisson[n=40k] x2     truth -0.10  recovered {c['x2']:+.3f}")
    assert abs(c["x1"] - 0.4) < 0.15 and abs(c["x2"] + 0.1) < 0.15

    # robust / cluster smoke
    s = glm(_fresh("y"), predictors=["x1", "x2"], family="gaussian", cov="robust")
    print(f"OLS-HC1  x1 se recovered {s.models['glm']['se']['x1']:.3f} (robust ok)")

    print()
    print("=" * 68)
    print("mlogit — coefficient direction (B rises in x1, C falls in x1)")
    print("=" * 68)
    s = mlogit(_fresh("choice"), predictors=["x1", "x2"])
    m = s.models["mlogit"]
    b_x1 = m["coef"]["B"]["x1"]["coef"]
    c_x1 = m["coef"]["C"]["x1"]["coef"]
    print(f"base={m['base']}  categories={m['categories']}  n={m['n']}")
    print(f"B: x1 coef {b_x1:+.3f}  (truth: > 0, B rises in x1)")
    print(f"C: x1 coef {c_x1:+.3f}  (truth: < 0, C falls in x1)")
    assert b_x1 > 0 and c_x1 < 0

    print()
    print("=" * 68)
    print("ologit — x1 coefficient positive (y_ord monotone in x1)")
    print("=" * 68)
    s = ologit(_fresh("y_ord"), predictors=["x1", "x2"], link="logit")
    o = s.models["ologit"]
    print(f"x1 coef {o['coef']['x1']:+.3f}  (truth: > 0)")
    print(f"x2 coef {o['coef']['x2']:+.3f}  (truth: < 0, from -0.5*x2)")
    print(f"thresholds {o['thresholds']}  n_cat={o['n_categories']}")
    assert o["coef"]["x1"] > 0

    # probit link smoke
    s = ologit(_fresh("y_ord"), predictors=["x1", "x2"], link="probit")
    print(f"oprobit x1 coef {s.models['ologit']['coef']['x1']:+.3f} (probit ok)")

    print()
    print("=" * 68)
    print("margins — AME after glm(binomial); x1 AME finite & positive")
    print("=" * 68)
    s = _fresh("y_bin")
    s = glm(s, predictors=["x1", "x2"], family="binomial")
    s = margins(s, model="glm", at="overall")
    mg = s.diagnostics["margins"]
    print(f"model={mg['model']}  at={mg['at']}  note={mg['note']}")
    print(f"AME x1 {mg['ame']['x1']:+.4f}  (finite & > 0 expected)")
    print(f"AME x2 {mg['ame']['x2']:+.4f}  (< 0 expected)")
    assert np.isfinite(mg["ame"]["x1"]) and mg["ame"]["x1"] > 0
    assert np.isfinite(mg["ame"]["x2"]) and mg["ame"]["x2"] < 0

    # margins on linear GLM (AME == slope) and model=None autoselect
    s2 = glm(_fresh("y"), predictors=["x1", "x2"], family="gaussian")
    s2 = margins(s2, model=None)
    print(f"linear AME x1 {s2.diagnostics['margins']['ame']['x1']:+.3f} "
          f"(== OLS slope 0.5)")

    # graceful degradation: datasets present (requires satisfied) but the named
    # outcome column is absent → empty model with an honest note, no raise.
    empty = glm(
        StudyState(sources={"datasets": df}, variables={"outcome": "nope"}),
        family="gaussian",
    )
    assert empty.models["glm"]["_fit"] is None
    print("\nmissing-outcome glm degrades gracefully:",
          empty.models["glm"]["note"])

    print("\nALL SELF-TESTS PASSED")
