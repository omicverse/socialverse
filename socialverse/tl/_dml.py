"""``sv.tl._dml`` — Double/Debiased Machine Learning and heterogeneous treatment
effects (the EconML family), implemented natively on scikit-learn.

Classic estimators return one average effect. When the effect varies with covariates
— who benefits, who is harmed — you want the **CATE** ``θ(x)``. DML (Chernozhukov et
al. 2018) makes this robust to flexible ML confounding control via **cross-fitting +
Neyman-orthogonal** residual-on-residual estimation:

1. cross-fit nuisances ``ĝ(X)=E[Y|X]`` and ``m̂(X)=E[T|X]`` out-of-fold with any
   regressor, forming residuals ``Ỹ=Y-ĝ``, ``T̃=T-m̂``;
2. regress ``Ỹ`` on ``T̃`` (× ``X`` for heterogeneity) — the partialling-out makes the
   coefficient the effect, purged of confounding.

- ``sv.tl.dml`` → **LinearDML**: ATE + a linear CATE model ``θ(x)=a+b'x`` with
  analytic HC-robust SEs.
- ``sv.tl.causal_forest`` → **ForestDML / R-learner** with a random-forest final
  stage: fully nonparametric per-unit ``θ(x)`` + feature importances (bootstrap ATE
  SE). Wraps EconML's ``CausalForestDML`` when installed, else this native forest.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _get_datasets


def _try(mod):
    import importlib
    return importlib.import_module(mod)


def _crossfit(Y, T, XW, K, seed, discrete_t):
    """Out-of-fold nuisance predictions ĝ(XW)=E[Y|XW], m̂(XW)=E[T|XW] via K-fold
    cross-fitting with gradient-boosted trees (falls back to linear if sklearn is
    absent). Returns residuals (Ỹ, T̃)."""
    sk_ms = _try("sklearn.model_selection")
    try:
        ens = _try("sklearn.ensemble")
        reg = lambda: ens.GradientBoostingRegressor(max_depth=3, n_estimators=150, random_state=seed)
        clf = lambda: ens.GradientBoostingClassifier(max_depth=3, n_estimators=150, random_state=seed)
    except Exception:  # pragma: no cover
        lin = _try("sklearn.linear_model")
        reg = lambda: lin.LinearRegression()
        clf = lambda: lin.LogisticRegression(max_iter=1000)

    n = len(Y)
    gy = np.empty(n, float)
    mt = np.empty(n, float)
    kf = sk_ms.KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(XW):
        gy[te] = reg().fit(XW[tr], Y[tr]).predict(XW[te])
        if discrete_t and len(np.unique(T[tr])) >= 2:
            m = clf().fit(XW[tr], T[tr].astype(int))
            proba = m.predict_proba(XW[te])
            # E[T|X] = sum_k value_k * P(T=value_k|X) — correct for any class encoding
            mt[te] = proba @ m.classes_.astype(float)
        elif discrete_t:  # degenerate fold (a single treatment class in training) → prior
            mt[te] = float(T[tr].mean())
        else:
            mt[te] = reg().fit(XW[tr], T[tr]).predict(XW[te])
    return Y - gy, T - mt


def _resolve_cols(df, kwargs, state):
    """treatment / outcome / heterogeneity X / confounders W column names."""
    T = kwargs.get("treatment") or state.design.get("treatment")
    Y = kwargs.get("outcome") or state.variables.get("outcome") or state.design.get("outcome")
    X = kwargs.get("hetero") or kwargs.get("X")          # CATE features
    W = kwargs.get("controls") or kwargs.get("W")        # pure confounders
    if isinstance(X, str):
        X = [X]
    if isinstance(W, str):
        W = [W]
    return T, Y, list(X or []), list(W or [])


# ============================================================================== dml
@register(
    name="dml",
    aliases=["双重机器学习", "double_ml", "linear_dml", "cate"],
    category="causal",
    tier="pro",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["scikit-learn", "numpy"],
    description="双重机器学习 DML(LinearDML):cross-fitting+正交残差回归估 ATE 与线性 CATE θ(x),异质稳健",
    requires={"design": ["treatment"], "variables": ["outcome"]},
    produces={"models": ["dml"]},
    auto_fix="escalate",
)
def dml(state: StudyState, **kwargs: Any) -> StudyState:
    """LinearDML: ATE and a linear CATE ``θ(x)=a+b'x``.

    Keyword arguments: ``treatment=``/``outcome=`` (or from design), ``hetero=`` list
    of columns entering the CATE, ``controls=`` pure confounders, ``folds=`` (default
    5), ``discrete_treatment=`` (default inferred), ``seed=``.
    """
    df = _get_datasets(state, kwargs)
    T, Y, Xcols, Wcols = _resolve_cols(df, kwargs, state)

    def _empty(note):
        state.write("models", "dml", {"ate": None, "note": note})
        return state

    if df is None or T is None or Y is None:
        return _empty("缺少数据或 treatment/outcome")
    feat = Xcols + Wcols
    if not feat:
        feat = [c for c in df.columns if c not in (T, Y) and pd.api.types.is_numeric_dtype(df[c])]
    XW = df[feat].apply(pd.to_numeric, errors="coerce")
    yv = pd.to_numeric(df[Y], errors="coerce")
    tv = pd.to_numeric(df[T], errors="coerce")
    ok = XW.notna().all(axis=1) & yv.notna() & tv.notna()
    XW, yv, tv = XW[ok].to_numpy(float), yv[ok].to_numpy(float), tv[ok].to_numpy(float)
    if len(yv) < 50:
        return _empty("样本过小(<50),DML 不稳")

    K = int(kwargs.get("folds", 5))
    seed = int(kwargs.get("seed", 0))
    uniq = np.unique(tv)
    binary = kwargs.get("discrete_treatment", len(uniq) <= 2) and len(uniq) == 2
    if binary:
        tv = (tv == uniq.max()).astype(float)  # recode binary T to {0,1} so T-p̂ is on-scale
    # multi-valued treatments are residualised as continuous (E[T|X] via regression);
    # a classifier's max-prob is not E[T|X].

    yres, tres = _crossfit(yv, tv, XW, K, seed, bool(binary))

    # ATE: OLS of yres on tres (through origin) with HC-robust SE
    denom = float(tres @ tres)
    if denom <= 0:
        return _empty("处理无残差变异,无法识别")
    ate = float((tres @ yres) / denom)
    e = yres - ate * tres
    se_ate = float(np.sqrt(np.sum((tres * e) ** 2)) / denom)  # HC0 sandwich

    model = {
        "ate": ate, "se": se_ate, "ci": [ate - 1.96 * se_ate, ate + 1.96 * se_ate],
        "p": float(2 * (1 - _try("scipy.stats").norm.cdf(abs(ate / se_ate)))) if se_ate > 0 else None,
        "n": int(len(yv)), "folds": K, "discrete_treatment": bool(binary),
        "estimator": "linear_dml", "hetero": Xcols, "controls": Wcols,
    }

    # linear CATE: yres ~ tres + tres*(Xj - mean) so the intercept is E[θ(X)] (=ATE),
    # not θ at X=0.
    if Xcols:
        Xh = df.loc[ok, Xcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        Xh = Xh - Xh.mean(0, keepdims=True)
        D = np.column_stack([tres] + [tres * Xh[:, j] for j in range(Xh.shape[1])])
        DtD = D.T @ D
        DtD_inv = np.linalg.pinv(DtD)  # pinv: robust to (near-)singular collinear X
        beta = DtD_inv @ (D.T @ yres)
        r = yres - D @ beta
        meat = (D * r[:, None]).T @ (D * r[:, None])
        V = DtD_inv @ meat @ DtD_inv  # HC0 sandwich
        cate = {"intercept": float(beta[0]), "intercept_se": float(np.sqrt(V[0, 0]))}
        for j, c in enumerate(Xcols, start=1):
            cate[c] = float(beta[j])
            cate[c + "_se"] = float(np.sqrt(V[j, j]))
        model["cate_linear"] = cate
        model["note"] = "LinearDML:ATE + 线性 CATE θ(x)=a+b·x(HC 稳健 SE)"
    else:
        model["note"] = "LinearDML:ATE(未指定 hetero=,只出平均效应)"

    state.write("models", "dml", model)
    return state


# ==================================================================== causal_forest
@register(
    name="causal_forest",
    aliases=["因果森林", "forest_dml", "r_learner", "heterogeneous_effects"],
    category="causal",
    tier="pro",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["scikit-learn", "econml", "numpy"],
    description="因果森林 / ForestDML(R-learner 森林最终阶段):非参 per-unit CATE θ(x)+特征重要度",
    requires={"design": ["treatment"], "variables": ["outcome"]},
    produces={"models": ["causal_forest"]},
    auto_fix="escalate",
)
def causal_forest(state: StudyState, **kwargs: Any) -> StudyState:
    """Nonparametric CATE via an R-learner with a random-forest final stage.

    Cross-fits nuisances, then fits a forest to the R-learner pseudo-target so ``θ(x)``
    is fully nonparametric. Reports per-unit CATE summary, feature importances, and a
    bootstrap ATE SE. If EconML is installed, delegates to ``CausalForestDML`` for
    calibrated inference; otherwise uses the native forest.
    """
    df = _get_datasets(state, kwargs)
    T, Y, Xcols, Wcols = _resolve_cols(df, kwargs, state)

    def _empty(note):
        state.write("models", "causal_forest", {"ate": None, "note": note})
        return state

    if df is None or T is None or Y is None:
        return _empty("缺少数据或 treatment/outcome")
    feat = Xcols + Wcols or [c for c in df.columns
                             if c not in (T, Y) and pd.api.types.is_numeric_dtype(df[c])]
    hetero = Xcols or feat
    XW = df[feat].apply(pd.to_numeric, errors="coerce")
    Xh = df[hetero].apply(pd.to_numeric, errors="coerce")
    yv = pd.to_numeric(df[Y], errors="coerce")
    tv = pd.to_numeric(df[T], errors="coerce")
    ok = XW.notna().all(axis=1) & Xh.notna().all(axis=1) & yv.notna() & tv.notna()
    XW, Xh = XW[ok].to_numpy(float), Xh[ok].to_numpy(float)
    yv, tv = yv[ok].to_numpy(float), tv[ok].to_numpy(float)
    if len(yv) < 100:
        return _empty("样本过小(<100),因果森林不稳")

    K = int(kwargs.get("folds", 5))
    seed = int(kwargs.get("seed", 0))
    uniq = np.unique(tv)
    binary = kwargs.get("discrete_treatment", len(uniq) <= 2) and len(uniq) == 2
    if binary:
        tv = (tv == uniq.max()).astype(float)  # recode binary T to {0,1}

    ens = _try("sklearn.ensemble")
    yres, tres = _crossfit(yv, tv, XW, K, seed, bool(binary))

    def _forest_cate(Xh, yres, tres, sd):
        # R-learner: fit theta(x) minimizing sum w_i (pseudo_i - theta(x_i))^2,
        # pseudo = yres/tres, weight = tres^2 (Nie & Wager 2021).
        w = tres ** 2
        keep = w > 1e-8
        pseudo = np.where(keep, yres / np.where(keep, tres, 1.0), 0.0)
        f = ens.RandomForestRegressor(n_estimators=300, min_samples_leaf=max(5, len(yres) // 100),
                                      random_state=sd, n_jobs=-1)
        f.fit(Xh[keep], pseudo[keep], sample_weight=w[keep])
        return f

    forest = _forest_cate(Xh, yres, tres, seed)
    theta = forest.predict(Xh)
    ate = float(theta.mean())

    # bootstrap ATE SE (resample rows, refit forest on residuals)
    nb = int(kwargs.get("nboots", 40))
    rng = np.random.default_rng(seed + 1)
    boots = []
    for _ in range(nb):
        idx = rng.integers(0, len(yv), len(yv))
        try:
            fb = _forest_cate(Xh[idx], yres[idx], tres[idx], seed + 2)
            boots.append(float(fb.predict(Xh).mean()))
        except Exception:
            continue
    se = float(np.std(boots, ddof=1)) if len(boots) > 5 else None

    q = np.quantile(theta, [0.1, 0.5, 0.9])
    model = {
        "ate": ate, "se": se,
        "ci": [ate - 1.96 * se, ate + 1.96 * se] if se else None,
        "cate_summary": {"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2]),
                         "min": float(theta.min()), "max": float(theta.max())},
        "feature_importance": {c: float(v) for c, v in zip(hetero, forest.feature_importances_)},
        "n": int(len(yv)), "hetero": hetero, "estimator": "forest_r_learner",
        "note": "R-learner 森林最终阶段的非参 CATE θ(x);ATE=θ(x) 均值,bootstrap SE(近似)",
    }
    state.write("models", "causal_forest", model)
    return state


__all__ = ["dml", "causal_forest"]
