"""``sv.tl._hte`` — heterogeneous & distributional treatment effects beyond DML:
**meta-learners** (S/T/X-learner) for the CATE, and **quantile treatment effects**
for the effect across the outcome distribution.

- ``metalearners`` — Künzel et al. (2019): estimate ``θ(x)`` by combining base ML
  regressors under the S- (single model with T as a feature), T- (separate treated /
  control models), or X-learner (T-learner + propensity-weighted imputation) meta-
  strategy. A model-agnostic complement to ``dml`` / ``causal_forest``.
- ``qte`` — quantile treatment effects: the treatment's effect at each quantile of the
  outcome distribution (inequality / distributional questions the mean effect hides),
  with an optional propensity re-weighting for observational data. Bootstrap SEs.
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


def _cols_hte(df, kwargs, state):
    T = kwargs.get("treatment") or state.design.get("treatment")
    Y = kwargs.get("outcome") or state.variables.get("outcome")
    X = kwargs.get("hetero") or kwargs.get("X") or kwargs.get("features")
    if isinstance(X, str):
        X = [X]
    if not X and df is not None:
        X = [c for c in df.columns if c not in (T, Y) and pd.api.types.is_numeric_dtype(df[c])]
    return T, Y, list(X or [])


# ==================================================================== metalearners
@register(
    name="metalearners",
    aliases=["元学习器", "meta_learners", "t_learner", "x_learner", "s_learner"],
    category="causal",
    tier="pro",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["scikit-learn", "numpy"],
    description="元学习器 S/T/X-learner:用基学习器估异质处理效应 CATE θ(x)(dml/causal_forest 的模型无关补充)",
    requires={"design": ["treatment"], "variables": ["outcome"]},
    produces={"models": ["metalearners"]},
    auto_fix="escalate",
)
def metalearners(state: StudyState, **kwargs: Any) -> StudyState:
    """S/T/X meta-learners for the CATE.

    Keyword arguments: ``learner=`` one of ``"T"`` (default), ``"S"``, ``"X"``,
    ``"all"``; ``hetero=`` feature columns; ``treatment=``/``outcome=`` (binary
    treatment). Reports ATE (mean CATE), the per-unit CATE spread, and — for the X-
    learner — propensity-weighted combination.
    """
    df = _get_datasets(state, kwargs)
    T, Y, Xcols = _cols_hte(df, kwargs, state)

    def _empty(note):
        state.write("models", "metalearners", {"ate": None, "note": note})
        return state

    if df is None or T is None or Y is None or not Xcols:
        return _empty("缺少 data / treatment / outcome / 特征")
    ens = _try("sklearn.ensemble")
    X = df[Xcols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[Y], errors="coerce")
    t = pd.to_numeric(df[T], errors="coerce")
    ok = X.notna().all(axis=1) & y.notna() & t.notna()
    X, y, t = X[ok].to_numpy(float), y[ok].to_numpy(float), t[ok].to_numpy(float)
    uniq = np.unique(t)
    if len(uniq) != 2:
        return _empty("metalearners 需二值处理")
    t = (t == uniq.max()).astype(int)
    if t.sum() < 10 or (1 - t).sum() < 10:
        return _empty("某组样本过小")

    learner = str(kwargs.get("learner", "T")).upper()
    seed = int(kwargs.get("seed", 0))

    def reg():
        return ens.GradientBoostingRegressor(max_depth=3, n_estimators=150, random_state=seed)

    cates = {}
    # T-learner
    m1 = reg().fit(X[t == 1], y[t == 1])
    m0 = reg().fit(X[t == 0], y[t == 0])
    cates["T"] = m1.predict(X) - m0.predict(X)
    # S-learner
    if learner in ("S", "ALL"):
        Xt = np.column_stack([X, t])
        ms = reg().fit(Xt, y)
        cates["S"] = (ms.predict(np.column_stack([X, np.ones(len(y))]))
                      - ms.predict(np.column_stack([X, np.zeros(len(y))])))
    # X-learner
    if learner in ("X", "ALL"):
        d1 = y[t == 1] - m0.predict(X[t == 1])   # treated: obs - imputed control
        d0 = m1.predict(X[t == 0]) - y[t == 0]   # control: imputed treated - obs
        tau1 = reg().fit(X[t == 1], d1)
        tau0 = reg().fit(X[t == 0], d0)
        clf = ens.GradientBoostingClassifier(max_depth=3, n_estimators=150, random_state=seed)
        e = clf.fit(X, t).predict_proba(X)[:, 1]
        cates["X"] = e * tau0.predict(X) + (1 - e) * tau1.predict(X)

    chosen = cates.get(learner, cates["T"])
    q = np.quantile(chosen, [0.1, 0.5, 0.9])
    model = {
        "ate": float(chosen.mean()),
        "cate_summary": {"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2]),
                         "min": float(chosen.min()), "max": float(chosen.max())},
        "learner": learner if learner in cates else "T",
        "ate_by_learner": {k: float(v.mean()) for k, v in cates.items()},
        "n": int(len(y)), "hetero": Xcols, "estimator": f"{learner.lower()}_learner",
        "note": "元学习器 CATE θ(x)(S/T/X);ATE=θ(x) 均值。异质效应用 sv.pl.forest 可视化",
    }
    state.write("models", "metalearners", model)
    return state


# ============================================================================== qte
@register(
    name="qte",
    aliases=["分位处理效应", "quantile_treatment_effect", "quantile_te"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "scikit-learn"],
    description="分位处理效应 QTE:处理在结果分布各分位的效应(不平等/分布视角),可倾向加权 + bootstrap SE",
    requires={"design": ["treatment"], "variables": ["outcome"]},
    produces={"models": ["qte"]},
    auto_fix="escalate",
)
def qte(state: StudyState, **kwargs: Any) -> StudyState:
    """Quantile treatment effects across the outcome distribution.

    For each quantile τ reports ``Q_τ(Y|treated) - Q_τ(Y|control)``. With covariates
    (``adjust=`` columns) it re-weights by the estimated propensity (IPW) so the QTE is
    identified under selection-on-observables; otherwise it is the unconditional QTE
    (valid under random assignment). Bootstrap standard errors over units.

    Keyword arguments: ``quantiles=`` (default ``[.1,.25,.5,.75,.9]``), ``adjust=``
    covariates for IPW, ``nboots=`` (default 200), ``seed=``.
    """
    df = _get_datasets(state, kwargs)
    T = kwargs.get("treatment") or state.design.get("treatment")
    Y = kwargs.get("outcome") or state.variables.get("outcome")
    adjust = kwargs.get("adjust") or []
    if isinstance(adjust, str):
        adjust = [adjust]
    qs = list(kwargs.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]))
    nb = int(kwargs.get("nboots", 200))
    seed = int(kwargs.get("seed", 42))

    def _empty(note):
        state.write("models", "qte", {"qte": {}, "note": note})
        return state

    if df is None or T is None or Y is None:
        return _empty("缺少 data / treatment / outcome")
    y = pd.to_numeric(df[Y], errors="coerce")
    t = pd.to_numeric(df[T], errors="coerce")
    A = (df[list(adjust)].apply(pd.to_numeric, errors="coerce") if adjust
         else pd.DataFrame(index=df.index))
    ok = y.notna() & t.notna() & (A.notna().all(axis=1) if adjust else True)
    y, t = y[ok].to_numpy(float), t[ok].to_numpy(float)
    A = A[ok].to_numpy(float) if adjust else None
    uniq = np.unique(t)
    if len(uniq) != 2:
        return _empty("qte 需二值处理")
    t = (t == uniq.max()).astype(int)

    def _wquantile(v, q, w=None):
        if w is None:
            return float(np.quantile(v, q))
        idx = np.argsort(v)
        v, w = v[idx], w[idx]
        cw = np.cumsum(w) - 0.5 * w
        cw /= w.sum()
        return float(np.interp(q, cw, v))

    def _ps(A, t):
        if A is None:
            return None
        clf = _try("sklearn.linear_model").LogisticRegression(max_iter=1000)
        return clf.fit(A, t).predict_proba(A)[:, 1]

    def _point(y, t, A):
        e = _ps(A, t) if A is not None else None
        w1 = 1.0 / np.clip(e, 0.02, 0.98) if e is not None else None
        w0 = 1.0 / np.clip(1 - e, 0.02, 0.98) if e is not None else None
        out = {}
        for q in qs:
            q1 = _wquantile(y[t == 1], q, None if w1 is None else w1[t == 1])
            q0 = _wquantile(y[t == 0], q, None if w0 is None else w0[t == 0])
            out[q] = q1 - q0
        return out

    point = _point(y, t, A)
    rng = np.random.default_rng(seed)
    boot = {q: [] for q in qs}
    n = len(y)
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        try:
            b = _point(y[idx], t[idx], A[idx] if A is not None else None)
            for q in qs:
                boot[q].append(b[q])
        except Exception:
            continue
    from scipy import stats
    res = {}
    for q in qs:
        arr = np.array(boot[q])
        se = float(np.std(arr, ddof=1)) if arr.size > 5 else None
        p = float(2 * (1 - stats.norm.cdf(abs(point[q] / se)))) if se and se > 0 else None
        res[str(q)] = {"qte": float(point[q]), "se": se,
                       "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
                       if arr.size > 5 else None, "p": p}
    state.write("models", "qte", {
        "qte": res, "quantiles": qs, "adjusted": bool(adjust), "n": int(n),
        "estimator": "ipw_qte" if adjust else "unconditional_qte",
        "note": "分位处理效应:各分位 Q_τ(Y|1)-Q_τ(Y|0)"
                + ("(倾向加权 IPW)" if adjust else "(无条件,随机分配下有效)"),
    })
    return state


__all__ = ["metalearners", "qte"]
