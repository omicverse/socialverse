"""``sv.tl._matching`` — registered implementation for the causal-inference gap:
**propensity-score matching / inverse-probability weighting (PSM / IPW)**.

The workhorse of observational causal inference in the social sciences, ported to
the ``StudyState`` / ``registry`` spine with *real* estimation (recovers a known
average treatment effect on the treated, ATT), no placeholders.

Champion packages this file mirrors
-----------------------------------
* ``psm`` — R's ``MatchIt`` (nearest-neighbour matching), ``Matching`` (Abadie-
  Imbens), and ``twang`` / ``WeightIt`` (IPW); Python ``causalinference`` /
  ``DoWhy``. The propensity score ``p = P(treat = 1 | X)`` is estimated by a
  logistic regression (``statsmodels.Logit``, or a numpy IRLS fallback if
  ``statsmodels`` is unavailable). Two estimators of the ATT are provided:

  - ``method="nn"`` — 1:1 nearest-neighbour matching on the (linear-predictor,
    or raw-probability) propensity score, with an optional caliper. The ATT is
    the mean paired difference ``mean(y_treated - y_matched_control)``, with a
    bootstrap standard error over resampled treated units.
  - ``method="ipw"`` — inverse-probability weighting with **ATT weights**
    (weight 1 for treated units, ``p/(1-p)`` for controls); the ATT is the
    weighted mean difference, with an analytic (weighted, robust-ish) SE plus a
    bootstrap cross-check.

  Covariate balance is reported as the standardized mean difference (SMD) of each
  covariate before and after adjustment — R's ``cobalt::bal.tab`` / ``MatchIt``'s
  ``summary`` love-plot statistic. A well-specified match/weight drives every
  post-adjustment SMD toward zero.

The registry contract: ``psm`` ``requires`` a working ``sources['datasets']``
frame, a declared ``design['treatment']`` (or a ``treatment=`` kwarg), and a
``variables['outcome']``; it ``produces`` the fitted ``models['psm']`` (with the
ATT, its SE, the naive treated-minus-control difference, and match bookkeeping)
plus ``diagnostics['balance']`` (pre/post SMD) — so a resolver can refuse to
report an ATT as "credible" until balance has actually been checked.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["psm"]


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


def _fit_propensity(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Estimate ``p = P(treat=1 | X)`` by logistic regression.

    Prefers the parity-gated pure-Python ``pymatchit.glm_logit_ps`` port (an
    R ``glm.fit`` IRLS reproduction, verified to 1e-6 against R's MatchIt); if
    that raises for any reason, falls back to ``statsmodels.Logit`` and, failing
    that, a plain numpy IRLS (Newton) fit of the same model. Returns fitted
    probabilities clipped away from 0/1 for numerical stability of the IPW
    weights.
    """
    # -- faithful port first (proven 1e-6 vs R MatchIt) --------------------
    try:
        from ..external.pymatchit import glm_logit_ps

        _, p_port = glm_logit_ps(np.asarray(X, float), np.asarray(t, float))
        p_port = np.asarray(p_port, dtype=float)
        if p_port.shape == (len(t),) and np.all(np.isfinite(p_port)):
            return np.clip(p_port, 1e-6, 1.0 - 1e-6)
    except Exception:
        pass

    Xd = np.column_stack([np.ones(len(t)), X])
    sm = _try_import("statsmodels.api")
    p = None
    if sm is not None:
        try:
            res = sm.Logit(t.astype(float), Xd).fit(disp=0, maxiter=100)
            p = np.asarray(res.predict(Xd), dtype=float)
        except Exception:
            p = None
    if p is None:
        # numpy IRLS fallback
        beta = np.zeros(Xd.shape[1])
        for _ in range(100):
            eta = Xd @ beta
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            W = mu * (1.0 - mu)
            W = np.clip(W, 1e-9, None)
            z = eta + (t - mu) / W
            XtW = Xd.T * W
            try:
                beta_new = np.linalg.solve(XtW @ Xd, XtW @ z)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(beta_new - beta)) < 1e-10:
                beta = beta_new
                break
            beta = beta_new
        eta = Xd @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _smd(x_t: np.ndarray, x_c: np.ndarray,
         w_t: np.ndarray | None = None, w_c: np.ndarray | None = None) -> float:
    """Standardized mean difference of a covariate between treated and control.

    ``SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)``, with the pooled SD
    taken from the *unweighted* (pre-adjustment) groups so the pre/post numbers
    share a common yardstick (R's ``cobalt`` default, ``s.d.denom = "pooled"``).
    Optional weights allow the post-adjustment (weighted) means.
    """
    if w_t is None:
        m_t = float(np.mean(x_t))
    else:
        m_t = float(np.average(x_t, weights=w_t))
    if w_c is None:
        m_c = float(np.mean(x_c))
    else:
        m_c = float(np.average(x_c, weights=w_c))
    v_t = float(np.var(x_t, ddof=1)) if len(x_t) > 1 else 0.0
    v_c = float(np.var(x_c, ddof=1)) if len(x_c) > 1 else 0.0
    pooled = np.sqrt((v_t + v_c) / 2.0)
    if pooled <= 0:
        return 0.0
    return (m_t - m_c) / pooled


# --------------------------------------------------------------------------- psm
@register(
    name="psm",
    aliases=["倾向得分匹配", "PSM", "psmatch", "matching"],
    category="causal",
    tier="plus",
    skill="(倾向得分匹配 缺口)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="倾向得分匹配/IPW:logit 估倾向得分 → 最近邻 1:1 匹配(可选卡尺)或 ATT 权重 IPW → ATT + 匹配前后协变量平衡(SMD)",
    requires={"sources": ["datasets"], "design": ["treatment"], "variables": ["outcome"]},
    produces={"models": ["psm"], "diagnostics": ["balance"]},
    auto_fix="escalate",
)
def psm(state: StudyState, **kwargs: Any) -> StudyState:
    """Propensity-score matching / IPW estimate of the ATT.

    Estimates the propensity score ``p = P(treat=1 | covariates)`` by logistic
    regression, then estimates the average treatment effect on the treated (ATT)
    either by 1:1 nearest-neighbour matching (``method="nn"``) or by inverse-
    probability weighting with ATT weights (``method="ipw"``). Covariate balance
    (standardized mean difference of each covariate) is reported before and after
    adjustment.

    kwargs
    ------
    treatment : str
        Binary (0/1) treatment column. Default: ``design['treatment']`` → ``"treat"``.
    outcome : str
        Outcome column. Default: ``variables['outcome']`` → ``"y"``.
    covariates : list[str]
        Confounders entering the propensity model. Default: numeric columns other
        than treatment/outcome.
    method : {"nn", "ipw"}
        Estimator. ``"nn"`` = nearest-neighbour matching; ``"ipw"`` = ATT-weighted
        difference. Default ``"nn"``.
    caliper : float, optional
        Maximum absolute propensity-score distance for a valid NN match (on the
        raw-probability scale). Unmatched treated units are dropped.
    n_boot : int
        Bootstrap replications for the SE. Default 500.
    """
    df = _get_datasets(state, kwargs)
    method = str(kwargs.get("method", "nn")).lower()

    def _empty(note: str) -> StudyState:
        state.write("models", "psm", {
            "att": None, "se": None, "naive_diff": None, "method": method,
            "n_treated": 0, "n_matched": 0, "note": note,
        })
        state.write("diagnostics", "balance", {
            "smd_before": {}, "smd_after": {}, "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法进行倾向得分匹配")

    treatment = (kwargs.get("treatment")
                 or state.design.get("treatment") or "treat")
    outcome = (kwargs.get("outcome")
               or state.variables.get("outcome") or "y")
    if treatment not in df.columns:
        return _empty(f"缺少处理变量(treatment='{treatment}')")
    if outcome not in df.columns:
        return _empty(f"缺少结果变量(outcome='{outcome}')")

    covariates = list(kwargs.get("covariates") or [])
    if not covariates:
        covariates = [
            c for c in df.columns
            if c not in (treatment, outcome) and pd.api.types.is_numeric_dtype(df[c])
        ]
    covariates = [c for c in covariates if c in df.columns]
    if not covariates:
        return _empty("没有可用协变量(covariates),无法估计倾向得分")

    work = df[[outcome, treatment] + covariates].copy().dropna()
    # coerce treatment to 0/1
    t_raw = work[treatment].to_numpy()
    uniq = np.unique(t_raw)
    if set(np.unique(t_raw.astype(float))) <= {0.0, 1.0}:
        t = work[treatment].to_numpy(dtype=float)
    else:
        # map the larger-mean level to treated
        t = (t_raw == uniq[-1]).astype(float)
    if t.sum() == 0 or (1 - t).sum() == 0:
        return _empty("处理组或对照组为空,无法估计 ATT")

    y = work[outcome].to_numpy(dtype=float)
    X = work[covariates].to_numpy(dtype=float)

    treated_mask = t == 1
    control_mask = t == 0
    y_t, y_c = y[treated_mask], y[control_mask]
    X_t, X_c = X[treated_mask], X[control_mask]
    n_treated = int(treated_mask.sum())

    naive_diff = float(np.mean(y_t) - np.mean(y_c))

    # -- propensity score ------------------------------------------------------
    p = _fit_propensity(X, t)
    p_t, p_c = p[treated_mask], p[control_mask]

    # -- pre-adjustment balance ------------------------------------------------
    smd_before = {c: _smd(X_t[:, j], X_c[:, j]) for j, c in enumerate(covariates)}

    rng = np.random.default_rng(0)
    n_boot = int(kwargs.get("n_boot", 500))
    caliper = kwargs.get("caliper")

    if method == "ipw":
        # ATT weights: treated=1, control = p/(1-p)
        w = np.where(treated_mask, 1.0, p / (1.0 - p))
        w_t = w[treated_mask]
        w_c = w[control_mask]
        att = (float(np.average(y_t, weights=w_t))
               - float(np.average(y_c, weights=w_c)))
        n_matched = int(control_mask.sum())

        # analytic weighted SE (treated-mean var + weighted-control-mean var)
        var_t = float(np.var(y_t, ddof=1)) / max(len(y_t), 1)
        wc = w_c / w_c.sum()
        mc = float(np.average(y_c, weights=w_c))
        var_c = float(np.sum(wc**2 * (y_c - mc) ** 2))
        se_analytic = float(np.sqrt(max(var_t + var_c, 0.0)))

        # bootstrap cross-check (resample units, refit propensity)
        boot = []
        idx_all = np.arange(len(y))
        for _ in range(n_boot):
            bi = rng.choice(idx_all, size=len(idx_all), replace=True)
            tb, yb, Xb = t[bi], y[bi], X[bi]
            if tb.sum() == 0 or (1 - tb).sum() == 0:
                continue
            pb = _fit_propensity(Xb, tb)
            tm, cm = tb == 1, tb == 0
            wb_c = pb[cm] / (1.0 - pb[cm])
            try:
                att_b = (float(np.average(yb[tm]))
                         - float(np.average(yb[cm], weights=wb_c)))
            except ZeroDivisionError:
                continue
            boot.append(att_b)
        se_boot = float(np.std(boot, ddof=1)) if len(boot) > 1 else None
        se = se_boot if se_boot is not None else se_analytic

        # post-adjustment balance = weighted SMD
        smd_after = {
            c: _smd(X_t[:, j], X_c[:, j], w_t=w_t, w_c=w_c)
            for j, c in enumerate(covariates)
        }
        balance_note = "IPW(ATT 权重):加权后协变量标准化均值差(SMD)"

    else:  # nearest-neighbour 1:1 matching on the propensity score
        c_idx = np.arange(len(p_c))
        matched_c_y = []
        matched_c_rows = []  # indices into control arrays, for balance
        matched_t_rows = []

        # Greedy 1:1 nearest-neighbour on the propensity score WITH replacement
        # — socialverse psm's documented default, robust when n_treated exceeds
        # n_control (every treated unit gets its nearest control). The parity-
        # gated ``pymatchit.nearest_match`` provides R MatchIt's exact WITHOUT-
        # replacement greedy match for callers needing R-identical matched sets;
        # psm keeps with-replacement so it never drops treated units.
        for i in range(n_treated):
            dist = np.abs(p_c - p_t[i])
            j = int(np.argmin(dist))
            if caliper is not None and dist[j] > float(caliper):
                continue
            matched_c_y.append(y_c[j])
            matched_c_rows.append(j)
            matched_t_rows.append(i)
        n_matched = len(matched_t_rows)
        if n_matched == 0:
            return _empty("卡尺过严:没有匹配到任何对照单元")

        mt = np.array(matched_t_rows)
        mc = np.array(matched_c_rows)
        diffs = y_t[mt] - y_c[mc]
        att = float(np.mean(diffs))

        # bootstrap SE over resampled matched pairs
        boot = []
        for _ in range(n_boot):
            bi = rng.integers(0, len(diffs), size=len(diffs))
            boot.append(float(np.mean(diffs[bi])))
        se_boot = float(np.std(boot, ddof=1)) if len(boot) > 1 else None
        # analytic paired SE as a cross-check / fallback
        se_analytic = (float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                       if len(diffs) > 1 else None)
        se = se_boot if se_boot is not None else se_analytic

        # post-adjustment balance: treated used vs matched controls
        X_t_used = X_t[mt]
        X_c_used = X_c[mc]
        smd_after = {
            c: _smd(X_t_used[:, j], X_c_used[:, j])
            for j, c in enumerate(covariates)
        }
        balance_note = "最近邻 1:1 匹配:匹配后(处理 vs 匹配对照)协变量 SMD"

    max_smd_before = float(np.max(np.abs(list(smd_before.values())))) if smd_before else None
    max_smd_after = float(np.max(np.abs(list(smd_after.values())))) if smd_after else None

    # -- port-backed extra balance columns (Var. Ratio + eCDF mean/max) --------
    # Faithful MatchIt::summary(standardize=TRUE) columns via the parity-gated
    # ``pymatchit.balance_table``. Adds diagnostics; never touches SMD keys.
    balance_ext: dict[str, Any] = {}
    ps_weights_summary: dict[str, Any] | None = None
    try:
        from ..external.pymatchit import balance_table, get_w_from_ps

        # adjustment weights aligned to the full ``work`` frame (treated first).
        if method == "ipw":
            w_full = w  # ATT weights already computed above
        else:
            # NN 1:1 (with replacement): treated weight 1, each control weighted
            # by how many times it was used as a match.
            w_full = np.zeros(len(t), float)
            t_positions = np.flatnonzero(treated_mask)
            c_positions = np.flatnonzero(control_mask)
            w_full[t_positions[mt]] = 1.0
            for cj in mc:
                w_full[c_positions[cj]] += 1.0

        bt_before = balance_table(X, t, weights=None, covariates=covariates)
        bt_after = balance_table(X, t, weights=w_full, covariates=covariates)

        def _col(bt: dict, key: str) -> dict[str, float]:
            return {c: (float(v) if np.isfinite(v) else None)
                    for c, v in zip(bt["vars"], np.asarray(bt[key], float))}

        balance_ext = {
            "var_ratio_before": _col(bt_before, "var_ratio"),
            "var_ratio_after": _col(bt_after, "var_ratio"),
            "ecdf_mean_before": _col(bt_before, "ecdf_mean"),
            "ecdf_mean_after": _col(bt_after, "ecdf_mean"),
            "ecdf_max_before": _col(bt_before, "ecdf_max"),
            "ecdf_max_after": _col(bt_after, "ecdf_max"),
        }

        # ps balancing-weight summary from get_w_from_ps (ATT estimand, the
        # focal quantity psm targets).
        w_att = np.asarray(get_w_from_ps(p, t, estimand="ATT", treated=1), float)
        w_att_c = w_att[control_mask]
        n_eff_c = (float(np.sum(w_att_c) ** 2 / np.sum(w_att_c ** 2))
                   if np.sum(w_att_c ** 2) > 0 else 0.0)
        ps_weights_summary = {
            "estimand": "ATT",
            "min": float(np.min(w_att)),
            "max": float(np.max(w_att)),
            "mean": float(np.mean(w_att)),
            "sum": float(np.sum(w_att)),
            "control_min": float(np.min(w_att_c)) if w_att_c.size else None,
            "control_max": float(np.max(w_att_c)) if w_att_c.size else None,
            "control_mean": float(np.mean(w_att_c)) if w_att_c.size else None,
            "n_eff_control": n_eff_c,
            "note": ("ATT 平衡权重(处理组=1,对照=p/(1-p));"
                     "n_eff_control 为对照有效样本量,极端权重会显著压低它"),
        }
    except Exception as _exc:  # pragma: no cover - graceful degradation
        balance_ext = {"note_ext": f"扩展平衡统计不可用(port 调用失败:{_exc})"}
        ps_weights_summary = None

    state.write("models", "psm", {
        "att": float(att),
        "se": (float(se) if se is not None else None),
        "naive_diff": naive_diff,
        "method": method,
        "n_treated": n_treated,
        "n_matched": int(n_matched),
        "covariates": covariates,
        "caliper": (float(caliper) if caliper is not None else None),
        "estimator": ("statsmodels.Logit 倾向得分 + "
                      + ("IPW(ATT 权重)加权差" if method == "ipw"
                         else "最近邻 1:1 匹配配对差")),
        "note": ("ATT = 处理组结果 − 匹配/加权对照组结果;"
                 "naive_diff(未调整处理−对照差)有混杂偏误"),
        "backend": "pymatchit",
        "ps_weights_summary": ps_weights_summary,
    })
    state.write("diagnostics", "balance", {
        "smd_before": smd_before,
        "smd_after": smd_after,
        "max_smd_before": max_smd_before,
        "max_smd_after": max_smd_after,
        "note": (balance_note
                 + ";|SMD|<0.1 通常视为平衡良好,匹配/加权后应显著小于匹配前"),
        "backend": "pymatchit",
        **balance_ext,
    })
    return state


# ------------------------------------------------------------------- self-test
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    # confounders drive BOTH treatment assignment and outcome
    logit = -0.2 + 1.0 * x1 + 0.8 * x2 - 0.6 * x3
    p_true = 1.0 / (1.0 + np.exp(-logit))
    treat = (rng.uniform(size=n) < p_true).astype(int)
    TRUE_ATT = 2.0
    y = (1.0 + 1.5 * x1 + 1.0 * x2 + 0.5 * x3
         + TRUE_ATT * treat + rng.normal(0, 1, n))
    df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2, "x3": x3})

    st = StudyState()
    st.sources["datasets"] = df
    st.design["treatment"] = "treat"
    st.variables["outcome"] = "y"

    print("TRUE ATT =", TRUE_ATT)
    for meth in ("nn", "ipw"):
        s = StudyState()
        s.sources["datasets"] = df
        s.design["treatment"] = "treat"
        s.variables["outcome"] = "y"
        psm(s, method=meth, covariates=["x1", "x2", "x3"], n_boot=300)
        m = s.models["psm"]
        b = s.diagnostics["balance"]
        print(f"\n[method={meth}]")
        print(f"  ATT recovered = {m['att']:.3f}  (SE {m['se']:.3f})  vs TRUE {TRUE_ATT}")
        print(f"  naive_diff    = {m['naive_diff']:.3f}  (biased, should differ from {TRUE_ATT})")
        print(f"  n_treated={m['n_treated']}  n_matched={m['n_matched']}")
        print(f"  max|SMD| before = {b['max_smd_before']:.3f}  ->  after = {b['max_smd_after']:.3f}")
        assert abs(m["att"] - TRUE_ATT) < 0.4, f"{meth} ATT off: {m['att']}"
        assert abs(m["naive_diff"] - TRUE_ATT) > 0.3, "naive should be biased"
        assert b["max_smd_after"] < b["max_smd_before"], "balance should improve"
    print("\nALL SELF-TESTS PASSED")
