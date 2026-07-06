"""``sv.tl._quasi`` — registered implementations for the quasi-experimental gap.

Two designs that the DID family does not cover, ported to the ``StudyState`` /
``registry`` spine with *real* computation (able to recover the known DGP
parameters), not placeholders:

    rdd                 synthetic_control
    (local-linear RDD)  (donor-pool synthetic control)

The real-world champions this module mirrors:

* ``rdd`` — the local-linear, triangular-kernel sharp RDD of R's **rdrobust**
  (Calonico–Cattaneo–Titiunik) and Python's ``rdrobust``. We fit weighted least
  squares separately on each side of the cutoff inside a data-driven bandwidth
  (an Imbens–Kalyanaraman-lite / rule-of-thumb selector) and read the jump as
  the difference of the two intercepts. There is no heavy dependency: the whole
  estimator is plain ``statsmodels`` WLS / ``numpy``.

* ``synthetic_control`` — Abadie–Diamond–Hainmueller synthetic control, the
  engine of R's **Synth** and Python's ``pysyncon`` / ``SparseSC``. Donor
  weights are non-negative and sum to one, chosen by ``scipy.optimize`` SLSQP to
  minimize pre-treatment fit; the counterfactual is the weighted donor path and
  the treatment effect is the treated-minus-synthetic gap. ``cvxpy`` would give a
  slicker convex solve when installed, but SLSQP recovers the same optimum and is
  always available.

Only ``numpy`` / ``pandas`` / ``scipy`` / ``statsmodels`` are hard dependencies;
optional accelerators are lazy-imported and the code degrades to the fallback
without them.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["rdd", "synthetic_control"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional accelerator."""
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


def _pick_outcome(df: pd.DataFrame, state: StudyState, kwargs: dict[str, Any],
                  exclude: list[str]) -> str | None:
    """Resolve the outcome column: kwarg/variables slot, else first numeric non-excluded."""
    y = kwargs.get("outcome") or state.variables.get("outcome")
    if y is not None and y in df.columns:
        return y
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular (edge) kernel weight ``max(1 - |u|, 0)`` — the rdrobust default."""
    return np.clip(1.0 - np.abs(u), 0.0, None)


def _wls_intercept(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Local-linear WLS of ``y`` on ``x`` (with intercept); return (intercept, se).

    Falls back to a closed-form weighted normal-equations solve if statsmodels is
    unavailable, so the boundary estimate is always produced.
    """
    sm = _try_import("statsmodels.api")
    X = np.column_stack([np.ones_like(x), x])
    if sm is not None:
        res = sm.WLS(y, X, weights=w).fit()
        return float(res.params[0]), float(res.bse[0])
    # closed-form weighted least squares
    W = np.diag(w)
    xtwx = X.T @ W @ X
    xtwy = X.T @ W @ y
    beta = np.linalg.solve(xtwx, xtwy)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float((w * resid ** 2).sum() / dof)
    cov = sigma2 * np.linalg.inv(xtwx)
    return float(beta[0]), float(np.sqrt(cov[0, 0]))


def _rot_bandwidth(r: np.ndarray, y: np.ndarray, cutoff: float) -> float:
    """Imbens–Kalyanaraman-lite rule-of-thumb bandwidth.

    A pragmatic selector: a Silverman-style pilot scaled by the running-variable
    spread, floored so that each side retains enough mass for a stable local-linear
    fit. This is deliberately simpler than the full IK plug-in (which the
    ``rdrobust`` champion uses) but lands in the same ballpark on smooth DGPs.
    """
    n = len(r)
    sd = float(np.std(r))
    # Silverman pilot ~ 1.84 * sd * n^(-1/5); widen for the boundary local-linear fit
    h = 1.84 * sd * n ** (-1.0 / 5.0)
    h *= 2.5
    span = float(np.max(r) - np.min(r)) or 1.0
    h = min(max(h, 0.10 * span), 0.9 * span)
    # ensure both sides carry data
    for _ in range(6):
        left = int(np.sum((r < cutoff) & (r >= cutoff - h)))
        right = int(np.sum((r >= cutoff) & (r < cutoff + h)))
        if left >= 10 and right >= 10:
            break
        h *= 1.3
    return float(h)


# ------------------------------------------------------------------------ rdd
@register(
    name="rdd",
    aliases=["断点回归", "RDD", "regression_discontinuity"],
    category="quasi",
    tier="plus",
    skill="(RDD 缺口)",
    languages=["Python"],
    key_tools=["statsmodels", "numpy"],
    description="锐性断点回归:三角核局部线性 WLS 双侧拟合 + 数据驱动带宽,估断点跳",
    requires={
        "sources": ["datasets"],
        "variables": ["outcome"],
        "estimand": ["target"],
    },
    produces={"models": ["rdd"], "diagnostics": ["bandwidth"]},
    auto_fix="escalate",
)
def rdd(state: StudyState, **kwargs: Any) -> StudyState:
    """Sharp regression-discontinuity: local-linear, triangular-kernel WLS.

    We take the running variable and the cutoff, select a data-driven bandwidth
    (rule-of-thumb / IK-lite), and fit a *separate* weighted linear regression on
    each side of the cutoff using triangular kernel weights that decay to zero at
    the bandwidth edge. The RD estimate is the difference of the two boundary
    intercepts (right minus left), i.e. the jump in the conditional mean at the
    threshold. Its SE is the root-sum-of-squares of the two intercept SEs.

    Recovers the sharp-RDD DGP of :func:`socialverse.datasets.load_rdd` (true jump
    ``tau = 3``).
    """
    df = _get_datasets(state, kwargs)
    running = kwargs.get("running", "running")
    cutoff = float(kwargs.get("cutoff", 0.0))

    if df is None or running not in (df.columns if df is not None else []):
        model = {"jump": None, "note": "缺少数据或 running 变量,无法估计 RDD"}
        state.write("models", "rdd", model)
        state.write("diagnostics", "bandwidth", {"h": None, "note": model["note"]})
        return state

    ycol = _pick_outcome(df, state, kwargs, exclude=[running, "treat"])
    if ycol is None:
        model = {"jump": None, "note": "找不到结果变量"}
        state.write("models", "rdd", model)
        state.write("diagnostics", "bandwidth", {"h": None, "note": model["note"]})
        return state

    r = pd.to_numeric(df[running], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    ok = np.isfinite(r) & np.isfinite(y)
    r, y = r[ok], y[ok]

    h = float(kwargs.get("bandwidth") or _rot_bandwidth(r, y, cutoff))

    def _side(mask: np.ndarray) -> tuple[float, float, int]:
        rr, yy = r[mask], y[mask]
        w = _triangular_kernel((rr - cutoff) / h)
        keep = w > 0
        rr, yy, w = rr[keep], yy[keep], w[keep]
        if len(rr) < 3:
            return float("nan"), float("nan"), len(rr)
        a, se = _wls_intercept(rr - cutoff, yy, w)  # center at cutoff → intercept = boundary
        return a, se, len(rr)

    left_int, left_se, n_left = _side(r < cutoff)
    right_int, right_se, n_right = _side(r >= cutoff)

    jump = float(right_int - left_int)
    se = float(np.sqrt(np.nan_to_num(left_se) ** 2 + np.nan_to_num(right_se) ** 2))
    t = jump / se if se > 0 else float("nan")

    model = {
        "jump": jump,
        "se": se,
        "t": float(t),
        "left_intercept": float(left_int),
        "right_intercept": float(right_int),
        "cutoff": cutoff,
        "running": running,
        "outcome": ycol,
        "kernel": "triangular",
        "estimator": "local-linear WLS (sharp RDD)",
        "note": "锐性 RDD:三角核局部线性,右-左截距之差为断点跳",
    }
    bandwidth = {
        "h": h,
        "selector": "IK-lite rule-of-thumb" if kwargs.get("bandwidth") is None else "user",
        "n_left": n_left,
        "n_right": n_right,
        "note": "三角核带宽内每侧有效样本量",
    }
    state.write("models", "rdd", model)
    state.write("diagnostics", "bandwidth", bandwidth)
    return state


# --------------------------------------------------------- synthetic control
def _synth_weights(X0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Non-negative, sum-to-one donor weights minimizing pre-period MSE.

    Solves ``min_w ||x1 - X0 w||^2  s.t. w >= 0, sum(w) = 1`` via ``scipy`` SLSQP,
    seeded from the uniform simplex point (deterministic). Falls back to a
    projected-gradient simplex descent if scipy is unavailable.

    Parameters
    ----------
    X0 : ``(T_pre, J)`` pre-period outcomes of the ``J`` donors.
    x1 : ``(T_pre,)`` pre-period outcomes of the treated unit.
    """
    J = X0.shape[1]
    w0 = np.full(J, 1.0 / J)

    def obj(w: np.ndarray) -> float:
        d = x1 - X0 @ w
        return float(d @ d)

    opt = _try_import("scipy.optimize")
    if opt is not None:
        cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
        bounds = [(0.0, 1.0)] * J

        def grad(w: np.ndarray) -> np.ndarray:
            return -2.0 * X0.T @ (x1 - X0 @ w)

        res = opt.minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        w = np.clip(np.asarray(res.x, float), 0.0, None)
        s = w.sum()
        return w / s if s > 0 else w0

    # projected-gradient fallback on the simplex
    w = w0.copy()
    lr = 1.0 / (np.linalg.norm(X0, 2) ** 2 + 1e-9)
    for _ in range(5000):
        g = -2.0 * X0.T @ (x1 - X0 @ w)
        w = _project_simplex(w - lr * g)
    return w


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex (Duchi et al.)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.clip(v - theta, 0.0, None)


@register(
    name="synthetic_control",
    aliases=["合成控制", "SCM", "synth"],
    category="quasi",
    tier="pro",
    skill="(合成控制 缺口)",
    languages=["Python"],
    key_tools=["scipy", "numpy"],
    description="合成控制:SLSQP 求非负和为1权重最小化前期 MSE,构造反事实路径与 gap",
    requires={
        "sources": ["datasets"],
        "design": ["treatment", "time"],
        "variables": ["outcome"],
        "estimand": ["target"],
    },
    produces={"models": ["synth"], "diagnostics": ["pre_fit"]},
    auto_fix="escalate",
)
def synthetic_control(state: StudyState, **kwargs: Any) -> StudyState:
    """Abadie–Diamond–Hainmueller synthetic control on a donor pool.

    From a long panel ``(unit, time, outcome)`` we pivot to a ``time × unit``
    outcome matrix, split at ``treat_time``, and choose donor weights (non-negative,
    summing to one) that minimize the pre-treatment MSE between the treated unit and
    its synthetic counterpart. The counterfactual is the weighted donor path; the
    per-period **gap** (treated − synthetic) is zero in expectation before treatment
    and tracks the treatment effect after.

    Recovers the treatment effect of :func:`socialverse.datasets.load_did_panel`
    (``att = -0.8``): pre-period RMSE is small and the post-period mean gap ≈ att.
    """
    df = _get_datasets(state, kwargs)
    unit = kwargs.get("unit") or state.design.get("panel_id") or "unit"
    time = kwargs.get("time") or state.design.get("time") or "time"
    treatment = kwargs.get("treatment") or state.design.get("treatment")
    treated_unit = kwargs.get("treated_unit", 0)
    treat_time = kwargs.get("treat_time")

    def _fail(note: str) -> StudyState:
        state.write("models", "synth", {"att": None, "note": note})
        state.write("diagnostics", "pre_fit", {"pre_rmse": None, "note": note})
        return state

    if df is None or unit not in df.columns or time not in df.columns:
        return _fail("缺少面板数据或 unit/time 列,无法做合成控制")

    ycol = _pick_outcome(df, state, kwargs, exclude=[unit, time, "treat", "post",
                                                     "treat_post", "first_treated"])
    if ycol is None:
        return _fail("找不到结果变量")

    # wide time × unit outcome matrix (mean-collapse any duplicate cells)
    wide = df.pivot_table(index=time, columns=unit, values=ycol, aggfunc="mean")
    wide = wide.sort_index()
    if treated_unit not in wide.columns:
        return _fail(f"处理单元 {treated_unit!r} 不在面板中")

    times = wide.index.to_numpy()
    if treat_time is None:
        treat_time = times[len(times) // 2]

    # ADH requires a clean, *untreated* donor pool: if a treatment indicator is
    # available, exclude every unit that is ever treated (so a treated donor's own
    # post-period effect does not contaminate the synthetic counterfactual).
    treated_units: set = set()
    if treatment is not None and treatment in df.columns:
        ever = df.groupby(unit)[treatment].max()
        treated_units = {u for u, v in ever.items() if v and u != treated_unit}

    donors = [c for c in wide.columns if c != treated_unit and c not in treated_units]
    # keep donors with complete series
    valid = [c for c in donors if wide[c].notna().all()]
    donors = valid or donors
    sub = wide[[treated_unit] + donors].dropna()
    times = sub.index.to_numpy()

    pre_mask = times < treat_time
    post_mask = ~pre_mask
    if pre_mask.sum() < 2 or len(donors) < 1:
        return _fail("前期观测或对照池不足")

    y1 = sub[treated_unit].to_numpy(float)
    Y0 = sub[donors].to_numpy(float)  # (T, J)

    w = _synth_weights(Y0[pre_mask], y1[pre_mask])

    synth_path = Y0 @ w
    gap = y1 - synth_path

    pre_gap = gap[pre_mask]
    post_gap = gap[post_mask]
    pre_rmse = float(np.sqrt(np.mean(pre_gap ** 2)))
    att = float(np.mean(post_gap)) if post_mask.any() else float("nan")

    weights = {str(d): float(wi) for d, wi in zip(donors, w) if wi > 1e-4}

    model = {
        "att": att,
        "treated_unit": treated_unit,
        "treat_time": (treat_time.item() if hasattr(treat_time, "item") else treat_time),
        "weights": weights,
        "n_donors": len(donors),
        "path": {
            "time": [t.item() if hasattr(t, "item") else t for t in times],
            "treated": [float(v) for v in y1],
            "synthetic": [float(v) for v in synth_path],
            "gap": [float(v) for v in gap],
        },
        "outcome": ycol,
        "estimator": "ADH synthetic control (SLSQP simplex weights)",
        "note": "合成控制:非负和为1权重最小化前期 MSE;post 期平均 gap 为处理效应",
    }
    pre_fit = {
        "pre_rmse": pre_rmse,
        "n_pre": int(pre_mask.sum()),
        "n_post": int(post_mask.sum()),
        "note": "前期 treated 与 synthetic 的 RMSE(越小拟合越好)",
    }
    state.write("models", "synth", model)
    state.write("diagnostics", "pre_fit", pre_fit)
    return state
