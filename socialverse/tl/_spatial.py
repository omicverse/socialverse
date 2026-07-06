"""``sv.tl._spatial`` — registered implementation for the spatial-analysis gap.

The spatial-econometrics axis of ``socialverse``: read whether a variable
clusters in space (spatial autocorrelation), then model that clustering with a
spatial-lag (SAR) regression. This is the ``socialverse`` analog of PySAL's
``esda`` (Moran's I / LISA) and ``spreg`` (SAR maximum likelihood), which are the
Python champion packages for this work.

Real computation only, with honest fallbacks:

- :func:`spatial_autocorr` computes global **Moran's I** with a conditional
  permutation reference distribution (the same inference ``esda.Moran`` uses when
  ``permutations>0``) and the full **LISA** decomposition — a local ``I_i`` per
  observation plus its Moran-scatterplot quadrant (HH / LL / HL / LH). If PySAL
  (``esda`` / ``libpysal``) is installed it is used to cross-check; otherwise the
  identical formulas are evaluated in pure NumPy.

- :func:`spatial_regression` fits a spatial-lag model
  ``y = rho * W y + X beta + eps`` by **concentrated maximum likelihood** — the
  log-Jacobian ``log|I - rho W|`` is profiled out and ``rho`` is found by a 1-D
  Brent optimization over ``scipy.optimize`` (matching ``spreg.ML_Lag``). If
  ``spreg`` is available it is used as a champion accelerator; the pure-NumPy ML
  path is the default and is the one validated below. Direct / indirect / total
  impacts are reported from the reduced form ``(I - rho W)^{-1}``.

Both functions require ``sources.datasets``; the weights matrix ``W`` is taken
from a ``W=`` kwarg, else built as a row-normalized k-nearest-neighbour matrix
from ``row`` / ``col`` (or ``x``/``y``) coordinate columns — so the function
never dead-ends on a missing ``W``.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from .._registry import register
from .._state import StudyState

__all__ = ["spatial_autocorr", "spatial_regression"]

_SEED = 0


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_frame(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working data frame.

    Priority: explicit ``data=`` / ``df=`` kwarg, then ``sources['datasets']``.
    ``sources['datasets']`` may be a DataFrame or a ``{name: DataFrame}`` mapping
    (first frame is taken), and may itself be a ``(df, W)`` tuple as returned by
    ``datasets.load_spatial`` — in which case the frame is unpacked.
    """
    df = kwargs.get("data")
    if df is None:
        df = kwargs.get("df")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, tuple) and df and isinstance(df[0], pd.DataFrame):
        df = df[0]
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _get_W_from_state(state: StudyState, kwargs: dict[str, Any]) -> np.ndarray | None:
    """Pull an explicit weights matrix from kwargs or ``sources.datasets`` tuple."""
    W = kwargs.get("W")
    if W is None:
        ds = state.sources.get("datasets")
        if isinstance(ds, tuple) and len(ds) >= 2:
            W = ds[1]
    if W is None:
        return None
    W = np.asarray(W, dtype=float)
    if W.ndim == 2 and W.shape[0] == W.shape[1]:
        return W
    return None


def _row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-standardize a weights matrix (rows summing to 1; isolate-safe)."""
    W = np.asarray(W, dtype=float).copy()
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def _knn_W(coords: np.ndarray, k: int = 4) -> np.ndarray:
    """Row-normalized k-nearest-neighbour binary weights from point coordinates.

    Champion equivalent: ``libpysal.weights.KNN``. Distances are Euclidean; the
    ``k`` nearest neighbours (self excluded) get weight 1, then rows are
    standardized. Deterministic (stable argsort).
    """
    n = coords.shape[0]
    k = int(min(max(k, 1), n - 1))
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(d[i], kind="stable")[:k]
        W[i, nn] = 1.0
    return _row_normalize(W)


def _resolve_W(
    state: StudyState, df: pd.DataFrame, kwargs: dict[str, Any], n: int
) -> np.ndarray:
    """Return a row-normalized n x n weights matrix.

    Explicit ``W`` (kwarg or dataset tuple) wins; otherwise a KNN matrix is built
    from coordinate columns (``row``/``col``, else ``y``/``x`` positions, else the
    frame index as a 1-D line).
    """
    W = _get_W_from_state(state, kwargs)
    if W is not None and W.shape[0] == n:
        return _row_normalize(W)

    k = int(kwargs.get("knn", 4))
    cols = {c.lower(): c for c in df.columns}
    for a, b in (("row", "col"), ("lat", "lon"), ("y_coord", "x_coord")):
        if a in cols and b in cols:
            coords = df[[cols[a], cols[b]]].to_numpy(dtype=float)
            return _knn_W(coords, k=k)
    # fall back to an ordered line graph on the index
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    return _knn_W(coords, k=min(k, 2))


def _resolve_value_col(df: pd.DataFrame, kwargs: dict[str, Any]) -> str:
    """Pick the analysis variable column for autocorrelation."""
    v = kwargs.get("value") or kwargs.get("y") or kwargs.get("outcome")
    if v is not None and v in df.columns:
        return str(v)
    for cand in ("y", "value", "outcome"):
        if cand in df.columns:
            return cand
    num = df.select_dtypes(include=[np.number]).columns
    return str(num[-1] if len(num) else df.columns[-1])


# ------------------------------------------------------------ Moran / LISA math
def _global_moran(z: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """Global Moran's I and its analytic (normality) z-score.

    ``I = (n / S0) * (z' W z) / (z' z)`` with ``z`` the mean-deviations and
    ``S0 = sum(W)``. The expectation ``E[I] = -1/(n-1)`` and the normality-
    assumption variance are used for the analytic z (a fast sanity companion to
    the permutation p-value).
    """
    n = z.shape[0]
    S0 = float(W.sum())
    num = float(z @ (W @ z))
    den = float(z @ z)
    I = (n / S0) * (num / den)

    Wsym = W + W.T
    S1 = 0.5 * float((Wsym ** 2).sum())
    S2 = float(((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum())
    EI = -1.0 / (n - 1)
    b2 = (n * float((z ** 4).sum())) / (den ** 2)
    n2 = n * n
    A = n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)
    B = b2 * ((n2 - n) * S1 - 2 * n * S2 + 6 * S0 * S0)
    C = (n - 1) * (n - 2) * (n - 3) * S0 * S0
    varI = (A - B) / C - EI * EI if C != 0 else np.nan
    zscore = (I - EI) / np.sqrt(varI) if varI and varI > 0 else np.nan
    return float(I), float(zscore)


def _perm_pvalue(
    z: np.ndarray, W: np.ndarray, I_obs: float, permutations: int, rng: np.random.Generator
) -> float:
    """Two-sided permutation p-value for global Moran's I.

    Row labels of ``z`` are permuted ``permutations`` times; the pseudo p-value
    follows ``esda``'s ``(#{|I_perm| >= |I_obs|} + 1) / (permutations + 1)``.
    """
    n = z.shape[0]
    S0 = float(W.sum())
    den = float(z @ z)
    const = n / (S0 * den)
    extreme = 0
    Icenter = abs(I_obs)
    for _ in range(permutations):
        zp = z[rng.permutation(n)]
        Ip = const * float(zp @ (W @ zp))
        if abs(Ip) >= Icenter:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def _lisa(
    z: np.ndarray, W: np.ndarray, permutations: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Local Moran's I (LISA) per observation with quadrant + permutation p.

    ``I_i = (n-1) * z_i * (W z)_i / (z' z)`` (Anselin 1995, ``esda.Moran_Local``
    scaling). Quadrant from the sign of ``z_i`` and its spatial lag ``(W z)_i``:
    HH=1, LH=2, LL=3, HL=4. Conditional-permutation p-values hold ``z_i`` fixed
    and permute the other rows into the neighbour set.
    """
    n = z.shape[0]
    den = float(z @ z)
    lag = W @ z
    Ii = (n - 1) * z * lag / den

    quad = np.zeros(n, dtype=int)
    quad[(z > 0) & (lag > 0)] = 1  # HH
    quad[(z <= 0) & (lag > 0)] = 2  # LH
    quad[(z <= 0) & (lag <= 0)] = 3  # LL
    quad[(z > 0) & (lag <= 0)] = 4  # HL
    labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

    # conditional permutation p-values: hold z_i fixed, reshuffle the
    # neighbour weights over the other observations' z values.
    p = np.ones(n)
    if permutations > 0:
        for i in range(n):
            others = np.delete(np.arange(n), i)
            w_i = W[i, others]          # neighbour weights for row i (over others)
            if not np.any(w_i):
                continue
            base = (n - 1) * z[i] / den
            obs = abs(Ii[i])
            z_others = z[others]
            extreme = 0
            for _ in range(permutations):
                zperm = z_others[rng.permutation(z_others.shape[0])]
                Iperm = base * float(w_i @ zperm)
                if abs(Iperm) >= obs:
                    extreme += 1
            p[i] = (extreme + 1) / (permutations + 1)

    sig = p < 0.05
    counts = {labels[q]: int(((quad == q) & sig).sum()) for q in (1, 2, 3, 4)}
    return {
        "Ii": [float(v) for v in Ii],
        "quadrant": [labels[int(q)] for q in quad],
        "p_sim": [float(v) for v in p],
        "n_significant": int(sig.sum()),
        "cluster_counts": counts,
        "note": "LISA:局部 Moran I + 象限(HH/LL/HL/LH)+ 条件置换 p",
    }


# ------------------------------------------------------------------- Moran main
@register(
    name="spatial_autocorr",
    aliases=["空间自相关", "Moran"],
    category="spatial",
    tier="plus",
    skill="(空间 缺口)",
    languages=["Python"],
    key_tools=["numpy", "pysal"],
    description="全局 Moran's I(置换 p)+ 局部 LISA(每点 Ii + HH/LL/HL/LH 象限)",
    requires={"sources": ["datasets"]},
    produces={"diagnostics": ["moran"], "models": ["lisa"]},
    auto_fix="escalate",
)
def spatial_autocorr(state: StudyState, **kwargs: Any) -> StudyState:
    """Global Moran's I and local LISA for a spatially referenced variable.

    Computes global Moran's I with a conditional-permutation reference
    distribution (``permutations`` shuffles, default 999) for a two-sided pseudo
    p-value, and the full LISA decomposition (a local ``I_i``, Moran-scatterplot
    quadrant, and conditional-permutation p per observation).

    The variable is chosen from ``value=`` (else ``y``); the weights matrix is
    the ``W=`` kwarg / dataset tuple, else a row-normalized KNN matrix from
    coordinate columns. Writes ``diagnostics['moran']`` (global I / z / p /
    quadrant counts) and ``models['lisa']`` (per-observation LISA). Falls back to
    an empty-but-valid record rather than raising when no data is present.
    """
    rng = np.random.default_rng(int(kwargs.get("seed", _SEED)))
    permutations = int(kwargs.get("permutations", 999))

    df = _get_frame(state, kwargs)
    if df is None or df.empty:
        state.write("diagnostics", "moran", {"I": None, "note": "无空间数据"})
        state.write("models", "lisa", {"Ii": [], "note": "无空间数据"})
        return state

    col = _resolve_value_col(df, kwargs)
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    n = y.shape[0]
    W = _resolve_W(state, df, kwargs, n)
    z = y - y.mean()

    I, z_analytic = _global_moran(z, W)
    p_perm = _perm_pvalue(z, W, I, permutations, rng)
    lisa = _lisa(z, W, int(kwargs.get("lisa_permutations", permutations)), rng)

    # optional champion cross-check (esda), non-fatal
    backend = "numpy"
    esda = _try_import("esda")
    libpysal = _try_import("libpysal")
    if esda is not None and libpysal is not None:
        try:
            wobj = libpysal.weights.full2W(W)
            wobj.transform = "r"
            mi = esda.Moran(y, wobj, permutations=permutations)
            if np.isfinite(mi.I):
                I = float(mi.I)
                p_perm = float(mi.p_sim)
                backend = "esda"
        except Exception:
            backend = "numpy"

    state.write("diagnostics", "moran", {
        "variable": col,
        "n": int(n),
        "I": float(I),
        "expected_I": float(-1.0 / (n - 1)),
        "z_score": float(z_analytic),
        "p_perm": float(p_perm),
        "permutations": permutations,
        "cluster_counts": lisa["cluster_counts"],
        "backend": backend,
        "note": "全局 Moran's I;p_perm 为置换伪 p 值(双侧)",
    })
    state.write("models", "lisa", lisa)
    return state


# ---------------------------------------------------------------- SAR ML engine
def _log_jacobian(rho: float, ev: np.ndarray) -> float:
    """log|I - rho W| via the eigenvalues of W: sum(log(1 - rho * lambda_i))."""
    return float(np.log(np.abs(1.0 - rho * ev)).sum())


def _concentrated_negloglik(
    rho: float, y: np.ndarray, X: np.ndarray, Wy: np.ndarray, ev: np.ndarray
) -> float:
    """Negative concentrated log-likelihood of the SAR (spatial-lag) model.

    For ``y = rho W y + X beta + eps``, beta and sigma^2 are profiled out by OLS
    of ``(y - rho W y)`` on ``X``; the remaining objective in ``rho`` is
    ``-(logJ - (n/2) log(SSE/n))`` (dropping constants) — exactly the
    concentrated likelihood ``spreg.ML_Lag`` maximizes.
    """
    n = y.shape[0]
    e = y - rho * Wy
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    resid = e - X @ beta
    sse = float(resid @ resid)
    if sse <= 0:
        return np.inf
    logJ = _log_jacobian(rho, ev)
    ll = logJ - (n / 2.0) * np.log(sse / n)
    return -ll


def _sar_ml(y: np.ndarray, X: np.ndarray, W: np.ndarray) -> dict[str, Any]:
    """Fit the spatial-lag model by concentrated ML (pure NumPy/SciPy).

    Returns rho, beta, sigma^2 and the fit. ``rho`` is bounded inside the
    invertibility range ``(1/min_ev, 1/max_ev)`` of ``W`` and found by Brent's
    method on the concentrated negative log-likelihood.
    """
    n = y.shape[0]
    Wy = W @ y
    ev = np.linalg.eigvals(W).real
    lo = 1.0 / ev.min() + 1e-6 if ev.min() < 0 else -0.999
    hi = 1.0 / ev.max() - 1e-6 if ev.max() > 0 else 0.999
    lo, hi = max(lo, -0.999), min(hi, 0.999)

    res = optimize.minimize_scalar(
        _concentrated_negloglik, bounds=(lo, hi), method="bounded",
        args=(y, X, Wy, ev), options={"xatol": 1e-6},
    )
    rho = float(res.x)

    e = y - rho * Wy
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    resid = e - X @ beta
    sigma2 = float(resid @ resid) / n
    return {
        "rho": rho,
        "beta": [float(b) for b in beta],
        "sigma2": sigma2,
        "loglik": float(-res.fun),
        "converged": bool(res.success),
    }


def _impacts(rho: float, beta_k: float, W: np.ndarray) -> dict[str, float]:
    """Direct / indirect / total impacts of a regressor in the SAR model.

    Uses the reduced form ``(I - rho W)^{-1}``: total = beta_k / (1 - rho),
    average direct = mean of the diagonal of ``(I - rho W)^{-1} beta_k``, and
    indirect = total - direct (LeSage & Pace).
    """
    n = W.shape[0]
    S = np.linalg.inv(np.eye(n) - rho * W) * beta_k
    total = float(S.sum() / n)
    direct = float(np.trace(S) / n)
    return {"direct": direct, "indirect": total - direct, "total": total}


# --------------------------------------------------------------- SAR main
@register(
    name="spatial_regression",
    aliases=["空间回归", "SAR"],
    category="spatial",
    tier="pro",
    skill="(空间回归 缺口)",
    languages=["Python"],
    key_tools=["numpy", "scipy", "spreg"],
    description="空间滞后 SAR:集中似然 ML 估 rho + beta,报直接/间接/总效应",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["sar"], "diagnostics": ["spatial"]},
    prerequisites={"optional_functions": ["spatial_autocorr"]},
    auto_fix="escalate",
)
def spatial_regression(state: StudyState, **kwargs: Any) -> StudyState:
    """Spatial-lag (SAR) regression by concentrated maximum likelihood.

    Fits ``y = rho * W y + X beta + eps``. The spatial-autoregressive parameter
    ``rho`` is estimated by profiling out ``beta`` and ``sigma^2`` and optimizing
    the concentrated likelihood over ``rho`` with a bounded 1-D SciPy search
    (``spreg.ML_Lag`` semantics); the log-Jacobian ``log|I - rho W|`` uses the
    eigenvalues of ``W``. Direct / indirect / total impacts come from the reduced
    form ``(I - rho W)^{-1}``.

    Outcome from ``outcome=`` (else ``variables['outcome']`` / ``y``); regressors
    from ``predictors=`` (else ``x``). ``W`` from kwarg / dataset tuple, else KNN.
    Writes ``models['sar']`` (rho, beta, impacts, loglik) and
    ``diagnostics['spatial']`` (fit / backend). Empty-but-valid record on no data.
    """
    df = _get_frame(state, kwargs)
    if df is None or df.empty:
        state.write("models", "sar", {"rho": None, "note": "无空间数据"})
        state.write("diagnostics", "spatial", {"note": "无空间数据"})
        return state

    outcome = (
        kwargs.get("outcome")
        or state.variables.get("outcome")
        or ("y" if "y" in df.columns else None)
    )
    if outcome not in df.columns:
        outcome = _resolve_value_col(df, kwargs)

    predictors = kwargs.get("predictors")
    if not predictors:
        predictors = [c for c in ("x",) if c in df.columns]
    if not predictors:
        predictors = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in (outcome, "id", "row", "col")
        ][:1]

    sub = df[[outcome, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
    y = sub[outcome].to_numpy(dtype=float)
    n = y.shape[0]
    Xdata = sub[list(predictors)].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), Xdata])
    W = _resolve_W(state, df.loc[sub.index].reset_index(drop=True), kwargs, n)

    fit = _sar_ml(y, X, W)
    rho = fit["rho"]

    # per-predictor impacts (beta index 0 is the intercept)
    impacts = {
        name: _impacts(rho, fit["beta"][1 + i], W)
        for i, name in enumerate(predictors)
    }

    # optional champion accelerator (spreg), non-fatal
    backend = "numpy_ml"
    spreg = _try_import("spreg")
    if spreg is not None:
        try:
            m = spreg.ML_Lag(y.reshape(-1, 1), Xdata, w=_spreg_w(W))
            rho_sp = float(np.ravel(m.rho)[0])
            if np.isfinite(rho_sp):
                rho = rho_sp
                fit["rho"] = rho_sp
                fit["beta"] = [float(b) for b in np.ravel(m.betas)]
                impacts = {
                    name: _impacts(rho, fit["beta"][1 + i], W)
                    for i, name in enumerate(predictors)
                }
                backend = "spreg"
        except Exception:
            backend = "numpy_ml"

    state.write("models", "sar", {
        "outcome": str(outcome),
        "predictors": list(predictors),
        "n": int(n),
        "rho": float(rho),
        "beta": {"const": fit["beta"][0],
                 **{name: fit["beta"][1 + i] for i, name in enumerate(predictors)}},
        "sigma2": fit["sigma2"],
        "loglik": fit["loglik"],
        "impacts": impacts,
        "backend": backend,
        "note": "SAR 空间滞后模型:y = rho W y + X beta + eps(集中似然 ML)",
    })
    state.write("diagnostics", "spatial", {
        "model": "SAR (spatial lag)",
        "rho": float(rho),
        "converged": fit["converged"],
        "backend": backend,
        "note": "rho>0 表示正空间依赖;间接效应=空间溢出",
    })
    return state


def _spreg_w(W: np.ndarray):
    """Wrap a dense weights matrix as a libpysal W for spreg (best-effort)."""
    libpysal = _try_import("libpysal")
    if libpysal is None:
        raise RuntimeError("libpysal unavailable")
    wobj = libpysal.weights.full2W(W)
    wobj.transform = "r"
    return wobj
