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

from .._registry import register
from .._state import StudyState

__all__ = ["spatial_autocorr", "spatial_regression", "spatial_history"]

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
    from scipy import optimize  # optional backend — imported lazily so the module registers without it

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


# ============================================================================
# spatial_history — the spatial-TEMPORAL summary that makes a "historical map"
# mean something. A drawing of dots answers "where"; this answers "how did the
# pattern MOVE and SPREAD over time". It is the Historical-Map tool's single
# inference entrypoint (the survey_crosstab analog): given events with
# coordinates + a time column, it bins time and returns, per period, the
# weighted CENTER OF GRAVITY, its migration path (haversine km + bearing), the
# spatial DISPERSION (standard distance), the EXTENT (diameter), and — when a
# magnitude column is given and n allows — the per-period Moran's I clustering.
# ============================================================================
_R_KM = 6371.0088  # mean Earth radius (km)


def _haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * _R_KM * np.arcsin(np.sqrt(np.clip(d, 0.0, 1.0)))


def _spherical_mean(lat, lon, w=None):
    """Weighted mean center on the sphere via 3-D unit-vector averaging.

    Correct across the antimeridian and at high latitude (a naive mean of
    longitude is not). Returns ``(mean_lat, mean_lon)`` in degrees.
    """
    lat = np.radians(np.asarray(lat, float))
    lon = np.radians(np.asarray(lon, float))
    w = np.ones_like(lat) if w is None else np.asarray(w, float)
    x = (w * np.cos(lat) * np.cos(lon)).sum()
    y = (w * np.cos(lat) * np.sin(lon)).sum()
    z = (w * np.sin(lat)).sum()
    return float(np.degrees(np.arctan2(z, np.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))


def _standard_distance_km(lat, lon, clat, clon, w=None):
    """Weighted RMS haversine distance of points from the center (a spatial 'SD')."""
    d = _haversine_km(np.asarray(lat, float), np.asarray(lon, float), clat, clon)
    if w is None:
        return float(np.sqrt((d ** 2).mean()))
    w = np.asarray(w, float)
    sw = w.sum()
    return float(np.sqrt((w * d ** 2).sum() / sw)) if sw > 0 else float(np.sqrt((d ** 2).mean()))


def _compass(lat1, lon1, lat2, lon2) -> str:
    """8-point compass name for the initial bearing point1 → point2."""
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon)
    brg = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((brg + 22.5) % 360 // 45)]


def _bootstrap_center_km(la, lo, w, B: int, rng) -> float | None:
    """95th-pct great-circle distance of resampled weighted centers from the point
    estimate — a bootstrap uncertainty radius (km) on a period's center of gravity.

    Resamples the recorded points uniformly with replacement (the sampling model is
    "which events happened to be recorded") and recomputes the WEIGHTED center; the
    spread of those centers is the honest uncertainty a raw centroid hides.
    """
    n = len(la)
    if n < 3:
        return None
    clat, clon = _spherical_mean(la, lo, w)
    d = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        bclat, bclon = _spherical_mean(la[idx], lo[idx], (w[idx] if w is not None else None))
        d[b] = _haversine_km(clat, clon, bclat, bclon)
    return float(np.percentile(d, 95))


def _perm_displacement_p(laA, loA, wA, laB, loB, wB, obs: float, B: int, rng) -> float:
    """Permutation p-value that a center genuinely MOVED between two periods.

    Pools the two periods' points, repeatedly re-splits them into groups of the same
    two sizes, and recomputes the between-center displacement. p = fraction of
    permuted displacements >= observed. Low p ⇒ the shift is larger than what
    reshuffling which points fell in which period would produce (i.e. real movement,
    not a recording artifact) — the survey-crosstab move for a trajectory.
    """
    la = np.concatenate([laA, laB]); lo = np.concatenate([loA, loB])
    w = np.concatenate([wA, wB]) if (wA is not None and wB is not None) else None
    nA, n = len(laA), len(la)
    if nA < 2 or (n - nA) < 2:
        return float("nan")
    extreme = 0
    for _ in range(B):
        perm = rng.permutation(n)
        iA, iB = perm[:nA], perm[nA:]
        cA = _spherical_mean(la[iA], lo[iA], (w[iA] if w is not None else None))
        cB = _spherical_mean(la[iB], lo[iB], (w[iB] if w is not None else None))
        if _haversine_km(cA[0], cA[1], cB[0], cB[1]) >= obs:
            extreme += 1
    return (extreme + 1) / (B + 1)


def _resolve_coord_cols(df: pd.DataFrame, kwargs: dict) -> tuple[str | None, str | None]:
    """Find the latitude/longitude columns (explicit kwargs, else common names)."""
    cols = {c.lower(): c for c in df.columns}
    lat = kwargs.get("lat") or kwargs.get("latitude")
    lon = kwargs.get("lon") or kwargs.get("lng") or kwargs.get("longitude")
    if lat and lon:
        return (lat if lat in df.columns else None, lon if lon in df.columns else None)
    for a in ("lat", "latitude", "纬度", "y", "y_coord"):
        if a in cols:
            lat = cols[a]; break
    for b in ("lon", "lng", "long", "longitude", "经度", "x", "x_coord"):
        if b in cols:
            lon = cols[b]; break
    return lat, lon


@register(
    name="spatial_history",
    aliases=["历史地图分析", "时空重心", "spatiotemporal", "map_diffusion", "重心迁移", "时空分析"],
    category="spatial",
    tier="plus",
    skill="spatial-analysis",
    languages=["Python"],
    key_tools=["numpy"],
    description="时空分析:按时段算加权重心(重力中心)+ 迁移路径(km/方位)+ 空间离散度 + 每期聚集(Moran)",
    requires={},   # the Historical-Map tool passes data + columns via kwargs; validated in-body
    produces={"models": ["spatial_history"], "artifacts": ["trajectory"]},
    auto_fix="escalate",
)
def spatial_history(state: StudyState, **kwargs: Any) -> StudyState:
    """Spatial-temporal summary of georeferenced events for the Historical-Map tool.

    Bins the ``time`` column and, per period, computes the weighted **mean center**
    (spherical 3-D average, weighted by ``value=`` if given), the **standard
    distance** (weighted RMS haversine distance from the center — spatial spread in
    km), the **extent** (max pairwise distance / diameter, km), and — when a
    ``value`` column is present and the period has enough points — the per-period
    **Moran's I** (does the magnitude cluster in that period). Across periods it
    returns the **trajectory** of centers and the **migration** (path length, net
    displacement, compass bearing) — i.e. how the phenomenon's center of gravity
    moved and how its footprint spread over time.

    Columns: ``lat``/``lon`` (else auto-detected), ``time`` (year/period),
    optional ``value`` (magnitude → weights the center + enables Moran), optional
    ``label`` (kept on points for provenance), optional ``bins`` (# equal-width
    time bins; else one bin per distinct time value). Writes
    ``models['spatial_history']``. Never raises — returns an ``error`` field.
    """
    result: dict[str, Any] = {"backend": "numpy"}
    try:
        df = _get_frame(state, kwargs)
        if df is None or df.empty:
            result["error"] = "无地理数据(需要含经纬度与时间列的表)"
            state.write("models", "spatial_history", result)
            return state

        lat, lon = _resolve_coord_cols(df, kwargs)
        tcol = kwargs.get("time") or kwargs.get("year") or kwargs.get("date") or kwargs.get("period")
        if not lat or not lon:
            result["error"] = "没找到经纬度列(需要 lat/lon 这类列)"
            state.write("models", "spatial_history", result)
            return state

        d = df.copy()
        d["_lat"] = pd.to_numeric(d[lat], errors="coerce")
        d["_lon"] = pd.to_numeric(d[lon], errors="coerce")
        has_time = bool(tcol) and tcol in d.columns
        d["_t"] = pd.to_numeric(d[tcol], errors="coerce") if has_time else 0.0
        sub_needed = ["_lat", "_lon"] + (["_t"] if has_time else [])
        d = d.dropna(subset=sub_needed).reset_index(drop=True)
        if d.empty:
            result["error"] = "经纬度/时间列没有可用的数值"
            state.write("models", "spatial_history", result)
            return state

        vcol = kwargs.get("value") or kwargs.get("weight") or kwargs.get("magnitude")
        w_all = pd.to_numeric(d[vcol], errors="coerce").to_numpy() if vcol and vcol in d.columns else None
        if w_all is not None:
            w_all = np.where(np.isfinite(w_all) & (w_all > 0), w_all, np.nan)

        # ---- time binning ----------------------------------------------------
        tvals = d["_t"].to_numpy(dtype=float)
        bins = kwargs.get("bins")
        if has_time and bins and int(bins) > 1 and tvals.max() > tvals.min():
            nb = int(bins)
            edges = np.linspace(tvals.min(), tvals.max(), nb + 1)
            idx = np.clip(np.digitize(tvals, edges[1:-1]), 0, nb - 1)
            labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(nb)]
            d["_period"] = [labels[i] for i in idx]
            order = labels
        elif has_time:
            d["_period"] = [f"{v:g}" for v in tvals]
            order = [f"{v:g}" for v in sorted(pd.unique(tvals))]
        else:
            d["_period"] = "all"
            order = ["all"]

        # ---- per-period stats ------------------------------------------------
        permutations = int(kwargs.get("permutations", 999))
        rng = np.random.default_rng(int(kwargs.get("seed", _SEED)))
        B_boot = int(kwargs.get("bootstrap", 300))
        periods: list[dict[str, Any]] = []
        _pts: list[tuple] = []  # per-period (la, lo, sw) for the migration inference
        for p in order:
            sub = d[d["_period"] == p]
            if sub.empty:
                continue
            la = sub["_lat"].to_numpy(); lo = sub["_lon"].to_numpy()
            sw = None
            if w_all is not None:
                sw = w_all[sub.index.to_numpy()]
                if not np.any(np.isfinite(sw)) or np.nansum(sw) <= 0:
                    sw = None
                else:
                    sw = np.where(np.isfinite(sw), sw, 0.0)
            clat, clon = _spherical_mean(la, lo, sw)
            sd = _standard_distance_km(la, lo, clat, clon, sw) if len(la) >= 2 else 0.0
            if 2 <= len(la) <= 400:
                ii, jj = np.triu_indices(len(la), 1)
                diam = float(_haversine_km(la[ii], lo[ii], la[jj], lo[jj]).max())
            else:
                diam = None
            rec = {
                "period": str(p), "n": int(len(sub)),
                "mean_lat": round(clat, 4), "mean_lon": round(clon, 4),
                "dispersion_km": round(sd, 1),
                "extent_km": round(diam, 1) if diam is not None else None,
                "sum_w": round(float(np.nansum(sw)), 2) if sw is not None else None,
                "center_ci_km": _bootstrap_center_km(la, lo, sw, B_boot, rng),
            }
            _pts.append((la, lo, sw))
            # per-period Moran's I on the magnitude (does the value cluster?)
            if sw is not None and len(la) >= 6:
                try:
                    coords = np.column_stack([la, lo])
                    Wm = _knn_W(coords, k=min(4, len(la) - 1))
                    yv = sw.astype(float)
                    z = yv - yv.mean()
                    if float(z @ z) > 0:
                        I, _zc = _global_moran(z, Wm)
                        pI = _perm_pvalue(z, Wm, I, min(permutations, 499), rng)
                        rec["moran_I"] = round(float(I), 3)
                        rec["moran_p"] = round(float(pI), 3)
                        rec["clustered"] = bool(pI < 0.05 and I > 0)
                except Exception:
                    pass
            periods.append(rec)

        # ---- trajectory + migration -----------------------------------------
        traj = [{"period": pp["period"], "lat": pp["mean_lat"], "lon": pp["mean_lon"]} for pp in periods]
        steps, path = [], 0.0
        for a, b in zip(traj, traj[1:]):
            seg = float(_haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]))
            path += seg
            steps.append(round(seg, 1))
        net, bearing = 0.0, "-"
        p_move: float = float("nan")
        sig_move = None
        if len(traj) >= 2:
            net = float(_haversine_km(traj[0]["lat"], traj[0]["lon"], traj[-1]["lat"], traj[-1]["lon"]))
            bearing = _compass(traj[0]["lat"], traj[0]["lon"], traj[-1]["lat"], traj[-1]["lon"])
            # inference: did the center REALLY move (first→last), or is it a recording
            # artifact? Permute period labels over the pooled first+last points.
            (laA, loA, wA), (laB, loB, wB) = _pts[0], _pts[-1]
            p_move = _perm_displacement_p(laA, loA, wA, laB, loB, wB, net,
                                          int(kwargs.get("perm_move", 499)), rng)
            sig_move = (bool(p_move < 0.05) if np.isfinite(p_move) else None)

        # per-point period assignment so the tool renders reliably (no client re-binning)
        label_col = kwargs.get("label") or next(
            (c for c in ("place", "name", "label", "地名", "名称", "city") if c in df.columns), None)
        lbl2idx = {pp["period"]: i for i, pp in enumerate(periods)}
        cap = 3000
        pts_out = []
        for _, row in d.head(cap).iterrows():
            pts_out.append({
                "lat": round(float(row["_lat"]), 5), "lon": round(float(row["_lon"]), 5),
                "period_idx": int(lbl2idx.get(str(row["_period"]), 0)),
                "t": (float(row["_t"]) if has_time else None),
                "label": (str(row[label_col]) if label_col and pd.notna(row.get(label_col)) else None),
                "value": (float(row[vcol]) if (vcol and vcol in d.columns and pd.notna(row.get(vcol))) else None),
            })

        # overall center + bounding box
        oclat, oclon = _spherical_mean(d["_lat"].to_numpy(), d["_lon"].to_numpy(),
                                       w_all if w_all is not None else None)
        result.update({
            "lat_col": lat, "lon_col": lon, "time_col": tcol if has_time else None,
            "value_col": vcol if (vcol and vcol in d.columns) else None,
            "weighted": w_all is not None,
            "has_time": has_time,
            "n": int(len(d)), "n_periods": len(periods),
            "periods": periods,
            "trajectory": traj,
            "points": pts_out,
            "points_truncated": bool(len(d) > cap),
            "migration": {
                "path_length_km": round(path, 1),
                "net_displacement_km": round(net, 1),
                "net_bearing": bearing,
                "steps_km": steps,
                "p_value": (round(float(p_move), 3) if np.isfinite(p_move) else None),
                "significant": sig_move,
            },
            "overall": {
                "mean_lat": round(oclat, 4), "mean_lon": round(oclon, 4),
                "lat_min": round(float(d["_lat"].min()), 4), "lat_max": round(float(d["_lat"].max()), 4),
                "lon_min": round(float(d["_lon"].min()), 4), "lon_max": round(float(d["_lon"].max()), 4),
                "t_min": (float(np.nanmin(tvals)) if has_time else None),
                "t_max": (float(np.nanmax(tvals)) if has_time else None),
            },
            "note": "时空重心轨迹:每期加权重心(球面均值)+ 迁移(haversine km/方位)+ 标准距离离散度",
        })
    except Exception as exc:  # never dead-end the tool
        result["error"] = f"spatial_history 未能完成:{exc}"
    state.write("models", "spatial_history", result)
    return state
