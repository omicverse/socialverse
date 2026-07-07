"""``sv.tl._fect`` — the counterfactual (imputation) estimator for panel causal
inference, a.k.a. **FEct** (Liu, Wang & Xu 2024, *AJPS*).

Two-way fixed-effects DID (``sv.tl.did``) averages a single ATT and, on
staggered-adoption panels with dynamic/heterogeneous effects, that average is a
possibly-biased weighted combination of period-specific effects (the Goodman-Bacon
"negative weighting" problem). The counterfactual estimator sidesteps this: it fits
the outcome model **only on untreated observations** (control units + the
pre-treatment periods of treated units), uses it to **impute the counterfactual**
``Y_it(0)`` for every treated observation, and averages the individual effects
``delta_it = Y_it - Yhat_it(0)``. This is the imputation estimator of the modern
heterogeneity-robust DID family (Borusyak-Jaravel-Spiess, Gardner two-stage).

- ``r = 0`` (default) → **additive FEct**: ``Y_it(0) = alpha_i + xi_t`` (unit + time
  FE), estimated on untreated cells by alternating projections. Robust and exact.
- ``r >= 1`` → **IFEct**: adds ``r`` latent interactive factors
  ``lambda_i' f_t`` (Bai 2009 / gsynth), estimated by alternating least squares on
  the untreated cells. This absorbs unobserved time-varying confounders but needs
  each treated unit to have enough pre-periods to identify its loading — it is
  guarded and falls back with a warning when that fails.

Inference is a nonparametric **block bootstrap over units** (resample whole unit
time-series with replacement; duplicated units become distinct pseudo-units).
``placebo=True`` runs the fect placebo test (hold out the pre-treatment window,
impute it out-of-sample, DIM z-test that the pseudo-effect is zero).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _cols, _get_datasets, _pick_outcome


# --------------------------------------------------------------------- matrices
def _build_matrices(df, cols, y_col):
    """Long panel → (Y, D, E) matrices [n_units × n_times] + index bookkeeping.

    ``Y`` = outcome (NaN where the cell is absent), ``D`` = treat_post indicator,
    ``E`` = cell-present mask. ``treat_post`` is ``treatment × (time >= first_treated)``
    when ``first_treated`` is available, else the raw ``treatment`` indicator.
    """
    work = df.copy()
    treat = pd.to_numeric(work[cols["treatment"]], errors="coerce").fillna(0)
    ft_col = cols["first_treated"]
    if ft_col is not None and ft_col in work.columns:
        t = pd.to_numeric(work[cols["time"]], errors="coerce")
        ft = pd.to_numeric(work[ft_col], errors="coerce")
        post = (t >= ft) & np.isfinite(ft)
        tp = (treat * post.astype(float)).astype(float)
        if tp.abs().sum() == 0:
            tp = treat.astype(float)
    else:
        tp = treat.astype(float)
    work["_tp"] = (tp > 0.5).astype(float)

    units = np.sort(work[cols["panel_id"]].unique())
    times = np.sort(pd.to_numeric(work[cols["time"]], errors="coerce").unique())
    ui = {u: i for i, u in enumerate(units)}
    ti = {t: j for j, t in enumerate(times)}
    N, T = len(units), len(times)

    Y = np.full((N, T), np.nan)
    D = np.zeros((N, T))
    E = np.zeros((N, T), dtype=bool)
    tnum = pd.to_numeric(work[cols["time"]], errors="coerce")
    for u, tt, yy, dd in zip(work[cols["panel_id"]], tnum, work[y_col], work["_tp"]):
        if u in ui and tt in ti and np.isfinite(yy):
            i, j = ui[u], ti[tt]
            Y[i, j] = float(yy)
            D[i, j] = float(dd)
            E[i, j] = True

    # onset (time index of first treatment) per unit — anchor for relative event
    # time. Prefer the authoritative first_treated column (robust to a missing onset
    # cell in an unbalanced panel); else fall back to the first observed treated cell.
    onset = np.full(N, -1, dtype=int)
    if ft_col is not None and ft_col in work.columns:
        fyu = (pd.to_numeric(work[ft_col], errors="coerce")
               .groupby(work[cols["panel_id"]]).first())
        for u, i in ui.items():
            fy = fyu.get(u, np.nan)
            if np.isfinite(fy) and fy > 0:
                pos = int(np.searchsorted(times, fy))
                if pos < T:
                    onset[i] = pos
    for i in range(N):
        if onset[i] < 0:
            row = (D[i] > 0.5) & E[i]
            if row.any():
                onset[i] = int(np.argmax(row))
    return Y, D, E, units, times, onset


# ------------------------------------------------------------ additive two-way FE
def _fe_levels(Y, O, maxit=5000, tol=1e-10):
    """Additive two-way FE levels (alpha_i, xi_t) fit on the mask ``O`` by
    alternating projections (Gauss-Seidel on the LS normal equations). Handles
    unbalanced masks; only ``alpha_i + xi_t`` is identified, which is all imputation
    needs."""
    N, T = Y.shape
    Yf = np.nan_to_num(Y)
    a = np.zeros(N)
    g = np.zeros(T)
    cu = O.sum(1).astype(float)
    ct = O.sum(0).astype(float)
    prev = None
    for _ in range(maxit):
        a = np.where(cu > 0, np.where(O, Yf - g[None, :], 0.0).sum(1) / np.where(cu > 0, cu, 1), a)
        g = np.where(ct > 0, np.where(O, Yf - a[:, None], 0.0).sum(0) / np.where(ct > 0, ct, 1), g)
        fit = a[:, None] + g[None, :]
        if prev is not None and np.max(np.abs(fit - prev)) < tol:
            break
        prev = fit
    return a, g


def _ifect_factors(Y, O, r, a, g, inner=150, tol=1e-7):
    """One IFEct outer pass: given additive FE (a, g), estimate ``r`` interactive
    factors on the untreated cells by alternating least squares (Bai 2009), returning
    the fitted factor matrix ``lambda_i' f_t``."""
    N, T = Y.shape
    Of = O.astype(float)
    R = np.nan_to_num(Y) - a[:, None] - g[None, :]
    R = np.where(O, R, 0.0)
    # init factors from the SVD of the untreated-filled residual
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    F = Vt[:r].T * np.sqrt(T)  # (T, r)
    I = np.eye(r)
    Lam = np.zeros((N, r))
    for _ in range(inner):
        G = np.einsum("it,tr,ts->irs", Of, F, F) + 1e-8 * I
        b = np.einsum("it,tr,it->ir", Of, F, R)
        Lam = np.linalg.solve(G, b[..., None])[..., 0]
        G2 = np.einsum("it,ir,is->trs", Of, Lam, Lam) + 1e-8 * I
        b2 = np.einsum("it,ir,it->tr", Of, Lam, R)
        Fn = np.linalg.solve(G2, b2[..., None])[..., 0]
        Fn = Fn * np.sqrt(T) / np.sqrt((Fn ** 2).sum(0, keepdims=True) + 1e-12)
        if np.max(np.abs(Fn - F)) < tol:
            F = Fn
            break
        F = Fn
    return Lam @ F.T


def _fect_fit(Y, D, E, r=0, outer=200, tol=1e-6):
    """Fit the counterfactual model and return (Yhat0, converged).

    ``Yhat0`` is the imputed untreated potential outcome on every cell. Treated cells
    ``M = E & (D==1)`` are the ones that get a counterfactual; the fit uses only the
    untreated observed cells ``O = E & (D==0)``.
    """
    O = E & (D < 0.5)
    if r <= 0:
        a, g = _fe_levels(Y, O)
        return a[:, None] + g[None, :], True
    # IFEct: alternate additive FE and factor estimation
    a, g = _fe_levels(Y, O)
    fac = np.zeros_like(Y)
    conv = False
    for _ in range(outer):
        fac = _ifect_factors(Y, O, r, a, g)
        a2, g2 = _fe_levels(Y - fac, O)
        if np.max(np.abs(a2 - a)) < tol and np.max(np.abs(g2 - g)) < tol:
            a, g = a2, g2
            conv = True
            break
        a, g = a2, g2
    return a[:, None] + g[None, :] + fac, conv


# ------------------------------------------------------------------- helper stats
def _att_from(Y, Yhat0, M):
    return float((Y - Yhat0)[M].mean()) if M.any() else float("nan")


def _keep_estimable(D, E, r):
    """Boolean unit mask: keep units with at least one untreated observed cell (needed
    to identify the counterfactual), and — for IFEct — enough pre/untreated cells to
    identify an ``r``-dim loading."""
    O = E & (D < 0.5)
    n_untreated = O.sum(1)
    need = max(1, r + 2) if r > 0 else 1
    return n_untreated >= need


# ============================================================================ fect
@register(
    name="fect",
    aliases=["反事实估计", "counterfactual_estimator", "imputation_did", "ifect"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description="反事实(插补)估计量 FEct/IFEct:异质稳健 DID,untreated 拟合两向FE(+r因子)→插补 Y(0)→ATT+块bootstrap",
    requires={
        "design": ["panel_id", "time", "treatment"],
        "variables": ["outcome"],
    },
    produces={"models": ["fect"]},
    auto_fix="escalate",
)
def fect(state: StudyState, **kwargs: Any) -> StudyState:
    """Counterfactual (imputation) estimator for staggered/heterogeneous DID.

    Fits the untreated potential-outcome model on untreated observations (unit + time
    FE, plus ``r`` interactive factors when ``r>=1``), imputes ``Y_it(0)`` for treated
    observations, and averages the individual effects into the ATT. Robust to the
    dynamic-heterogeneity bias that afflicts two-way-FE DID.

    Keyword arguments
    -----------------
    r : int, default 0
        Number of interactive fixed-effect factors. ``0`` = additive FEct (robust,
        exact). ``>=1`` = IFEct (needs adequate pre-periods per treated unit).
    nboots : int, default 200
        Block-bootstrap replications (resample units). Set ``1000`` to match published
        SEs; the point estimate is deterministic and unaffected.
    placebo : bool, default False
        Run the placebo test on the pre-treatment window.
    placebo_periods : int, default 3
        Width (in periods just before onset) of the placebo/held-out window.
    seed : int, default 42
        Bootstrap RNG seed (results are otherwise reproducible).
    alpha : float, default 0.05
        Two-sided level for the bootstrap percentile CI.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)

    def _empty(note):
        model = {"att": None, "se": None, "ci": None, "estimator": "fect", "note": note}
        state.write("models", "fect", model)
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time", "treatment")):
        return _empty("缺少面板数据或设计列(panel_id/time/treatment),无法估计")

    r = int(kwargs.get("r", 0))
    nboots = int(kwargs.get("nboots", 200))
    seed = int(kwargs.get("seed", 42))
    alpha = float(kwargs.get("alpha", 0.05))
    do_placebo = bool(kwargs.get("placebo", False))
    placebo_periods = int(kwargs.get("placebo_periods", 3))

    y_col = _pick_outcome(df, cols, exclude=[c for c in cols.values() if c])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    Y, D, E, units, times, onset = _build_matrices(df, cols, y_col)

    # drop units without an estimable counterfactual (always-treated etc.)
    keep = _keep_estimable(D, E, r)
    n_dropped = int((~keep).sum())
    Y, D, E, onset = Y[keep], D[keep], E[keep], onset[keep]
    units = units[keep]
    N, T = Y.shape
    M = E & (D > 0.5)
    if M.sum() == 0:
        return _empty("没有可估计的处理观测(可能全部单位无处理前期)")

    notes = []
    Yhat0, conv = _fect_fit(Y, D, E, r=r)
    if r > 0 and not conv:
        notes.append("IFEct 未完全收敛,结果谨慎解读(建议改用 r=0 或减小 r)")
    att = _att_from(Y, Yhat0, M)
    estimator = "fect_additive_imputation" if r == 0 else f"ifect_r{r}"

    def _rel_matrix(onset_vec):
        """Relative event time per cell = column index - unit onset (sentinel for
        never-treated units). Anchored on the authoritative onset, not argmax(D)."""
        rel = np.full((len(onset_vec), T), np.iinfo(np.int32).min)
        cols_ix = np.arange(T)
        for i, o in enumerate(onset_vec):
            if o >= 0:
                rel[i] = cols_ix - o
        return rel

    # ---- dynamic effects: att by relative period ----
    delta = Y - Yhat0
    rel_all = _rel_matrix(onset)
    att_by_period: dict[int, float] = {}
    n_by_period: dict[int, int] = {}
    for s in np.unique(rel_all[M]):
        cells = M & (rel_all == s)
        if cells.any():
            att_by_period[int(s)] = float(delta[cells].mean())
            n_by_period[int(s)] = int(cells.sum())

    # ---- block bootstrap over units ----
    rng = np.random.default_rng(seed)
    boots = np.empty(nboots)
    boot_by_period: dict[int, list] = {s: [] for s in att_by_period}
    for b in range(nboots):
        idx = rng.integers(0, N, size=N)  # resample unit rows w/ replacement (pseudo-ids)
        Yb, Db, Eb, onset_b = Y[idx], D[idx], E[idx], onset[idx]
        Mb = Eb & (Db > 0.5)
        if Mb.sum() == 0:
            boots[b] = np.nan
            continue
        try:
            Yh_b, _ = _fect_fit(Yb, Db, Eb, r=r)
            att_b = _att_from(Yb, Yh_b, Mb)
            # collect one period-ATT per resample atomically, so boot_by_period and
            # boots stay aligned even if this iteration raises partway through.
            db = Yb - Yh_b
            rel_b = _rel_matrix(onset_b)
            per_period = {}
            for s in boot_by_period:
                cells = Mb & (rel_b == s)
                if cells.any():
                    per_period[s] = float(db[cells].mean())
            boots[b] = att_b
            for s, v in per_period.items():
                boot_by_period[s].append(v)
        except Exception:
            boots[b] = np.nan
    boots = boots[np.isfinite(boots)]
    # only emit inference with enough valid replications (else it is noise)
    min_boots = max(30, nboots // 4)
    have_inf = boots.size >= min_boots
    if not have_inf and boots.size:
        notes.append(f"有效 bootstrap 复制仅 {boots.size}/{nboots}(<{min_boots}),不报 SE/CI")
    se = float(np.std(boots, ddof=1)) if have_inf else None
    ci = ([float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))]
          if have_inf else None)
    p = None
    if se and se > 0:
        from scipy import stats
        p = float(2 * (1 - stats.norm.cdf(abs(att / se))))

    # per-period SE only when a period appears in enough resamples to be meaningful
    period_se = {s: (float(np.std(v, ddof=1)) if len(v) >= min(10, min_boots) else None)
                 for s, v in boot_by_period.items()}

    # ---- placebo test (hold out the pre-treatment window, impute out-of-sample) ----
    placebo_res = None
    if do_placebo:
        placebo_res = _placebo_test(Y, D, E, r, placebo_periods, nboots, seed + 1)

    model = {
        "att": att,
        "se": se,
        "ci": ci,
        "p": p,
        "n_treated_obs": int(M.sum()),
        "n_units": int(N),
        "n_units_dropped": n_dropped,
        "n_boots": int(boots.size),
        "outcome": y_col,
        "r": r,
        "estimator": estimator,
        "att_by_period": att_by_period,
        "att_by_period_se": period_se,
        "n_by_period": n_by_period,
        "placebo": placebo_res,
        "note": "反事实插补 ATT(异质稳健);块 bootstrap SE(按单位重抽)"
                + ("。丢弃 %d 个无处理前期单位" % n_dropped if n_dropped else "")
                + ("。" + " ".join(notes) if notes else ""),
    }
    state.write("models", "fect", model)
    return state


def _placebo_test(Y, D, E, r, S, nboots, seed):
    """fect placebo test: mask the ``S`` pre-treatment periods just before onset,
    refit on the remaining untreated cells, impute the held-out window out-of-sample,
    and DIM z-test that the pseudo-effect is zero. A significant p flags a pre-trend
    (identification concern)."""
    N, T = Y.shape
    Dm = D.copy()
    Em = E.copy()
    treated_units = (D > 0.5).any(axis=1)
    placebo_mask = np.zeros((N, T), dtype=bool)
    for i in range(N):
        if not treated_units[i]:
            continue
        f0 = int(np.argmax(D[i] > 0.5))
        win = [t for t in range(f0 - S, f0) if 0 <= t and E[i, t] and D[i, t] < 0.5]
        for t in win:
            placebo_mask[i, t] = True
            Em[i, t] = False  # hold out of the fit
    if placebo_mask.sum() == 0:
        return {"note": "无可用的处理前窗口,placebo 未运行"}

    # Holding out the window can push a unit below the estimable threshold (esp.
    # IFEct r>=1, where too few untreated cells make the loading degenerate). Restrict
    # the placebo to units that stay estimable so we never impute from garbage.
    keep2 = _keep_estimable(Dm, Em, r)
    n_drop = int((placebo_mask.any(1) & ~keep2).sum())
    Y2, Dm2, Em2, pmask = Y[keep2], Dm[keep2], Em[keep2], placebo_mask[keep2]
    if pmask.sum() == 0:
        return {"note": "挖窗后无可估计的处理前单位,placebo 未运行"}
    N2 = Y2.shape[0]

    def _fit(Yx, Dx, Ex):
        return _fect_fit(Yx, Dx, Ex, r=r)[0]

    Yh = _fit(Y2, Dm2, Em2)
    att_p = float((Y2 - Yh)[pmask].mean())

    rng = np.random.default_rng(seed)
    pb = []
    for _ in range(nboots):
        idx = rng.integers(0, N2, size=N2)
        pm = pmask[idx]
        if pm.sum() == 0:
            continue
        try:  # a degenerate resample must skip, not crash the whole test
            Yh_b = _fit(Y2[idx], Dm2[idx], Em2[idx])
            pb.append(float((Y2[idx] - Yh_b)[pm].mean()))
        except Exception:
            continue
    pb = np.array(pb)
    ok = pb.size >= max(30, nboots // 4)
    se_p = float(np.std(pb, ddof=1)) if ok else None
    p = None
    if se_p and se_p > 0:
        from scipy import stats
        p = float(2 * (1 - stats.norm.cdf(abs(att_p / se_p))))
    note = "DIM 双侧 z 检验;p 大 = 未见处理前效应,支持识别假设"
    if n_drop:
        note += f"(挖窗后丢弃 {n_drop} 个不可估计单位)"
    return {"placebo_att": att_p, "placebo_se": se_p, "placebo_p": p,
            "window_periods": S, "n_placebo_cells": int(pmask.sum()),
            "n_boots": int(pb.size), "note": note}


__all__ = ["fect"]
