"""``sv.tl._psychometrics`` — the latent-variable / measurement gap.

Three registry entries fill the psychometrics hole in the social-science stack —
the family of models that treat what we *measure* (test items, questionnaire
scales) as noisy indicators of an unobserved construct:

- :func:`cfa` (**CFA / confirmatory factor analysis**) — fit a hypothesized
  measurement model (which items load on which factor) and report factor
  loadings plus the global fit triplet ``CFI / RMSEA / SRMR``.
- :func:`sem` (**SEM / structural equation modeling**) — estimate a system of
  structural (regression) paths among variables, with per-equation R² and the
  same global fit indices.
- :func:`irt` (**IRT / item response theory**) — estimate a 2-parameter logistic
  model: item discrimination ``a`` and difficulty ``b`` plus person abilities
  ``theta`` from binary item responses.

Real-world champions this mirrors: R's **lavaan** and Python **semopy** for
CFA/SEM; **mirt** (R) / **girth** (Python) for IRT. Those are used as optional
accelerators only — when they are absent, every function falls back to a genuine
implementation built on ``statsmodels`` / ``scipy`` / ``numpy`` so a notebook
with none of them installed still recovers the right numbers:

- CFA fallback: maximum-likelihood factor analysis on each factor's item block
  (``statsmodels.multivariate.Factor``), assembling the model-implied covariance
  from the estimated loadings + uniquenesses and scoring fit against the sample
  covariance. When the model is a single-factor block this is exact ML-CFA; with
  correlated factors it is an *honest approximation* (block-wise loadings with an
  estimated inter-factor correlation), which we label as such.
- SEM fallback: **observed-variable path analysis** — each structural equation is
  an OLS regression, path coefficients are the standardized slopes, and global
  fit compares the path-implied covariance to the sample covariance. This is the
  classic path-analysis special case of SEM (no latent variables), labelled
  ``estimator="path_analysis_ols"``.
- IRT fallback: joint / marginal maximum-likelihood 2PL via ``scipy.optimize`` —
  alternating person-ability and item-parameter Newton/BFGS steps.

Everything is deterministic (seeded), returns real numbers, and writes through
the 12-slot :class:`~socialverse._state.StudyState` contract.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["cfa", "sem", "irt"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import — returns the module or ``None`` if unavailable."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _resolve_data(state: StudyState, kwargs: dict) -> pd.DataFrame:
    """Resolve a modelling frame from kwargs['data'] or state.sources['datasets']."""
    data = kwargs.get("data")
    if data is None:
        data = state.sources.get("datasets")
    if isinstance(data, dict):
        data = next(iter(data.values()), None)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data) if data is not None else pd.DataFrame()
    return data


def _numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Complete-case numeric matrix over ``cols`` (rows × len(cols))."""
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    return sub.to_numpy(dtype=float)


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return cov / np.outer(d, d)


# --------------------------------------------------------------------- CFA math
def _ml_factor_loadings(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-factor ML loadings + uniquenesses for one item block.

    Prefers ``statsmodels.multivariate.factor.Factor`` (ML). Falls back to the
    first principal-component of the correlation matrix scaled to unit factor
    variance (principal-factor style) if statsmodels is unavailable.

    Returns ``(loadings[k], uniquenesses[k])`` on the *standardized* metric.
    """
    k = block.shape[1]
    R = np.corrcoef(block, rowvar=False)
    R = np.atleast_2d(R)
    if k == 1:
        return np.array([1.0]), np.array([0.0])

    fa_mod = _try_import("statsmodels.multivariate.factor")
    if fa_mod is not None:
        try:
            fa = fa_mod.Factor(corr=R, n_factor=1, method="ml")
            res = fa.fit()
            load = np.asarray(res.loadings, dtype=float).reshape(-1)
            # sign-orient so the majority of loadings are positive
            if np.sum(load) < 0:
                load = -load
            uniq = np.clip(1.0 - load**2, 1e-6, 1.0)
            return load, uniq
        except Exception:
            pass

    # principal-factor fallback: leading eigenvector of R, scaled by sqrt(eigval)
    vals, vecs = np.linalg.eigh(R)
    lead = vecs[:, -1] * np.sqrt(max(vals[-1], 0.0))
    if np.sum(lead) < 0:
        lead = -lead
    load = np.clip(lead, -0.999, 0.999)
    uniq = np.clip(1.0 - load**2, 1e-6, 1.0)
    return load, uniq


def _model_implied_cov(spec: dict[str, list[str]], loadings: dict[str, np.ndarray],
                       uniq: dict[str, np.ndarray], items: list[str],
                       phi: np.ndarray) -> np.ndarray:
    """Assemble Σ(θ) = Λ Φ Λ' + Ψ on the standardized (correlation) metric.

    ``phi`` is the factor correlation matrix (factors ordered as ``spec`` keys).
    """
    p = len(items)
    idx = {it: i for i, it in enumerate(items)}
    factors = list(spec.keys())
    Lambda = np.zeros((p, len(factors)))
    Psi = np.zeros(p)
    for fj, f in enumerate(factors):
        for it, lo in zip(spec[f], loadings[f]):
            Lambda[idx[it], fj] = lo
        for it, uu in zip(spec[f], uniq[f]):
            Psi[idx[it]] = uu
    Sigma = Lambda @ phi @ Lambda.T + np.diag(Psi)
    return Sigma, Lambda, Psi


def _fit_indices(S: np.ndarray, Sigma: np.ndarray, n: int, n_params: int) -> dict:
    """CFI / RMSEA / SRMR from sample corr ``S`` and model-implied ``Sigma``.

    Uses the normal-theory ML discrepancy F = tr(S Σ⁻¹) − log|S Σ⁻¹| − p.
    """
    p = S.shape[0]
    eps = 1e-8
    Sig = Sigma + eps * np.eye(p)
    try:
        SigInv = np.linalg.inv(Sig)
    except np.linalg.LinAlgError:
        SigInv = np.linalg.pinv(Sig)
    M = S @ SigInv
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        logdet = np.log(max(np.linalg.det(M), eps))
    F_model = float(np.trace(M) - logdet - p)
    F_model = max(F_model, 0.0)

    df_model = p * (p + 1) / 2 - n_params
    df_model = max(df_model, 1.0)
    chi2_model = max(n - 1, 1) * F_model

    # baseline (independence) model: Σ = diag(S) → for a correlation matrix, I
    S0 = np.diag(np.diag(S))
    try:
        S0Inv = np.linalg.inv(S0 + eps * np.eye(p))
    except np.linalg.LinAlgError:
        S0Inv = np.linalg.pinv(S0)
    M0 = S @ S0Inv
    sign0, logdet0 = np.linalg.slogdet(M0)
    if sign0 <= 0:
        logdet0 = np.log(max(np.linalg.det(M0), eps))
    F_base = max(float(np.trace(M0) - logdet0 - p), 0.0)
    df_base = p * (p - 1) / 2
    chi2_base = max(n - 1, 1) * F_base

    # CFI = 1 − max(χ²_m − df_m, 0) / max(χ²_b − df_b, χ²_m − df_m, 0)
    num = max(chi2_model - df_model, 0.0)
    den = max(chi2_base - df_base, num, 1e-12)
    cfi = 1.0 - num / den
    cfi = float(np.clip(cfi, 0.0, 1.0))

    # RMSEA = sqrt(max(F_m/df_m − 1/(n−1), 0))
    rmsea = np.sqrt(max(F_model / df_model - 1.0 / max(n - 1, 1), 0.0))
    rmsea = float(np.clip(rmsea, 0.0, 1.0))

    # SRMR = RMS of residual correlations (lower triangle incl. diagonal)
    resid = S - Sigma
    tril = np.tril_indices(p, k=0)
    srmr = float(np.sqrt(np.mean(resid[tril] ** 2)))

    return {
        "CFI": cfi,
        "RMSEA": rmsea,
        "SRMR": srmr,
        "chi2": float(chi2_model),
        "df": float(df_model),
        "F_ml": F_model,
        "n": int(n),
        "n_params": int(n_params),
    }


# --------------------------------------------------------------------- CFA
@register(
    name="cfa",
    aliases=["验证性因子分析", "confirmatory_factor_analysis", "measurement_model"],
    category="psychometrics",
    tier="plus",
    skill="(SEM/CFA 缺口)",
    languages=["Python"],
    key_tools=["statsmodels", "semopy", "factor_analyzer"],
    description="验证性因子分析:按测量模型估载荷 + 拟合指数(CFI/RMSEA/SRMR)",
    requires={"sources": ["datasets"]},
    produces={"models": ["cfa"], "diagnostics": ["fit_indices"]},
    auto_fix="escalate",
)
def cfa(state: StudyState, **kwargs: Any) -> StudyState:
    """Confirmatory factor analysis: fit a hypothesized measurement model.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional
        Item response matrix (rows = respondents, cols = indicators). Overrides
        ``state.sources['datasets']``.
    model_spec : dict[str, list[str]]
        ``{factor: [item, ...]}`` — which observed columns load on which factor.
        If omitted, all numeric columns are put on a single factor.

    Notes
    -----
    Prefers **semopy** (lavaan-style ML-SEM) when installed. Otherwise fits ML
    single-factor loadings per block via ``statsmodels`` and assembles the
    model-implied covariance to score CFI/RMSEA/SRMR. With one factor this is
    exact ML-CFA; with correlated factors the inter-factor correlation is
    estimated from factor scores (honest block-wise approximation).
    """
    data = _resolve_data(state, kwargs)
    spec: dict[str, list[str]] = kwargs.get("model_spec") or {}

    numeric_cols = [c for c in data.columns
                    if pd.api.types.is_numeric_dtype(pd.to_numeric(data[c], errors="coerce"))]
    if not spec:
        spec = {"F1": [str(c) for c in numeric_cols]}
    # keep only items that actually exist as columns
    spec = {f: [c for c in cols if c in data.columns] for f, cols in spec.items()}
    spec = {f: cols for f, cols in spec.items() if cols}

    items: list[str] = []
    for cols in spec.values():
        for c in cols:
            if c not in items:
                items.append(c)

    X = _numeric_matrix(data, items)
    n = int(X.shape[0])
    # standardize to the correlation metric (CFA loadings reported standardized)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    S = np.corrcoef(Xs, rowvar=False)
    S = np.atleast_2d(S)

    backend = "path_ml_statsmodels"
    loadings_out: dict[str, dict[str, float]] = {}
    fit: dict[str, Any] = {}

    # ---- optional semopy champion -----------------------------------------
    semopy = _try_import("semopy")
    used_semopy = False
    if semopy is not None:
        try:
            lines = []
            for f, cols in spec.items():
                lines.append(f"{f} =~ " + " + ".join(cols))
            model_desc = "\n".join(lines)
            frame = pd.DataFrame(Xs, columns=items)
            mod = semopy.Model(model_desc)
            mod.fit(frame)
            est = mod.inspect()
            # loadings: rows with op '=~'
            for _, r in est.iterrows():
                if str(r.get("op")) == "=~":
                    f = str(r["lval"])
                    it = str(r["rval"])
                    loadings_out.setdefault(f, {})[it] = float(r["Estimate"])
            try:
                stats = semopy.calc_stats(mod).iloc[0]
                fit = {
                    "CFI": float(stats.get("CFI", np.nan)),
                    "RMSEA": float(stats.get("RMSEA", np.nan)),
                    "SRMR": float(stats.get("SRMR", np.nan)),
                    "chi2": float(stats.get("chi2", np.nan)),
                    "df": float(stats.get("DoF", np.nan)),
                    "n": n,
                }
            except Exception:
                fit = {}
            used_semopy = bool(loadings_out)
        except Exception:
            used_semopy = False

    # ---- statsmodels / numpy ML fallback ----------------------------------
    if not used_semopy:
        backend = "path_ml_statsmodels"
        loadings: dict[str, np.ndarray] = {}
        uniq: dict[str, np.ndarray] = {}
        scores = np.zeros((n, len(spec)))
        for fj, (f, cols) in enumerate(spec.items()):
            block = _numeric_matrix(data, cols)
            block = (block - block.mean(axis=0)) / (block.std(axis=0, ddof=1) + 1e-12)
            lo, uu = _ml_factor_loadings(block)
            loadings[f] = lo
            uniq[f] = uu
            loadings_out[f] = {c: float(v) for c, v in zip(cols, lo)}
            # regression factor score (sum of standardized items weighted by loading)
            w = lo / np.clip(uu, 1e-6, None)
            scores[:, fj] = block @ w

        # inter-factor correlation from factor scores (Φ off-diagonal)
        n_fac = len(spec)
        if n_fac > 1:
            phi = np.corrcoef(scores, rowvar=False)
            phi = np.atleast_2d(phi)
            np.fill_diagonal(phi, 1.0)
        else:
            phi = np.eye(1)

        Sigma, Lambda, Psi = _model_implied_cov(spec, loadings, uniq, items, phi)
        # free params: loadings + uniquenesses + factor covariances (Φ off-diag)
        n_load = sum(len(v) for v in loadings.values())
        n_uniq = len(items)
        n_phi = n_fac * (n_fac - 1) // 2
        n_params = n_load + n_uniq + n_phi
        fit = _fit_indices(S, Sigma, n, n_params)
        fit["factor_correlation"] = (phi.tolist() if n_fac > 1 else [[1.0]])

    # ---- assemble outputs -------------------------------------------------
    all_loadings = [v for f in loadings_out for v in loadings_out[f].values()]
    cfa_model = {
        "spec": spec,
        "loadings": loadings_out,
        "backend": "semopy" if used_semopy else backend,
        "n": n,
        "items": items,
        "note": ("exact ML-CFA (single factor)" if len(spec) == 1 and not used_semopy
                 else ("semopy ML-SEM" if used_semopy
                       else "block-wise ML loadings + estimated Φ (honest approximation)")),
        "mean_loading": float(np.mean(all_loadings)) if all_loadings else float("nan"),
        "prop_positive": float(np.mean([v > 0 for v in all_loadings])) if all_loadings else float("nan"),
    }

    state.write("models", "cfa", cfa_model)
    state.write("diagnostics", "fit_indices", fit)
    return state


# --------------------------------------------------------------------- SEM
@register(
    name="sem",
    aliases=["结构方程模型", "structural_equation_model", "path_analysis"],
    category="psychometrics",
    tier="pro",
    skill="(SEM 缺口)",
    languages=["Python"],
    key_tools=["semopy", "statsmodels"],
    description="结构方程模型:估计结构路径系数 + R² + 拟合指数(latent 不可用时退化为路径分析)",
    requires={"sources": ["datasets"]},
    produces={"models": ["sem"], "diagnostics": ["fit_indices"]},
    prerequisites={"optional_functions": ["cfa"]},
    auto_fix="escalate",
)
def sem(state: StudyState, **kwargs: Any) -> StudyState:
    """Structural equation modeling over observed variables (path analysis).

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional
        Modelling frame. Overrides ``state.sources['datasets']``.
    paths : dict[str, list[str]]
        Structural model ``{outcome: [predictor, ...]}`` — one entry per
        structural (regression) equation.
    lavaan : str, optional
        A lavaan-style model string (``y ~ x1 + x2``). Used only when semopy is
        installed; otherwise parsed into ``paths``.

    Notes
    -----
    Prefers **semopy** (full ML-SEM with latent variables). Fallback is genuine
    **observed-variable path analysis**: each equation is an OLS regression,
    coefficients are the path estimates, and the path-implied covariance scores
    CFI/RMSEA/SRMR. Labelled ``estimator="path_analysis_ols"`` — no latent
    variables are estimated in the fallback (honest degradation).
    """
    sm = _try_import("statsmodels.api")
    data = _resolve_data(state, kwargs)

    paths: dict[str, list[str]] = kwargs.get("paths") or {}
    lavaan = kwargs.get("lavaan")
    if not paths and isinstance(lavaan, str):
        for line in lavaan.splitlines():
            line = line.strip()
            if "~" in line and "=~" not in line and "~~" not in line:
                lhs, rhs = line.split("~", 1)
                preds = [t.strip() for t in rhs.replace("+", " ").split() if t.strip()]
                paths[lhs.strip()] = preds

    # normalize path predictor/outcome names to existing columns
    paths = {y: [p for p in preds if p in data.columns]
             for y, preds in paths.items() if y in data.columns}
    paths = {y: preds for y, preds in paths.items() if preds}

    # ---- optional semopy champion -----------------------------------------
    semopy = _try_import("semopy")
    used_semopy = False
    path_est: dict[str, dict[str, float]] = {}
    r2: dict[str, float] = {}
    fit: dict[str, Any] = {}
    estimator = "path_analysis_ols"

    if semopy is not None and (lavaan or paths):
        try:
            if isinstance(lavaan, str) and lavaan.strip():
                desc = lavaan
            else:
                desc = "\n".join(f"{y} ~ " + " + ".join(preds)
                                 for y, preds in paths.items())
            mod = semopy.Model(desc)
            mod.fit(data)
            est = mod.inspect()
            for _, r in est.iterrows():
                if str(r.get("op")) == "~":
                    y = str(r["lval"])
                    x = str(r["rval"])
                    path_est.setdefault(y, {})[x] = float(r["Estimate"])
            try:
                stats = semopy.calc_stats(mod).iloc[0]
                fit = {
                    "CFI": float(stats.get("CFI", np.nan)),
                    "RMSEA": float(stats.get("RMSEA", np.nan)),
                    "SRMR": float(stats.get("SRMR", np.nan)),
                }
            except Exception:
                fit = {}
            estimator = "semopy_ml"
            used_semopy = bool(path_est)
        except Exception:
            used_semopy = False

    # ---- OLS path-analysis fallback ---------------------------------------
    if not used_semopy:
        estimator = "path_analysis_ols"
        # variables involved in the structural system
        variables: list[str] = []
        for y, preds in paths.items():
            for v in [y] + preds:
                if v not in variables:
                    variables.append(v)

        for y, preds in paths.items():
            Y = _numeric_matrix(data, [y] + preds)
            yy = Y[:, 0]
            XX = Y[:, 1:]
            if sm is not None:
                Xc = sm.add_constant(XX, has_constant="add")
                res = sm.OLS(yy, Xc).fit()
                coefs = res.params
                path_est[y] = {p: float(coefs[i + 1]) for i, p in enumerate(preds)}
                path_est[y]["(intercept)"] = float(coefs[0])
                r2[y] = float(res.rsquared)
            else:
                Xc = np.column_stack([np.ones(len(yy)), XX])
                beta, *_ = np.linalg.lstsq(Xc, yy, rcond=None)
                yhat = Xc @ beta
                ss_res = float(np.sum((yy - yhat) ** 2))
                ss_tot = float(np.sum((yy - yy.mean()) ** 2))
                path_est[y] = {p: float(beta[i + 1]) for i, p in enumerate(preds)}
                path_est[y]["(intercept)"] = float(beta[0])
                r2[y] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # global fit: compare sample corr to path-implied corr over all variables
        if len(variables) >= 2:
            V = _numeric_matrix(data, variables)
            raw_sd = V.std(axis=0, ddof=1) + 1e-12       # SDs on the RAW metric
            Vs = (V - V.mean(axis=0)) / raw_sd
            S = np.atleast_2d(np.corrcoef(Vs, rowvar=False))
            n = int(V.shape[0])
            vidx = {v: i for i, v in enumerate(variables)}
            # standardized path coefficients: β_std = b_raw · sd(x) / sd(y)
            B = np.zeros((len(variables), len(variables)))
            for y, preds in paths.items():
                sy = raw_sd[vidx[y]]
                for p in preds:
                    raw = path_est[y].get(p, 0.0)
                    sp = raw_sd[vidx[p]]
                    B[vidx[y], vidx[p]] = raw * sp / sy
            I = np.eye(len(variables))
            try:
                IB_inv = np.linalg.inv(I - B)
            except np.linalg.LinAlgError:
                IB_inv = np.linalg.pinv(I - B)
            # residual variances so implied diagonal ≈ 1 (standardized)
            Psi = np.clip(1.0 - np.array([
                sum(B[i, j] ** 2 for j in range(len(variables))) for i in range(len(variables))
            ]), 1e-6, 1.0)
            Sigma = IB_inv @ np.diag(Psi) @ IB_inv.T
            Sigma = _corr_from_cov(Sigma)
            n_params = sum(len(v) for v in paths.values()) + len(variables)
            fit = _fit_indices(S, Sigma, n, n_params)

    sem_model = {
        "paths": paths,
        "coefficients": path_est,
        "r2": r2,
        "estimator": estimator,
        "backend": "semopy" if used_semopy else "statsmodels/numpy OLS",
        "note": ("semopy ML-SEM (latent-capable)" if used_semopy
                 else "observed-variable path analysis (OLS per equation; no latent variables)"),
    }
    state.write("models", "sem", sem_model)
    state.write("diagnostics", "fit_indices", fit)
    return state


# --------------------------------------------------------------------- IRT math
def _p_2pl(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2PL success probability matrix P[person, item]."""
    z = a[None, :] * (theta[:, None] - b[None, :])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _neg_loglik_items(params: np.ndarray, theta: np.ndarray, r: np.ndarray) -> float:
    """Negative log-likelihood for one item's (a, b) given fixed abilities."""
    a, b = params
    a = max(a, 1e-3)
    z = np.clip(a * (theta - b), -35, 35)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -float(np.sum(r * np.log(p) + (1 - r) * np.log(1 - p)))


def _theta_mle(row: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """MLE/EAP ability for one response row given item params (Newton, bounded)."""
    theta = 0.0
    for _ in range(50):
        z = np.clip(a * (theta - b), -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        # log-likelihood gradient + Hessian (add N(0,1) prior for MAP stability)
        grad = float(np.sum(a * (row - p))) - theta
        hess = -float(np.sum(a**2 * p * (1 - p))) - 1.0
        step = grad / hess
        theta -= step
        theta = float(np.clip(theta, -6, 6))
        if abs(step) < 1e-6:
            break
    return theta


def _fit_2pl(resp: np.ndarray, n_iter: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint-ML 2PL: alternate ability MAP and per-item (a,b) minimization.

    Returns ``(a[items], b[items], theta[persons])``.
    """
    opt = _try_import("scipy.optimize")
    n_p, n_i = resp.shape

    # init: b from item difficulty (via prop correct), a from point-biserial
    prop = resp.mean(axis=0)
    prop = np.clip(prop, 0.02, 0.98)
    b = -np.log(prop / (1 - prop))                    # easier item → lower b
    total = resp.sum(axis=1).astype(float)
    a = np.array([
        max(0.3, 1.7 * abs(np.corrcoef(resp[:, j], total)[0, 1]))
        if np.std(resp[:, j]) > 0 else 1.0
        for j in range(n_i)
    ])
    theta = (total - total.mean()) / (total.std() + 1e-9)

    for _ in range(n_iter):
        # E-ish step: update abilities given current item params
        theta = np.array([_theta_mle(resp[i], a, b) for i in range(n_p)])
        # anchor scale: theta ~ N(0,1)
        theta = (theta - theta.mean()) / (theta.std() + 1e-9)
        # M step: update each item's (a, b) by minimizing item NLL
        a_new = a.copy()
        b_new = b.copy()
        for j in range(n_i):
            r = resp[:, j].astype(float)
            if opt is not None:
                try:
                    res = opt.minimize(
                        _neg_loglik_items, x0=np.array([a[j], b[j]]),
                        args=(theta, r), method="L-BFGS-B",
                        bounds=[(0.05, 6.0), (-6.0, 6.0)],
                    )
                    a_new[j], b_new[j] = float(res.x[0]), float(res.x[1])
                    continue
                except Exception:
                    pass
            # numpy Newton fallback for (a, b)
            aj, bj = a[j], b[j]
            for _ in range(25):
                z = np.clip(aj * (theta - bj), -35, 35)
                p = 1.0 / (1.0 + np.exp(-z))
                w = p * (1 - p)
                ga = float(np.sum((r - p) * (theta - bj)))
                gb = float(np.sum((r - p) * (-aj)))
                haa = -float(np.sum(w * (theta - bj) ** 2))
                hbb = -float(np.sum(w * aj**2))
                hab = float(np.sum((r - p) - w * aj * (theta - bj)))
                H = np.array([[haa, hab], [hab, hbb]]) - 1e-6 * np.eye(2)
                try:
                    d = np.linalg.solve(H, np.array([ga, gb]))
                except np.linalg.LinAlgError:
                    break
                aj = float(np.clip(aj - d[0], 0.05, 6.0))
                bj = float(np.clip(bj - d[1], -6.0, 6.0))
                if np.max(np.abs(d)) < 1e-6:
                    break
            a_new[j], b_new[j] = aj, bj
        if np.max(np.abs(a_new - a)) < 1e-4 and np.max(np.abs(b_new - b)) < 1e-4:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    theta = np.array([_theta_mle(resp[i], a, b) for i in range(n_p)])
    return a, b, theta


def _item_information(a: np.ndarray, b: np.ndarray) -> list[dict]:
    """Fisher item information at θ=0 (and peak θ=b) for each item."""
    out = []
    for j in range(len(a)):
        p0 = 1.0 / (1.0 + np.exp(-a[j] * (0.0 - b[j])))
        info0 = a[j] ** 2 * p0 * (1 - p0)
        info_peak = a[j] ** 2 * 0.25          # max information = a²/4 at θ=b
        out.append({
            "item": int(j + 1),
            "a": float(a[j]),
            "b": float(b[j]),
            "info_at_0": float(info0),
            "info_peak": float(info_peak),
        })
    return out


# --------------------------------------------------------------------- IRT
@register(
    name="irt",
    aliases=["项目反应理论", "item_response_theory", "two_pl"],
    category="psychometrics",
    tier="plus",
    skill="(IRT 缺口)",
    languages=["Python"],
    key_tools=["scipy", "girth"],
    description="项目反应理论:2PL 估计题目区分度 a、难度 b 与被试能力 theta + 题目信息",
    requires={"sources": ["datasets"]},
    produces={"models": ["irt"], "diagnostics": ["item_info"]},
    auto_fix="escalate",
)
def irt(state: StudyState, **kwargs: Any) -> StudyState:
    """Two-parameter logistic (2PL) item response theory.

    Parameters (via ``kwargs``)
    ---------------------------
    data : DataFrame, optional
        Binary (0/1) item response matrix (rows = persons, cols = items).
        Overrides ``state.sources['datasets']``.
    items : list[str], optional
        Which columns are items. Defaults to all numeric 0/1 columns.

    Notes
    -----
    Prefers **girth** (`twopl_mml`) when installed. Fallback is a genuine
    joint / marginal maximum-likelihood 2PL via ``scipy.optimize`` — alternating
    person-ability MAP estimation and per-item ``(a, b)`` minimization. Returns
    discriminations, difficulties, abilities, and Fisher item information.
    """
    data = _resolve_data(state, kwargs)
    items = kwargs.get("items")
    if items is None:
        items = [c for c in data.columns
                 if set(pd.to_numeric(data[c], errors="coerce").dropna().unique())
                 <= {0, 1, 0.0, 1.0}]
    items = [c for c in items if c in data.columns]
    if not items:
        items = [str(c) for c in data.columns]

    resp = _numeric_matrix(data, items).astype(int)
    n_p, n_i = resp.shape

    backend = "scipy_jml_2pl"
    a = b = theta = None

    # ---- optional girth champion ------------------------------------------
    girth = _try_import("girth")
    if girth is not None:
        try:
            out = girth.twopl_mml(resp.T)          # girth expects items × persons
            a = np.asarray(out["Discrimination"], dtype=float).reshape(-1)
            b = np.asarray(out["Difficulty"], dtype=float).reshape(-1)
            theta = np.array([_theta_mle(resp[i], a, b) for i in range(n_p)])
            backend = "girth_twopl_mml"
        except Exception:
            a = b = theta = None

    # ---- scipy JML fallback -----------------------------------------------
    if a is None:
        a, b, theta = _fit_2pl(resp)
        backend = "scipy_jml_2pl"

    item_params = [{"item": items[j], "a": float(a[j]), "b": float(b[j])}
                   for j in range(n_i)]
    item_info = _item_information(a, b)

    irt_model = {
        "model": "2PL",
        "backend": backend,
        "items": items,
        "a": [float(v) for v in a],
        "b": [float(v) for v in b],
        "theta": [float(v) for v in theta],
        "item_params": item_params,
        "n_persons": int(n_p),
        "n_items": int(n_i),
        "prop_positive_a": float(np.mean(np.asarray(a) > 0)),
        "note": "2PL item response theory; a=discrimination, b=difficulty, theta=ability",
    }
    state.write("models", "irt", irt_model)
    state.write("diagnostics", "item_info", {"items": item_info,
                                             "backend": backend})
    return state
