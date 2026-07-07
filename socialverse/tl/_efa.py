"""``sv.tl._efa`` — registered implementation for the exploratory measurement gap:
**exploratory factor analysis (EFA)**.

The exploratory dual of ``cfa`` (in ``_psychometrics``): instead of *testing* a
pre-specified measurement model, EFA *discovers* how many latent factors a set of
survey items reflects and how strongly each item loads on them. This is the
workhorse behind scale construction in psychology, sociology, marketing and
political science.

Champion package this file mirrors
----------------------------------
* ``efa`` — R's ``psych::fa`` / SPSS ``FACTOR``. Given a set of Likert / numeric
  items, it computes the item correlation matrix ``R``, extracts eigenvalues,
  applies the **Kaiser criterion** (retain factors with eigenvalue > 1) to choose
  the number of factors, screens the correlation matrix for factorability with
  the **Kaiser-Meyer-Olkin (KMO)** measure of sampling adequacy and **Bartlett's
  test of sphericity**, then estimates factor **loadings** with an oblique/
  orthogonal **rotation** (varimax by default) and reports the variance each
  factor explains.

  When ``factor_analyzer`` is installed we defer loadings and KMO/Bartlett to it
  (matching ``psych`` conventions exactly). Otherwise every number is produced
  with a real, self-contained numpy/scipy implementation: principal-component
  extraction of the loading matrix followed by a genuine Kaiser varimax rotation,
  a hand-rolled KMO from the anti-image (partial-correlation) matrix, and
  Bartlett's χ² from the determinant of ``R``. No placeholders — on the toy
  single-factor survey the Kaiser rule recovers ``n_factors == 1`` and the first
  eigenvalue dominates.

The registry contract: ``efa`` ``requires`` a working ``sources['datasets']``
frame and ``produces`` a fitted ``models['efa']`` carrying the factor count,
eigenvalues, loadings, explained variance and the factorability diagnostics — so
a resolver can ground scale-construction claims in an actual extraction rather
than a guess.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["efa"]


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


def _pick_items(df: pd.DataFrame, kwargs: dict[str, Any]) -> list[str]:
    """Resolve the item columns: explicit ``items=`` → columns starting with
    ``item`` → all numeric columns."""
    items = kwargs.get("items")
    if items:
        return [c for c in items if c in df.columns]
    prefixed = [c for c in df.columns if str(c).lower().startswith("item")]
    if prefixed:
        return prefixed
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _kmo(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Kaiser-Meyer-Olkin measure of sampling adequacy (overall + per-item).

    KMO compares the magnitude of the observed correlations to the partial
    correlations. Using the inverse of the correlation matrix, the partial
    correlation between i and j is ``-p_ij / sqrt(p_ii * p_jj)`` where ``P = R⁻¹``.
    ``KMO = Σ r² / (Σ r² + Σ partial²)`` over the off-diagonal, both overall and
    per variable. Higher (→1) ⇒ more factorable; < 0.5 is unacceptable.
    """
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    partial = -Rinv / d  # anti-image / partial correlations
    np.fill_diagonal(partial, 0.0)
    R2 = R.copy()
    np.fill_diagonal(R2, 0.0)
    sum_r2 = float(np.sum(R2 ** 2))
    sum_p2 = float(np.sum(partial ** 2))
    overall = sum_r2 / (sum_r2 + sum_p2) if (sum_r2 + sum_p2) > 0 else float("nan")
    # per-item KMO (columnwise)
    per = np.sum(R2 ** 2, axis=0)
    perp = np.sum(partial ** 2, axis=0)
    per_item = np.where((per + perp) > 0, per / (per + perp), np.nan)
    return float(overall), per_item


def _bartlett(R: np.ndarray, n: int) -> tuple[float, float, int]:
    """Bartlett's test of sphericity: is ``R`` distinguishable from identity?

    ``χ² = -(n - 1 - (2p + 5)/6) · ln|R|`` with ``df = p(p-1)/2``. A small p-value
    means the items are correlated enough to factor. Returns (chi2, p, df).
    """
    p = R.shape[0]
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        # near-singular / non-positive-definite → clamp with a tiny ridge
        Rr = R + 1e-8 * np.eye(p)
        sign, logdet = np.linalg.slogdet(Rr)
    chi2 = -(n - 1 - (2 * p + 5) / 6.0) * float(logdet)
    dof = p * (p - 1) // 2
    from scipy import stats as _st
    pval = float(_st.chi2.sf(max(chi2, 0.0), df=dof))
    return float(max(chi2, 0.0)), pval, int(dof)


def _varimax(loadings: np.ndarray, gamma: float = 1.0,
             iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Kaiser normalized varimax rotation of a ``(p, k)`` loading matrix.

    Iteratively rotates to maximize the variance of squared loadings within each
    factor (simple structure). Reduces to an identity rotation for a single
    factor (nothing to rotate).
    """
    p, k = loadings.shape
    if k < 2:
        return loadings.copy()
    # Kaiser normalization by communalities
    h = np.sqrt(np.sum(loadings ** 2, axis=1))
    h[h == 0] = 1.0
    L = loadings / h[:, None]
    Rot = np.eye(k)
    d_old = 0.0
    for _ in range(iters):
        Lam = L @ Rot
        u, s, vt = np.linalg.svd(
            L.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag(np.sum(Lam ** 2, axis=0)))
        )
        Rot = u @ vt
        d = float(np.sum(s))
        if d_old != 0 and d / d_old < 1 + tol:
            break
        d_old = d
    return (L @ Rot) * h[:, None]


# --------------------------------------------------------------------------- efa
@register(
    name="efa",
    aliases=["探索性因子分析", "efa", "factor_analysis"],
    category="psychometrics",
    tier="plus",
    skill="(测量缺口)",
    languages=["Python"],
    key_tools=["factor_analyzer", "numpy", "scipy"],
    description="探索性因子分析(EFA):相关阵→Kaiser 定因子数 + KMO/Bartlett 适切性 + 旋转载荷 + 解释方差",
    requires={"sources": ["datasets"]},
    produces={"models": ["efa"]},
    auto_fix="escalate",
)
def efa(state: StudyState, **kwargs: Any) -> StudyState:
    """Exploratory factor analysis of a set of survey items.

    Pipeline (R ``psych::fa`` / SPSS ``FACTOR``):

    1. Assemble the numeric item matrix and its correlation matrix ``R``.
    2. Eigen-decompose ``R``; the descending eigenvalues drive factor retention.
    3. **Kaiser criterion**: retain ``#{eigenvalue > 1}`` factors (used when
       ``n_factors`` is not supplied).
    4. **KMO** measure of sampling adequacy + **Bartlett's** test of sphericity —
       the two standard factorability checks.
    5. Extract factor **loadings** — via ``factor_analyzer`` when available, else a
       principal-component extraction with a genuine Kaiser **varimax** rotation.
    6. Report the variance each factor explains.

    kwargs
    ------
    items : list[str], optional
        Item columns. Default: columns starting with ``item`` (else all numeric).
    n_factors : int, optional
        Number of factors to extract. Default: Kaiser criterion.
    rotation : str
        Rotation method (``"varimax"`` default; passed through to
        ``factor_analyzer`` when installed).
    """
    df = _get_datasets(state, kwargs)
    rotation = kwargs.get("rotation", "varimax")

    def _empty(note: str) -> StudyState:
        state.write("models", "efa", {
            "n_factors": 0, "eigenvalues": [], "loadings": {},
            "variance_explained": [], "kmo": None,
            "bartlett_chi2": None, "bartlett_p": None,
            "rotation": rotation, "n": 0, "estimator": None, "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法做探索性因子分析")

    items = _pick_items(df, kwargs)
    if len(items) < 2:
        return _empty(f"可用于因子分析的项目不足(items={items})")

    work = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        return _empty("有效样本量不足,无法估计相关阵")

    X = work.to_numpy(dtype=float)
    n, p = X.shape
    # drop constant items (zero variance breaks the correlation matrix)
    sd = X.std(axis=0)
    keep = sd > 0
    if not keep.all():
        items = [c for c, k in zip(items, keep) if k]
        X = X[:, keep]
        p = X.shape[1]
        if p < 2:
            return _empty("去除常数项后项目不足")

    R = np.corrcoef(X, rowvar=False)

    # -- eigenvalues (descending) → Kaiser criterion ---------------------------
    eigvals = np.linalg.eigvalsh(R)[::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    kaiser_n = int(np.sum(eigvals > 1.0))

    n_factors = kwargs.get("n_factors")
    if n_factors is None:
        n_factors = kaiser_n
    n_factors = int(max(1, min(int(n_factors), p)))

    # -- factorability diagnostics --------------------------------------------
    try:
        kmo_overall, _ = _kmo(R)
    except Exception:
        kmo_overall = None
    try:
        bart_chi2, bart_p, _ = _bartlett(R, n)
    except Exception:
        bart_chi2 = bart_p = None

    # -- loadings + explained variance ----------------------------------------
    fa_lib = _try_import("factor_analyzer")
    loadings = None
    estimator = None
    if fa_lib is not None:
        try:
            FA = fa_lib.FactorAnalyzer(
                n_factors=n_factors, rotation=rotation, method="principal"
            )
            FA.fit(X)
            loadings = np.asarray(FA.loadings_, dtype=float)
            # prefer factor_analyzer's KMO / Bartlett for exact psych parity
            try:
                _, kmo_all = fa_lib.factor_analyzer.calculate_kmo(X)
                kmo_overall = float(kmo_all)
            except Exception:
                pass
            try:
                bc, bp = fa_lib.factor_analyzer.calculate_bartlett_sphericity(X)
                bart_chi2, bart_p = float(bc), float(bp)
            except Exception:
                pass
            estimator = f"factor_analyzer.FactorAnalyzer(principal, rotation={rotation})"
        except Exception:
            loadings = None

    if loadings is None:
        # principal-component extraction + Kaiser varimax (self-contained)
        vals, vecs = np.linalg.eigh(R)
        idx = np.argsort(vals)[::-1][:n_factors]
        lam = np.clip(vals[idx], 0.0, None)
        V = vecs[:, idx]
        raw = V * np.sqrt(lam)[None, :]  # unrotated PC loadings (p, k)
        loadings = _varimax(raw) if rotation == "varimax" else raw
        estimator = f"principal-component extraction + {rotation} (numpy/scipy)"

    # variance explained per factor = Σ loading² / p (proportion of total variance)
    ss = np.sum(loadings ** 2, axis=0)
    variance_explained = (ss / p).tolist()

    loadings_map = {
        it: [float(v) for v in loadings[i]] for i, it in enumerate(items)
    }

    state.write("models", "efa", {
        "n_factors": int(n_factors),
        "kaiser_n_factors": kaiser_n,
        "eigenvalues": [float(v) for v in eigvals],
        "loadings": loadings_map,
        "variance_explained": [float(v) for v in variance_explained],
        "kmo": None if kmo_overall is None else float(kmo_overall),
        "bartlett_chi2": None if bart_chi2 is None else float(bart_chi2),
        "bartlett_p": None if bart_p is None else float(bart_p),
        "rotation": rotation,
        "items": list(items),
        "n": int(n),
        "estimator": estimator,
        "note": ("探索性因子分析:Kaiser 准则(特征值>1)定因子数 = "
                 f"{kaiser_n};KMO 抽样适切性 + Bartlett 球形检验判可因子性;"
                 f"载荷经 {rotation} 旋转,方差解释为各因子载荷平方和 / 项目数"),
    })
    return state
