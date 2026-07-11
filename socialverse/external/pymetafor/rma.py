"""Pure-Python reconstruction of ``metafor::rma.uni`` (Viechtbauer 2010, JSS).

Reference-driven port under the Omicverse-RebuildR protocol: the R source is the
executable spec, and ``tests/test_parity.py`` gates this module element-wise
against ``metafor::rma`` on the canonical ``dat.bcg`` fixture (class-1
deterministic-numerical parity, tol 1e-6).

Faithful to metafor's actual algorithm, not a look-alike:
- τ² by metafor's **Fisher-scoring** iteration (REML/ML), DerSimonian-Laird and
  equal-effects in closed form — matching metafor's ``rma.uni`` control defaults
  (``stepadj=1``, ``maxiter=100``, ``threshold=1e-5``).
- I²/H² via the Higgins-Thompson "typical within-study variance" s².
- SE(τ²) from the inverse expected Fisher information at convergence.
- Wald / Knapp-Hartung inference; Q_E, Q_M omnibus; prediction interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import linalg, stats

__all__ = ["rma", "RMAResult", "blup", "BLUPResult"]


@dataclass
class RMAResult:
    beta: np.ndarray
    se: np.ndarray
    zval: np.ndarray
    pval: np.ndarray
    ci_lb: np.ndarray
    ci_ub: np.ndarray
    tau2: float
    se_tau2: float | None
    I2: float
    H2: float
    QE: float
    QEp: float
    k: int
    p: int
    method: str
    test: str
    dfs: float | None = None
    QM: float | None = None
    QMp: float | None = None
    s2w: float = field(default=float("nan"))  # typical within-study variance
    # --- inputs retained for downstream predictors (blup, etc.) ---
    yi: np.ndarray | None = None            # observed effects
    vi: np.ndarray | None = None            # sampling variances
    X: np.ndarray | None = None             # design matrix (incl. intercept)
    vb: np.ndarray | None = None            # Var(β̂), un-shrunk fixed-effect part

    def predict(self, level: float = 95.0):
        """Fitted value + confidence & prediction interval for the average effect
        (intercept-only model). Mirrors ``predict.rma``."""
        a = (1 - level / 100.0) / 2.0
        se_pred = float(np.sqrt(self.se[0] ** 2 + self.tau2))
        if self.test == "knha":
            crit = stats.t.ppf(1 - a, self.dfs)
        else:
            crit = stats.norm.ppf(1 - a)
        b = float(self.beta[0])
        return {
            "pred": b,
            "ci_lb": b - crit * float(self.se[0]),
            "ci_ub": b + crit * float(self.se[0]),
            "pi_lb": b - crit * se_pred,
            "pi_ub": b + crit * se_pred,
        }


def _qr_weighted(X, wi):
    """Thin QR of the weight-scaled design W^½X. Returns (Q, R, sw).

    Everything downstream is expressed through Q/R so we never form the
    ill-conditioned (XᵀWX)⁻¹ explicitly — this is what lets the port match
    metafor to 1e-6 even when cond(XᵀWX) ~ 1e11 (uncentred moderators)."""
    sw = np.sqrt(wi)
    Q, R = linalg.qr(sw[:, None] * X, mode="economic")
    return Q, R, sw


def _invcalc(X, wi):
    """(XᵀWX)⁻¹ via QR: R⁻¹R⁻ᵀ (stable, no explicit matrix inverse)."""
    _, R, _ = _qr_weighted(X, wi)
    Rinv = linalg.solve_triangular(R, np.eye(R.shape[0]))
    return Rinv @ Rinv.T


def _fit_beta(yi, X, wi):
    """Weighted least squares β̂ = (XᵀWX)⁻¹XᵀWy via QR + its covariance."""
    Q, R, sw = _qr_weighted(X, wi)
    z = sw * yi
    beta = linalg.solve_triangular(R, Q.T @ z)
    Rinv = linalg.solve_triangular(R, np.eye(R.shape[0]))
    return beta, Rinv @ Rinv.T


def _P_matrix(X, wi):
    """REML projection P = W − WX(XᵀWX)⁻¹XᵀW = W^½(I − QQᵀ)W^½,
    built from the QR factor Q so no ill-conditioned inverse is formed."""
    Q, _, sw = _qr_weighted(X, wi)
    k = X.shape[0]
    return sw[:, None] * (np.eye(k) - Q @ Q.T) * sw[None, :]


def _tau2_iterative(yi, vi, X, method, maxiter=100, threshold=1e-5, stepadj=1.0):
    """metafor Fisher-scoring for REML/ML τ² (rma.uni inner loop).

    Replicates metafor's exact iteration AND its default convergence tolerance
    (``threshold=1e-5``), so the reported τ² matches metafor's reported value
    (which is itself 1e-5-converged, not the exact root). Intercept-only fits
    agree to 1e-6 on all derived quantities; flat/ill-conditioned
    meta-regressions agree to metafor's own convergence bound (see
    RECONSTRUCTION_REPORT.md §Known limitations)."""
    k, p = X.shape
    y = yi
    tau2 = 0.0
    for _ in range(maxiter):
        old = tau2
        wi = 1.0 / (vi + tau2)
        P = _P_matrix(X, wi)
        Py = P @ y
        if method == "REML":
            PP = P @ P
            adj = float(y @ (P @ Py) - np.trace(P)) / float(np.trace(PP))
        elif method == "ML":
            # score 0.5(y'PPy − tr(W) + tr((X'WX)^-1 X'W²X)); info 0.5 tr(PP-ish)
            W = np.diag(wi)
            stXWX = _invcalc(X, wi)
            XtW2X = X.T @ ((wi ** 2)[:, None] * X)
            PP = P @ P
            trP_ml = float(np.sum(wi) - np.trace(stXWX @ XtW2X))
            adj = float(y @ (P @ Py) - trP_ml) / float(np.trace(PP))
        else:
            raise ValueError(method)
        adj *= stepadj
        while tau2 + adj < 0:
            adj /= 2.0
        tau2 = tau2 + adj
        if abs(old - tau2) < threshold:
            break
    return max(tau2, 0.0)


def _tau2_dl(yi, vi, X):
    """DerSimonian-Laird (closed form, general moderator version)."""
    k, p = X.shape
    wi = 1.0 / vi
    beta, stXWX = _fit_beta(yi, X, wi)
    resid = yi - X @ beta
    RSS = float(np.sum(wi * resid ** 2))           # == Q_E
    XtW2X = X.T @ ((wi ** 2)[:, None] * X)
    trace_term = float(np.sum(wi) - np.trace(stXWX @ XtW2X))
    tau2 = (RSS - (k - p)) / trace_term
    return max(tau2, 0.0)


def _typical_s2(vi, X):
    """Higgins-Thompson typical within-study variance s² (for I²/H²)."""
    k, p = X.shape
    wi = 1.0 / vi
    stXWX = _invcalc(X, wi)
    XtW2X = X.T @ ((wi ** 2)[:, None] * X)
    trace_term = float(np.sum(wi) - np.trace(stXWX @ XtW2X))
    return (k - p) / trace_term


def rma(yi, vi, mods=None, method="REML", test="z", level=95.0, add_intercept=True):
    """Random/mixed/equal-effects meta-analysis — ``metafor::rma`` parity.

    Parameters
    ----------
    yi, vi : array-like  observed effects and their sampling variances.
    mods   : array-like or None  moderator matrix (without intercept column).
    method : {"REML","ML","DL","EE"}  τ² estimator ("EE"/"FE" = equal-effects).
    test   : {"z","knha"}  Wald normal vs Knapp-Hartung t adjustment.
    """
    yi = np.asarray(yi, float).ravel()
    vi = np.asarray(vi, float).ravel()
    k = yi.size
    method = method.upper()
    if method in ("FE", "CE"):
        method = "EE"

    # design matrix
    if mods is None:
        X = np.ones((k, 1))
    else:
        M = np.asarray(mods, float)
        if M.ndim == 1:
            M = M[:, None]
        X = np.hstack([np.ones((k, 1)), M]) if add_intercept else M
    p = X.shape[1]

    # --- τ² ---
    if method == "EE":
        tau2 = 0.0
    elif method == "DL":
        tau2 = _tau2_dl(yi, vi, X)
    else:
        tau2 = _tau2_iterative(yi, vi, X, method)

    # --- β and its covariance ---
    wi = 1.0 / (vi + tau2)
    beta, stXWX = _fit_beta(yi, X, wi)
    vb = stXWX.copy()

    # --- Q_E (heterogeneity) on the equal-effects residuals ---
    wf = 1.0 / vi
    beta_fe, stXWX_fe = _fit_beta(yi, X, wf)
    resid_fe = yi - X @ beta_fe
    QE = float(np.sum(wf * resid_fe ** 2))
    QEdf = k - p
    QEp = float(stats.chi2.sf(QE, QEdf)) if QEdf > 0 else float("nan")

    # --- I², H² ---
    s2 = _typical_s2(vi, X)
    if method == "EE":
        I2 = 100.0 * max(QE - QEdf, 0.0) / QE if QE > 0 else 0.0
        H2 = QE / QEdf if QEdf > 0 else float("nan")
    else:
        I2 = 100.0 * tau2 / (tau2 + s2)
        H2 = (tau2 + s2) / s2

    # --- Knapp-Hartung / Wald inference ---
    dfs = None
    if test == "knha":
        dfs = k - p
        # KH scaling: vb *= (resid' W resid)/(k-p) with W = diag(1/(vi+tau2))
        resid = yi - X @ beta
        s2_kh = float(np.sum(wi * resid ** 2) / (k - p))
        vb = vb * s2_kh
    se = np.sqrt(np.diag(vb))
    tval = beta / se
    a = (1 - level / 100.0) / 2.0
    if test == "knha":
        pval = 2.0 * stats.t.sf(np.abs(tval), dfs)
        crit = stats.t.ppf(1 - a, dfs)
    else:
        pval = 2.0 * stats.norm.sf(np.abs(tval))
        crit = stats.norm.ppf(1 - a)
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    # --- SE(τ²) via inverse Fisher information ---
    se_tau2 = None
    if method in ("REML", "ML"):
        P = _P_matrix(X, wi)
        trPP = float(np.trace(P @ P))
        se_tau2 = float(np.sqrt(2.0 / trPP))
    elif method == "DL":
        # metafor's DL variance of τ² (Q-based, large-sample)
        wf_ = 1.0 / vi
        sw, sw2, sw3 = np.sum(wf_), np.sum(wf_ ** 2), np.sum(wf_ ** 3)
        cc = sw - sw2 / sw
        # Var(Q) under the DL/random model → delta-method on τ² = (Q-(k-p))/c
        A = sw - 2.0 * sw2 / sw + sw2 * sw / (sw ** 2)
        varQ = 2.0 * (k - p) + 4.0 * A * tau2 + 2.0 * (sw2 - 2.0 * sw3 / sw + (sw2 ** 2) / (sw ** 2)) * tau2 ** 2
        se_tau2 = float(np.sqrt(varQ) / cc) if cc > 0 else None

    # --- Q_M omnibus for moderators (excludes intercept) ---
    QM = QMp = None
    if p > 1:
        idx = np.arange(1, p)
        bsub = beta[idx]
        vsub = vb[np.ix_(idx, idx)]
        QM = float(bsub @ linalg.solve(vsub, bsub))
        if test == "knha":
            QM = QM / (p - 1)
            QMp = float(stats.f.sf(QM, p - 1, dfs))
        else:
            QMp = float(stats.chi2.sf(QM, p - 1))

    return RMAResult(
        beta=beta, se=se, zval=tval, pval=pval, ci_lb=ci_lb, ci_ub=ci_ub,
        tau2=float(tau2), se_tau2=se_tau2, I2=float(I2), H2=float(H2),
        QE=QE, QEp=QEp, k=k, p=p, method=method, test=test, dfs=dfs,
        QM=QM, QMp=QMp, s2w=s2,
        yi=yi, vi=vi, X=X, vb=vb,
    )


@dataclass
class BLUPResult:
    """Per-study best linear unbiased predictors (empirical-Bayes shrinkage).

    Mirrors ``metafor::blup.rma.uni``: element-wise ``pred``, its ``se``, and the
    prediction interval (``pi_lb``/``pi_ub``)."""

    pred: np.ndarray
    se: np.ndarray
    pi_lb: np.ndarray
    pi_ub: np.ndarray


def blup(res: RMAResult, level: float = 95.0):
    """Best linear unbiased predictors — ``metafor::blup.rma.uni`` parity.

    For each study *i*, shrinks the observed effect toward its fitted value by the
    reliability ``li = τ²/(τ²+v_i)`` (Robinson 1991; Viechtbauer 2010)::

        pred_i  = li·y_i + (1−li)·X_i β̂
                = X_i β̂ + li·(y_i − X_i β̂)              # equivalent form
        var_i   = li·v_i + (1−li)²·X_i Var(β̂) X_iᵀ      (li·v_i when li==1)
        se_i    = √var_i
        pi_i    = pred_i ± crit·se_i

    ``crit`` is the standard-normal quantile (or a t-quantile with ``k−p`` df when
    the fit used the Knapp-Hartung adjustment), exactly as in metafor.

    Parameters
    ----------
    res   : RMAResult   a fit from :func:`rma` (must retain ``yi``/``vi``/``X``/``vb``).
    level : float       confidence level for the prediction interval (percent).
    """
    if res.yi is None or res.vi is None or res.X is None or res.vb is None:
        raise ValueError("blup needs an rma() fit that retained yi/vi/X/vb")
    yi = np.asarray(res.yi, float).ravel()
    vi = np.asarray(res.vi, float).ravel()
    X = np.asarray(res.X, float)
    vb = np.asarray(res.vb, float)
    beta = np.asarray(res.beta, float).ravel()
    tau2 = float(res.tau2)

    Xbeta = X @ beta                         # fitted (marginal) value per study
    # reliability / shrinkage weight; li==1 when tau2 is infinite (not used here)
    li = tau2 / (tau2 + vi)
    pred = li * yi + (1.0 - li) * Xbeta

    # Var(pred_i): within-study shrinkage term + uncertainty in the mean.
    # metafor special-cases li==1 (τ²=∞) to li·v_i; here τ² is finite so the
    # general branch applies, but we keep the guard for exact parity.
    quad = np.einsum("ij,jk,ik->i", X, vb, X)   # X_i Var(β̂) X_iᵀ per study
    vpred = np.where(li == 1.0, li * vi, li * vi + (1.0 - li) ** 2 * quad)
    se = np.sqrt(vpred)

    a = (1 - level / 100.0) / 2.0
    if res.test == "knha":
        crit = stats.t.ppf(1 - a, res.dfs)
    else:
        crit = stats.norm.ppf(1 - a)
    pi_lb = pred - crit * se
    pi_ub = pred + crit * se
    return BLUPResult(pred=pred, se=se, pi_lb=pi_lb, pi_ub=pi_ub)
