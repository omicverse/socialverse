"""pymada -- pure-Python reconstruction of R ``mada`` Reitsma bivariate model.

Ports ``mada::reitsma`` (via ``mvmeta::mvmeta.fit``) for the diagnostic
test-accuracy meta-analysis: a bivariate random-effects model on the logit
scale for sensitivity and false-positive-rate, fitted by profiled REML.

Algorithm faithfully mirrors the R source:

* Data prep (``reitsma.default``): logit link (``talpha(1)`` == ``make.link
  ("logit")``, the default ``alphasens = alphafpr = 1``).  Within-study
  variances are the binomial variances of the proportions mapped through the
  square of the logit jacobian ``d = 1/x + 1/(1-x)``.  The between-outcome
  within-study covariance is fixed at 0 (sens & fpr come from disjoint
  patient groups).  Continuity correction: when ANY of TP/FN/FP/TN equals 0,
  add ``correction`` (default 0.5) to ALL four cells of ALL studies
  (``correction.control = "all"``).

* Fit (``mvmeta.reml``): unstructured between-study covariance ``Psi`` (k=2 ->
  3 free parameters = the lower-triangular Cholesky factor).  The profiled
  restricted log-likelihood is maximized over those 3 parameters with BFGS
  (``optim`` ``fnscale=-1``, ``reltol=sqrt(eps)``), starting from 10 IGLS
  iterations off ``diag(0.001)``.  Given ``Psi`` the fixed effects are the GLS
  estimate and ``vcov`` its model covariance.

Public entry point: :func:`reitsma`.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# logit link (talpha(1)) and its jacobian d/dx logit(x) = 1/x + 1/(1-x)
# ---------------------------------------------------------------------------
def _logit(x):
    return np.log(x) - np.log(1.0 - x)


def _plogis(x):
    return 1.0 / (1.0 + np.exp(-x))


def _jacobian(x):
    # jacobiantrafo(alpha=1, x) = 1/x + 1/(1-x)
    return 1.0 / x + 1.0 / (1.0 - x)


# ---------------------------------------------------------------------------
# Psi parametrisation (unstructured, k=2): par = vech of lower-tri Cholesky L
#   par2Psi:  L[lower.tri diag=TRUE] <- par ; Psi = L L'
# For k=2, par = (L11, L21, L22) filled column-major over the lower triangle,
# which R fills as row(col) with lower.tri(diag=TRUE) in column order:
#   L[1,1]=par[1], L[2,1]=par[2], L[2,2]=par[3]
# ---------------------------------------------------------------------------
def _par2Psi(par, k):
    L = np.zeros((k, k))
    idx = np.tril_indices(k)  # (rows, cols) in row-major lower-tri order
    # R's L[lower.tri(L, diag=TRUE)] <- par assigns column-major.  Build the
    # column-major lower-tri index order to match exactly.
    order = _lower_tri_colmajor(k)
    for val, (r, c) in zip(par, order):
        L[r, c] = val
    return L @ L.T


def _lower_tri_colmajor(k):
    """Indices of the lower triangle (incl. diag) in R column-major order."""
    out = []
    for c in range(k):
        for r in range(c, k):
            out.append((r, c))
    return out


def _vech_lower_colmajor(M):
    """vech of a matrix's lower triangle (incl diag) in column-major order."""
    k = M.shape[0]
    return np.array([M[r, c] for (r, c) in _lower_tri_colmajor(k)])


# ---------------------------------------------------------------------------
# GLS fit given Psi:  per-study Sigma_i = S_i + Psi ; whiten with chol.
# ---------------------------------------------------------------------------
def _glsfit(Xlist, ylist, Slist, Psi):
    invtUX_blocks = []
    invtUy_blocks = []
    Ulist = []
    invtUXlist = []
    for X, y, S in zip(Xlist, ylist, Slist):
        Sigma = S + Psi
        U = np.linalg.cholesky(Sigma).T          # upper-tri, R chol convention
        invU = _backsolve(U)                     # U^{-1}
        invtUX = invU.T @ X
        invtUy = invU.T @ y
        Ulist.append(U)
        invtUXlist.append(invtUX)
        invtUX_blocks.append(invtUX)
        invtUy_blocks.append(invtUy)
    invtUX = np.vstack(invtUX_blocks)
    invtUy = np.vstack(invtUy_blocks)
    coef, *_ = np.linalg.lstsq(invtUX, invtUy, rcond=None)
    coef = coef.ravel()
    return {
        "coef": coef,
        "Ulist": Ulist,
        "invtUXlist": invtUXlist,
        "invtUX": invtUX,
        "invtUy": invtUy,
    }


def _backsolve(U):
    """Inverse of upper-triangular U (== R backsolve(U, I))."""
    return np.linalg.solve(U, np.eye(U.shape[0]))


# ---------------------------------------------------------------------------
# Profiled restricted log-likelihood (remlprof.fn), sign as in R (maximize).
# ---------------------------------------------------------------------------
def _remlprof_fn(par, Xlist, ylist, Slist, k, nall):
    Psi = _par2Psi(par, k)
    gls = _glsfit(Xlist, ylist, Slist, Psi)
    coef = gls["coef"]
    ncoef = len(coef)
    pconst = -0.5 * (nall - ncoef) * np.log(2.0 * np.pi)
    resid = gls["invtUy"] - gls["invtUX"] @ coef.reshape(-1, 1)
    pres = -0.5 * float(resid.T @ resid)
    pdet1 = -sum(np.sum(np.log(np.diag(U))) for U in gls["Ulist"])
    tXWXtot = sum(iX.T @ iX for iX in gls["invtUXlist"])
    pdet2 = -np.sum(np.log(np.diag(np.linalg.cholesky(tXWXtot).T)))
    return float(pconst + pdet1 + pdet2 + pres)


# ---------------------------------------------------------------------------
# Analytic REML gradient wrt the Cholesky-vech par (mvmeta gradchol.reml),
# needed to reproduce R's optim(method="BFGS") stationary point exactly.
# ---------------------------------------------------------------------------
def _remlprof_gr(par, Xlist, ylist, Slist, k):
    L = np.zeros((k, k))
    for val, (r, c) in zip(par, _lower_tri_colmajor(k)):
        L[r, c] = val
    U = L.T                       # R: U <- t(L)
    Psi = U.T @ U                 # crossprod(U) = L L'
    gls = _glsfit(Xlist, ylist, Slist, Psi)
    coef = gls["coef"].reshape(-1, 1)
    invtXWXtot = np.linalg.inv(sum(iX.T @ iX for iX in gls["invtUXlist"]))
    invSigmalist = [invU @ invU.T for invU in
                    [_backsolve(Uc) for Uc in gls["Ulist"]]]
    reslist = [y - X @ coef for X, y in zip(Xlist, ylist)]
    # ind1 = rep(1:k, k:1); ind2 = unlist(sapply(1:k, seq, to=k)) -- 1-based
    ind1, ind2 = [], []
    for a in range(k):
        for b in range(a, k):
            ind1.append(a)        # 0-based row of L used
            ind2.append(b)
    grad = np.empty(len(par))
    for i in range(len(par)):
        A = np.zeros((k, k)); B = np.zeros((k, k)); C = np.zeros((k, k))
        A[ind2[i], :] = U[ind1[i], :]
        B[:, ind2[i]] = U[ind1[i], :]
        C[ind2[i], :] = 1.0
        C[:, ind2[i]] = 1.0
        D = C * A + C * B
        g = 0.0
        for X, invSigma, res in zip(Xlist, invSigmalist, reslist):
            E = invSigma @ D @ invSigma
            F = float(res.T @ E @ res)
            G = np.trace(invSigma @ D)
            H = np.trace(invtXWXtot @ (X.T @ E @ X))
            g += 0.5 * (F - G + H)
        grad[i] = g
    return grad


# ---------------------------------------------------------------------------
# IGLS starting value (initpar / iter.igls) -- 10 iterations from diag(0.001).
# ---------------------------------------------------------------------------
def _xpnd(theta, k):
    """Expand a vech (lower-tri, column-major, incl diag) to a symmetric mat."""
    M = np.zeros((k, k))
    for val, (r, c) in zip(theta, _lower_tri_colmajor(k)):
        M[r, c] = val
        M[c, r] = val
    return M


def _iter_igls(Psi, Xlist, ylist, Slist, k):
    gls = _glsfit(Xlist, ylist, Slist, Psi)
    coef = gls["coef"].reshape(-1, 1)
    npar = k * (k + 1) // 2
    # indMat: symmetric matrix of parameter indices (1..npar), column-major vech
    indMat = _xpnd(np.arange(1, npar + 1), k)
    Zlist, flist = [], []
    Sigmalist = [S + Psi for S in Slist]
    for X, y, S, Sigma in zip(Xlist, ylist, Slist, Sigmalist):
        r = (y - X @ coef).ravel()
        f = np.outer(r, r).ravel(order="F") - S.ravel(order="F")
        flist.append(f.reshape(-1, 1))
        z = indMat.ravel(order="F")
        Z = np.column_stack([(z == (j + 1)).astype(float) for j in range(npar)])
        Zlist.append(Z)
    # whiten by eSigma = Sigma (x) Sigma
    invteUZ_blocks, invteUf_blocks = [], []
    for Sigma, Z, f in zip(Sigmalist, Zlist, flist):
        eSigma = np.kron(Sigma, Sigma)
        eU = np.linalg.cholesky(eSigma).T
        inveU = _backsolve(eU)
        invteUZ_blocks.append(inveU.T @ Z)
        invteUf_blocks.append(inveU.T @ f)
    invteUZ = np.vstack(invteUZ_blocks)
    invteUf = np.vstack(invteUf_blocks)
    theta, *_ = np.linalg.lstsq(invteUZ, invteUf, rcond=None)
    Psi_new = _xpnd(theta.ravel(), k)
    # project to PD: eigen-floor at 1e-8
    vals, vecs = np.linalg.eigh(Psi_new)
    vals = np.maximum(vals, 1e-8)
    return (vecs * vals) @ vecs.T


def _initpar(Xlist, ylist, Slist, k, igls_iter=10):
    Psi = np.diag(np.full(k, 0.001))
    for _ in range(igls_iter):
        Psi = _iter_igls(Psi, Xlist, ylist, Slist, k)
    # par = vech(t(chol(Psi))) -- lower Cholesky factor, column-major vech
    L = np.linalg.cholesky(Psi)  # numpy returns lower L with Psi = L L'
    return _vech_lower_colmajor(L)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reitsma(data=None, TP=None, FN=None, FP=None, TN=None,
            correction=0.5, correction_control="all"):
    """Fit the Reitsma bivariate random-effects model (REML).

    Parameters
    ----------
    data : mapping or DataFrame, optional
        If given, columns ``TP``, ``FN``, ``FP``, ``TN`` are pulled from it.
    TP, FN, FP, TN : array-like, optional
        The 2x2 cell counts per study (used when ``data`` is None).
    correction : float
        Continuity correction added on zero cells (default 0.5).
    correction_control : {"all", "single", "none"}
        ``"all"`` adds the correction to every cell of every study when any
        cell is zero (mada default).

    Returns
    -------
    dict with keys:
        ``coefficients``      pooled logit(sens), logit(fpr)   (length 2)
        ``vcov``              2x2 covariance of the fixed effects
        ``Psi``               2x2 between-study covariance
        ``se``                std errors of the coefficients
        ``sensitivity``       pooled sensitivity (inv-logit of coef[0])
        ``false_pos_rate``    pooled FPR       (inv-logit of coef[1])
        ``logLik``            restricted log-likelihood (reitsma-adjusted)
        ``par``               optimized Cholesky-vech parameters
    """
    if data is not None:
        TP = np.asarray(_get(data, "TP"), float)
        FN = np.asarray(_get(data, "FN"), float)
        FP = np.asarray(_get(data, "FP"), float)
        TN = np.asarray(_get(data, "TN"), float)
    else:
        TP = np.asarray(TP, float)
        FN = np.asarray(FN, float)
        FP = np.asarray(FP, float)
        TN = np.asarray(TN, float)

    N = len(TP)

    # Raw (uncorrected) counts -- mada stores these in fit$freqdata and derives
    # the observed FPR range (used for the partial-AUC integration) from them.
    FP_raw = FP.copy()
    TN_raw = TN.copy()

    # --- continuity correction ---
    if correction_control == "all":
        if np.any(np.concatenate([TP, FN, FP, TN]) == 0):
            TP = TP + correction
            FN = FN + correction
            FP = FP + correction
            TN = TN + correction
    elif correction_control == "single":
        flag = ((TP == 0) | (FN == 0) | (FP == 0) | (TN == 0)) * correction
        TP = TP + flag
        FN = FN + flag
        FP = FP + flag
        TN = TN + flag

    n_pos = TP + FN
    n_neg = FP + TN
    sens = TP / n_pos
    fpr = FP / n_neg

    trafo_sens = _logit(sens)
    trafo_fpr = _logit(fpr)
    var_ts = (sens * (1 - sens) / n_pos) * (_jacobian(sens) ** 2)
    var_tf = (fpr * (1 - fpr) / n_neg) * (_jacobian(fpr) ** 2)

    # --- assemble per-study lists (k=2, p=1 intercept-only) ---
    k = 2
    Xlist = [np.array([[1.0, 0.0], [0.0, 1.0]]) for _ in range(N)]  # I2 (x) X[i]
    ylist = [np.array([[trafo_sens[i]], [trafo_fpr[i]]]) for i in range(N)]
    Slist = [np.array([[var_ts[i], 0.0], [0.0, var_tf[i]]]) for i in range(N)]
    nall = 2 * N

    # --- REML fit ---
    par0 = _initpar(Xlist, ylist, Slist, k)
    # Maximize the profiled REML with R's own algorithm: optim(method="BFGS")
    # (vmmin) using the analytic gradient and reltol=sqrt(eps).  Replicating
    # vmmin's stopping rule reaches R's exact stationary point (not a sharper
    # over-converged one that scipy's gtol would find).
    fn = lambda p: _remlprof_fn(p, Xlist, ylist, Slist, k, nall)      # maximize
    gr = lambda p: _remlprof_gr(p, Xlist, ylist, Slist, k)           # d(logLik)/dp
    par = _vmmin_maximize(par0, fn, gr, reltol=np.sqrt(np.finfo(float).eps))
    Psi = _par2Psi(par, k)
    gls = _glsfit(Xlist, ylist, Slist, Psi)
    coef = gls["coef"]

    # vcov = (X' Sigma^{-1} X)^{-1} via QR of the whitened design (as R)
    invtUX = gls["invtUX"]
    Q, R = np.linalg.qr(invtUX)
    Rinv = np.linalg.solve(R, np.eye(R.shape[0]))
    vcov = Rinv @ Rinv.T

    se = np.sqrt(np.diag(vcov))
    # profiled restricted log-likelihood at the optimum (mvmeta scale; the
    # reitsma jacobian adjustment sum(log|d_sens|+log|d_fpr|) is added below).
    loglik_mvmeta = _remlprof_fn(par, Xlist, ylist, Slist, k, nall)
    jac_adj = float(np.sum(np.log(_jacobian(sens)) + np.log(_jacobian(fpr))))
    return {
        "coefficients": coef,
        "vcov": vcov,
        "Psi": Psi,
        "se": se,
        "sensitivity": float(_plogis(coef[0])),
        "false_pos_rate": float(_plogis(coef[1])),
        "logLik": loglik_mvmeta + jac_adj,
        "par": par,
        "alphasens": 1.0,
        "alphafpr": 1.0,
        # raw observed false-positive rates (FP/(FP+TN)) on the UNCORRECTED
        # cell counts -- mada's fit$freqdata; drives the partial-AUC range.
        "freq_fpr": FP_raw / (FP_raw + TN_raw),
    }


# ---------------------------------------------------------------------------
# HSROC coefficients and area under the Rutter-Gatsonis SROC curve (AUC.reitsma)
# ---------------------------------------------------------------------------
def calc_hsroc_coef(fit):
    """Rutter-Gatsonis HSROC coefficients from a fitted Reitsma model.

    Mirrors ``mada:::calc_hsroc_coef`` for the no-covariate (df == 5) case.
    """
    coef = np.asarray(fit["coefficients"], float)
    Psi = np.asarray(fit["Psi"], float)
    ran_sd = np.sqrt(np.diag(Psi))          # (sd_sens, sd_fpr)
    sd1, sd2 = ran_sd[0], ran_sd[1]
    Theta = 0.5 * (np.sqrt(sd2 / sd1) * coef[0] + np.sqrt(sd1 / sd2) * coef[1])
    Lambda = np.sqrt(sd2 / sd1) * coef[0] - np.sqrt(sd1 / sd2) * coef[1]
    sigma2theta = 0.5 * (sd1 * sd2 + Psi[0, 1])
    sigma2alpha = 2.0 * (sd1 * sd2 - Psi[0, 1])
    beta = np.log(sd2 / sd1)
    return {
        "Theta": float(Theta),
        "Lambda": float(Lambda),
        "beta": float(beta),
        "sigma2theta": float(sigma2theta),
        "sigma2alpha": float(sigma2alpha),
    }


def _sroc_ruttergatsonis(fit):
    """Return the Rutter-Gatsonis SROC function sens = f(fpr).

    ``f(x) = inv.trafo(alphasens, Lambda*exp(-beta/2) + exp(-beta)*trafo(alphafpr, x))``.
    With ``alphasens = alphafpr = 1`` the transforms are the plain logit /
    inverse-logit (``talpha(1)``).
    """
    hs = calc_hsroc_coef(fit)
    Lambda, beta = hs["Lambda"], hs["beta"]
    alphasens = fit.get("alphasens", 1.0)
    alphafpr = fit.get("alphafpr", 1.0)

    def f(x):
        x = np.asarray(x, float)
        # trafo(alphafpr, x) with alpha==1 is logit(x)
        lin = Lambda * np.exp(-beta / 2.0) + np.exp(-beta) * _trafo(alphafpr, x)
        return _inv_trafo(alphasens, lin)

    return f


def _trafo(alpha, x):
    # talpha(alpha)$linkfun; alpha == 1 => logit
    if alpha == 1.0:
        return _logit(x)
    raise NotImplementedError("only alpha == 1 (logit) is supported")


def _inv_trafo(alpha, x):
    # talpha(alpha)$linkinv; alpha == 1 => plogis
    if alpha == 1.0:
        return _plogis(x)
    raise NotImplementedError("only alpha == 1 (logit) is supported")


def _auc_default(sroc, fpr):
    """Trapezoidal area under ``sroc`` sampled at the vector ``fpr``.

    Ports ``mada:::AUC.default``:
        ``AUC = (s[0]/2 + sum(s[1:n-1]) + s[n-1]/2) / n``   (n = len(fpr))
    """
    fpr = np.asarray(fpr, float)
    if fpr.ndim != 1:
        raise ValueError("fpr must be a vector")
    if np.any(fpr < 0) or np.any(fpr > 1):
        raise ValueError("fpr values must lie in [0, 1]")
    n = len(fpr)
    if n < 10:
        raise ValueError("specify at least 10 FPR values!")
    s = np.asarray(sroc(fpr), float)
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("sroc values must lie in [0, 1]")
    return float((s[0] / 2.0 + np.sum(s[1:n - 1]) + s[n - 1] / 2.0) / n)


def AUC(fit, fpr=None, sroc_type="ruttergatsonis"):
    """Area (and partial area) under the Rutter-Gatsonis SROC curve.

    Ports ``mada:::AUC.reitsma`` for the no-covariate model.  Returns a dict
    ``{"AUC": ..., "pAUC": ...}`` where

    * ``AUC``  integrates the SROC over ``fpr = 1:99/100`` (the mada default).
    * ``pAUC`` integrates over 99 equally-spaced FPR values across the OBSERVED
      false-positive-rate range (``fit["freq_fpr"]``), clamped to ``[0.01,
      0.99]`` exactly as mada does.

    Parameters
    ----------
    fit : dict
        A fit returned by :func:`reitsma`.
    fpr : array-like, optional
        FPR grid for the full AUC (default ``arange(1, 100)/100``).
    sroc_type : {"ruttergatsonis", "naive"}
        Only ``"ruttergatsonis"`` is implemented (the AUC.reitsma default).
    """
    if sroc_type != "ruttergatsonis":
        raise NotImplementedError("only sroc_type='ruttergatsonis' is supported")
    if fpr is None:
        fpr = np.arange(1, 100) / 100.0

    rsroc = _sroc_ruttergatsonis(fit)
    auc = _auc_default(rsroc, fpr)

    # partial AUC over the observed (clamped) FPR range
    obs = np.asarray(fit["freq_fpr"], float)
    lo, hi = float(np.min(obs)), float(np.max(obs))
    lo = max(0.01, lo)
    hi = min(0.99, hi)
    obsfpr = np.linspace(lo, hi, 99)
    pauc = _auc_default(rsroc, obsfpr)

    return {"AUC": auc, "pAUC": pauc}


def _vmmin_maximize(b0, fn, gr, reltol, maxit=100, abstol=-np.inf):
    """Port of R's ``vmmin`` (the BFGS behind optim), maximizing ``fn``.

    R's ``optim(method="BFGS")`` with ``fnscale=-1`` minimizes ``-fn``; vmmin's
    stopping rule is ``(f - fmin) <= reltol*(|f|+reltol)`` on that minimized
    objective.  We minimize ``F = -fn`` (grad ``-gr``) so the maximizer of ``fn``
    and the exact stopping point match R.
    """
    stepredn, acctol, reltest = 0.2, 1e-4, 10.0
    n = len(b0)
    b = np.array(b0, float)

    def F(x):
        return -fn(x)

    def G(x):
        return -np.asarray(gr(x), float)

    f = F(b)
    g = G(b)
    B = np.eye(n)                 # inverse-Hessian approx
    ilast = 0
    iter_ = 0
    while True:
        if ilast == iter_:
            B = np.eye(n)
        X = b.copy(); c = g.copy()
        t = -B @ g               # search direction
        gradproj = float(g @ t)
        steplength = 1.0
        accpoint = False
        if gradproj < 0.0:
            while True:
                count = 0
                b = X + steplength * t
                # reltest convergence check (elementwise)
                if np.all(reltest + X == reltest + b):
                    count = n
                if count < n:
                    f2 = F(b)
                    if np.isfinite(f2) and f2 <= f + gradproj * steplength * acctol:
                        accpoint = True
                        break
                    else:
                        steplength *= stepredn
                else:
                    break
                if steplength < 1e-18:
                    break
        if gradproj >= 0.0 or not accpoint:
            # cannot find a lower point along this direction
            if ilast < iter_:
                ilast = iter_       # reset to steepest descent and retry
                iter_ += 1
                continue
            break
        # accepted a lower point
        f_old = f
        f = f2
        g2 = G(b)
        # BFGS inverse-Hessian update
        t_step = steplength * t          # s = b - X
        c_diff = g2 - c                  # y = g2 - g
        D1 = float(t_step @ c_diff)
        if D1 > 0.0:
            Bc = B @ c_diff
            D2 = float(c_diff @ Bc)
            D2 = 1.0 + D2 / D1
            B = (B
                 - (np.outer(t_step, Bc) + np.outer(Bc, t_step)
                    - D2 * np.outer(t_step, t_step)) / D1)
        else:
            ilast = iter_
        g = g2
        iter_ += 1
        if f <= abstol:
            break
        if f_old - f <= reltol * (abs(f_old) + reltol):
            break
        if iter_ >= maxit:
            break
    return b


def _get(data, key):
    try:
        return data[key]
    except (KeyError, TypeError):
        return getattr(data, key)
