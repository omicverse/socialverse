"""pylavaan — pure numpy/scipy reconstruction of R lavaan's ML confirmatory
factor analysis (cfa()).

Faithfully replicates lavaan 0.6.x defaults:
  * estimator = "ML" (maximum likelihood, normal theory)
  * marker-variable identification: the FIRST indicator of each factor has its
    loading fixed to 1.0 (std.lv = FALSE)
  * no mean structure (covariance-only)
  * sample covariance uses the *biased* ML divisor N (not N-1)
  * discrepancy F_ML = tr(S Sigma^-1) - log|S Sigma^-1| - p
  * chisq = N * F_ML ; df = p(p+1)/2 - npar
  * baseline (independence) model = diagonal Sigma; its ML fit is closed-form
  * CFI/TLI/RMSEA/SRMR per lavaan's fitMeasures formulas
  * standard errors from the expected (Fisher) information matrix

Only the standardized + unstandardized loadings, chisq, df, CFI, TLI, RMSEA and
SRMR are gated to 1e-6 element-wise against R.  SEs are provided but replicated
with the same expected-information approach lavaan uses.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, linalg


# --------------------------------------------------------------------------
# model spec parsing (lavaan-style "f =~ x1 + x2 + ...")
# --------------------------------------------------------------------------
def parse_model(model: str):
    """Parse a lavaan measurement model string into factor -> [indicators].

    Only the ``=~`` (measured-by) operator is supported, which is all cfa()
    with a pure measurement model needs.  Factor covariances are added
    automatically (all factors correlated), as cfa() does by default.
    """
    factors = {}  # ordered dict {factor: [indicators]}
    for raw in model.replace(";", "\n").split("\n"):
        line = raw.split("#", 1)[0].strip()
        if not line or "=~" not in line:
            continue
        lhs, rhs = line.split("=~", 1)
        fac = lhs.strip()
        inds = [t.strip() for t in rhs.replace("+", " ").split() if t.strip()]
        factors.setdefault(fac, [])
        for it in inds:
            if it not in factors[fac]:
                factors[fac].append(it)
    return factors


# --------------------------------------------------------------------------
# CFA model container
# --------------------------------------------------------------------------
class _CFAModel:
    """Marker-variable ML CFA (covariance structure only)."""

    def __init__(self, factors, obs_names):
        self.factors = factors
        self.factor_names = list(factors.keys())
        self.obs_names = list(obs_names)
        self.p = len(self.obs_names)
        self.m = len(self.factor_names)
        self.idx = {v: i for i, v in enumerate(self.obs_names)}

        # free loading positions (row=indicator, col=factor); first indicator
        # of each factor is fixed to 1.
        self.free_load = []          # list of (row, col)
        self.fixed_load = []         # list of (row, col) fixed at 1
        for c, f in enumerate(self.factor_names):
            for k, ind in enumerate(self.factors[f]):
                r = self.idx[ind]
                if k == 0:
                    self.fixed_load.append((r, c))
                else:
                    self.free_load.append((r, c))

        # free residual (theta) variances: one per observed variable (diagonal)
        self.free_theta = list(range(self.p))

        # free factor (psi) variances + covariances: full symmetric m x m
        self.free_psi_var = list(range(self.m))
        self.free_psi_cov = [(i, j) for i in range(self.m) for j in range(i + 1, self.m)]

        # parameter layout
        self.n_load = len(self.free_load)
        self.n_theta = len(self.free_theta)
        self.n_psivar = len(self.free_psi_var)
        self.n_psicov = len(self.free_psi_cov)
        self.npar = self.n_load + self.n_theta + self.n_psivar + self.n_psicov

    # ---- pack / unpack -------------------------------------------------
    def matrices(self, theta):
        """Build Lambda, Theta, Psi from a flat parameter vector."""
        o = 0
        Lam = np.zeros((self.p, self.m))
        for (r, c) in self.fixed_load:
            Lam[r, c] = 1.0
        for (r, c) in self.free_load:
            Lam[r, c] = theta[o]; o += 1
        Th = np.zeros((self.p, self.p))
        for i in self.free_theta:
            Th[i, i] = theta[o]; o += 1
        Psi = np.zeros((self.m, self.m))
        for i in self.free_psi_var:
            Psi[i, i] = theta[o]; o += 1
        for (i, j) in self.free_psi_cov:
            Psi[i, j] = Psi[j, i] = theta[o]; o += 1
        return Lam, Th, Psi

    def sigma(self, theta):
        Lam, Th, Psi = self.matrices(theta)
        return Lam @ Psi @ Lam.T + Th

    # ---- start values (lavaan-like) -----------------------------------
    def start(self, S):
        theta = np.zeros(self.npar)
        o = 0
        # loadings: simple regression-like start = ratio of covariances; lavaan
        # uses a fabin-style start but any reasonable start converges to the ML
        # optimum, so use cov(ind, marker)/var(marker) heuristic clipped.
        for (r, c) in self.free_load:
            marker = self.fixed_load[c][0]
            denom = S[marker, marker]
            val = S[r, marker] / denom if denom > 0 else 1.0
            theta[o] = val; o += 1
        # residual variances: half the observed variance
        for i in self.free_theta:
            theta[o] = 0.5 * S[i, i]; o += 1
        # factor variances: half the marker variance
        for c, i in enumerate(self.free_psi_var):
            marker = self.fixed_load[c][0]
            theta[o] = 0.5 * S[marker, marker]; o += 1
        # factor covariances: 0
        for _ in self.free_psi_cov:
            theta[o] = 0.0; o += 1
        return theta


# --------------------------------------------------------------------------
# discrepancy function and derivatives
# --------------------------------------------------------------------------
def _fml(S, Sig, p):
    """Normal-theory ML discrepancy F = tr(S Sig^-1) - log|S Sig^-1| - p."""
    L = linalg.cho_factor(Sig, lower=True)
    Sinv_S = linalg.cho_solve(L, S)          # Sig^-1 S
    tr = np.trace(Sinv_S)
    # log|S Sig^-1| = log|S| - log|Sig|
    sign_s, logdet_s = np.linalg.slogdet(S)
    logdet_sig = 2.0 * np.sum(np.log(np.diag(L[0])))
    return tr - (logdet_s - logdet_sig) - p


def _fit_ml(model: _CFAModel, S: np.ndarray):
    """Minimise F_ML over free parameters using L-BFGS-B then polish."""
    p = model.p

    def objective(th):
        Sig = model.sigma(th)
        # guard against non-PD proposals
        try:
            return _fml(S, Sig, p)
        except linalg.LinAlgError:
            return 1e10

    x0 = model.start(S)
    # residual/factor variances must stay positive
    bounds = []
    o = 0
    for _ in model.free_load:
        bounds.append((None, None)); o += 1
    for _ in model.free_theta:
        bounds.append((1e-6, None)); o += 1
    for _ in model.free_psi_var:
        bounds.append((1e-6, None)); o += 1
    for _ in model.free_psi_cov:
        bounds.append((None, None)); o += 1

    res = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
    # polish with Nelder-Mead-free Newton-ish via BFGS unbounded from optimum
    res2 = optimize.minimize(objective, res.x, method="BFGS",
                             options=dict(maxiter=5000, gtol=1e-12))
    best = res2 if res2.fun < res.fun else res
    # Newton polish on the analytic gradient to drive ||grad F_ML|| -> ~machine
    # eps.  L-BFGS/BFGS leave a residual gradient (~1e-7) that, while harmless
    # for the fit indices, is amplified by N in the score-test modification
    # indices; a few Fisher-scoring steps recover lavaan's exact ML optimum.
    x = _newton_polish(model, S, best.x)
    return x, objective(x)


def _fml_gradient(model: _CFAModel, S, theta):
    """Analytic gradient of F_ML = tr(S Sig^-1) - log|S Sig^-1| - p.

    dF/dtheta_k = -tr( Sig^-1 (S - Sig) Sig^-1 dSig/dtheta_k ).
    """
    p = model.p
    Sig = model.sigma(theta)
    Sinv = np.linalg.inv(Sig)
    M = Sinv @ (S - Sig) @ Sinv          # symmetric
    g = np.zeros(model.npar)
    for k, dS in enumerate(_dsigma_columns(model, theta)):
        g[k] = -np.sum(M * dS)
    return g


def _newton_polish(model: _CFAModel, S, theta, iters=50, tol=1e-13):
    """Fisher-scoring refinement of the ML estimates from a good start."""
    p = model.p
    x = theta.astype(float).copy()
    for _ in range(iters):
        g = _fml_gradient(model, S, x)
        if np.linalg.norm(g) < tol:
            break
        # Fisher information for F_ML (= 2 x per-obs expected info / ... ),
        # step = H^-1 g with H the Gauss-Newton (expected) Hessian.
        H = _fml_expected_hessian(model, x)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x_new = x - step
        # keep variances positive; if a full step invalidates PD-ness, damp it
        f_ok = False
        alpha = 1.0
        for _ in range(30):
            cand = x - alpha * step
            try:
                Sig = model.sigma(cand)
                np.linalg.cholesky(Sig)
                f_ok = True
                break
            except np.linalg.LinAlgError:
                alpha *= 0.5
        if not f_ok:
            break
        x = cand
    return x


def _fml_expected_hessian(model: _CFAModel, theta):
    """Gauss-Newton (expected) Hessian of F_ML: 2 * Delta' (Sig^-1 (x) Sig^-1)
    Delta contracted per column, i.e. H_kl = 2 tr(Sig^-1 dS_k Sig^-1 dS_l)."""
    Sig = model.sigma(theta)
    Sinv = np.linalg.inv(Sig)
    cols = list(_dsigma_columns(model, theta))
    n = len(cols)
    # precompute Sinv dS_k Sinv
    A = [Sinv @ dS @ Sinv for dS in cols]
    H = np.zeros((n, n))
    for k in range(n):
        for l in range(k, n):
            v = np.sum(A[k] * cols[l])
            H[k, l] = H[l, k] = v
    return H


def _dsigma_columns(model: _CFAModel, theta):
    """Yield analytic dSigma/dtheta_k (p x p) for each free parameter, in the
    model's parameter layout order [free_load, free_theta, psi_var, psi_cov]."""
    p, mm = model.p, model.m
    Lam, Th, Psi = model.matrices(theta)
    for (r, c) in model.free_load:
        E = np.zeros((p, mm)); E[r, c] = 1.0
        yield E @ Psi @ Lam.T + Lam @ Psi @ E.T
    for i in model.free_theta:
        M = np.zeros((p, p)); M[i, i] = 1.0
        yield M
    for a in model.free_psi_var:
        E = np.zeros((mm, mm)); E[a, a] = 1.0
        yield Lam @ E @ Lam.T
    for (a, b) in model.free_psi_cov:
        E = np.zeros((mm, mm)); E[a, b] = 1.0; E[b, a] = 1.0
        yield Lam @ E @ Lam.T


# --------------------------------------------------------------------------
# expected-information standard errors
# --------------------------------------------------------------------------
def _duplication(p):
    """Duplication matrix D_p mapping vech->vec (p^2 x p*(p+1)/2)."""
    q = p * (p + 1) // 2
    D = np.zeros((p * p, q))
    col = 0
    for j in range(p):
        for i in range(j, p):
            u = np.zeros(q); u[col] = 1
            E = np.zeros((p, p)); E[i, j] = 1; E[j, i] = 1
            D[:, col] = E.flatten(order="F")
            col += 1
    return D


def _jacobian_sigma(model: _CFAModel, theta):
    """d vech(Sigma) / d theta  (numerical, central difference)."""
    p = model.p
    q = p * (p + 1) // 2
    tril = np.tril_indices(p)

    def vech_sigma(th):
        Sig = model.sigma(th)
        return Sig[tril]

    J = np.zeros((q, model.npar))
    base = vech_sigma(theta)
    for k in range(model.npar):
        h = 1e-6 * max(1.0, abs(theta[k]))
        tp = theta.copy(); tp[k] += h
        tm = theta.copy(); tm[k] -= h
        J[:, k] = (vech_sigma(tp) - vech_sigma(tm)) / (2 * h)
    return J


def _expected_se(model: _CFAModel, theta, N):
    """Standard errors via expected (Fisher) information.

    Fisher info per lavaan (ML, covariance only):
        I(theta) = N/2 * Delta^T (D^T (Sig^-1 (x) Sig^-1) D) Delta
    where Delta = d vech(Sigma)/d theta and D the duplication matrix.
    """
    p = model.p
    Sig = model.sigma(theta)
    Sinv = np.linalg.inv(Sig)
    D = _duplication(p)
    W = D.T @ np.kron(Sinv, Sinv) @ D      # (q x q)
    Delta = _jacobian_sigma_vech(model, theta)  # (q x npar) using vech order matching D
    Info = (N / 2.0) * (Delta.T @ W @ Delta)
    cov = np.linalg.pinv(Info)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return se


def _jacobian_sigma_vech(model, theta):
    """d vech(Sigma)/d theta in the SAME vech order the duplication matrix uses
    (column-major lower triangle: for j, for i>=j)."""
    p = model.p
    order = []
    for j in range(p):
        for i in range(j, p):
            order.append((i, j))
    q = len(order)

    def vech_col(th):
        Sig = model.sigma(th)
        return np.array([Sig[i, j] for (i, j) in order])

    J = np.zeros((q, model.npar))
    for k in range(model.npar):
        h = 1e-6 * max(1.0, abs(theta[k]))
        tp = theta.copy(); tp[k] += h
        tm = theta.copy(); tm[k] -= h
        J[:, k] = (vech_col(tp) - vech_col(tm)) / (2 * h)
    return J


# --------------------------------------------------------------------------
# fit measures
# --------------------------------------------------------------------------
def _baseline_fml(S):
    """Independence (diagonal) model: Sigma = diag(S).  Closed-form F_ML."""
    p = S.shape[0]
    d = np.diag(S).copy()
    Sig = np.diag(d)
    return _fml(S, Sig, p)


def _zeroin(f, lo, hi, tol=None, maxit=1000):
    """Faithful port of R's C `zeroin2` (Brent's method) used by uniroot().

    Reproduces R's root finder bit-for-bit so that the noncentrality-parameter
    inversions behind rmsea.ci.lower/upper match lavaan exactly (lavaan calls
    stats::uniroot with its default tol = .Machine$double.eps^0.25)."""
    if tol is None:
        tol = np.finfo(float).eps ** 0.25
    a, b = lo, hi
    fa, fb = f(a), f(b)
    c, fc = a, fa
    EPS = np.finfo(float).eps
    for _ in range(maxit):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * EPS * abs(b) + tol / 2.0
        new_step = (c - b) / 2.0
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if p < 0.75 * cb * q - abs(tol_act * q) / 2.0 and p < abs(prev_step * q / 2.0):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0.0) == (fc > 0.0):
            c, fc = a, fa
    return b


def _rmsea_ci(chisq, df, N, level=0.90):
    """lavaan's rmsea CI via noncentral chi-square ncp inversion (R uniroot)."""
    from scipy.stats import ncx2
    if df < 1 or not np.isfinite(chisq):
        return float("nan"), float("nan")
    upper_perc = 1.0 - (1.0 - level) / 2.0     # 0.95
    lower_perc = (1.0 - level) / 2.0           # 0.05

    def lower_lambda(lam):
        return ncx2.cdf(chisq, df, lam) - upper_perc

    def upper_lambda(lam):
        return ncx2.cdf(chisq, df, lam) - lower_perc

    if lower_lambda(0.0) < 0.0:
        lo = 0.0
    else:
        lam_l = _zeroin(lower_lambda, 0.0, chisq)
        lo = np.sqrt(lam_l / (N * df))

    N_rmsea = max(N, chisq * 4.0)
    if upper_lambda(N_rmsea) > 0.0 or upper_lambda(0.0) < 0.0:
        hi = 0.0
    else:
        lam_u = _zeroin(upper_lambda, 0.0, N_rmsea)
        hi = np.sqrt(lam_u / (N * df))
    return lo, hi


def _fit_measures(S, Sig, N, npar):
    p = S.shape[0]
    fml = _fml(S, Sig, p)
    chisq = N * fml
    df = p * (p + 1) // 2 - npar

    # baseline / independence model
    fml_b = _baseline_fml(S)
    chisq_b = N * fml_b
    df_b = p * (p - 1) // 2

    # CFI
    num = max(chisq - df, 0.0)
    den = max(chisq - df, chisq_b - df_b, 0.0)
    cfi = 1.0 - num / den if den > 0 else 1.0

    # TLI (NNFI)
    ratio_b = chisq_b / df_b
    ratio_t = chisq / df
    tli = (ratio_b - ratio_t) / (ratio_b - 1.0)

    # RMSEA (per-group N divisor; single group)
    rmsea = np.sqrt(max(chisq - df, 0.0) / (df * N)) if df > 0 else 0.0
    rmsea_lo, rmsea_hi = _rmsea_ci(chisq, df, N)
    # H0: rmsea <= 0.05  -> pvalue = P(chisq' > obs | ncp0), ncp0 = 0.05^2*df*N
    from scipy.stats import ncx2, chi2
    ncp0 = (0.05 ** 2) * df * N
    rmsea_pvalue = 1.0 - ncx2.cdf(chisq, df, ncp0) if df > 0 else float("nan")

    # chisq p-value (central)
    pvalue = 1.0 - chi2.cdf(chisq, df) if df > 0 else float("nan")

    # SRMR (Bentler correlation-based; RMS of lower-tri incl diag of
    # standardized covariance residuals)
    D = np.sqrt(np.diag(S))
    Rs = S / np.outer(D, D)
    Dm = np.sqrt(np.diag(Sig))
    Rm = Sig / np.outer(Dm, Dm)
    E = Rs - Rm
    tril = np.tril_indices(p)
    srmr = np.sqrt(np.sum(E[tril] ** 2) / (p * (p + 1) / 2))

    # log-likelihood of the fitted model (multivariate-normal, no mean struct):
    #   logl = -N p/2 log(2 pi) - N/2 log|Sig| - N/2 tr(Sig^-1 S)
    Sinv = np.linalg.inv(Sig)
    sign, logdet_sig = np.linalg.slogdet(Sig)
    logl = -N * p / 2.0 * np.log(2 * np.pi) - N / 2.0 * logdet_sig \
        - N / 2.0 * np.trace(Sinv @ S)
    # unrestricted (saturated) logl uses Sig = S
    signs, logdet_s = np.linalg.slogdet(S)
    unrestricted_logl = -N * p / 2.0 * np.log(2 * np.pi) - N / 2.0 * logdet_s \
        - N * p / 2.0
    aic = -2.0 * logl + 2.0 * npar
    bic = -2.0 * logl + npar * np.log(N)
    # sample-size-adjusted BIC (Sclove): N* = (N + 2) / 24
    n_star = (N + 2.0) / 24.0
    bic2 = -2.0 * logl + npar * np.log(n_star)

    # NFI = (chisq_b - chisq) / chisq_b
    nfi = (chisq_b - chisq) / chisq_b if chisq_b > 0 else float("nan")

    # GFI = 1 - tr((W - I)^2) / tr(W^2), W = Sig^-1 S
    W = Sinv @ S
    WI = W - np.eye(p)
    gfi = 1.0 - np.trace(WI @ WI) / np.trace(W @ W)
    # AGFI = 1 - (p(p+1) / (2 df)) (1 - GFI)
    agfi = 1.0 - (p * (p + 1) / (2.0 * df)) * (1.0 - gfi) if df > 0 else float("nan")

    return dict(chisq=chisq, df=int(df), pvalue=pvalue, cfi=cfi, tli=tli,
                rmsea=rmsea, rmsea_ci_lower=rmsea_lo, rmsea_ci_upper=rmsea_hi,
                rmsea_pvalue=rmsea_pvalue, srmr=srmr, fmin=0.5 * fml,
                baseline_chisq=chisq_b, baseline_df=int(df_b),
                logl=logl, unrestricted_logl=unrestricted_logl,
                npar=int(npar), aic=aic, bic=bic, bic2=bic2,
                nfi=nfi, gfi=gfi, agfi=agfi)


# --------------------------------------------------------------------------
# modification indices (univariate score / Lagrange-multiplier test + EPC)
# --------------------------------------------------------------------------
def _vech_order(p):
    """Column-major lower-triangle index order used by the duplication matrix."""
    return [(i, j) for j in range(p) for i in range(j, p)]


def _dsigma_vech(model, theta, order):
    """Analytic d vech(Sigma)/d theta for the model's free parameters, columns
    ordered per the model layout, rows in the given vech `order`."""
    cols = []
    for dS in _dsigma_columns(model, theta):
        cols.append(np.array([dS[i, j] for (i, j) in order]))
    return np.array(cols).T if cols else np.zeros((len(order), 0))


def _modification_indices(model, theta, S, N):
    """lavaan modindices(): univariate score-test MI + EPC for every fixed
    parameter (all cross-loadings not already in the model + all residual
    covariances), using the expected information matrix.

    Returns a list of dicts {lhs, op, rhs, mi, epc} in factor/indicator natural
    order (loadings then residual covariances), matching lavaan's row layout.
    """
    p, mm = model.p, model.m
    Sig = model.sigma(theta)
    Sinv = np.linalg.inv(Sig)
    order = _vech_order(p)
    D = _duplication(p)
    W = D.T @ np.kron(Sinv, Sinv) @ D          # q x q, per-obs vech info weight

    # model-parameter columns of d vech(Sigma)/d theta
    Delta_model = _dsigma_vech(model, theta, order)

    # candidate "extra" (currently-fixed) parameters, in lavaan's order:
    #   cross-loadings f =~ indicator for every (factor, indicator) not present,
    #   then residual covariances x_i ~~ x_j for i < j.
    extra_meta = []
    extra_cols = []
    existing = set(model.fixed_load) | set(model.free_load)
    Lam, _, Psi = model.matrices(theta)
    for c, f in enumerate(model.factor_names):
        for r in range(p):
            if (r, c) in existing:
                continue
            # dSigma / dLam[r,c] = E Psi Lam' + Lam Psi E'
            E = np.zeros((p, mm)); E[r, c] = 1.0
            dS = E @ Psi @ Lam.T + Lam @ Psi @ E.T
            extra_cols.append(np.array([dS[i, j] for (i, j) in order]))
            extra_meta.append((f, "=~", model.obs_names[r]))
    for i in range(p):
        for j in range(i + 1, p):
            M = np.zeros((p, p)); M[i, j] = 1.0; M[j, i] = 1.0
            extra_cols.append(np.array([M[a, b] for (a, b) in order]))
            extra_meta.append((model.obs_names[i], "~~", model.obs_names[j]))
    Delta_extra = np.array(extra_cols).T

    # full (per-obs) expected information over [model | extra]
    Delta_all = np.hstack([Delta_model, Delta_extra])
    Info = 0.5 * (Delta_all.T @ W @ Delta_all)
    nmod = Delta_model.shape[1]
    nex = Delta_extra.shape[1]
    midx = np.arange(nmod)
    eidx = np.arange(nmod, nmod + nex)
    I11 = Info[np.ix_(eidx, eidx)]
    I12 = Info[np.ix_(eidx, midx)]
    I21 = Info[np.ix_(midx, eidx)]
    I22 = Info[np.ix_(midx, midx)]
    I22inv = np.linalg.inv(I22)
    V = I11 - I12 @ I22inv @ I21
    Vdiag = np.diag(V).copy()
    Vdiag[Vdiag < np.finfo(float).eps ** (1.0 / 3.0)] = np.nan

    # per-obs score (gradient of logl) for the extra parameters:
    #   g_k = 1/2 * (dvech Sigma_k)' D'(Sig^-1 (x) Sig^-1) vec(S - Sigma)
    gvec = D.T @ np.kron(Sinv, Sinv) @ (S - Sig).flatten(order="F")
    score = 0.5 * (Delta_extra.T @ gvec)

    mi = N * (score * score) / Vdiag
    # epc = mi / d,  d = -N * (lavaan's sign-flipped score) = N * score  ->
    # epc = N score^2 / Vdiag / (N score) = score / Vdiag
    epc = score / Vdiag

    rows = []
    for k in range(nex):
        lhs, op, rhs = extra_meta[k]
        rows.append(dict(lhs=lhs, op=op, rhs=rhs,
                         mi=float(mi[k]), epc=float(epc[k])))
    return rows


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
class CFAResult:
    def __init__(self, model, theta, S, N):
        self.model = model
        self.theta = theta
        self.S = S
        self.N = N
        self.Sigma = model.sigma(theta)
        self._se = None

    # -- unstandardized parameter table ---------------------------------
    def parameter_estimates(self):
        """Return a list of dicts mirroring lavaan parameterEstimates() rows.

        Ordering matches lavaan: all loadings (=~) first in factor/indicator
        order, then residual variances (~~ on same var), then factor variances,
        then factor covariances.
        """
        m = self.model
        Lam, Th, Psi = m.matrices(self.theta)
        se = self.standard_errors()
        rows = []
        o = 0  # index into free-parameter SE vector, matching layout order

        # SE bookkeeping: layout is [free_load, free_theta, free_psivar, free_psicov]
        se_load = {}
        idx = 0
        for (r, c) in m.free_load:
            se_load[(r, c)] = se[idx]; idx += 1
        se_theta = {}
        for i in m.free_theta:
            se_theta[i] = se[idx]; idx += 1
        se_pvar = {}
        for k, i in enumerate(m.free_psi_var):
            se_pvar[i] = se[idx]; idx += 1
        se_pcov = {}
        for (i, j) in m.free_psi_cov:
            se_pcov[(i, j)] = se[idx]; idx += 1

        # standardized (std.all) helpers
        Dm = np.sqrt(np.diag(self.Sigma))     # model-implied obs SDs
        psi_sd = np.sqrt(np.diag(Psi))        # factor SDs

        # loadings
        for c, f in enumerate(m.factor_names):
            for k, ind in enumerate(m.factors[f]):
                r = m.idx[ind]
                est = Lam[r, c]
                s = 0.0 if (r, c) in m.fixed_load else se_load[(r, c)]
                std_lv = est * psi_sd[c]
                std_all = std_lv / Dm[r]
                rows.append(dict(lhs=f, op="=~", rhs=ind, est=est, se=s,
                                 std_lv=std_lv, std_all=std_all))
        # residual variances
        for ind in m.obs_names:
            i = m.idx[ind]
            est = Th[i, i]
            std_all = est / (Dm[i] ** 2)
            rows.append(dict(lhs=ind, op="~~", rhs=ind, est=est,
                             se=se_theta[i], std_lv=est, std_all=std_all))
        # factor variances
        for c, f in enumerate(m.factor_names):
            est = Psi[c, c]
            rows.append(dict(lhs=f, op="~~", rhs=f, est=est, se=se_pvar[c],
                             std_lv=1.0, std_all=1.0))
        # factor covariances
        for (i, j) in m.free_psi_cov:
            est = Psi[i, j]
            std = est / (psi_sd[i] * psi_sd[j])
            rows.append(dict(lhs=m.factor_names[i], op="~~",
                             rhs=m.factor_names[j], est=est,
                             se=se_pcov[(i, j)], std_lv=std, std_all=std))
        return rows

    def standard_errors(self):
        if self._se is None:
            self._se = _expected_se(self.model, self.theta, self.N)
        return self._se

    def fit_measures(self):
        return _fit_measures(self.S, self.Sigma, self.N, self.model.npar)

    def modification_indices(self, sort=True, minimum_value=0.0):
        """Univariate score-test modification indices + EPC for all fixed
        parameters, mirroring lavaan::modindices()."""
        rows = _modification_indices(self.model, self.theta, self.S, self.N)
        if minimum_value > 0.0:
            rows = [r for r in rows if not (r["mi"] < minimum_value)]
        if sort:
            rows = sorted(rows, key=lambda r: (-(r["mi"] if r["mi"] == r["mi"] else -np.inf)))
        return rows


def cfa(model: str, data, meanstructure=False):
    """Fit a confirmatory factor analysis by ML, mirroring lavaan::cfa().

    Parameters
    ----------
    model : str
        lavaan measurement-model syntax using ``=~`` lines.
    data : mapping name->1D array (or pandas.DataFrame / 2D array with
        obs_names supplied via the model order).

    Returns
    -------
    CFAResult
    """
    factors = parse_model(model)
    # observed variable order = order of first appearance in the model
    obs = []
    for f in factors:
        for ind in factors[f]:
            if ind not in obs:
                obs.append(ind)

    # assemble data matrix in obs order, listwise-complete rows
    cols = []
    for v in obs:
        col = np.asarray(data[v], dtype=float)
        cols.append(col)
    X = np.column_stack(cols)
    mask = ~np.any(np.isnan(X), axis=1)
    X = X[mask]
    N = X.shape[0]

    # ML sample covariance uses divisor N (biased), lavaan default
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / N

    m = _CFAModel(factors, obs)
    theta, fmin = _fit_ml(m, S)
    return CFAResult(m, theta, S, N)


# --------------------------------------------------------------------------
# R-style free functions taking a fitted result (mirror fitMeasures/modindices)
# --------------------------------------------------------------------------
def fit_measures(fitted_cfa):
    """Full fit-index battery for a fitted CFA, mirroring lavaan::fitMeasures().

    Returns a dict keyed like lavaan's names (dots replaced by underscores),
    e.g. ``rmsea_ci_lower``, ``bic2``, ``logl``, ``gfi``.
    """
    return fitted_cfa.fit_measures()


def modification_indices(fitted_cfa, sort=True, minimum_value=0.0):
    """Univariate modification indices + EPC, mirroring lavaan::modindices()."""
    return fitted_cfa.modification_indices(sort=sort, minimum_value=minimum_value)
