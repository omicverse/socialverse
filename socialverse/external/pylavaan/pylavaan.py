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
    return best.x, best.fun


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

    # SRMR (Bentler correlation-based; RMS of lower-tri incl diag of
    # standardized covariance residuals)
    D = np.sqrt(np.diag(S))
    Rs = S / np.outer(D, D)
    Dm = np.sqrt(np.diag(Sig))
    Rm = Sig / np.outer(Dm, Dm)
    E = Rs - Rm
    tril = np.tril_indices(p)
    srmr = np.sqrt(np.sum(E[tril] ** 2) / (p * (p + 1) / 2))

    return dict(chisq=chisq, df=int(df), cfi=cfi, tli=tli, rmsea=rmsea,
                srmr=srmr, fmin=0.5 * fml, baseline_chisq=chisq_b,
                baseline_df=int(df_b))


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
