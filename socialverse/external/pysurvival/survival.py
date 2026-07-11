"""Pure-Python reconstruction of R **survival** (Therneau): Kaplan-Meier +
Cox proportional-hazards.

Reference-driven port under the Omicverse-RebuildR protocol; parity-gated against
survival 3.8.3 (``tests/test_parity.py``) on the canonical ``lung`` dataset at
1e-6. Cox uses the Newton-Raphson partial-likelihood with both **Efron** (R
default) and **Breslow** tie handling.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg, stats

__all__ = ["km", "coxph", "clogit", "survreg",
           "KMResult", "CoxResult", "ClogitResult", "SurvregResult"]


@dataclass
class KMResult:
    time: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    surv: np.ndarray
    std_err: np.ndarray        # SE of the cumulative hazard (R survfit convention)
    lower: np.ndarray
    upper: np.ndarray
    median: float


@dataclass
class CoxResult:
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    pval: np.ndarray
    vcov: np.ndarray
    loglik: tuple            # (null, fitted)
    concordance: float
    n: int
    n_event: int
    ties: str
    iter: int


def km(time, event, conf_level=0.95):
    """Kaplan-Meier estimator (matches ``survfit(Surv(time,event)~1)``).

    Rows at every unique observation time; ``std_err`` is the SE of the
    cumulative hazard (= Greenwood on the log scale), as R's ``survfit`` reports.
    """
    time = np.asarray(time, float)
    event = np.asarray(event).astype(int)
    ut = np.unique(time)
    n_risk = np.array([np.sum(time >= t) for t in ut], float)
    n_event = np.array([np.sum((time == t) & (event == 1)) for t in ut], float)
    # survival step + Greenwood cumulative-hazard variance
    frac = np.where(n_risk > 0, 1.0 - n_event / n_risk, 1.0)
    surv = np.cumprod(frac)
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = np.where(n_risk * (n_risk - n_event) > 0,
                       n_event / (n_risk * (n_risk - n_event)), 0.0)
    std_err = np.sqrt(np.cumsum(inc))
    z = stats.norm.ppf(1 - (1 - conf_level) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.clip(np.exp(np.log(surv) - z * std_err), 0, 1)
        upper = np.clip(np.exp(np.log(surv) + z * std_err), 0, 1)
    below = np.where(surv <= 0.5)[0]
    median = float(ut[below[0]]) if below.size else float("nan")
    return KMResult(ut, n_risk.astype(int), n_event.astype(int), surv, std_err,
                    lower, upper, median)


def _cox_nll(beta, time, event, X, ties):
    """Partial log-likelihood, score, and observed information at ``beta``."""
    eta = X @ beta
    w = np.exp(eta)
    n, p = X.shape
    ll = 0.0
    U = np.zeros(p)
    I = np.zeros((p, p))
    ev_times = np.unique(time[event == 1])
    for t in ev_times:
        risk = time >= t
        ev = (time == t) & (event == 1)
        d = int(ev.sum())
        wR, xR = w[risk], X[risk]
        sR = wR.sum()
        sRx = wR @ xR
        sRxx = (wR[:, None, None] * xR[:, :, None] * xR[:, None, :]).sum(0)
        xDsum = X[ev].sum(0)
        ll += eta[ev].sum()
        U += xDsum
        if ties == "breslow":
            m = sRx / sR
            ll -= d * np.log(sR)
            U -= d * m
            I += d * (sRxx / sR - np.outer(m, m))
        else:  # efron
            wD, xD = w[ev], X[ev]
            sD = wD.sum(); sDx = wD @ xD
            sDxx = (wD[:, None, None] * xD[:, :, None] * xD[:, None, :]).sum(0)
            for l in range(d):
                f = l / d
                dn = sR - f * sD
                nu = sRx - f * sDx
                nuu = sRxx - f * sDxx
                m = nu / dn
                ll -= np.log(dn)
                U -= m
                I += nuu / dn - np.outer(m, m)
    return ll, U, I


def _concordance(time, event, risk):
    """Harrell's C (matches ``coxph``'s concordance for uncensored-first pairs)."""
    time = np.asarray(time, float); event = np.asarray(event).astype(int)
    risk = np.asarray(risk, float)
    conc = disc = tie = 0.0
    n = time.size
    for i in range(n):
        if event[i] != 1:
            continue
        # j comparable if it outlives i's event time (or ties in time but censored)
        comp = (time > time[i]) | ((time == time[i]) & (event == 0))
        comp[i] = False
        ri, rj = risk[i], risk[comp]
        conc += np.sum(ri > rj)   # higher risk fails first = concordant
        disc += np.sum(ri < rj)
        tie += np.sum(ri == rj)
    tot = conc + disc + tie
    return float((conc + 0.5 * tie) / tot) if tot else float("nan")


def coxph(time, event, X, ties="efron", maxiter=30, eps=1e-9):
    """Cox proportional-hazards model — ``coxph`` parity (Newton-Raphson)."""
    time = np.asarray(time, float)
    event = np.asarray(event).astype(int)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    beta = np.zeros(p)
    ll0, _, _ = _cox_nll(beta, time, event, X, ties)   # null log-likelihood
    ll_old = ll0
    it = 0
    for it in range(1, maxiter + 1):
        ll, U, I = _cox_nll(beta, time, event, X, ties)
        if it > 1 and abs(ll - ll_old) < eps * (abs(ll) + eps):
            break                                        # converged (ll at current β)
        ll_old = ll
        beta = beta + linalg.solve(I, U, assume_a="sym")
    ll_fit, U, I = _cox_nll(beta, time, event, X, ties)
    vcov = linalg.inv(I)
    se = np.sqrt(np.diag(vcov))
    z = beta / se
    pval = 2 * stats.norm.sf(np.abs(z))
    C = _concordance(time, event, X @ beta)
    return CoxResult(coef=beta, se=se, z=z, pval=pval, vcov=vcov,
                     loglik=(float(ll0), float(ll_fit)), concordance=C,
                     n=n, n_event=int((event == 1).sum()), ties=ties, iter=it)


@dataclass
class ClogitResult:
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    pval: np.ndarray
    vcov: np.ndarray
    loglik: tuple            # (null, fitted) conditional log-likelihood
    n: int
    n_event: int
    iter: int


def _clogit_ll(beta, groups, ties="exact"):
    """Conditional partial log-likelihood, score and information for clogit.

    ``groups`` is a list of ``(Xg, casemask)`` per stratum, where ``Xg`` is the
    design matrix for that stratum and ``casemask`` marks the cases (y==1). This
    is the stratified Cox partial likelihood with every stratum a single risk
    set (cases = simultaneous "events"). ``exact`` sums the conditional
    likelihood over all size-``d`` subsets of the risk set (matches
    ``survival::clogit`` default, method="exact").
    """
    p = groups[0][0].shape[1]
    ll = 0.0
    U = np.zeros(p)
    I = np.zeros((p, p))
    for Xg, cm in groups:
        d = int(cm.sum())
        m = Xg.shape[0]
        if d == 0 or d == m:
            continue                      # non-informative stratum
        eta = Xg @ beta
        # numerator: cases contribute directly
        ll += eta[cm].sum()
        U += Xg[cm].sum(0)
        # denominator: sum over all size-d subsets S of exp(sum eta_S)
        # compute e_k = elementary-symmetric-style weighted moments via DP.
        # We accumulate, over subsets of size d, the total weight W, the
        # weighted sum of (sum_{i in S} x_i) and of its outer product.
        w = np.exp(eta)
        # DP over items; state indexed by chosen-count k (0..d)
        # W[k]      = sum over size-k subsets of prod w
        # G[k]      = sum over size-k subsets of prod w * (sum x)
        # H[k]      = sum over size-k subsets of prod w * (sum x)(sum x)^T
        W = np.zeros(d + 1)
        G = np.zeros((d + 1, p))
        H = np.zeros((d + 1, p, p))
        W[0] = 1.0
        for i in range(m):
            wi = w[i]; xi = Xg[i]
            xio = np.outer(xi, xi)
            kmax = min(i + 1, d)
            for k in range(kmax, 0, -1):
                # add item i to size-(k-1) subsets -> size-k subsets
                Wp = W[k - 1]; Gp = G[k - 1]; Hp = H[k - 1]
                H[k] += wi * (Hp + np.outer(xi, Gp) + np.outer(Gp, xi) + Wp * xio)
                G[k] += wi * (Gp + Wp * xi)
                W[k] += wi * Wp
        Wd, Gd, Hd = W[d], G[d], H[d]
        ll -= np.log(Wd)
        mu = Gd / Wd
        U -= mu
        I += Hd / Wd - np.outer(mu, mu)
    return ll, U, I


def clogit(y, strata, X, maxiter=30, eps=1e-9):
    """Conditional (fixed-effects) logistic regression — ``clogit`` parity.

    Fits the exact conditional partial likelihood for stratum-matched data by
    Newton-Raphson (equivalent to a stratified Cox model with method="exact").
    ``y`` is the 0/1 case indicator, ``strata`` the matched-set id per row.
    """
    y = np.asarray(y).astype(int)
    strata = np.asarray(strata)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    groups = []
    for s in np.unique(strata):
        idx = strata == s
        cm = (y[idx] == 1)
        groups.append((X[idx], cm))
    beta = np.zeros(p)
    ll0, _, _ = _clogit_ll(beta, groups)
    ll_old = ll0
    it = 0
    for it in range(1, maxiter + 1):
        ll, U, I = _clogit_ll(beta, groups)
        if it > 1 and abs(ll - ll_old) < eps * (abs(ll) + eps):
            break
        ll_old = ll
        beta = beta + linalg.solve(I, U, assume_a="sym")
    ll_fit, U, I = _clogit_ll(beta, groups)
    vcov = linalg.inv(I)
    se = np.sqrt(np.diag(vcov))
    z = beta / se
    pval = 2 * stats.norm.sf(np.abs(z))
    return ClogitResult(coef=beta, se=se, z=z, pval=pval, vcov=vcov,
                        loglik=(float(ll0), float(ll_fit)),
                        n=n, n_event=int((y == 1).sum()), iter=it)


@dataclass
class SurvregResult:
    coef: np.ndarray          # regression coefficients (incl. intercept)
    scale: float              # scale parameter (fixed at 1 for exponential)
    se: np.ndarray            # SE of [coef..., log(scale)] (R vcov ordering)
    loglik: float             # maximised log-likelihood
    dist: str
    iter: int
    vcov: np.ndarray


def _survreg_negll(params, time, status, X, dist, fix_logscale=None):
    """Negative log-likelihood, gradient and Hessian for a parametric AFT model.

    Parameterisation matches ``survreg``: linear predictor ``eta = X @ b`` on the
    log-time scale, with standardized residual ``z = (log t - eta)/sigma`` and
    ``sigma = exp(log_scale)``. Optimised parameter vector is
    ``[b..., log_scale]`` (log_scale dropped when ``fix_logscale`` is given, e.g.
    exponential with sigma==1). Weibull/exponential use the extreme-value error;
    lognormal uses the Gaussian error.
    """
    X = np.asarray(X, float)
    p = X.shape[1]
    b = params[:p]
    if fix_logscale is None:
        ls = params[p]
    else:
        ls = fix_logscale
    sigma = np.exp(ls)
    status = np.asarray(status, float)
    logt = np.log(time)
    eta = X @ b
    z = (logt - eta) / sigma

    if dist in ("weibull", "exponential"):
        # extreme-value: event density g(z)=exp(z-e^z); survivor S(z)=exp(-e^z)
        ez = np.exp(z)
        logf = z - ez - np.log(sigma)      # log density on time scale
        logS = -ez
        # derivatives wrt z
        # event: dl/dz = 1 - ez ; d2l/dz2 = -ez
        # cens : dl/dz = -ez    ; d2l/dz2 = -ez
        dz_ev = 1.0 - ez
        dz_ce = -ez
        d2z_ev = -ez
        d2z_ce = -ez
    elif dist == "lognormal":
        # gaussian error on log-time
        logphi = -0.5 * z * z - 0.5 * np.log(2 * np.pi)
        logf = logphi - np.log(sigma)
        Phi = stats.norm.cdf(-z)           # survivor S(z)=1-Phi(z)=Phi(-z)
        logS = np.log(Phi)
        phi = np.exp(logphi)
        h = phi / Phi                      # inverse Mills ratio phi(z)/Phi(-z)
        dz_ev = -z
        dz_ce = -h                         # d/dz log S(z) = -phi(z)/Phi(-z)
        d2z_ev = -np.ones_like(z)
        d2z_ce = -h * (h - z)              # d/dz of (-h)
    else:
        raise ValueError(f"unsupported dist {dist!r}")

    ev = status == 1
    ll = np.where(ev, logf, logS).sum()

    # gradient / Hessian via chain rule.  dz/db = -X/sigma ; dz/dls = -z
    dl_dz = np.where(ev, dz_ev, dz_ce)
    d2l_dz2 = np.where(ev, d2z_ev, d2z_ce)
    # extra explicit sigma term in the event density: -log(sigma)
    # d(-log sigma)/dls = -1 (only for events)
    n_params = p + (0 if fix_logscale is not None else 1)
    g = np.zeros(n_params)
    Hess = np.zeros((n_params, n_params))

    dz_db = -X / sigma                     # (n,p)
    # d z / d ls = -z
    dz_dls = -z

    # first derivatives
    g[:p] = (dl_dz[:, None] * dz_db).sum(0)
    if fix_logscale is None:
        g[p] = (dl_dz * dz_dls).sum() - float(ev.sum())  # event term d(-log sigma)/dls

    # second derivatives.  d2z/db2 = 0 ; d2z/(db dls) = X/sigma = -dz_db ;
    # d2z/dls2 = -dz_dls = z
    # Hess_bb = sum d2l_dz2 * dz_db dz_db^T
    Hess[:p, :p] = (d2l_dz2[:, None, None] * dz_db[:, :, None] * dz_db[:, None, :]).sum(0)
    if fix_logscale is None:
        d2z_db_dls = -dz_db                # (n,p)
        Hbs = (d2l_dz2 * dz_db.T * dz_dls).sum(1) + (dl_dz * d2z_db_dls.T).sum(1)
        Hess[:p, p] = Hbs
        Hess[p, :p] = Hbs
        d2z_dls2 = -dz_dls                 # = z
        Hess[p, p] = (d2l_dz2 * dz_dls * dz_dls).sum() + (dl_dz * d2z_dls2).sum()

    return -ll, -g, -Hess


def survreg(time, status, X, dist="weibull", maxiter=50, eps=1e-10):
    """Parametric AFT model MLE — ``survreg`` parity (Weibull/exponential/lognormal).

    ``X`` should already include an intercept column if desired (matches the R
    design matrix from ``~ age + sex``, i.e. leading column of ones). Optimises
    ``[coef..., log(scale)]`` by Newton-Raphson on the exact log-likelihood; for
    ``dist='exponential'`` the scale is fixed at 1. Returns coefficients, scale
    and the SE vector in R's ``vcov`` ordering ``[coef..., Log(scale)]``.
    """
    time = np.asarray(time, float)
    status = np.asarray(status).astype(int)
    # R's Surv() convention: {1,2} coding means 1=censored, 2=event; otherwise
    # 1=event, 0=censored.
    if set(np.unique(status)) <= {1, 2}:
        status = (status == 2).astype(int)
    else:
        status = (status == 1).astype(int)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    exponential = dist == "exponential"
    fix_ls = 0.0 if exponential else None

    # initialise: OLS of log-time on X for coef, residual sd -> log scale
    logt = np.log(time)
    b0, *_ = np.linalg.lstsq(X, logt, rcond=None)
    resid = logt - X @ b0
    ls0 = np.log(resid.std() + 1e-6) if not exponential else 0.0
    params = np.concatenate([b0, [] if exponential else [ls0]])

    it = 0
    for it in range(1, maxiter + 1):
        nll, g, H = _survreg_negll(params, time, status, X, dist, fix_ls)
        step = linalg.solve(H, g, assume_a="sym")
        # damped Newton to stay in a sane region
        lam = 1.0
        while lam > 1e-6:
            trial = params - lam * step
            ntrial = _survreg_negll(trial, time, status, X, dist, fix_ls)[0]
            if np.isfinite(ntrial) and ntrial <= nll + 1e-12:
                break
            lam *= 0.5
        params = params - lam * step
        if np.max(np.abs(lam * step)) < eps:
            break

    nll, g, H = _survreg_negll(params, time, status, X, dist, fix_ls)
    vcov = linalg.inv(H)
    se = np.sqrt(np.diag(vcov))
    b = params[:p]
    scale = 1.0 if exponential else float(np.exp(params[p]))
    # ``_survreg_negll`` works on the log-time scale; R's survreg reports the
    # log-likelihood on the original time scale, which adds the Jacobian term
    # -log(t) for every event (a parameter-free constant, so it leaves the MLE,
    # gradient and Hessian untouched).
    jac = -np.log(time[status == 1]).sum()
    return SurvregResult(coef=b, scale=scale, se=se, loglik=float(-nll + jac),
                         dist=dist, iter=it, vcov=vcov)
