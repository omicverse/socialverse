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

__all__ = ["km", "coxph", "KMResult", "CoxResult"]


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
