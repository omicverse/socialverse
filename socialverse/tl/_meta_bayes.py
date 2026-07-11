"""``sv.tl._meta_bayes`` — semi-analytic Bayesian meta-analysis (Tier-3).

**No MCMC.** With a flat (or normal) prior on the mean and a prior on the
heterogeneity τ, the conditional posterior of the mean given τ is normal
(conjugate); the marginal posterior integrates that over a 1-D τ grid by
quadrature. Fully faithful to the ``bayesmeta`` R package. Also a Bayesian
meta-regression (same trick, multivariate conditional posterior of β).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._registry import register
from .._state import StudyState
from ._meta import _effects, _design


def _tau_grid(v, scale, n=500):
    hi = max(10 * np.sqrt(np.median(v)), 5 * scale, 1.0)
    return np.linspace(1e-5, hi, n)


def _tau_prior_logpdf(tau, kind, scale):
    if kind == "uniform":
        return np.zeros_like(tau)
    if kind == "half-cauchy":
        return np.log(2 / (np.pi * scale * (1 + (tau / scale) ** 2)))
    # half-normal (default)
    return -0.5 * (tau / scale) ** 2


@register(
    name="bayesmeta", aliases=["贝叶斯meta", "bayesian_meta"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="半解析贝叶斯随机效应 meta(无 MCMC):flat μ 先验 + τ 先验(half-normal/half-cauchy/uniform),μ 条件共轭 + τ 一维求积;产出 μ/τ 后验 + 可信区间 + 预测区间",
    requires={"models": ["meta_effects"]}, produces={"models": ["bayesmeta"]},
)
def bayesmeta(state: StudyState, **kwargs: Any) -> StudyState:
    """Semi-analytic Bayesian random-effects meta-analysis (no MCMC).

    kwargs: ``tau_prior='half-normal'``|``'half-cauchy'``|``'uniform'``,
    ``tau_scale=`` (default a data-scaled value). Reports posterior mean/median +
    95% credible interval for μ and τ, and a posterior predictive interval."""
    eff = _effects(state)
    if eff is None:
        state.write("models", "bayesmeta", {"note": "no meta_effects"})
        return state
    from scipy import stats
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    scale = float(kwargs.get("tau_scale", np.sqrt(np.median(v)) + np.std(y)))
    prior = str(kwargs.get("tau_prior", "half-normal"))
    taus = _tau_grid(v, scale)
    t2 = taus ** 2
    muhat = np.empty_like(taus); muvar = np.empty_like(taus); loglik = np.empty_like(taus)
    for i, tt in enumerate(t2):
        w = 1.0 / (v + tt); Sw = np.sum(w); mh = np.sum(w * y) / Sw
        muhat[i] = mh; muvar[i] = 1.0 / Sw
        loglik[i] = -0.5 * np.sum(np.log(v + tt)) - 0.5 * np.log(Sw) - 0.5 * np.sum(w * (y - mh) ** 2)
    logpost = loglik + _tau_prior_logpdf(taus, prior, scale)
    logpost -= logpost.max()
    post = np.exp(logpost); post /= np.trapz(post, taus)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(taus))])
    def q(p, grid): return float(np.interp(p, cdf, grid))
    tau_mean = float(np.trapz(taus * post, taus))
    mu_mean = float(np.trapz(muhat * post, taus))
    mu_var = float(np.trapz((muvar + muhat ** 2) * post, taus) - mu_mean ** 2)
    mu_grid = np.linspace(mu_mean - 6 * np.sqrt(mu_var), mu_mean + 6 * np.sqrt(mu_var), 600)
    mix_cdf = np.array([np.trapz(stats.norm.cdf(mg, muhat, np.sqrt(muvar)) * post, taus) for mg in mu_grid])
    def mq(p): return float(np.interp(p, mix_cdf, mu_grid))
    # posterior predictive for a new study: N(mu, tau²) integrated
    pred_var = float(np.trapz((muvar + t2 + muhat ** 2) * post, taus) - mu_mean ** 2)
    state.write("models", "bayesmeta", {
        "mu_mean": mu_mean, "mu_median": mq(0.5), "mu_ci": [mq(0.025), mq(0.975)], "mu_sd": float(np.sqrt(mu_var)),
        "tau_mean": tau_mean, "tau_median": q(0.5, taus), "tau_ci": [q(0.025, taus), q(0.975, taus)],
        "prediction_interval": [mu_mean - 1.96 * np.sqrt(pred_var), mu_mean + 1.96 * np.sqrt(pred_var)],
        "prob_positive": float(np.trapz((1 - stats.norm.cdf(0, muhat, np.sqrt(muvar))) * post, taus)),
        "tau_prior": prior, "k": len(y),
    })
    return state


@register(
    name="bayes_metareg", aliases=["贝叶斯元回归", "bayesian_metareg"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy", "scipy"],
    description="半解析贝叶斯元回归(无 MCMC):flat β 先验条件下 β|τ 多元正态共轭 + τ 一维求积;各系数后验均值/可信区间",
    requires={"models": ["meta_effects"]}, produces={"models": ["bayes_metareg"]},
)
def bayes_metareg(state: StudyState, **kwargs: Any) -> StudyState:
    """Semi-analytic Bayesian meta-regression (no MCMC). Moderators via ``moderators=[...]``."""
    eff = _effects(state)
    if eff is None:
        state.write("models", "bayes_metareg", {"note": "no meta_effects"})
        return state
    y = eff["yi"].to_numpy(float); v = eff["vi"].to_numpy(float)
    X, Xcols, mods = _design(state, eff, kwargs)
    scale = float(kwargs.get("tau_scale", np.sqrt(np.median(v)) + np.std(y)))
    prior = str(kwargs.get("tau_prior", "half-normal"))
    taus = _tau_grid(v, scale); t2 = taus ** 2; p = X.shape[1]
    betas = np.empty((len(taus), p)); covs = np.empty((len(taus), p, p)); loglik = np.empty(len(taus))
    for i, tt in enumerate(t2):
        w = 1.0 / (v + tt)
        A = np.linalg.pinv(X.T @ (w[:, None] * X))
        b = A @ (X.T @ (w * y))
        betas[i] = b; covs[i] = A
        resid = y - X @ b
        sign, logdetA = np.linalg.slogdet(X.T @ (w[:, None] * X))
        loglik[i] = -0.5 * np.sum(np.log(v + tt)) - 0.5 * logdetA - 0.5 * np.sum(w * resid ** 2)
    logpost = loglik + _tau_prior_logpdf(taus, prior, scale)
    logpost -= logpost.max(); postw = np.exp(logpost); postw /= np.trapz(postw, taus)
    from scipy import stats
    coefs = {}
    for j, name in enumerate(Xcols):
        mj = betas[:, j]; vj = covs[:, j, j]
        mean = float(np.trapz(mj * postw, taus))
        var = float(np.trapz((vj + mj ** 2) * postw, taus) - mean ** 2)
        sd = np.sqrt(var)
        coefs[name] = {"mean": mean, "sd": float(sd), "ci_lb": mean - 1.96 * sd, "ci_ub": mean + 1.96 * sd,
                       "prob_positive": float(np.trapz((1 - stats.norm.cdf(0, mj, np.sqrt(vj))) * postw, taus))}
    state.write("models", "bayes_metareg", {"coefs": coefs, "terms": Xcols, "moderators": mods,
                                            "tau_mean": float(np.trapz(taus * postw, taus)), "k": len(y)})
    return state
