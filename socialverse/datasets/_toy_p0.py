"""Toy datasets for the P0 regression / IV / matching / mediation methods.

Every dataset embeds a *known* data-generating process so the P0 functions can be
tested by parameter recovery (see tests/test_p0_methods.py). Pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_regression(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Cross-section for glm / mlogit / ologit / margins.

    Truths — OLS ``y``: const 1.0, x1 0.5, x2 -0.4 · logit ``y_bin``: x1 0.8,
    x2 -0.5 · poisson ``y_count``: x1 0.4 · ordered ``y_ord`` monotone in x1.
    ``choice`` (A/B/C) is a nominal outcome whose B/C utilities rise/fall in x1.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + 0.5 * x1 - 0.4 * x2 + rng.normal(0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x1 - 0.5 * x2)))
    y_bin = rng.binomial(1, p)
    mu = np.exp(0.2 + 0.4 * x1 - 0.1 * x2)
    y_count = rng.poisson(mu)
    latent = 0.7 * x1 - 0.5 * x2 + rng.logistic(0, 1, n)
    y_ord = np.digitize(latent, [-0.4, 0.9])  # 0 / 1 / 2
    # nominal choice among A/B/C via Gumbel-max (multinomial logit DGP)
    u = np.stack([
        rng.gumbel(0, 1, n),                 # A (base)
        0.9 * x1 + rng.gumbel(0, 1, n),      # B rises in x1
        -0.7 * x1 + rng.gumbel(0, 1, n),     # C falls in x1
    ], axis=1)
    choice = np.array(["A", "B", "C"])[u.argmax(axis=1)]
    return pd.DataFrame({
        "x1": x1, "x2": x2, "y": y, "y_bin": y_bin,
        "y_count": y_count, "y_ord": y_ord, "choice": choice,
    })


def load_iv(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Instrumental-variables cross-section.

    Endogenous ``x`` is confounded by unobserved ``u`` (also drives ``y``), so OLS
    is biased *up*; instrument ``z`` is excluded from the outcome. Truth: causal
    effect of ``x`` on ``y`` = 1.5 (2SLS recovers it, OLS overstates).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)          # instrument
    w = rng.normal(0, 1, n)          # exogenous control
    u = rng.normal(0, 1, n)          # unobserved confounder
    x = 0.7 * z + 0.6 * u + 0.3 * w + rng.normal(0, 0.5, n)   # endogenous
    y = 1.0 + 1.5 * x + 0.4 * w + 2.0 * u + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "w": w})


def load_treatment(n: int = 700, seed: int = 0) -> pd.DataFrame:
    """Observational treatment data for propensity-score matching / IPW.

    Treatment assignment depends on covariates x1..x3 (selection on observables);
    the same covariates drive ``y``, so a naive treated-minus-control difference is
    biased. Truth: ATT = 2.0 (PSM / IPW on x1..x3 recovers it).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.binomial(1, 0.5, n).astype(float)
    ps = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.5 * x2 + 0.4 * x3)))
    treat = rng.binomial(1, ps).astype(float)
    y = 1.0 + 1.2 * x1 + 0.7 * x2 - 0.5 * x3 + 2.0 * treat + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2, "x3": x3})


def load_mediation(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Single-mediator data: x -> m -> y plus a direct x -> y path.

    Truths: a (x->m) = 0.6, b (m->y) = 0.7, direct (x->y | m) = 0.3.
    ACME (indirect) = a*b = 0.42 · ADE (direct) = 0.30 · total = 0.72.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = 0.5 + 0.6 * x + rng.normal(0, 1.0, n)
    y = 1.0 + 0.3 * x + 0.7 * m + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "x": x, "m": m})


def load_ratings(n: int = 120, raters: int = 3, categories: int = 4,
                 agree: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Inter-rater / inter-coder data: ``n`` subjects each coded by ``raters``.

    Each subject has a hidden true category; every rater reports the truth with
    probability ``agree``, else a uniformly random other category. With
    ``agree=0.8`` over 4 categories the raters are in *substantial* agreement, so
    Cohen's/Fleiss' κ and Krippendorff's α should land well above chance (≈0.6–0.8).
    Columns: ``rater_1 … rater_k`` (categorical codes).
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, categories, n)
    cols = {}
    for r in range(1, raters + 1):
        keep = rng.random(n) < agree
        noise = rng.integers(0, categories, n)
        cols[f"rater_{r}"] = np.where(keep, truth, noise)
    return pd.DataFrame(cols)
