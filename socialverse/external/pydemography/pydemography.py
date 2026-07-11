"""pydemography -- pure numpy reconstruction of R `demography` life-table core
plus the standard Kitagawa rate decomposition and Oaxaca-Blinder decomposition.

The life table replicates demography:::lt (the numerical engine behind
demography::lifetable) exactly: the Chiang/Keyfitz separation factors a0, a1,
the qx = nx*mx / (1 + (nx-ax)*mx) closure, the open-ended final interval
Lx[n] = lx[n]/mx[n], and the ex = Tx/lx chain.

Kitagawa and Oaxaca-Blinder are not part of the R demography package; they are
implemented here from their published closed forms and gated against reference
values computed with the same formulas in the R driver.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Period life table (demography:::lt)
# --------------------------------------------------------------------------- #
def life_table(mx, sex="total", startage=0, agegroup=1):
    """Period life table from an age-specific mortality schedule ``mx``.

    Faithful port of demography:::lt. Returns a dict with the actuarial columns
    ax, mx, qx, lx, dx, Lx, Tx, ex, nx (each a numpy array) and scalar ``e0``.

    Parameters
    ----------
    mx : array-like
        Central death rates nMx by age interval.
    sex : {"female", "male", "total"}
        Selects the infant/child separation-factor (a0, a1) branch.
    startage : int
        Age of the first interval (0 or positive).
    agegroup : {1, 5}
        Single-year (nx=1) or five-year (nx = 1,4,5,5,...) intervals.
    """
    mx = np.asarray(mx, dtype=float).copy()
    if mx.size and np.isnan(mx[0]):
        mx[0] = 0.0
    # truncate at first NA (mirrors R's firstmiss logic)
    nan_idx = np.where(np.isnan(mx))[0]
    if nan_idx.size:
        mx = mx[: nan_idx[0]]
    nn = mx.size
    if nn < 1:
        raise ValueError("Not enough data to proceed")

    if agegroup == 1:
        nx = np.concatenate([np.ones(nn - 1), [np.inf]]) if nn > 1 else np.array([np.inf])
    elif agegroup == 5:
        if startage > 0 and startage < 5:
            raise ValueError("0 < startage < 5 not supported for 5-year age groups")
        nx = np.concatenate([[1.0, 4.0], np.full(max(nn - 3, 0), 5.0), [np.inf]])[:nn]
    else:
        raise ValueError("agegroup must be either 1 or 5")

    # a0: infant separation factor
    if startage == 0:
        m0 = mx[0]
        if sex == "female":
            a0 = 0.053 + 2.8 * m0 if m0 < 0.107 else 0.35
        elif sex == "male":
            a0 = 0.045 + 2.684 * m0 if m0 < 0.107 else 0.33
        else:
            a0 = 0.049 + 2.742 * m0 if m0 < 0.107 else 0.34
    elif startage > 0:
        a0 = 0.5
    else:
        raise ValueError("startage must be non-negative")

    # ax vector
    if agegroup == 1:
        if nn > 1:
            ax = np.concatenate([[a0], np.full(nn - 2, 0.5), [np.inf]])
        else:
            ax = np.array([np.inf])
    elif agegroup == 5 and startage == 0:
        m0 = mx[0]
        if sex == "female":
            a1 = 1.522 - 1.518 * m0 if m0 < 0.107 else 1.361
        elif sex == "male":
            a1 = 1.651 - 2.816 * m0 if m0 < 0.107 else 1.352
        else:
            a1 = 1.5865 - 2.167 * m0 if m0 < 0.107 else 1.3565
        ax = np.concatenate([[a0, a1], np.full(max(nn - 3, 0), 2.6), [np.inf]])[:nn]
    else:
        ax = np.concatenate([np.full(nn - 1, 2.6), [np.inf]])
        nx = np.full(nn, 5.0)

    # qx with open-ended closure. The final (open) interval has nx=ax=Inf;
    # its qx is forced to 1, so evaluate the closed intervals only to avoid
    # an Inf-Inf intermediate.
    qx = np.empty(nn)
    if nn > 1:
        qx[: nn - 1] = nx[: nn - 1] * mx[: nn - 1] / (
            1.0 + (nx[: nn - 1] - ax[: nn - 1]) * mx[: nn - 1]
        )
    qx[nn - 1] = 1.0

    if nn > 1:
        lx = np.concatenate([[1.0], np.cumprod(1.0 - qx[: nn - 1])])
        dx = -np.diff(np.concatenate([lx, [0.0]]))
    else:
        lx = np.array([1.0])
        dx = np.array([1.0])

    Lx = np.empty(nn)
    if nn > 1:
        Lx[: nn - 1] = (
            nx[: nn - 1] * lx[: nn - 1] - dx[: nn - 1] * (nx[: nn - 1] - ax[: nn - 1])
        )
    Lx[nn - 1] = lx[nn - 1] / mx[nn - 1]  # open interval
    Tx = np.cumsum(Lx[::-1])[::-1]
    ex = Tx / lx

    return {
        "ax": ax, "mx": mx, "qx": qx, "lx": lx, "dx": dx,
        "Lx": Lx, "Tx": Tx, "ex": ex, "nx": nx,
        "e0": float(ex[0]),
    }


def life_expectancy(mx, sex="total", startage=0, agegroup=1, age=0):
    """Life expectancy e_x at a given ``age`` (default e0)."""
    lt = life_table(mx, sex=sex, startage=startage, agegroup=agegroup)
    return float(lt["ex"][int(age) - int(startage)])


# --------------------------------------------------------------------------- #
# Kitagawa rate decomposition
# --------------------------------------------------------------------------- #
def kitagawa(c1, r1, c2, r2):
    """Kitagawa (1955) decomposition of the crude-rate difference R2 - R1.

    ``c1, c2`` are group compositional shares (each summing to 1); ``r1, r2``
    are the corresponding group-specific rates.

    Returns dict with R1, R2, total, rate_effect, composition_effect where
        rate_effect = sum (r2 - r1) * (c1 + c2)/2
        composition_effect = sum (c2 - c1) * (r1 + r2)/2
        total = rate_effect + composition_effect = R2 - R1.
    """
    c1 = np.asarray(c1, float); r1 = np.asarray(r1, float)
    c2 = np.asarray(c2, float); r2 = np.asarray(r2, float)
    R1 = float(np.sum(c1 * r1))
    R2 = float(np.sum(c2 * r2))
    rate_effect = float(np.sum((r2 - r1) * (c1 + c2) / 2.0))
    comp_effect = float(np.sum((c2 - c1) * (r1 + r2) / 2.0))
    return {
        "R1": R1, "R2": R2, "total": R2 - R1,
        "rate_effect": rate_effect, "composition_effect": comp_effect,
    }


# --------------------------------------------------------------------------- #
# Oaxaca-Blinder decomposition
# --------------------------------------------------------------------------- #
def _ols(X, y):
    """OLS coefficients via least squares (matches R lm)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def oaxaca(yA, xA, yB, xB):
    """Twofold Oaxaca-Blinder decomposition of the mean-outcome gap A - B.

    Uses group B's coefficients as the reference (non-discriminatory) structure,
    matching the R driver:
        gap = (Xbar_A - Xbar_B) . beta_B    [explained / endowments]
            +  Xbar_A . (beta_A - beta_B)    [unexplained / coefficients]

    ``xA, xB`` are covariate matrices WITHOUT the intercept column (it is added
    internally). Returns dict with betaA, betaB, meanYA, meanYB, gap, explained,
    unexplained.
    """
    yA = np.asarray(yA, float); yB = np.asarray(yB, float)
    xA = np.atleast_2d(np.asarray(xA, float))
    xB = np.atleast_2d(np.asarray(xB, float))
    if xA.shape[0] != yA.shape[0]:
        xA = xA.T
    if xB.shape[0] != yB.shape[0]:
        xB = xB.T

    XA = np.column_stack([np.ones(xA.shape[0]), xA])
    XB = np.column_stack([np.ones(xB.shape[0]), xB])
    bA = _ols(XA, yA)
    bB = _ols(XB, yB)
    XbarA = XA.mean(axis=0)
    XbarB = XB.mean(axis=0)

    meanYA = float(yA.mean()); meanYB = float(yB.mean())
    explained = float(np.sum((XbarA - XbarB) * bB))
    unexplained = float(np.sum(XbarA * (bA - bB)))
    return {
        "betaA": bA, "betaB": bB,
        "meanYA": meanYA, "meanYB": meanYB,
        "gap": meanYA - meanYB,
        "explained": explained, "unexplained": unexplained,
    }
