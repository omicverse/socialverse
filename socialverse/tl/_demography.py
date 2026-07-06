"""``sv.tl._demography`` — registered implementations for the *demography* gap.

Two registry entries fill the formal-demography hole in ``socialverse``:

- :func:`life_table` (人口学 缺口) — a **period life table**: convert age-specific
  mortality rates ``mx`` into ``qx → lx → ndx → nLx → Tx → ex``, yielding life
  expectancy ``e(x)`` at every age (including ``e0``, expectation of life at
  birth). This is the demographer's canonical column-by-column construction.
- :func:`decomposition` (分解 / Kitagawa / Oaxaca) — decompose the *difference in
  crude death rates* between two populations into a **rate effect** (differences
  in age-specific mortality) and a **composition effect** (differences in age
  structure), the classic Kitagawa (1955) additive split, with an optional
  Oaxaca–Blinder regression decomposition as a companion.

Champion references these mirror: the life-table column algebra corresponds to
R's ``demography::lifetable`` / ``MortalityLaws`` and Python's ``pyliftover``-era
demographic toolkits; the Kitagawa/Oaxaca split corresponds to R ``DemoDecomp``
and Python ``statsmodels`` OLS for the regression variant. Everything here is
plain NumPy / pandas (with ``statsmodels`` used only for the optional OLS
Oaxaca), so the functions always return a real, deterministic result with no
optional dependency required.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["life_table", "decomposition"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_datasets(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working frame: explicit ``data=`` kwarg, else ``sources['datasets']``.

    ``sources['datasets']`` may itself be a DataFrame or a ``{name: DataFrame}``
    mapping; in the latter case the first frame is taken.
    """
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _numeric(series: pd.Series) -> np.ndarray:
    """Coerce a column to a 1-D float array (non-numeric → NaN)."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _default_widths(n: int) -> np.ndarray:
    """Fallback interval widths when none supplied: unit-width intervals."""
    return np.ones(n, dtype=float)


def _build_life_table(
    mx: np.ndarray,
    width: np.ndarray,
    *,
    radix: float = 100_000.0,
) -> dict[str, np.ndarray]:
    """Construct the standard period life-table columns from ``mx`` and ``n``.

    Columns (Preston, Heuveline & Guillot, *Demography*, ch. 3):

    * ``ax``  — average person-years lived in ``[x, x+n)`` by those who die there.
      Uses ``a0 ≈ 0.1`` for the (special) infant interval and ``ax = n/2`` for the
      others; the open-ended final interval uses ``a = 1/m`` (all remaining years).
    * ``qx``  — probability of dying in the interval,
      ``qx = n·mx / (1 + (n − ax)·mx)``; the open interval has ``qx = 1``.
    * ``px``  — survival probability ``1 − qx``.
    * ``lx``  — survivors at exact age ``x`` (radix at age 0).
    * ``ndx`` — deaths in the interval, ``lx − l(x+n)``.
    * ``nLx`` — person-years lived in the interval,
      ``n·l(x+n) + ax·ndx`` (open interval: ``lx / mx``).
    * ``Tx``  — person-years lived above age ``x`` (reverse cumsum of ``nLx``).
    * ``ex``  — expectation of life at age ``x``, ``Tx / lx``.
    """
    n = width.astype(float)
    k = mx.shape[0]

    # -- ax: average years lived by decedents in each interval ---------------
    ax = n / 2.0
    ax[0] = 0.1 if n[0] <= 1.0 else n[0] / 2.0          # infant interval special-case
    # open-ended final interval: mean years lived = 1/m (exp. of remaining life)
    if mx[-1] > 0:
        ax[-1] = 1.0 / mx[-1]
    else:
        ax[-1] = n[-1]

    # -- qx: probability of dying in [x, x+n) --------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        qx = (n * mx) / (1.0 + (n - ax) * mx)
    qx = np.clip(qx, 0.0, 1.0)
    qx[-1] = 1.0                                        # everyone eventually dies
    px = 1.0 - qx

    # -- lx: survivors at exact age x (radix at birth) -----------------------
    lx = np.empty(k, dtype=float)
    lx[0] = radix
    for i in range(1, k):
        lx[i] = lx[i - 1] * px[i - 1]

    # -- ndx: deaths in interval ---------------------------------------------
    ndx = lx * qx

    # -- nLx: person-years lived in interval ---------------------------------
    nLx = np.empty(k, dtype=float)
    lx_next = np.concatenate([lx[1:], [0.0]])
    nLx[:-1] = n[:-1] * lx_next[:-1] + ax[:-1] * ndx[:-1]
    # open-ended interval: all remaining person-years = l / m
    if mx[-1] > 0:
        nLx[-1] = lx[-1] / mx[-1]
    else:
        nLx[-1] = lx[-1] * n[-1]

    # -- Tx: person-years lived above age x ----------------------------------
    Tx = np.flip(np.cumsum(np.flip(nLx)))

    # -- ex: expectation of life at age x ------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        ex = np.where(lx > 0, Tx / lx, 0.0)

    return {
        "n": n, "ax": ax, "mx": mx.astype(float), "qx": qx, "px": px,
        "lx": lx, "ndx": ndx, "nLx": nLx, "Tx": Tx, "ex": ex,
    }


def _crude_rate(mx: np.ndarray, pop: np.ndarray) -> float:
    """Population-weighted crude death rate ``Σ mx·pop / Σ pop``."""
    total = float(pop.sum())
    if total <= 0:
        return float("nan")
    return float((mx * pop).sum() / total)


# ---------------------------------------------------------------------- life table
@register(
    name="life_table",
    aliases=["生命表"],
    category="demography",
    tier="plus",
    skill="(人口学 缺口)",
    languages=["Python"],
    key_tools=["numpy"],
    description="周期生命表:年龄别死亡率 mx→qx→lx→ndx→nLx→Tx→ex,给出各年龄预期寿命(含 e0)",
    requires={"sources": ["datasets"]},
    produces={"models": ["life_table"]},
    auto_fix="none",
)
def life_table(state: StudyState, **kwargs: Any) -> StudyState:
    """Build a **period life table** from age-specific mortality rates.

    Parameters (via ``kwargs``)
    ---------------------------
    age : str
        Column naming the age group (default ``"age_group"``).
    mx : str
        Column holding the age-specific central death rate (default ``"mx_A"``).
    width : str
        Column holding the interval width ``n`` in years (default ``"n_years"``).
    radix : float
        Life-table radix ``l0`` (default ``100000``).

    Writes ``models.life_table`` — a dict with the tidy table (``DataFrame``),
    ``e0`` (expectation of life at birth), and the input column mapping.
    """
    df = _get_datasets(state, kwargs)
    age_col = kwargs.get("age", "age_group")
    mx_col = kwargs.get("mx", "mx_A")
    width_col = kwargs.get("width", "n_years")
    radix = float(kwargs.get("radix", 100_000.0))

    if df is None or mx_col not in df.columns:
        empty = {
            "table": pd.DataFrame(),
            "e0": float("nan"),
            "columns": {"age": age_col, "mx": mx_col, "width": width_col},
            "note": "no datasets / mx column missing — nothing to build",
        }
        state.write("models", "life_table", empty)
        return state

    mx = _numeric(df[mx_col])
    if width_col in df.columns:
        width = _numeric(df[width_col])
    else:
        width = _default_widths(mx.shape[0])
    # guard against NaN widths
    width = np.where(np.isfinite(width) & (width > 0), width, 1.0)

    lt = _build_life_table(mx, width, radix=radix)

    ages = (df[age_col].astype(str).to_numpy()
            if age_col in df.columns else np.arange(mx.shape[0]).astype(str))
    table = pd.DataFrame({
        "age": ages,
        "n": lt["n"],
        "mx": lt["mx"],
        "ax": lt["ax"],
        "qx": lt["qx"],
        "lx": lt["lx"],
        "ndx": lt["ndx"],
        "nLx": lt["nLx"],
        "Tx": lt["Tx"],
        "ex": lt["ex"],
    })

    e0 = float(lt["ex"][0])
    life_table_model = {
        "table": table,
        "e0": e0,
        "ex": {str(a): float(e) for a, e in zip(ages, lt["ex"])},
        "radix": radix,
        "columns": {"age": age_col, "mx": mx_col, "width": width_col},
        "note": "period life table; a0≈0.1, ax=n/2 elsewhere, open interval a=1/m",
    }
    state.write("models", "life_table", life_table_model)
    return state


# -------------------------------------------------------------------- decomposition
@register(
    name="decomposition",
    aliases=["分解", "Kitagawa", "Oaxaca"],
    category="demography",
    tier="pro",
    skill="(分解 缺口)",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="粗率差分解:Kitagawa 将两人群粗死亡率差拆为率效应+年龄构成效应(相加=总差);附 Oaxaca-Blinder 回归分解",
    requires={"sources": ["datasets"]},
    produces={"models": ["decomposition"], "diagnostics": ["components"]},
    auto_fix="escalate",
)
def decomposition(state: StudyState, **kwargs: Any) -> StudyState:
    """**Kitagawa** decomposition of a crude-death-rate difference.

    The difference in population-weighted crude rates between population *B* and
    population *A* is split additively (Kitagawa 1955):

    * **rate effect**        ``Σ (mB − mA) · (cA + cB)/2``
    * **composition effect** ``Σ (cB − cA) · (mA + mB)/2``

    where ``c`` is the age-composition share ``pop / Σ pop``. The two effects sum
    exactly to ``crude_B − crude_A``.

    Parameters (via ``kwargs``)
    ---------------------------
    mx_a, mx_b : str
        Age-specific rate columns for populations A and B
        (defaults ``"mx_A"`` / ``"mx_B"``).
    pop_a, pop_b : str
        Population (exposure) columns for A and B
        (defaults ``"pop_A"`` / ``"pop_B"``).

    Writes ``models.decomposition`` (the Kitagawa split + optional Oaxaca) and
    ``diagnostics.components`` (per-age contributions + the adding-up check).
    """
    df = _get_datasets(state, kwargs)
    mx_a_col = kwargs.get("mx_a", "mx_A")
    mx_b_col = kwargs.get("mx_b", "mx_B")
    pop_a_col = kwargs.get("pop_a", "pop_A")
    pop_b_col = kwargs.get("pop_b", "pop_B")

    needed = {mx_a_col, mx_b_col, pop_a_col, pop_b_col}
    if df is None or not needed.issubset(df.columns):
        empty = {
            "crude_A": float("nan"), "crude_B": float("nan"),
            "total_diff": float("nan"),
            "rate_effect": float("nan"), "composition_effect": float("nan"),
            "note": "no datasets / required columns missing",
        }
        state.write("models", "decomposition", empty)
        state.write("diagnostics", "components", pd.DataFrame())
        return state

    mA = _numeric(df[mx_a_col])
    mB = _numeric(df[mx_b_col])
    pA = _numeric(df[pop_a_col])
    pB = _numeric(df[pop_b_col])

    # -- age-composition shares ---------------------------------------------
    cA = pA / pA.sum()
    cB = pB / pB.sum()

    crude_A = _crude_rate(mA, pA)
    crude_B = _crude_rate(mB, pB)
    total_diff = crude_B - crude_A

    # -- Kitagawa additive split --------------------------------------------
    rate_contrib = (mB - mA) * (cA + cB) / 2.0
    comp_contrib = (cB - cA) * (mA + mB) / 2.0
    rate_effect = float(rate_contrib.sum())
    composition_effect = float(comp_contrib.sum())

    check_residual = float(total_diff - (rate_effect + composition_effect))

    decomposition_model: dict[str, Any] = {
        "method": "Kitagawa",
        "crude_A": float(crude_A),
        "crude_B": float(crude_B),
        "total_diff": float(total_diff),
        "rate_effect": rate_effect,
        "composition_effect": composition_effect,
        "adding_up_residual": check_residual,
        "note": "rate_effect + composition_effect = crude_B − crude_A (exact)",
    }

    # -- optional Oaxaca–Blinder (regression) companion ----------------------
    oaxaca = _oaxaca_blinder(mA, mB, cA, cB)
    if oaxaca is not None:
        decomposition_model["oaxaca"] = oaxaca

    # per-age contributions + the adding-up diagnostic
    age_col = kwargs.get("age", "age_group")
    ages = (df[age_col].astype(str).to_numpy()
            if age_col in df.columns else np.arange(mA.shape[0]).astype(str))
    components = pd.DataFrame({
        "age": ages,
        "mx_A": mA, "mx_B": mB,
        "share_A": cA, "share_B": cB,
        "rate_contribution": rate_contrib,
        "composition_contribution": comp_contrib,
    })

    state.write("models", "decomposition", decomposition_model)
    state.write("diagnostics", "components", components)
    return state


def _oaxaca_blinder(
    mA: np.ndarray, mB: np.ndarray, cA: np.ndarray, cB: np.ndarray
) -> dict[str, float] | None:
    """A regression-flavoured companion split of ``crude_B − crude_A``.

    We regress age-specific rates on the age index for each population (a simple
    linear age profile), then apply the standard twofold Oaxaca–Blinder
    decomposition to the *composition-weighted* means:

    * **endowments** ``(x̄_B − x̄_A)ᵀ β_A``  — the part explained by differences in
      the (share-weighted) age covariate.
    * **coefficients** ``x̄_Bᵀ (β_B − β_A)`` — the part due to differing rate
      profiles (returns to age).

    Uses ``statsmodels`` OLS when available; ``None`` if it cannot be built.
    """
    sm = _try_import("statsmodels.api")
    if sm is None:
        return None
    try:
        k = mA.shape[0]
        idx = np.arange(k, dtype=float)
        X = sm.add_constant(idx)
        # weight each age by its population share so the fitted profile reflects
        # the population that actually experiences it
        wa = cA / cA.sum()
        wb = cB / cB.sum()
        bA = sm.WLS(mA, X, weights=wa).fit().params
        bB = sm.WLS(mB, X, weights=wb).fit().params
        xbarA = np.array([1.0, float((idx * wa).sum())])
        xbarB = np.array([1.0, float((idx * wb).sum())])
        endowments = float((xbarB - xbarA) @ bA)
        coefficients = float(xbarB @ (bB - bA))
        return {
            "endowments": endowments,
            "coefficients": coefficients,
            "note": "twofold Oaxaca-Blinder on share-weighted linear age profile",
        }
    except Exception:
        return None
