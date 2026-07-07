"""``sv.tl._mediation`` — registered implementation for the **causal mediation**
gap.

Causal mediation decomposes the total effect of a treatment/exposure ``x`` on an
outcome ``y`` into an **indirect** effect that runs through a mediator ``m``
(ACME — average causal mediation effect) and a **direct** effect that does not
(ADE — average direct effect). This is the Baron-Kenny / Imai-Keele-Tingley
product-of-coefficients decomposition, ported to the ``StudyState`` / ``registry``
spine with *real* estimation (recovers a known DGP) and a genuine nonparametric
bootstrap confidence interval — no placeholders.

Champion packages this file mirrors
-----------------------------------
* R's ``mediation::mediate`` (Imai, Keele, Tingley, Yamamoto) and Python
  ``statsmodels.stats.mediation.Mediation``. Two OLS models are fitted:

    m ~ x + covariates            (the "mediator model"; the ``x`` coefficient = a)
    y ~ x + m + covariates        (the "outcome model"; ``m`` coef = b, ``x`` = c')

  For the linear/no-interaction case the causal quantities are the classic
  product-of-coefficients:

    ACME  (indirect) = a * b
    ADE   (direct)   = c'
    total effect     = ACME + ADE
    proportion_mediated = ACME / total

  The confidence interval for ACME comes from a nonparametric (case-resampling)
  bootstrap — the same interval ``mediation::mediate`` reports by default. When
  ``statsmodels.stats.mediation.Mediation`` is importable we run it as an
  independent cross-check and record its ACME, but every headline number comes
  from the in-file OLS + bootstrap so the notebook runs without that submodule.

The registry contract: ``mediation`` ``requires`` a working
``sources['datasets']`` frame and a declared ``variables['outcome']``, and
``produces`` the fitted mediation model — so a resolver can refuse to report an
"indirect effect" until the mediator and outcome models have actually been fit.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["mediation"]


# --------------------------------------------------------------------- helpers
def _try_import(name: str):
    """Lazy, fail-soft import of an optional heavy dependency."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _get_datasets(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working frame: explicit ``data=`` kwarg, else ``sources['datasets']``.

    ``sources['datasets']`` may be a DataFrame or a ``{name: DataFrame}`` mapping;
    in the latter case the first frame is taken.
    """
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Ordinary least squares coefficients via a stable lstsq solve.

    ``X`` already includes any intercept column. Returns the coefficient vector.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _fit_paths(
    work: pd.DataFrame,
    treatment: str,
    mediator: str,
    outcome: str,
    covariates: list[str],
) -> tuple[float, float, float]:
    """Fit the two OLS models and return ``(a, b, direct)``.

    * ``a``      = coefficient of ``treatment`` in ``mediator ~ treatment + covariates``
    * ``b``      = coefficient of ``mediator``  in ``outcome ~ treatment + mediator + covariates``
    * ``direct`` = coefficient of ``treatment`` in that same outcome model (c')
    """
    n = len(work)
    ones = np.ones(n)
    cov = [work[c].to_numpy(dtype=float) for c in covariates]

    # mediator model: m ~ 1 + x + covariates  → a is the x coefficient (index 1)
    Xm = np.column_stack([ones, work[treatment].to_numpy(dtype=float), *cov])
    beta_m = _ols(work[mediator].to_numpy(dtype=float), Xm)
    a = float(beta_m[1])

    # outcome model: y ~ 1 + x + m + covariates → direct=x coef(idx1), b=m coef(idx2)
    Xy = np.column_stack(
        [
            ones,
            work[treatment].to_numpy(dtype=float),
            work[mediator].to_numpy(dtype=float),
            *cov,
        ]
    )
    beta_y = _ols(work[outcome].to_numpy(dtype=float), Xy)
    direct = float(beta_y[1])
    b = float(beta_y[2])
    return a, b, direct


# --------------------------------------------------------------------- mediation
@register(
    name="mediation",
    aliases=["中介分析", "mediation", "mediate"],
    category="causal",
    tier="plus",
    skill="(中介 缺口)",
    languages=["Python"],
    key_tools=["statsmodels"],
    description="因果中介分析:中介模型(a)+ 结果模型(b, direct)→ ACME=a·b / ADE / 总效应 + bootstrap CI",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["mediation"], "diagnostics": ["mediation_paths"]},
    auto_fix="escalate",
)
def mediation(state: StudyState, **kwargs: Any) -> StudyState:
    """Causal mediation analysis (product-of-coefficients + nonparametric bootstrap).

    Fits the mediator model ``m ~ x + covariates`` (recovering path ``a``) and the
    outcome model ``y ~ x + m + covariates`` (recovering the mediator path ``b`` and
    the direct path ``c'``), then decomposes the total effect:

    * ``ACME`` (average causal mediation / indirect effect) ``= a * b``
    * ``ADE``  (average direct effect)                       ``= c'``
    * ``total``                                               ``= ACME + ADE``
    * ``proportion_mediated``                                ``= ACME / total``

    A nonparametric case-resampling bootstrap (``boot`` replicates, ``seed`` for
    reproducibility) gives the percentile 95% CI for ACME. When
    ``statsmodels.stats.mediation.Mediation`` is available it is run as an
    independent cross-check.

    kwargs
    ------
    treatment : str
        Treatment / exposure column ``x``. Default ``"x"``.
    mediator : str
        Mediator column ``m``. Default ``"m"``.
    outcome : str
        Outcome column ``y``. Default: ``variables['outcome']`` else ``"y"``.
    covariates : list[str]
        Optional adjustment covariates entered in both models. Default ``[]``.
    boot : int
        Number of bootstrap replicates for the ACME CI. Default ``1000``.
    seed : int
        RNG seed for the bootstrap. Default ``0``.
    """
    df = _get_datasets(state, kwargs)

    def _empty(note: str) -> StudyState:
        model = {
            "acme": None, "ade": None, "total": None, "prop_mediated": None,
            "a": None, "b": None, "direct": None, "ci_acme": None,
            "n": 0, "boot": 0, "note": note,
        }
        state.write("models", "mediation", model)
        state.write("diagnostics", "mediation_paths", {
            "a_path": None, "b_path": None, "direct_path": None, "note": note,
        })
        return state

    if df is None:
        return _empty("缺少数据(sources['datasets']),无法进行中介分析")

    treatment = kwargs.get("treatment", "x")
    mediator = kwargs.get("mediator", "m")
    outcome = kwargs.get("outcome") or state.variables.get("outcome") or "y"
    covariates = [c for c in (kwargs.get("covariates") or []) if c in df.columns]
    boot = int(kwargs.get("boot", 1000))
    seed = int(kwargs.get("seed", 0))

    missing = [c for c in (treatment, mediator, outcome) if c not in df.columns]
    if missing:
        return _empty(
            f"缺少中介分析所需列: {missing}"
            f"(treatment='{treatment}', mediator='{mediator}', outcome='{outcome}')"
        )

    keep = [outcome, treatment, mediator] + covariates
    work = df[keep].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < max(10, len(covariates) + 4):
        return _empty(f"有效样本量不足(n={len(work)}),无法稳健估计中介效应")

    # point estimates
    a, b, direct = _fit_paths(work, treatment, mediator, outcome, covariates)
    acme = a * b
    ade = direct
    total = acme + ade
    prop = float(acme / total) if total != 0 else None

    # nonparametric case-resampling bootstrap for the ACME CI
    rng = np.random.default_rng(seed)
    n = len(work)
    idx_all = np.arange(n)
    boot_acme: list[float] = []
    boot_ade: list[float] = []
    boot_total: list[float] = []
    for _ in range(max(boot, 0)):
        idx = rng.choice(idx_all, size=n, replace=True)
        sub = work.iloc[idx]
        try:
            ai, bi, di = _fit_paths(sub, treatment, mediator, outcome, covariates)
        except Exception:
            continue
        boot_acme.append(ai * bi)
        boot_ade.append(di)
        boot_total.append(ai * bi + di)

    ci_acme = None
    ci_ade = None
    ci_total = None
    boot_se = None
    if boot_acme:
        arr = np.asarray(boot_acme, dtype=float)
        ci_acme = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        boot_se = float(np.std(arr, ddof=1)) if arr.size > 1 else None
        ade_arr = np.asarray(boot_ade, dtype=float)
        ci_ade = [float(np.percentile(ade_arr, 2.5)), float(np.percentile(ade_arr, 97.5))]
        tot_arr = np.asarray(boot_total, dtype=float)
        ci_total = [float(np.percentile(tot_arr, 2.5)), float(np.percentile(tot_arr, 97.5))]

    # optional independent cross-check via statsmodels.stats.mediation
    crosscheck: dict[str, Any] | None = None
    smm = _try_import("statsmodels.stats.mediation")
    smf = _try_import("statsmodels.formula.api")
    sm = _try_import("statsmodels.api")
    if smm is not None and smf is not None and sm is not None:
        try:
            cov_terms = (" + " + " + ".join(covariates)) if covariates else ""
            out_model = smf.ols(
                f"{outcome} ~ {treatment} + {mediator}{cov_terms}", data=work
            )
            med_model = smf.ols(
                f"{mediator} ~ {treatment}{cov_terms}", data=work
            )
            med = smm.Mediation(
                out_model, med_model, treatment, mediator
            )
            mres = med.fit(n_rep=min(max(boot, 100), 500))
            summ = mres.summary()
            # ACME (average) row label used by statsmodels
            acme_row = None
            for label in ("ACME (average)", "ACME (control)"):
                if label in summ.index:
                    acme_row = label
                    break
            if acme_row is not None:
                crosscheck = {
                    "acme": float(summ.loc[acme_row, "Estimate"]),
                    "source": "statsmodels.stats.mediation.Mediation",
                }
        except Exception:
            crosscheck = None

    model = {
        "acme": float(acme),
        "ade": float(ade),
        "total": float(total),
        "prop_mediated": prop,
        "a": float(a),
        "b": float(b),
        "direct": float(direct),
        "ci_acme": ci_acme,
        "ci_ade": ci_ade,
        "ci_total": ci_total,
        "boot_se_acme": boot_se,
        "n": int(n),
        "boot": int(len(boot_acme)),
        "seed": seed,
        "treatment": treatment,
        "mediator": mediator,
        "outcome": outcome,
        "covariates": covariates,
        "crosscheck": crosscheck,
        "estimator": "OLS 中介/结果模型 + 系数乘积(a·b)+ 非参数 bootstrap CI",
        "note": (
            "ACME(间接)=a·b;ADE(直接)=c';总效应=ACME+ADE;"
            "比例中介=ACME/总效应。CI 为 percentile bootstrap(95%)。"
            "product-of-coefficients 仅在线性、无 x·m 交互假设下等于因果中介效应。"
        ),
    }
    state.write("models", "mediation", model)
    state.write("diagnostics", "mediation_paths", {
        "a_path": float(a),
        "b_path": float(b),
        "direct_path": float(direct),
        "note": (
            f"a: {treatment}→{mediator} 的系数;"
            f"b: {mediator}→{outcome}(控制 {treatment})的系数;"
            f"direct: {treatment}→{outcome}(控制 {mediator})的系数。"
        ),
    })
    return state


# ------------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Toy DGP matching ds.load_mediation():
    #   x ~ N(0,1); m = 0.6 x + e_m ; y = 0.3 x + 0.7 m + e_y
    #   → a=0.6, b=0.7, direct=0.3, ACME=a*b=0.42, ADE=0.30, total=0.72
    rng = np.random.default_rng(20260706)
    N = 4000
    x = rng.normal(0, 1, N)
    m = 0.6 * x + rng.normal(0, 1, N)
    y = 0.3 * x + 0.7 * m + rng.normal(0, 1, N)
    toy = pd.DataFrame({"y": y, "x": x, "m": m})

    st = StudyState()
    st.sources["datasets"] = toy
    st.variables["outcome"] = "y"

    st = mediation(st, treatment="x", mediator="m", outcome="y", boot=800, seed=0)
    mdl = st.models["mediation"]
    dia = st.diagnostics["mediation_paths"]

    truth = {"a": 0.6, "b": 0.7, "direct": 0.3, "acme": 0.42, "ade": 0.30, "total": 0.72}
    print("=== mediation self-test (recovered vs truth) ===")
    for k in ("a", "b", "direct", "acme", "ade", "total"):
        print(f"  {k:7s} recovered={mdl[k]:+.4f}  truth={truth[k]:+.3f}  "
              f"|Δ|={abs(mdl[k]-truth[k]):.4f}")
    print(f"  prop_mediated recovered={mdl['prop_mediated']:.4f} (truth≈{0.42/0.72:.4f})")
    print(f"  ci_acme(95%)={mdl['ci_acme']}  boot_se={mdl['boot_se_acme']}")
    print(f"  n={mdl['n']}  boot={mdl['boot']}  crosscheck={mdl['crosscheck']}")

    tol = {"a": 0.1, "b": 0.1, "direct": 0.1, "acme": 0.1, "ade": 0.1, "total": 0.1}
    ok = all(abs(mdl[k] - truth[k]) <= tol[k] for k in tol)
    # CI should cover the true ACME
    ci_ok = mdl["ci_acme"][0] <= 0.42 <= mdl["ci_acme"][1]
    print(f"  within-tol={ok}  ci_covers_true_ACME={ci_ok}")
    assert ok, "recovered params outside tolerance"
    assert ci_ok, "bootstrap CI does not cover true ACME"
    print("SELF-TEST PASSED")
