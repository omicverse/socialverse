"""P0 regression-base methods recover their known data-generating-process truths.

Every dataset (socialverse.datasets._toy_p0) embeds a known DGP; these tests are
parameter-recovery checks with tolerances loose enough for one finite sample but
tight enough to catch a wrong implementation.
"""
import warnings

import pytest

import socialverse as sv
from socialverse import datasets as ds

warnings.simplefilter("ignore")


def _state(df, outcome=None, **design):
    s = sv.StudyState()
    sv.pp.ingest(s, data=df)
    if outcome:
        s.write("variables", "outcome", outcome)
    for k, v in design.items():
        s.write("design", k, v)
    return s


# --------------------------------------------------------------- registration
def test_all_p0_functions_registered():
    for name in ["glm", "mlogit", "ologit", "margins", "iv_regress", "psm", "mediation"]:
        assert sv.registry.get(name) is not None, f"{name} not registered"


# ------------------------------------------------------------------------ glm
def test_glm_ols_recovers_coefficients():
    s = _state(ds.load_regression(), "y")
    sv.tl.glm(s, predictors=["x1", "x2"], family="gaussian")
    coef = s.models["glm"]["coef"]
    assert coef["x1"] == pytest.approx(0.5, abs=0.15)   # truth 0.5
    assert coef["x2"] == pytest.approx(-0.4, abs=0.15)  # truth -0.4


def test_glm_logit_recovers_coefficients():
    s = _state(ds.load_regression(), "y_bin")
    sv.tl.glm(s, predictors=["x1", "x2"], family="binomial")
    coef = s.models["glm"]["coef"]
    assert coef["x1"] == pytest.approx(0.8, abs=0.3)    # logit, noisier
    assert coef["x2"] < 0


def test_glm_poisson_recovers_rate():
    s = _state(ds.load_regression(), "y_count")
    sv.tl.glm(s, predictors=["x1", "x2"], family="poisson")
    assert s.models["glm"]["coef"]["x1"] == pytest.approx(0.4, abs=0.15)


def test_margins_after_logit_is_finite_and_signed():
    s = _state(ds.load_regression(), "y_bin")
    sv.tl.glm(s, predictors=["x1", "x2"], family="binomial")
    sv.tl.margins(s, model="glm")
    ame = s.diagnostics["margins"]["ame"]
    assert ame["x1"] > 0 and ame["x2"] < 0            # same sign as coefs
    assert abs(ame["x1"]) < 1.0                        # a probability marginal effect


# -------------------------------------------------------------- mlogit/ologit
def test_mlogit_fits():
    s = _state(ds.load_regression(), "choice")
    sv.tl.mlogit(s, predictors=["x1"])
    assert s.models["mlogit"]["n"] == 600


def test_ologit_x1_positive():
    s = _state(ds.load_regression(), "y_ord")
    sv.tl.ologit(s, predictors=["x1", "x2"])
    assert s.models["ologit"]["coef"]["x1"] > 0


# ------------------------------------------------------------------ iv_regress
def test_iv_recovers_effect_and_beats_biased_ols():
    s = _state(ds.load_iv(), "y")
    sv.tl.iv_regress(s, endogenous="x", instruments=["z"], exog=["w"])
    iv = s.models["iv"]["coef"]["x"]
    fs = s.diagnostics["first_stage"]
    assert iv == pytest.approx(1.5, abs=0.25)          # truth 1.5
    assert fs["ols_endog_coef"] > iv + 0.3             # OLS biased upward
    assert fs["F"] > 10                                # strong instrument


# ------------------------------------------------------------------------ psm
@pytest.mark.parametrize("method", ["nn", "ipw"])
def test_psm_recovers_att(method):
    s = _state(ds.load_treatment(), "y", treatment="treat")
    sv.tl.psm(s, covariates=["x1", "x2", "x3"], method=method)
    m = s.models["psm"]
    assert m["att"] == pytest.approx(2.0, abs=0.45)    # truth ATT 2.0
    assert abs(m["naive_diff"] - 2.0) > abs(m["att"] - 2.0) - 0.5  # naive worse/biased


def test_psm_improves_balance():
    s = _state(ds.load_treatment(), "y", treatment="treat")
    sv.tl.psm(s, covariates=["x1", "x2", "x3"], method="nn")
    bal = s.diagnostics["balance"]
    before = max(abs(v) for v in bal["smd_before"].values())
    after = max(abs(v) for v in bal["smd_after"].values())
    assert after <= before


# ------------------------------------------------------------------ mediation
def test_mediation_recovers_acme_ade_total():
    s = _state(ds.load_mediation(), "y")
    sv.tl.mediation(s, treatment="x", mediator="m", boot=300, seed=0)
    m = s.models["mediation"]
    assert m["acme"] == pytest.approx(0.42, abs=0.12)   # a*b = 0.42
    assert m["ade"] == pytest.approx(0.30, abs=0.12)    # direct 0.30
    assert m["total"] == pytest.approx(0.72, abs=0.15)


# --------------------------------------------------------- compat py- aliases
@pytest.mark.parametrize("alias,expected", [
    ("py-logit", "glm"), ("py-poisson", "glm"), ("py-regress", "glm"),
    ("py-mlogit", "mlogit"), ("py-polr", "ologit"), ("py-margins", "margins"),
    ("py-ivregress", "iv_regress"), ("py-psmatch2", "psm"), ("py-mediate", "mediation"),
])
def test_p0_py_aliases_resolve(alias, expected):
    e = sv.registry.get(alias)
    assert e is not None and e.name == expected
