import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pymada import reitsma, AUC, calc_hsroc_coef

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _fit():
    d = REF["data"]
    return reitsma(TP=d["TP"], FN=d["FN"], FP=d["FP"], TN=d["TN"])


def test_coefficients():
    r = _fit(); ref = REF["reitsma"]
    _c("coef", r["coefficients"], ref["coef"])


def test_std_errors():
    r = _fit(); ref = REF["reitsma"]
    _c("se", r["se"], ref["se"])


def test_vcov():
    r = _fit(); ref = REF["reitsma"]
    # reference vcov is column-major 2x2 flattened
    got = np.asarray(r["vcov"], float).ravel(order="F")
    _c("vcov", got, ref["vcov"])


def test_Psi():
    r = _fit(); ref = REF["reitsma"]
    got = np.asarray(r["Psi"], float).ravel(order="F")
    _c("Psi", got, ref["Psi"])


def test_pooled_sensitivity_specificity():
    r = _fit(); ref = REF["reitsma"]
    _c("sensitivity", r["sensitivity"], ref["sensitivity"])
    _c("false_pos_rate", r["false_pos_rate"], ref["false_pos_rate"])


def test_hsroc_coefficients():
    r = _fit(); ref = REF["hsroc"]
    hs = calc_hsroc_coef(r)
    for k in ("Theta", "Lambda", "beta", "sigma2theta", "sigma2alpha"):
        _c(f"hsroc.{k}", hs[k], ref[k])


def test_auc():
    r = _fit(); ref = REF["auc"]
    a = AUC(r)
    _c("AUC", a["AUC"], ref["AUC"])


def test_partial_auc():
    r = _fit(); ref = REF["auc"]
    a = AUC(r)
    _c("pAUC", a["pAUC"], ref["pAUC"])
