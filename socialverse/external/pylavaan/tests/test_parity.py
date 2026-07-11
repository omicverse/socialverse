import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pylavaan import cfa

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6

MODEL = """visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9"""


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _fit():
    data = {k: np.asarray(v, float) for k, v in REF["data"].items()}
    return cfa(MODEL, data)


# ---- align Python parameter rows to R's parameterEstimates row order ----
def _py_param_table(res):
    """Return est/std_lv/std_all keyed by (lhs, op, rhs) to match R rows."""
    tbl = {}
    for row in res.parameter_estimates():
        tbl[(row["lhs"], row["op"], row["rhs"])] = row
    return tbl


def test_unstandardized_loadings_and_variances():
    res = _fit()
    tbl = _py_param_table(res)
    p = REF["params"]
    got, exp = [], []
    for lhs, op, rhs, est in zip(p["lhs"], p["op"], p["rhs"], p["est"]):
        got.append(tbl[(lhs, op, rhs)]["est"])
        exp.append(est)
    _c("est", got, exp)


def test_standardized_std_lv():
    res = _fit()
    tbl = _py_param_table(res)
    p = REF["params"]
    got, exp = [], []
    for lhs, op, rhs, v in zip(p["lhs"], p["op"], p["rhs"], p["std_lv"]):
        got.append(tbl[(lhs, op, rhs)]["std_lv"])
        exp.append(v)
    _c("std_lv", got, exp)


def test_standardized_std_all():
    res = _fit()
    tbl = _py_param_table(res)
    p = REF["params"]
    got, exp = [], []
    for lhs, op, rhs, v in zip(p["lhs"], p["op"], p["rhs"], p["std_all"]):
        got.append(tbl[(lhs, op, rhs)]["std_all"])
        exp.append(v)
    _c("std_all", got, exp)


def test_fit_measures():
    res = _fit()
    fm = res.fit_measures()
    ref = REF["fit"]
    _c("chisq", fm["chisq"], ref["chisq"])
    assert fm["df"] == int(ref["df"]), (fm["df"], ref["df"])
    _c("cfi", fm["cfi"], ref["cfi"])
    _c("tli", fm["tli"], ref["tli"])
    _c("rmsea", fm["rmsea"], ref["rmsea"])
    _c("srmr", fm["srmr"], ref["srmr"])
