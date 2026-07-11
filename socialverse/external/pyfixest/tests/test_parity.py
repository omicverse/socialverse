import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pyfixest import feols, fepois, newey_west

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _load():
    d = REF["data"]
    return (np.asarray(d["y"], float), np.asarray(d["x"], float),
            np.asarray(d["id"]), np.asarray(d["time"]))


def test_oneway_id():
    y, x, idv, timev = _load()
    r = feols(y, x, fe=idv, cluster=idv)
    ref = REF["oneway_id"]
    _c("oneway.coef", r["coef"], ref["coef"])
    _c("oneway.se", r["se"], ref["se"])
    _c("oneway.wr2", r["within_r2"], ref["within_r2"])
    assert r["nobs"] == ref["nobs"], (r["nobs"], ref["nobs"])
    assert r["nparams"] == ref["nparams"], (r["nparams"], ref["nparams"])


def test_oneway_id_clustertime():
    y, x, idv, timev = _load()
    r = feols(y, x, fe=idv, cluster=timev)
    ref = REF["oneway_id_clustertime"]
    _c("oneway_ct.coef", r["coef"], ref["coef"])
    _c("oneway_ct.se", r["se"], ref["se"])
    _c("oneway_ct.wr2", r["within_r2"], ref["within_r2"])


def test_twoway():
    y, x, idv, timev = _load()
    r = feols(y, x, fe=[idv, timev], cluster=idv)
    ref = REF["twoway"]
    _c("twoway.coef", r["coef"], ref["coef"])
    _c("twoway.se", r["se"], ref["se"])
    _c("twoway.wr2", r["within_r2"], ref["within_r2"])
    assert r["nparams"] == ref["nparams"], (r["nparams"], ref["nparams"])


def test_fepois_oneway_id():
    d = REF["fepois_data"]
    y = np.asarray(d["y"], float)
    x = np.asarray(d["x"], float)
    idv = np.asarray(d["id"])
    r = fepois(y, x, fe=idv, cluster=idv)
    ref = REF["fepois"]
    _c("fepois.coef", r["coef"], ref["coef"])
    _c("fepois.se", r["se"], ref["se"])
    _c("fepois.deviance", r["deviance"], ref["deviance"])
    assert r["nobs"] == ref["nobs"], (r["nobs"], ref["nobs"])


def test_newey_west_lag3():
    d = REF["nw_data"]
    y = np.asarray(d["y"], float)
    X = np.column_stack([np.asarray(d["x1"], float), np.asarray(d["x2"], float)])
    t = np.asarray(d["t"])
    r = newey_west(y, X, lag=3, order=t)
    ref = REF["nw_lag3"]
    _c("nw3.coef", r["coef"], ref["coef"])
    _c("nw3.se", r["se"], ref["se"])


def test_newey_west_lag2():
    d = REF["nw_data"]
    y = np.asarray(d["y"], float)
    X = np.column_stack([np.asarray(d["x1"], float), np.asarray(d["x2"], float)])
    t = np.asarray(d["t"])
    r = newey_west(y, X, lag=2, order=t)
    ref = REF["nw_lag2"]
    _c("nw2.coef", r["coef"], ref["coef"])
    _c("nw2.se", r["se"], ref["se"])
