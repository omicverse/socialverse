import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pyfixest import feols

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
