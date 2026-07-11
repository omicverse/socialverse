import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pysurvey import svydesign, svymean, svytotal, svyglm
REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6

def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float)); exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"

def _design(tag):
    d = REF[tag]["data"]
    ids = d.get("dnum")            # cluster PSU, or None → element sampling
    strata = d.get("stype")
    return svydesign(d, weights=d["pw"], ids=ids, strata=strata, fpc=d["fpc"]), d

def test_apistrat_svymean():
    ds, d = _design("apistrat"); r = svymean("api00", ds); ref = REF["apistrat"]["svymean"]
    _c("strat.mean.est", r["estimate"], ref["est"]); _c("strat.mean.se", r["se"], ref["se"])
    assert r["df"] == ref["df"], (r["df"], ref["df"])

def test_apistrat_svytotal():
    ds, d = _design("apistrat"); r = svytotal("api00", ds); ref = REF["apistrat"]["svytotal"]
    _c("strat.tot.est", r["estimate"], ref["est"], tol=1e-4); _c("strat.tot.se", r["se"], ref["se"], tol=1e-4)

def test_apistrat_svyglm():
    ds, d = _design("apistrat")
    r = svyglm("api00", np.column_stack([d["ell"], d["meals"]]), ds); ref = REF["apistrat"]["svyglm"]
    _c("strat.glm.coef", r["coef"], ref["coef"]); _c("strat.glm.se", r["se"], ref["se"])
    assert r["df"] == ref["df"], (r["df"], ref["df"])

def test_apiclus1_svymean():
    ds, d = _design("apiclus1"); r = svymean("api00", ds); ref = REF["apiclus1"]["svymean"]
    _c("clus.mean.est", r["estimate"], ref["est"]); _c("clus.mean.se", r["se"], ref["se"])
    assert r["df"] == ref["df"], (r["df"], ref["df"])

def test_apiclus1_svyglm():
    ds, d = _design("apiclus1")
    r = svyglm("api00", np.asarray(d["ell"]), ds); ref = REF["apiclus1"]["svyglm"]
    _c("clus.glm.coef", r["coef"], ref["coef"]); _c("clus.glm.se", r["se"], ref["se"])
    assert r["df"] == ref["df"], (r["df"], ref["df"])
