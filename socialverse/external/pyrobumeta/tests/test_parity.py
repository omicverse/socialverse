import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pyrobumeta import robu
REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def test_corr():
    d = REF["corr"]["data"]
    r = robu(d["effectsize"], d["var"], d["studyid"],
             [d["males"], d["college"], d["binge"]],
             modelweights="CORR", rho=REF["corr"]["rho"], small=True)
    ref = REF["corr"]
    _c("corr.b", r["b"], ref["b"])
    _c("corr.SE", r["SE"], ref["SE"])
    _c("corr.dfs", r["dfs"], ref["dfs"])
    _c("corr.t", r["t"], ref["t"])
    _c("corr.prob", r["prob"], ref["prob"])
    _c("corr.CI.L", r["CI_L"], ref["CI.L"])
    _c("corr.CI.U", r["CI_U"], ref["CI.U"])
    _c("corr.tau.sq", r["tau_sq"], ref["tau.sq"])
    _c("corr.I2", r["I2"], ref["I.2"])
    assert r["N"] == ref["N"] and r["M"] == ref["M"] and r["p"] == ref["p"]


def test_hier():
    d = REF["hier"]["data"]
    r = robu(d["effectsize"], d["var"], d["studyid"],
             [d["binge"], d["sreport"], d["males"], d["age"], d["followup"]],
             modelweights="HIER", small=True)
    ref = REF["hier"]
    _c("hier.b", r["b"], ref["b"])
    _c("hier.SE", r["SE"], ref["SE"])
    _c("hier.dfs", r["dfs"], ref["dfs"])
    _c("hier.t", r["t"], ref["t"])
    _c("hier.prob", r["prob"], ref["prob"])
    _c("hier.CI.L", r["CI_L"], ref["CI.L"])
    _c("hier.CI.U", r["CI_U"], ref["CI.U"])
    _c("hier.tau.sq", r["tau_sq"], ref["tau.sq"])
    _c("hier.omega.sq", r["omega_sq"], ref["omega.sq"])
    assert r["N"] == ref["N"] and r["M"] == ref["M"] and r["p"] == ref["p"]
