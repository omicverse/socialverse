import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pysurvival import km, coxph, clogit, survreg
REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6

def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float)); exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got[:6]}\n exp={exp[:6]}"

def _event(status):  # lung: 1=censored, 2=event
    status = np.asarray(status, int)
    return (status == 2).astype(int) if set(np.unique(status)) <= {1, 2} else (status == 1).astype(int)

def test_km():
    d = REF["km_data"]; r = km(d["time"], _event(d["status"])); ref = REF["km"]
    _c("km.time", r.time, ref["time"]); _c("km.n_risk", r.n_risk, ref["n.risk"])
    _c("km.n_event", r.n_event, ref["n.event"]); _c("km.surv", r.surv, ref["surv"])
    _c("km.std_err", r.std_err, ref["std.err"])
    assert r.median == ref["median"], (r.median, ref["median"])

def test_km_ci():
    d = REF["km_data"]; r = km(d["time"], _event(d["status"])); ref = REF["km"]
    _c("km.lower", r.lower, ref["lower"], tol=1e-6); _c("km.upper", r.upper, ref["upper"], tol=1e-6)

def _X():
    d = REF["cox_data"]
    return np.column_stack([d["age"], d["sex"], d["ph.ecog"]]).astype(float), d

def test_cox_efron():
    X, d = _X(); r = coxph(d["time"], _event(d["status"]), X, ties="efron"); ref = REF["cox_efron"]
    _c("cox.efron.coef", r.coef, ref["coef"]); _c("cox.efron.se", r.se, ref["se"])
    _c("cox.efron.z", r.z, ref["z"]); _c("cox.efron.loglik", r.loglik, ref["loglik"])

def test_cox_breslow():
    X, d = _X(); r = coxph(d["time"], _event(d["status"]), X, ties="breslow"); ref = REF["cox_breslow"]
    _c("cox.breslow.coef", r.coef, ref["coef"]); _c("cox.breslow.loglik", r.loglik, ref["loglik"])

def test_cox_concordance():
    X, d = _X(); r = coxph(d["time"], _event(d["status"]), X, ties="efron")
    _c("cox.concordance", r.concordance, REF["cox_efron"]["concordance"], tol=1e-3)

# --- clogit: conditional logistic (matched case-control, infert) ---
def test_clogit():
    d = REF["clogit_data"]; ref = REF["clogit"]
    X = np.column_stack([d["spontaneous"], d["induced"]]).astype(float)
    r = clogit(d["case"], d["stratum"], X)
    _c("clogit.coef", r.coef, ref["coef"])
    _c("clogit.se", r.se, ref["se"])
    _c("clogit.loglik", r.loglik, ref["loglik"])

# --- survreg: parametric AFT MLE (Weibull / exponential / lognormal, lung) ---
def _survreg_X():
    d = REF["survreg_data"]
    X = np.column_stack([np.ones_like(d["time"], float), d["age"], d["sex"]])
    return X, d

def test_survreg_weibull():
    X, d = _survreg_X(); ref = REF["survreg_weibull"]
    r = survreg(d["time"], d["status"], X, dist="weibull")
    _c("survreg.wb.coef", r.coef, ref["coef"])
    _c("survreg.wb.scale", r.scale, ref["scale"])
    _c("survreg.wb.se", r.se, ref["se"])
    _c("survreg.wb.loglik", r.loglik, ref["loglik"])

def test_survreg_exponential():
    X, d = _survreg_X(); ref = REF["survreg_exp"]
    r = survreg(d["time"], d["status"], X, dist="exponential")
    _c("survreg.exp.coef", r.coef, ref["coef"])
    _c("survreg.exp.se", r.se, ref["se"])
    _c("survreg.exp.loglik", r.loglik, ref["loglik"])

def test_survreg_lognormal():
    X, d = _survreg_X(); ref = REF["survreg_lognormal"]
    r = survreg(d["time"], d["status"], X, dist="lognormal")
    _c("survreg.ln.coef", r.coef, ref["coef"])
    _c("survreg.ln.scale", r.scale, ref["scale"])
    _c("survreg.ln.se", r.se, ref["se"])
    _c("survreg.ln.loglik", r.loglik, ref["loglik"])
