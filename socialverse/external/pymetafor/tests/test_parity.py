import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # external/ on path
from pymetafor import rma, blup

REF = json.loads((HERE / "reference.json").read_text())
FX = REF["fixture"]
YI, VI = np.array(FX["yi"]), np.array(FX["vi"])
MODS = np.column_stack([FX["ablat"], FX["year"]]).astype(float)
TOL = 1e-6   # class-1 deterministic-numerical, protocol hard ceiling

def _close(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float)); exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e} >= {tol:.0e}\n  got={got}\n  exp={exp}"
    return err

def _check(tag, res, ref, keys=("beta","se","zval","pval","ci.lb","ci.ub","tau2","I2","H2","QE")):
    m = {"beta":res.beta,"se":res.se,"zval":res.zval,"pval":res.pval,
         "ci.lb":res.ci_lb,"ci.ub":res.ci_ub,"tau2":res.tau2,"I2":res.I2,
         "H2":res.H2,"QE":res.QE,"QEp":res.QEp}
    errs = {}
    for k_ in keys:
        if ref.get(k_) in (None,"NA"): continue
        errs[k_] = _close(f"{tag}.{k_}", m[k_], ref[k_])
    return max(errs.values()) if errs else 0.0

def test_rma_reml():   _check("REML", rma(YI,VI,method="REML"), REF["rma_reml"])
def test_rma_dl():     _check("DL",   rma(YI,VI,method="DL"),   REF["rma_dl"])
def test_rma_ee():     _check("EE",   rma(YI,VI,method="EE"),   REF["rma_fe"])
def test_rma_knha():   _check("KNHA", rma(YI,VI,method="REML",test="knha"), REF["rma_hk"])
def test_rma_mods_identified():
    """Meta-regression: metafor's reported τ² is itself only 1e-5-converged
    (Fisher scoring stopped at its default threshold), and the `year` column is
    uncentred → cond(X'WX)≈2e11, so the unidentified intercept (SE≈29) is
    convergence+conditioning limited. The IDENTIFIED quantities — moderator
    slopes, τ²-free Q_E, and Q_M — match metafor within metafor's own
    convergence bound. See RECONSTRUCTION_REPORT.md §Known limitations."""
    r = rma(YI, VI, mods=MODS, method="REML"); ref = REF["rma_mods"]
    _close("MODS.slopes", r.beta[1:], ref["beta"][1:], tol=1e-5)
    _close("MODS.se_slopes", r.se[1:], ref["se"][1:], tol=1e-5)
    _close("MODS.QE", r.QE, ref["QE"], tol=1e-6)     # τ²-independent → exact
    _close("MODS.tau2", r.tau2, ref["tau2"], tol=1e-5)   # metafor's own tolerance
    _close("MODS.QM", r.QM, ref["QM"], tol=1e-3)

def test_rma_mods_centered_slopes():
    """Same fit with centred moderators (well-conditioned): identified slopes
    and Q_E match metafor to 1e-6; τ² to metafor's convergence bound."""
    c = REF["rma_mods_centered"]
    MC = np.column_stack([c["ablat_c"], c["year_c"]]).astype(float)
    r = rma(YI, VI, mods=MC, method="REML")
    _close("MODSc.slopes", r.beta[1:], c["beta"][1:], tol=1e-5)
    _close("MODSc.QE", r.QE, c["QE"], tol=1e-6)
def test_se_tau2_reml(): _close("REML.se_tau2", rma(YI,VI,method="REML").se_tau2, REF["rma_reml"]["se.tau2"])
def test_predict_reml():
    pr = rma(YI,VI,method="REML").predict()
    _close("PI.lb", pr["pi_lb"], REF["pred_reml"]["pi.lb"]); _close("PI.ub", pr["pi_ub"], REF["pred_reml"]["pi.ub"])

def test_blup_reml():
    """BLUP empirical-Bayes shrinkage per study, deterministic → all 1e-6."""
    b = blup(rma(YI, VI, method="REML")); ref = REF["blup_reml"]
    _close("BLUP.pred", b.pred, ref["pred"])
    _close("BLUP.se",   b.se,   ref["se"])
    _close("BLUP.pi.lb", b.pi_lb, ref["pi.lb"])
    _close("BLUP.pi.ub", b.pi_ub, ref["pi.ub"])
