"""Numerical parity gate for pypsych against R psych 2.6.5.

Fixture: psych::bfi first 5 items (A1..A5), complete cases (2709 subjects), dumped
raw into reference.json so Python reads the SAME item matrix.

Class-1 (exact, 1e-6):
  * cronbach_alpha: raw_alpha / std_alpha / G6 / average_r
  * fa(fm='pa', nfactors=1): loadings / communality / uniqueness
  * omega_total (McDonald ω_tot from the PA communalities), matched to the R driver
    computing the SAME closed form on psych's PA solution.

Documented reference-tolerance limitation (NOT gated at 1e-6):
  * psych::omega()'s own ``omega.tot`` differs (~2e-4) because psych's omega runs a
    separate internal pipeline — automatic item key-reversal + GPArotation minres —
    rather than the public fa(fm='pa') one-factor solution.  We assert our ω_tot is
    within 1e-3 of psych's omega.tot to confirm it is the right quantity, and gate
    our definition exactly against the R driver's matching computation.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pypsych import cronbach_alpha, fa_pa, omega_total, ICC, corr_test  # noqa: E402

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _items():
    return np.asarray(REF["data"]["items"], float)


def _corr():
    X = _items()
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def test_cronbach_alpha():
    r = cronbach_alpha(_items())
    ref = REF["alpha"]
    _c("alpha.raw", r["raw_alpha"], ref["raw_alpha"])
    _c("alpha.std", r["std_alpha"], ref["std_alpha"])
    _c("alpha.G6", r["G6"], ref["G6"])
    _c("alpha.avg_r", r["average_r"], ref["average_r"])


def test_fa_pa_loadings():
    f = fa_pa(_corr(), nfactors=1)
    ref = REF["fa"]
    _c("fa.loadings", f["loadings"], ref["loadings"])
    _c("fa.communality", f["communality"], ref["communality"])
    _c("fa.uniqueness", f["uniqueness"], ref["uniqueness"])


def test_omega_total_definition():
    # Gate our omega_total exactly against the R driver's matching PA-form computation.
    R = _corr()
    h2 = np.asarray(REF["fa"]["communality"], float)
    om = omega_total(R, communality=h2)
    _c("omega.paform", om, REF["omega"]["omega_tot_paform"])


def _corr_keyrev():
    """Correlation matrix after reversing the negatively-keyed items psych's
    omega() would auto-reverse (item scale 1..6 -> 7-x)."""
    X = _items().copy()
    keyrev = REF["omega"]["keyrev_items"]
    for j in np.atleast_1d(keyrev):             # 1-based (auto_unbox may scalarize)
        X[:, j - 1] = 7.0 - X[:, j - 1]
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def test_omega_keyreversed_definition():
    # After the SAME key reversal, gate our omega_total exactly (1e-6) vs the driver.
    om = omega_total(_corr_keyrev(), nfactors=1)
    _c("omega.keyrev", om, REF["omega"]["omega_tot_keyrev"])


def test_ICC_point_estimates():
    # Gate the six ICC point estimates + F / df / p to 1e-6 (class-1) against
    # psych::ICC(lmer=FALSE) on the fixed 6x4 ratings fixture.
    ic = REF["icc"]
    r = ICC(np.asarray(ic["ratings"], float))
    assert r["type"] == ic["type"]
    _c("icc.ICC", r["ICC"], ic["ICC"])
    _c("icc.F", r["F"], ic["F"])
    _c("icc.df1", r["df1"], ic["df1"])
    _c("icc.df2", r["df2"], ic["df2"])
    _c("icc.p", r["p"], ic["p"])
    _c("icc.MSW", r["MSW"], ic["MSW"])


def test_ICC_confidence_bounds():
    # ICC CIs are deterministic closed forms (qf/Satterthwaite df); gate to 1e-6.
    ic = REF["icc"]
    r = ICC(np.asarray(ic["ratings"], float))
    _c("icc.lower", r["lower"], ic["lower"])
    _c("icc.upper", r["upper"], ic["upper"])


def test_corr_test_r_and_p():
    # Gate corr.test correlation matrix + RAW two-sided p (adjust="none") + t + se
    # to 1e-6 against psych::corr.test on the same 5-item complete-case fixture.
    ct = REF["corr_test"]
    r = corr_test(_items())
    assert r["n"] == int(ct["n"])
    _c("corr.r", r["r"], np.asarray(ct["r"], float))
    _c("corr.p", r["p"], np.asarray(ct["p"], float))
    _c("corr.se", r["se"], np.asarray(ct["se"], float))
    # t has +Inf on the diagonal in both (r=1 -> t=Inf); gate the finite entries
    # and confirm both mark the same diagonal as +Inf.
    gt = np.asarray(r["t"], float)
    et = np.asarray(ct["t"], float)          # "Inf" strings coerce to +inf
    assert np.array_equal(np.isinf(gt), np.isinf(et))
    fin = np.isfinite(et)
    _c("corr.t", gt[fin], et[fin])


def test_omega_close_to_psych_reference():
    # Documented reference tolerance: psych::omega() auto-reverses A1 then factors
    # with GPArotation minres.  Reproducing only the key reversal, our McDonald
    # omega_total lands within 1e-3 of psych's omega.tot (residual gap = psych's
    # minres vs our PA factoring), confirming it is the same quantity.  The raw
    # (un-reversed) fixture omega legitimately differs by ~0.13 and is NOT gated here.
    om = omega_total(_corr_keyrev())
    _c("omega.vs_psych", om, REF["omega"]["psych_omega_tot"], tol=1e-3)
