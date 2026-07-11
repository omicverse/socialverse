import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pydemography import life_table, life_expectancy, kitagawa, oaxaca

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    # open-ended interval carries ax = Inf in R; compare only finite entries
    finite = np.isfinite(exp)
    err = float(np.max(np.abs(got[finite] - exp[finite]))) if finite.any() else 0.0
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


# --------------------------------------------------------------------------- #
# Life table (all three sex branches)
# --------------------------------------------------------------------------- #
def _check_lifetable(sex):
    ref = REF["lifetable"][sex]
    mx = REF["lifetable"]["input"]["mx"]
    lt = life_table(mx, sex=sex, startage=0, agegroup=1)
    for col in ("ax", "mx", "qx", "lx", "dx", "Lx", "Tx", "ex", "nx"):
        _c(f"lifetable.{sex}.{col}", lt[col], ref[col])
    _c(f"lifetable.{sex}.e0", lt["e0"], ref["e0"])


def test_lifetable_female():
    _check_lifetable("female")


def test_lifetable_male():
    _check_lifetable("male")


def test_lifetable_total():
    _check_lifetable("total")


def test_life_expectancy_e0():
    mx = REF["lifetable"]["input"]["mx"]
    for sex in ("female", "male", "total"):
        e0 = life_expectancy(mx, sex=sex, startage=0, agegroup=1, age=0)
        _c(f"e0.{sex}", e0, REF["lifetable"][sex]["e0"])


# --------------------------------------------------------------------------- #
# Kitagawa
# --------------------------------------------------------------------------- #
def test_kitagawa():
    ki = REF["kitagawa"]
    inp = ki["input"]
    r = kitagawa(inp["c1"], inp["r1"], inp["c2"], inp["r2"])
    _c("kitagawa.R1", r["R1"], ki["R1"])
    _c("kitagawa.R2", r["R2"], ki["R2"])
    _c("kitagawa.total", r["total"], ki["total"])
    _c("kitagawa.rate_effect", r["rate_effect"], ki["rate_effect"])
    _c("kitagawa.composition_effect", r["composition_effect"], ki["composition_effect"])
    # decomposition must be exact
    _c("kitagawa.sum", r["rate_effect"] + r["composition_effect"], r["total"])


# --------------------------------------------------------------------------- #
# Oaxaca-Blinder
# --------------------------------------------------------------------------- #
def test_oaxaca():
    ox = REF["oaxaca"]
    inp = ox["input"]
    r = oaxaca(inp["yA"], inp["xA"], inp["yB"], inp["xB"])
    _c("oaxaca.betaA", r["betaA"], ox["betaA"])
    _c("oaxaca.betaB", r["betaB"], ox["betaB"])
    _c("oaxaca.meanYA", r["meanYA"], ox["meanYA"])
    _c("oaxaca.meanYB", r["meanYB"], ox["meanYB"])
    _c("oaxaca.gap", r["gap"], ox["gap"])
    _c("oaxaca.explained", r["explained"], ox["explained"])
    _c("oaxaca.unexplained", r["unexplained"], ox["unexplained"])
    # explained + unexplained must reproduce the gap exactly
    _c("oaxaca.sum", r["explained"] + r["unexplained"], r["gap"])
