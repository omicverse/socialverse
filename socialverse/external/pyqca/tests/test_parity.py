import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pyqca import truth_table, minimize

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    ok = err <= tol if tol == 0 else err < tol
    assert ok, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _data():
    d = REF["data"]
    conds = d["conditions"]
    data = {c: d[c] for c in conds}
    data["SURV"] = d["SURV"]
    return data, conds, d["incl.cut"]


def _tt():
    data, conds, incl_cut = _data()
    return truth_table(data, outcome="SURV", conditions=conds, incl_cut=incl_cut)


# ---- truth table: OUT / n / incl / PRI, matched to the SAME observed rows ----
def test_truthtable_rows():
    tt = _tt()
    ref = REF["truthTable"]
    # observed row ids must match exactly (same construction + ordering)
    assert list(tt.rownames) == list(ref["rownames"]), (list(tt.rownames), ref["rownames"])
    _c("tt.OUT", tt.OUT, ref["OUT"], tol=0)
    _c("tt.n", tt.n, ref["n"], tol=0)


def test_truthtable_incl_pri():
    tt = _tt()
    ref = REF["truthTable"]
    _c("tt.incl", tt.incl, ref["incl"])
    _c("tt.PRI", tt.PRI, ref["PRI"])


def test_truthtable_condition_bits():
    tt = _tt()
    ref = REF["truthTable"]
    for j, c in enumerate(["DEV", "URB", "LIT", "IND", "STB"]):
        _c(f"tt.bits.{c}", tt.rows[:, j], ref[c], tol=0)


# ---- minimization: term strings + per-term parameters of fit ----
def test_minimize_terms():
    m = minimize(_tt())
    ref = REF["minimize"]
    assert m["terms"] == ref["terms"], (m["terms"], ref["terms"])


def test_minimize_pof():
    m = minimize(_tt())
    ref = REF["minimize"]
    _c("min.inclS", m["inclS"], ref["inclS"])
    _c("min.PRI", m["PRI"], ref["PRI"])
    _c("min.covS", m["covS"], ref["covS"])
    _c("min.covU", m["covU"], ref["covU"])


def test_solution_overall_pof():
    m = minimize(_tt())
    ref = REF["overall"]
    _c("sol.inclS", m["overall"]["inclS"], ref["inclS"])
    _c("sol.PRI", m["overall"]["PRI"], ref["PRI"])
    _c("sol.covS", m["overall"]["covS"], ref["covS"])
