import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from pynetmeta import netmeta

REF = json.loads((HERE / "reference.json").read_text())
TOL = 1e-6


def _c(name, got, exp, tol=TOL):
    got = np.atleast_1d(np.asarray(got, float))
    exp = np.atleast_1d(np.asarray(exp, float))
    err = float(np.max(np.abs(got - exp)))
    assert err < tol, f"{name}: max_abs_err={err:.2e}\n got={got}\n exp={exp}"


def _fit():
    d = REF["data"]
    return netmeta(d["TE"], d["seTE"], d["treat1"], d["treat2"], d["studlab"],
                   reference_group="plac")


def test_trts_order():
    assert _fit().trts == REF["trts"]


def test_TE_fixed_matrix():
    net = _fit()
    _c("TE.fixed", net.TE_fixed, REF["TE_fixed"])


def test_seTE_fixed_matrix():
    net = _fit()
    _c("seTE.fixed", net.seTE_fixed, REF["seTE_fixed"])


def test_TE_random_matrix():
    net = _fit()
    _c("TE.random", net.TE_random, REF["TE_random"])


def test_seTE_random_matrix():
    net = _fit()
    _c("seTE.random", net.seTE_random, REF["seTE_random"])


def test_Q():
    net = _fit()
    _c("Q", net.Q, REF["Q"])
    _c("df.Q", net.df_Q, REF["df_Q"])
    _c("pval.Q", net.pval_Q, REF["pval_Q"])


def test_tau2():
    net = _fit()
    _c("tau2", net.tau2, REF["tau2"])
    _c("tau", net.tau, REF["tau"])


def test_reference_column():
    # metf vs plac, both effects, against the R matrices
    net = _fit()
    ref_te_f = np.asarray(REF["TE_fixed"], float)
    ref_te_r = np.asarray(REF["TE_random"], float)
    trts = REF["trts"]
    i, j = trts.index("metf"), trts.index("plac")
    te_f, se_f = net.comparison("metf", "plac", random=False)
    te_r, se_r = net.comparison("metf", "plac", random=True)
    _c("metf.plac.fixed.TE", te_f, ref_te_f[i, j])
    _c("metf.plac.random.TE", te_r, ref_te_r[i, j])
