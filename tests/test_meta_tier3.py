"""Tier-3 meta-analysis — recovery + consistency checks for the advanced layer.

Network MA recovers a consistent network; Reitsma recovers generating sens/spec;
semi-analytic Bayes matches REML; dose-response recovers the slope; the rest get
run-and-shape coverage (validated statistically in the module smoke).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import socialverse as sv
from socialverse import StudyState, registry

warnings.simplefilter("ignore")


def test_all_tier3_registered():
    for n in ["nma_pairwise", "netmeta", "netrank", "nma_rankogram", "nma_inconsistency",
              "netsplit", "netcomb", "dta_descriptives", "dta_bivariate", "dta_glmm",
              "dosresmeta", "dosresmeta_spline", "ipd_twostage", "ipd_onestage",
              "bayesmeta", "bayes_metareg", "selection_model_stepfun", "pcurve", "puniform",
              "puniform_star", "pubbias_sensitivity", "pubbias_report", "metaforest",
              "ma_lrt", "ma_profile", "ma_cwb_test", "ma_cluster_influence",
              "metareg_multimodel", "ma_rho_sensitivity", "netgraph", "netheat", "sroc",
              "gosh", "dose_response_plot"]:
        assert registry.get(n) is not None, f"{n} not registered"


def _network(seed=0):
    rng = np.random.default_rng(seed); rows = []; sid = 0
    truth = {"A": -0.5, "B": 0.5, "C": 1.0}   # AB=1.0, BC=0.5, AC=1.5

    def arm(st, t):
        n = 250; p = 1 / (1 + np.exp(-truth[t])); return {"study": st, "treat": t, "events": int(rng.binomial(n, p)), "n": n}
    for pair in [("A", "B"), ("B", "C"), ("A", "C")]:
        for _ in range(8):
            sid += 1; rows += [arm(f"S{sid}", pair[0]), arm(f"S{sid}", pair[1])]
    s = StudyState(); s.write("sources", "datasets", pd.DataFrame(rows))
    sv.pp.nma_pairwise(s, study="study", treatment="treat", events="events", n="n")
    return s


def test_netmeta_recovers_consistent_network():
    s = _network()
    sv.tl.netmeta(s, reference="A")
    eff = s.models["nma"]["effects"]
    assert eff["B"]["vs_ref"] == pytest.approx(1.0, abs=0.25)
    assert eff["C"]["vs_ref"] == pytest.approx(1.5, abs=0.25)
    # consistency: no inconsistency on consistent data
    sv.tl.nma_inconsistency(s)
    assert s.diagnostics["nma_inconsistency"]["inconsistency_pval"] > 0.05


def test_netrank_and_rankogram_ordering():
    s = _network(); sv.tl.netmeta(s, reference="A")
    sv.tl.netrank(s, small_values="undesirable")   # higher logOR "better"
    ps = s.diagnostics["netrank"]["pscore"]
    assert ps["C"] > ps["B"] > ps["A"]
    sv.tl.nma_rankogram(s, small_values="undesirable", nsim=2000)
    su = s.diagnostics["rankogram"]["SUCRA"]
    assert su["C"] > su["A"]
    for v in su.values():
        assert 0 <= v <= 1


def test_netsplit_and_netcomb_run():
    s = _network(); sv.tl.netmeta(s, reference="A")
    sv.tl.netsplit(s)
    assert len(s.diagnostics["netsplit"]["comparisons"]) >= 1
    # component NMA on additive-named treatments
    data = s.models  # reuse arm data via a fresh mapping
    rng = np.random.default_rng(1); rows = []; sid = 0
    tr = {"X": -0.5, "X+Y": 0.5, "X+Z": 1.0}

    def arm(st, t):
        n = 250; p = 1 / (1 + np.exp(-tr[t])); return {"study": st, "treat": t, "events": int(rng.binomial(n, p)), "n": n}
    for pair in [("X", "X+Y"), ("X+Y", "X+Z"), ("X", "X+Z")]:
        for _ in range(8):
            sid += 1; rows += [arm(f"S{sid}", pair[0]), arm(f"S{sid}", pair[1])]
    s2 = StudyState(); s2.write("sources", "datasets", pd.DataFrame(rows))
    sv.pp.nma_pairwise(s2, study="study", treatment="treat", events="events", n="n")
    sv.tl.netcomb(s2)
    comp = s2.models["nma_components"]["effects"]
    assert comp["Y"]["estimate"] == pytest.approx(1.0, abs=0.3)   # X+Y vs X ≈ 1.0


def test_dta_bivariate_recovers_sens_spec():
    rng = np.random.default_rng(1); rows = []
    tse, tsp = 0.85, 0.80
    for _ in range(20):
        se_i = 1 / (1 + np.exp(-(np.log(tse / (1 - tse)) + rng.normal(0, 0.3))))
        sp_i = 1 / (1 + np.exp(-(np.log(tsp / (1 - tsp)) + rng.normal(0, 0.3))))
        npos, nneg = 100, 100
        tp = rng.binomial(npos, se_i); tn = rng.binomial(nneg, sp_i)
        rows.append({"tp": tp, "fn": npos - tp, "tn": tn, "fp": nneg - tn})
    s = StudyState(); s.write("sources", "datasets", pd.DataFrame(rows))
    sv.tl.dta_descriptives(s, tp="tp", fp="fp", fn="fn", tn="tn")
    sv.tl.dta_bivariate(s)
    b = s.models["dta_bivariate"]
    assert b["sensitivity"] == pytest.approx(0.85, abs=0.06)
    assert b["specificity"] == pytest.approx(0.80, abs=0.06)


def test_dosresmeta_recovers_slope():
    rng = np.random.default_rng(2); rows = []; slope = 0.15
    for st in range(12):
        for dose in [0, 5, 10, 20]:
            lr = slope * dose + rng.normal(0, 0.05)
            rows.append({"study": f"S{st}", "dose": dose,
                         "logrr": (lr if dose > 0 else 0.0), "se": (0.08 if dose > 0 else 0.01)})
    s = StudyState(); s.write("sources", "datasets", pd.DataFrame(rows))
    sv.tl.dosresmeta(s, study="study", dose="dose", logrr="logrr", se="se")
    assert s.models["dosres"]["slope_per_unit"] == pytest.approx(0.15, abs=0.03)


def test_bayesmeta_matches_reml():
    yi = np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0, 0.6, 0.3, 0.4, 0.7])
    vi = np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.05, 0.06, 0.04, 0.05])
    s = StudyState(); s.write("models", "meta_effects",
                              pd.DataFrame({"yi": yi, "vi": vi, "sei": np.sqrt(vi), "measure": "GEN"}))
    sv.tl.meta_random(s, method="REML"); reml = s.models["meta"]["estimate"]
    sv.tl.bayesmeta(s); bm = s.models["bayesmeta"]
    # flat-mean-prior Bayes posterior mean ≈ REML point estimate
    assert bm["mu_mean"] == pytest.approx(reml, abs=0.03)
    assert bm["mu_ci"][0] < bm["mu_mean"] < bm["mu_ci"][1]


def test_ipd_two_and_one_stage_agree():
    rng = np.random.default_rng(3); rows = []; te = 0.5
    for st in range(10):
        b = te + rng.normal(0, 0.15)
        for _ in range(60):
            t = rng.integers(0, 2); rows.append({"study": f"S{st}", "treat": t, "y": 1.0 + b * t + rng.normal(0, 1)})
    s = StudyState(); s.write("sources", "datasets", pd.DataFrame(rows))
    sv.tl.ipd_twostage(s, study="study", outcome="y", treatment="treat")
    two = s.models["meta"]["estimate"]
    sv.tl.ipd_onestage(s, study="study", outcome="y", treatment="treat")
    one = s.models["ipd"]["estimate"]
    assert abs(two - one) < 0.15


def test_selection_and_advanced_run():
    rng = np.random.default_rng(5); rows = []
    for st in range(14):
        for _ in range(rng.integers(1, 4)):
            x1 = rng.normal()
            rows.append({"yi": 0.3 + 0.4 * x1 + rng.normal(0, 0.2), "vi": rng.uniform(0.02, 0.06),
                         "study": f"S{st}", "x1": x1, "x2": rng.normal()})
    df = pd.DataFrame(rows); df["sei"] = np.sqrt(df["vi"]); df["measure"] = "GEN"
    s = StudyState(); s.write("models", "meta_effects", df)
    sv.tl.selection_model_stepfun(s)
    assert "mu_adjusted" in s.models["selection_model"]
    sv.tl.metaforest(s, moderators=["x1", "x2"])
    imp = s.diagnostics["metaforest"]["importance"]
    assert imp["x1"] > imp["x2"]                    # real moderator dominates
    sv.tl.ma_lrt(s, moderators=["x1", "x2"])
    assert s.diagnostics["ma_lrt"]["pval"] < 0.05
    sv.tl.ma_profile(s)
    lb, ub = s.diagnostics["ma_profile"]["tau2_ci"]; assert 0 <= lb <= ub
    sv.tl.metareg_multimodel(s, moderators=["x1", "x2"])
    assert s.diagnostics["metareg_multimodel"]["importance"]["x1"] > 0.5


def test_tier3_figures_emit(tmp_path):
    s = _network(); sv.tl.netmeta(s, reference="A")
    sv.pl.netgraph(s, out=str(tmp_path / "ng.png"))
    sv.pl.netheat(s, out=str(tmp_path / "nh.png"))
    for f in ("ng.png", "nh.png"):
        assert (tmp_path / f).stat().st_size > 1000
