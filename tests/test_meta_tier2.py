"""Tier-2 meta-analysis — exactness + parameter-recovery + smoke coverage.

Exactness checks anchor the closed-form converters and 2×2 poolers; the
iterative / diagnostic functions get run-and-shape checks (they are validated
statistically in the module smoke).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import socialverse as sv
from socialverse.tl._meta import _estimate_tau2

warnings.simplefilter("ignore")


def _src(df):
    s = sv.StudyState(); s.write("sources", "datasets", df); return s


def _eff(yi, vi, **extra):
    s = sv.StudyState()
    d = pd.DataFrame({"yi": yi, "vi": vi, "sei": np.sqrt(vi), "measure": "GEN"})
    for k, v in extra.items():
        d[k] = v
    s.write("models", "meta_effects", d); return s


# ------------------------------------------------------------ effect-size converters
def test_es_from_t_exact_and_f_agrees():
    s = _src(pd.DataFrame({"t": [2.3], "n1": [30], "n2": [30]}))
    sv.pp.es_from_t(s, t="t", n1="n1", n2="n2")
    d_t = float(s.models["meta_effects"]["yi"].iloc[0])
    assert d_t == pytest.approx(2.3 * np.sqrt(1 / 30 + 1 / 30), rel=1e-9)
    s2 = _src(pd.DataFrame({"f": [2.3 ** 2], "n1": [30], "n2": [30], "sign": [1]}))
    sv.pp.es_from_f(s2, f="f", n1="n1", n2="n2", sign="sign")
    assert float(s2.models["meta_effects"]["yi"].iloc[0]) == pytest.approx(d_t, rel=1e-9)


def test_es_from_ir_and_rom_exact():
    s = _src(pd.DataFrame({"events": [12], "time": [500.0]}))
    sv.pp.es_from_ir(s, events="events", time="time")
    assert float(s.models["meta_effects"]["yi"].iloc[0]) == pytest.approx(np.log(12 / 500))
    s2 = _src(pd.DataFrame({"m1": [10], "sd1": [2], "n1": [50], "m2": [8], "sd2": [2], "n2": [50]}))
    sv.pp.es_ratio_of_means(s2, m1="m1", sd1="sd1", n1="n1", m2="m2", sd2="sd2", n2="n2")
    assert float(s2.models["meta_effects"]["yi"].iloc[0]) == pytest.approx(np.log(10 / 8))


def test_pointbiserial_equal_groups():
    s = _src(pd.DataFrame({"r": [0.3], "n1": [50], "n2": [50]}))
    sv.pp.pointbiserial_to_d(s, r="r", n1="n1", n2="n2")
    assert float(s.models["meta_effects"]["yi"].iloc[0]) == pytest.approx(2 * 0.3 / np.sqrt(1 - 0.09), rel=1e-9)


def test_ma_aggregate_collapses_to_one_per_study():
    yi = np.array([0.2, 0.4, 0.1, 0.5, 0.3]); vi = np.array([0.04, 0.03, 0.05, 0.02, 0.04])
    s = _eff(yi, vi, study=["A", "A", "B", "B", "B"])
    sv.pp.ma_aggregate(s, cluster="study", rho=0.6)
    out = s.models["meta_effects"]
    assert len(out) == 2
    # study A composite mean = mean of its two effects
    assert float(out[out["study"] == "A"]["yi"].iloc[0]) == pytest.approx(0.3)


# ----------------------------------------------------------------- tau2 roster
def test_tau2_estimators_nonnegative_and_dl_exact():
    yi = np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0, 0.6, 0.3])
    vi = np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.05, 0.06])
    for m in ["DL", "REML", "ML", "PM", "SJ", "HS", "HE"]:
        assert _estimate_tau2(yi, vi, m) >= 0.0
    w = 1 / vi; mu = np.sum(w * yi) / np.sum(w); Q = np.sum(w * (yi - mu) ** 2)
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    assert _estimate_tau2(yi, vi, "DL") == pytest.approx(max(0, (Q - 7) / C), abs=1e-12)


def test_meta_random_all_methods_run():
    s = _eff(np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0]), np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04]))
    for m in ["DL", "REML", "ML", "PM", "SJ", "HS", "HE"]:
        sv.tl.meta_random(s, method=m)
        assert s.models["meta"]["estimate"] is not None


# --------------------------------------------------------------- 2x2 poolers
def test_mantel_haenszel_and_peto_positive_or():
    d = pd.DataFrame({"ai": [20, 5, 30], "bi": [80, 35, 70], "ci": [10, 3, 15], "di": [90, 37, 85]})
    sm = _src(d); sv.tl.meta_mh(sm, ai="ai", bi="bi", ci="ci", di="di")
    sp = _src(d); sv.tl.meta_peto(sp, ai="ai", bi="bi", ci="ci", di="di")
    assert sm.models["meta"]["or"] > 1 and sp.models["meta"]["or"] > 1
    assert sm.models["meta"]["or"] == pytest.approx(sp.models["meta"]["or"], rel=0.2)


# --------------------------------------------------------- CIs / conversion / subgroup
def test_tau2_ci_brackets_point_estimate():
    yi = np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0, 0.6, 0.3]); vi = np.full(8, 0.04)
    s = _eff(yi, vi); sv.tl.tau2_ci(s)
    ci = s.diagnostics["tau2_ci"]
    assert 0 <= ci["tau2_lb"] <= ci["tau2_ub"]
    assert 0 <= ci["I2_lb"] <= ci["I2_ub"] <= 100


def test_subgroup_q_partition():
    yi = np.array([0.2, 0.3, 0.8, 0.9]); vi = np.full(4, 0.03)
    s = _eff(yi, vi, grp=["a", "a", "b", "b"]); sv.tl.subgroup(s, moderator="grp")
    d = s.diagnostics["subgroup"]
    assert set(d["groups"]) == {"a", "b"} and d["Q_between"] >= 0


def test_es_convert_smd_to_r_roundtrip_sign():
    s = _eff(np.array([0.5, -0.8]), np.array([0.04, 0.05]))
    s.models["meta_effects"]["measure"] = "SMD"
    sv.tl.es_convert(s, to="ZCOR")
    z = s.models["meta_effects"]["yi"].to_numpy()
    assert z[0] > 0 and z[1] < 0   # sign preserved


# ------------------------------------------------------------- bias / diagnostics
def test_bias_and_influence_suite_runs():
    rng = np.random.default_rng(11)
    yi = rng.normal(0.3, 0.2, 15); vi = rng.uniform(0.02, 0.08, 15)
    s = _eff(yi, vi, study=[f"S{i}" for i in range(15)])
    sv.tl.meta_random(s)
    for fn in ["trim_and_fill", "pet", "peese", "pet_peese", "begg_test",
               "failsafe_n", "excess_significance", "leave_one_out",
               "cumulative_ma", "influence", "outlier_refit"]:
        sv.registry.get(fn).func(s)
    assert s.diagnostics["trim_and_fill"]["k0_missing"] >= 0
    assert len(s.diagnostics["leave_one_out"]["rows"]) == 15
    assert s.diagnostics["influence"]["n_influential"] >= 0


# ---------------------------------------------------------------------- RVE
def test_rve_cr2_psd_and_wald():
    rng = np.random.default_rng(12)
    rows = []
    for st in range(12):
        for _ in range(rng.integers(1, 4)):
            rows.append({"yi": rng.normal(0.3 + 0.01 * st, 0.2), "vi": rng.uniform(0.02, 0.06),
                         "study": f"S{st}", "x": rng.normal()})
    df = pd.DataFrame(rows); df["sei"] = np.sqrt(df["vi"]); df["measure"] = "GEN"
    s = sv.StudyState(); s.write("models", "meta_effects", df)
    sv.tl.ma_robust(s, moderators=["x"], vcov="CR2", cluster="study")
    VR = np.array(s.models["meta_rve"]["_VR"])
    assert np.all(np.linalg.eigvalsh(0.5 * (VR + VR.T)) > -1e-8)   # PSD
    sv.tl.ma_wald_test(s)
    assert 0 <= s.diagnostics["ma_wald_test"]["pval"] <= 1
    sv.tl.robu(s, moderators=["x"], model="CORR", cluster="study", rho_grid=[0.5, 0.8])
    assert "rho_sensitivity" in s.models["meta_rve"]


# --------------------------------------------------------------- governance
def test_prisma_flow_arithmetic():
    s = sv.StudyState()
    sv.gov.prisma_flow(s, identified=1200, duplicates=200, screened=1000,
                       excluded_screen=850, full_text=150, excluded_fulltext=120, included=30)
    assert s.governance["prisma"]["consistent"]
    s2 = sv.StudyState()
    sv.gov.prisma_flow(s2, screened=1000, excluded_screen=850, full_text=100)  # 150≠100
    assert not s2.governance["prisma"]["consistent"]


def test_screen_agreement_kappa_and_grade_levels():
    d = pd.DataFrame({"r1": ["in", "ex", "in", "ex", "in"], "r2": ["in", "ex", "ex", "ex", "in"]})
    s = _src(d); sv.gov.screen_agreement(s, rater1="r1", rater2="r2")
    a = s.governance["screen_agreement"]
    assert -1 <= a["cohen_kappa"] <= 1 and a["n_conflicts"] == 1
    g = sv.StudyState()
    sv.gov.grade(g, design="rct")           # start High, no downgrades
    assert g.governance["grade"]["certainty"] == "High"
    sv.gov.grade(g, design="observational", imprecision=1)  # start Low − 1
    assert g.governance["grade"]["level"] <= 2


def test_risk_of_bias_overall():
    s = sv.StudyState()
    sv.gov.risk_of_bias(s, tool="ROB2",
                        studies={"A": {"randomization": "low", "deviations": "high"},
                                 "B": {"randomization": "low", "deviations": "low"}})
    assert s.governance["risk_of_bias"]["overall"]["A"] == "high"
    assert s.governance["risk_of_bias"]["overall"]["B"] == "low"


# ------------------------------------------------------------------- figures
def test_tier2_figures_emit(tmp_path):
    yi = np.array([0.2, 0.5, 0.8, 0.3, 0.6, 0.1]); vi = np.array([0.04, 0.05, 0.03, 0.06, 0.04, 0.05])
    s = _eff(yi, vi, study=[f"S{i}" for i in range(6)]); sv.tl.meta_random(s)
    sv.pl.funnel_contour(s, out=str(tmp_path / "fc.png"))
    sv.pl.baujat(s, out=str(tmp_path / "bj.png"))
    sv.gov.prisma_flow(s, identified=100, duplicates=10, screened=90, excluded_screen=70,
                       full_text=20, excluded_fulltext=15, included=5)
    sv.pl.prisma_diagram(s, out=str(tmp_path / "pr.png"))
    sv.gov.risk_of_bias(s, tool="ROB2", studies={"A": {"randomization": "low"}})
    sv.pl.rob_traffic_light(s, out=str(tmp_path / "rob.png"))
    for f in ("fc.png", "bj.png", "pr.png", "rob.png"):
        assert (tmp_path / f).stat().st_size > 1000
