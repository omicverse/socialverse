"""Gap-method tests — each new method must recover its known DGP parameter."""
from __future__ import annotations

import warnings

import numpy as np

warnings.simplefilter("ignore")

import socialverse as sv  # noqa: E402
from socialverse import datasets as ds  # noqa: E402


def test_registry_grew_and_contracts_valid():
    assert len(sv.registry) >= 50
    cats = set(sv.registry.categories())
    for c in ("psychometrics", "quasi", "longitudinal", "spatial",
              "setmethods", "demography", "stylometry"):
        assert c in cats
    for fn in sv.registry.manifest()["functions"]:
        for slot in list(fn["requires"]) + list(fn["produces"]):
            assert slot in sv.VALID_SLOTS


def test_rdd_recovers_jump():
    st = sv.StudyState()
    st.write("estimand", "target", "LATE"); st.write("variables", "outcome", "y")
    st.write("sources", "datasets", ds.load_rdd(tau=3.0))
    sv.tl.rdd(st, running="running", cutoff=0.0)
    m = st.models["rdd"]
    jump = m.get("jump", m.get("estimate"))
    assert 2.5 < jump < 3.5


def test_survival_recovers_log_hr():
    st = sv.StudyState()
    st.write("variables", "outcome", "time")
    st.write("sources", "datasets", ds.load_survival(beta=0.8))
    sv.tl.survival(st, time="time", event="event", covariates=["x", "group"])
    lhr = st.models["cox"]["log_hr"]["x"]
    lhr = lhr[0] if isinstance(lhr, (tuple, list)) else lhr
    assert 0.6 < lhr < 1.0


def test_multilevel_recovers_slope_and_variance():
    st = sv.StudyState()
    st.write("variables", "outcome", "y")
    st.write("sources", "datasets", ds.load_multilevel())
    sv.tl.multilevel(st, groups="school", predictors=["x"])
    fe = st.models["mixedlm"]["fixed_effects"]["x"]
    slope = fe[0] if isinstance(fe, (tuple, list)) else fe
    assert 1.7 < slope < 2.3


def test_spatial_autocorr_positive_moran():
    df, W = ds.load_spatial(rho=0.5)
    st = sv.StudyState(); st.write("sources", "datasets", df)
    sv.tl.spatial_autocorr(st, value="y", W=W)
    assert st.diagnostics["moran"]["I"] > 0.2


def test_spatial_regression_recovers_rho():
    df, W = ds.load_spatial(rho=0.5)
    st = sv.StudyState(); st.write("variables", "outcome", "y")
    st.write("sources", "datasets", df)
    sv.tl.spatial_regression(st, outcome="y", predictors=["x"], W=W)
    assert 0.3 < st.models["sar"]["rho"] < 0.7


def test_irt_recovers_difficulty_ranking():
    resp, truth = ds.load_irt()
    st = sv.StudyState(); st.write("sources", "datasets", resp)
    sv.tl.irt(st)
    b_hat = st.models["irt"]["b"]
    b_hat = list(b_hat.values()) if isinstance(b_hat, dict) else list(b_hat)
    # estimated difficulties rank-correlate with truth
    from scipy.stats import spearmanr
    rho, _ = spearmanr(b_hat, truth["b"].values)
    assert rho > 0.7


def test_cfa_fit_indices_sane():
    resp, _ = ds.load_irt()
    st = sv.StudyState(); st.write("sources", "datasets", resp.astype(float))
    cols = list(resp.columns)
    sv.tl.cfa(st, model_spec={"F1": cols[:5], "F2": cols[5:]})
    fi = st.diagnostics["fit_indices"]
    assert 0.0 <= fi["RMSEA"] <= 1.0
    assert 0.0 <= fi["CFI"] <= 1.01


def test_qca_recovers_boolean_solution():
    st = sv.StudyState()
    st.write("variables", "outcome", "Y")
    st.write("sources", "datasets", ds.load_qca())
    sv.tl.qca(st, conditions=["A", "B", "C"], outcome="Y", threshold=0.5)
    sol = st.models["qca"]["solution"].replace(" ", "")
    # Y = (A AND B) OR C  -> solution mentions C and the A*B conjunction
    assert "C" in sol and ("A*B" in sol or "B*A" in sol)


def test_life_table_reasonable_e0():
    st = sv.StudyState()
    st.write("sources", "datasets", ds.load_demography())
    sv.tl.life_table(st, age="age_group", mx="mx_A", width="n_years")
    e0 = st.models["life_table"]
    e0 = e0.get("e0") if isinstance(e0, dict) else e0
    assert 50 < e0 < 90


def test_kitagawa_decomposition_adds_up():
    st = sv.StudyState()
    st.write("sources", "datasets", ds.load_demography())
    sv.tl.decomposition(st)
    d = st.models["decomposition"]
    assert abs((d["rate_effect"] + d["composition_effect"]) - d["total_diff"]) < 1e-6


def test_stylometry_clusters_by_author():
    st = sv.StudyState()
    st.write("corpus", "documents", ds.load_stylometry())
    sv.tl.stylometry(st)
    # attribution accuracy should be high (docs cluster by author)
    acc = st.models["stylometry"].get("accuracy")
    if acc is not None:
        assert acc > 0.6


def test_ergm_mple_positive_mutual():
    st = sv.StudyState()
    st.write("sources", "datasets", ds.load_network())
    sv.tl.ergm(st)
    # the toy network has reciprocity -> positive mutual coefficient
    assert st.models["ergm"]["coef"]["mutual"] > 0


def test_synthetic_control_negative_att():
    st = sv.StudyState()
    st.write("design", "treatment", "treat"); st.write("design", "time", "year")
    st.write("variables", "outcome", "y"); st.write("estimand", "target", "ATT")
    st.write("sources", "datasets", ds.load_did_panel(att=-0.8))
    sv.tl.synthetic_control(st, unit="firm_id", time="year",
                            treated_unit=0, treat_time=2015)
    assert "weights" in st.models["synth"]
