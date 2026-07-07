"""Modern causal-inference additions: DAG four-step, DML/CATE, modern DiD family.

Each test pins a known-truth synthetic DGP: identification returns the right
adjustment set and the estimators recover the true (possibly heterogeneous or
dynamic) effect where the naive/TWFE estimator is biased.
"""
import numpy as np
import pandas as pd
import pytest

import socialverse as sv
from socialverse import datasets


def _fit(fn, df, design=None, **kw):
    st = sv.StudyState()
    sv.pp.ingest(st, data=df)
    if design:
        for k, v in design.items():
            slot = "variables" if k == "outcome" else "design"
            st.write(slot, k, v)
    fn(st, **kw)
    return st


# ------------------------------------------------------------------- DAG four-step
def _confounded(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, n)
    T = 0.7 * Z + rng.normal(0, 1, n)
    Y = 1.5 * T + 2.0 * Z + rng.normal(0, 1, n)  # true ATE 1.5, backdoor {Z}
    return pd.DataFrame({"Z": Z, "T": T, "Y": Y})


def test_dag_identify_finds_backdoor_and_ate():
    st = _fit(sv.tl.dag_identify, _confounded(),
              design={"treatment": "T", "outcome": "Y"}, graph="Z->T; Z->Y; T->Y")
    est = st.identification["estimand"]
    assert est["strategy"] == "backdoor"
    assert est["adjustment_set"] == ["Z"]
    assert st.models["dag"]["ate"] == pytest.approx(1.5, abs=0.15)


def test_dag_refute_placebo_kills_effect():
    st = _fit(sv.tl.dag_identify, _confounded(),
              design={"treatment": "T", "outcome": "Y"}, graph="Z->T; Z->Y; T->Y")
    sv.tl.dag_refute(st, seed=1)
    ref = st.diagnostics["refutation"]
    checks = {c["refuter"]: c for c in ref["checks"]}
    assert abs(checks["placebo_treatment"]["new_estimate"]) < 0.15       # placebo ~ 0
    assert checks["random_common_cause"]["new_estimate"] == pytest.approx(1.5, abs=0.2)
    assert ref["verdict"] == "robust"


def test_dag_no_backdoor_without_confounder_control():
    # empty adjustment set must be rejected as insufficient when Z confounds
    from socialverse.tl._dag import _minimal_backdoor, _parse_graph
    ch = _parse_graph("Z->T; Z->Y; T->Y")
    mini, valid = _minimal_backdoor(ch, "T", "Y", {"Z"})
    assert mini == {"Z"} and set() not in valid


def test_dag_rejects_cyclic_graph():
    # a graph with a directed cycle is not a DAG -> refuse, don't emit a pseudo-effect
    st = _fit(sv.tl.dag_identify, _confounded(),
              design={"treatment": "T", "outcome": "Y"},
              graph="C1->C2; C2->C1; C1->T; C1->Y; T->Y")
    assert st.models["dag"]["ate"] is None
    assert "环" in st.models["dag"]["note"] or "DAG" in st.models["dag"]["note"]


def test_dag_string_binary_treatment_estimates():
    # string/categorical treatment must still yield an ATE (not silently drop to None)
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.normal(0, 1, n)
    Tb = (0.7 * Z + rng.normal(0, 1, n) > 0)
    Y = 1.5 * Tb + 2.0 * Z + rng.normal(0, 1, n)
    df = pd.DataFrame({"Z": Z, "T": np.where(Tb, "treated", "control"), "Y": Y})
    st = _fit(sv.tl.dag_identify, df, design={"treatment": "T", "outcome": "Y"},
              graph="Z->T; Z->Y; T->Y")
    assert st.models["dag"]["ate"] is not None
    sv.tl.dag_refute(st, seed=1)
    assert st.diagnostics["refutation"]["verdict"] == "robust"


def test_dag_frontdoor_multi_mediator():
    from socialverse.tl._dag import _frontdoor, _parse_graph
    # U latent confounds T,Y; two mediators fully intercept T->Y
    ch = _parse_graph("U->T; U->Y; T->M1; T->M2; M1->Y; M2->Y")
    M = _frontdoor(ch, "T", "Y", {"M1", "M2"})  # U unobserved
    assert M == {"M1", "M2"}


def test_dml_binary_non01_treatment():
    # treatment coded {1,2} must give the same ATE as {0,1} (internal recode)
    df, true_ate = _cate_dgp(n=2000)
    df["T"] = (df["T"] > df["T"].median()).astype(int) + 1  # -> {1,2}
    # rebuild Y consistent with a binary treatment effect of 2.0
    rng = np.random.default_rng(5)
    df["Y"] = 2.0 * (df["T"] - 1) + 2.0 * df["X0"] + rng.normal(0, 1, len(df))
    st = _fit(sv.tl.dml, df, design={"treatment": "T", "outcome": "Y"},
              hetero=["X0"], folds=3, seed=0)
    assert st.models["dml"]["ate"] == pytest.approx(2.0, abs=0.3)


# --------------------------------------------------------------------------- DML
def _cate_dgp(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    T = 0.8 * X[:, 0] + rng.normal(0, 1, n)
    theta = 1.0 + 0.5 * X[:, 0]                       # heterogeneous
    Y = theta * T + 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 1, n)
    return pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "T": T, "Y": Y}), float(theta.mean())


def test_dml_recovers_ate_and_cate():
    df, true_ate = _cate_dgp()
    st = _fit(sv.tl.dml, df, design={"treatment": "T", "outcome": "Y"},
              hetero=["X0"], controls=["X1"], folds=3, discrete_treatment=False, seed=0)
    m = st.models["dml"]
    assert m["ate"] == pytest.approx(true_ate, abs=0.2)
    # CATE slope on X0 recovered (true 0.5); naive OLS slope would be biased high
    assert m["cate_linear"]["X0"] == pytest.approx(0.5, abs=0.2)
    naive = float(np.polyfit(df["T"], df["Y"], 1)[0])
    assert abs(m["ate"] - true_ate) < abs(naive - true_ate)


def test_causal_forest_finds_effect_modifier():
    df, true_ate = _cate_dgp(n=2000)
    st = _fit(sv.tl.causal_forest, df, design={"treatment": "T", "outcome": "Y"},
              hetero=["X0", "X1"], folds=3, nboots=10, seed=0)
    m = st.models["causal_forest"]
    assert m["ate"] == pytest.approx(true_ate, abs=0.3)
    # X0 is the true effect modifier -> highest importance; CATE spreads
    assert m["feature_importance"]["X0"] > m["feature_importance"]["X1"]
    assert m["cate_summary"]["p90"] - m["cate_summary"]["p10"] > 0.3


# -------------------------------------------------------------------- modern DiD
def _staggered(seed=0):
    df = datasets.load_did_staggered(n_units=140, n_periods=12, att=2.0, seed=seed)
    return df, df.attrs["true_att"]


def _did_design(st):
    st.write("variables", "outcome", "y")
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")


def test_sun_abraham_flat_pretrends_growing_effect():
    df, _ = _staggered()
    st = sv.StudyState(); sv.pp.ingest(st, data=df); _did_design(st)
    sv.tl.sun_abraham(st)
    coefs = st.models["sun_abraham"]["coefs"]
    # pre-trends near zero
    for k in ("-3", "-2"):
        if k in coefs:
            assert abs(coefs[k][0]) < 0.3
    # dynamic effect grows post-treatment
    assert coefs["0"][0] < coefs["2"][0]
    assert coefs["2"][0] > 1.0


def test_did2s_recovers_att():
    df, true_att = _staggered(seed=2)
    st = sv.StudyState(); sv.pp.ingest(st, data=df); _did_design(st)
    sv.tl.did2s(st, nboots=60)
    m = st.models["did2s"]
    assert m["att"] == pytest.approx(true_att, abs=0.2)
    assert m["se"] is not None and m["se"] > 0


def test_local_projection_traces_irf():
    df, _ = _staggered(seed=1)
    st = sv.StudyState(); sv.pp.ingest(st, data=df); _did_design(st)
    sv.tl.local_projection(st, max_horizon=4)
    coefs = st.models["local_projection"]["coefs"]
    assert coefs["0"][0] < coefs["3"][0]      # impulse response grows
    assert coefs["3"][0] > 1.0
