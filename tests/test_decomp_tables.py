"""Decomposition estimators + publication tables.

The correctness proofs are exact identities: the Goodman-Bacon weights must
reconstruct the TWFE estimate, and the Oaxaca components must sum to the raw
between-group gap.
"""
import numpy as np
import pandas as pd
import pytest

import socialverse as sv
from socialverse import datasets


def _did_state(df):
    st = sv.StudyState()
    st.write("variables", "outcome", "y")
    st.write("estimand", "target", "ATT")
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")
    return st


def test_bacon_weights_reconstruct_twfe():
    df = datasets.load_did_staggered(n_units=150, n_periods=10, att=2.0, seed=1)
    st = _did_state(df)
    sv.tl.bacon_decompose(st)
    b = st.diagnostics["bacon"]
    # the decomposition is exact: Σ weight·(2x2) == TWFE
    assert b["reconstruction_ok"] is True
    assert b["reconstructed"] == pytest.approx(b["twfe_att"], abs=1e-6)
    assert 0.0 <= b["forbidden_weight"] <= 1.0
    assert abs(sum(c["weight"] for c in b["comparisons"]) - 1.0) < 1e-9
    assert "later_vs_earlier_forbidden" in b["by_type"]


def test_bacon_twfe_correct_on_unbalanced_panel():
    # dropping cells must not corrupt the reported TWFE (no nan_to_num on absent cells)
    import statsmodels.api as sm
    df = datasets.load_did_staggered(n_units=120, n_periods=10, att=2.0, seed=4)
    rng = np.random.default_rng(9)
    df = df[rng.random(len(df)) > 0.1].reset_index(drop=True)  # 10% missing -> unbalanced
    st = _did_state(df)
    sv.tl.bacon_decompose(st)
    got = st.diagnostics["bacon"]["twfe_att"]
    # independent TWFE: OLS of y on treat_post + unit + time dummies on observed rows
    X = pd.concat([df[["treat_post"]].astype(float),
                   pd.get_dummies(df["unit"], prefix="u", drop_first=True, dtype=float),
                   pd.get_dummies(df["period"], prefix="t", drop_first=True, dtype=float)], axis=1)
    ref = sm.OLS(df["y"].to_numpy(float), sm.add_constant(np.asarray(X, float))).fit().params[1]
    assert got == pytest.approx(float(ref), abs=1e-4)


def test_oaxaca_components_sum_to_gap():
    rng = np.random.default_rng(0)
    n = 3000
    gA = rng.integers(0, 2, n)
    edu = rng.normal(0, 1, n) + 0.5 * gA        # endowment gap
    wage = np.where(gA == 1, 1.0 + 2.0 * edu, 0.5 + 1.0 * edu) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"group": gA, "edu": edu, "wage": wage})
    st = sv.StudyState()
    st.write("variables", "outcome", "wage")
    sv.pp.ingest(st, data=df)
    sv.tl.oaxaca(st, group="group", predictors=["edu"])
    o = st.models["oaxaca"]
    assert o["threefold"]["sum"] == pytest.approx(o["gap"], abs=1e-6)
    assert o["twofold"]["sum"] == pytest.approx(o["gap"], abs=1e-6)
    # both endowment and coefficient components are positive in this DGP
    assert o["threefold"]["endowments"] > 0
    assert o["threefold"]["coefficients"] > 0


def test_regtable_renders_multi_model():
    df = datasets.load_did_staggered(n_units=120, n_periods=10, att=2.0, seed=2)
    st = _did_state(df)
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    sv.tl.fect(st, r=0, nboots=30)
    for fmt in ("text", "markdown", "latex"):
        sv.pl.regtable(st, models=[("TWFE", st.models["did"]),
                                   ("FEct", st.models["fect"])], format=fmt)
        content = st.artifacts["tables"]["content"]
        assert "TWFE" in content and "FEct" in content and "ATT" in content
    # latex uses booktabs
    sv.pl.regtable(st, models=[("TWFE", st.models["did"])], format="latex")
    assert r"\toprule" in st.artifacts["tables"]["content"]


def test_regtable_escapes_latex_specials():
    st = sv.StudyState()
    # a model whose coefficient term contains a LaTeX-special underscore
    model = {"coefficients": {"age_group": {"coef": 1.23, "se": 0.4, "p": 0.01}}, "n": 100}
    sv.pl.regtable(st, models=[("m1", model)], format="latex")
    out = st.artifacts["tables"]["content"]
    assert r"age\_group" in out and "age_group " not in out  # underscore escaped


def test_regtable_defaults_to_state_models():
    df = datasets.load_did_staggered(n_units=120, n_periods=10, seed=3)
    st = _did_state(df)
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    sv.pl.regtable(st)  # no models= -> use state.models
    assert st.artifacts["tables"]["content"]
