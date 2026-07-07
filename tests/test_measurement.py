"""Measurement-gap methods (EFA, scale reliability, inter-rater) recover their truths.

Contributed back after the humanities skills needed them: survey-design (EFA +
scale reliability) and qualitative content analysis (inter-coder reliability).
"""
import warnings

import pytest

import socialverse as sv
from socialverse import datasets as ds

warnings.simplefilter("ignore")


def _state(df):
    s = sv.StudyState()
    sv.pp.ingest(s, data=df)
    return s


def test_measurement_functions_registered():
    for name in ["efa", "reliability", "interrater"]:
        assert sv.registry.get(name) is not None, f"{name} not registered"


# ------------------------------------------------------------------------ EFA
def test_efa_recovers_single_factor():
    s = _state(ds.load_survey())          # item1..6 load on one factor
    sv.tl.efa(s)
    m = s.models["efa"]
    assert m["n_factors"] == 1                          # Kaiser: one eigenvalue > 1
    assert m["eigenvalues"][0] > 3 * m["eigenvalues"][1]  # first factor dominates
    assert m["kmo"] > 0.6                               # adequate sampling
    assert m["bartlett_p"] < 0.05                       # correlations not identity


# ---------------------------------------------------------------- reliability
def test_reliability_high_for_coherent_scale():
    s = _state(ds.load_survey())
    sv.tl.reliability(s)
    r = s.diagnostics["reliability"]
    assert 0.8 < r["cronbach_alpha"] < 1.0             # coherent 6-item scale
    assert r["mcdonald_omega"] == pytest.approx(r["cronbach_alpha"], abs=0.1)
    assert r["avg_inter_item_r"] > 0.3
    assert set(r["item_total"]) == {f"item{i}" for i in range(1, 7)}


# ------------------------------------------------------------------ interrater
def test_interrater_substantial_agreement():
    s = _state(ds.load_ratings())          # 3 raters, agree=0.8 → substantial
    sv.tl.interrater(s)
    i = s.diagnostics["interrater"]
    assert i["n_raters"] == 3
    assert 0.4 < i["fleiss_kappa"] < 0.85              # substantial, not perfect
    assert 0.4 < i["krippendorff_alpha"] < 0.85
    # the two nominal-agreement indices should roughly agree
    assert abs(i["fleiss_kappa"] - i["krippendorff_alpha"]) < 0.15


def test_interrater_perfect_agreement_is_one():
    import pandas as pd
    df = pd.DataFrame({"r1": [0, 1, 2, 1, 0], "r2": [0, 1, 2, 1, 0], "r3": [0, 1, 2, 1, 0]})
    s = _state(df)
    sv.tl.interrater(s, raters=["r1", "r2", "r3"])
    i = s.diagnostics["interrater"]
    assert i["percent_agreement"] == pytest.approx(1.0)
    assert i["krippendorff_alpha"] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------ py- aliases
@pytest.mark.parametrize("alias,expected", [
    ("py-factor", "efa"), ("py-fa", "efa"), ("py-principal", "efa"),
    ("py-omega", "reliability"), ("py-mcdonald", "reliability"),
    ("py-kappa", "interrater"), ("py-krippalpha", "interrater"), ("py-irr", "interrater"),
])
def test_measurement_aliases(alias, expected):
    e = sv.registry.get(alias)
    assert e is not None and e.name == expected
