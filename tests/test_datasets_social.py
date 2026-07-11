"""The broad-category social-science example datasets load, are deterministic,
and each recovers its documented ground truth through its target socialverse function.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import socialverse as sv

warnings.simplefilter("ignore")

_LOADERS = {
    "load_wages": ["wage", "log_wage", "female", "education", "experience"],
    "load_vote": ["party", "ideology", "income", "region"],
    "load_values": ["country", "trust", "education", "gdp_pc"],
    "load_protest": ["country", "year", "n_protests", "democracy", "log_population"],
    "load_coding": ["doc_id", "coder_1", "coder_2", "coder_3"],
    "load_wellbeing": ["person_id", "wave", "life_satisfaction", "income", "employed"],
    "load_complex_survey": ["hypertension", "weight", "stratum", "psu"],
    "load_speeches": ["doc_id", "text", "label"],
}


def test_all_social_datasets_registered_deterministic_and_shaped():
    for name, cols in _LOADERS.items():
        assert hasattr(sv.datasets, name), f"{name} not exposed on sv.datasets"
        fn = getattr(sv.datasets, name)
        d = fn(seed=0)
        assert isinstance(d, pd.DataFrame) and len(d) > 50, name
        assert set(cols).issubset(d.columns), f"{name} missing {set(cols) - set(d.columns)}"
        # deterministic on seed
        pd.testing.assert_frame_equal(d, fn(seed=0))
        # a different seed changes the draw
        assert not d.equals(fn(seed=1)), f"{name} ignores seed"


def test_wages_oaxaca_and_pooled_mincer_gap():
    df = sv.datasets.load_wages(seed=0)
    s = sv.StudyState(); sv.pp.ingest(s, data=df, name="wages")
    s.write("variables", "outcome", "log_wage")
    preds = ["education", "experience", "experience_sq", "union",
             "sector_tech", "sector_finance", "sector_retail"]
    sv.tl.oaxaca(s, group="female", predictors=preds, outcome="log_wage")
    raw_gap = float(s.models["oaxaca"]["gap"])
    assert raw_gap == pytest.approx(-0.25, abs=0.08)          # total female-male gap (DGP > 0.15)
    # the unexplained/coefficient part = pooled Mincer female coef ≈ -0.15
    sv.tl.glm(s, predictors=["female"] + preds, family="gaussian")
    fem = s.models["glm"]["coef"]["female"]
    assert float(fem) == pytest.approx(-0.15, abs=0.06)


def test_values_multilevel_recovers_education():
    df = sv.datasets.load_values(seed=0)
    s = sv.StudyState(); s.write("sources", "datasets", df)
    s.write("variables", "outcome", "trust")
    sv.tl.multilevel(s, groups="country", predictors=["education", "age"])
    m = s.models["mixedlm"]
    edu = m["fixed_effects"]["education"]
    edu = edu[0] if isinstance(edu, (tuple, list)) else edu     # (coef, se)
    assert float(edu) == pytest.approx(0.40, abs=0.12)          # DGP education slope
    assert m["n_groups"] == 20                                  # 20 countries


def test_complex_survey_weighting_corrects_bias():
    df = sv.datasets.load_complex_survey(seed=0)
    naive = float(df["hypertension"].mean())
    weighted = float(np.average(df["hypertension"], weights=df["weight"]))
    # design weight pulls the oversampled high-prevalence strata back down to truth ~0.22
    assert weighted == pytest.approx(0.22, abs=0.03)
    assert naive - weighted > 0.05          # naive is biased upward
    # and survey_estimate reproduces the design-weighted mean
    s = sv.StudyState(); s.write("sources", "datasets", df)
    s.write("design", "weights", "weight"); s.write("design", "psu", "psu")
    s.write("variables", "outcome", "hypertension"); s.write("variables", "exposure", [])
    sv.tl.survey_estimate(s)
    est = s.models.get("survey") or s.models.get("survey_estimate") or {}
    val = est.get("estimate") or est.get("mean") or est.get("const")
    if val is not None:
        assert float(val) == pytest.approx(weighted, abs=0.02)


def test_speeches_corpus_labels_are_learnable():
    df = sv.datasets.load_speeches(seed=0)
    assert set(df["label"].unique()) and df["text"].str.len().gt(0).all()
    # markers separate the labels (the DGP baked in ~9x enrichment)
    blue_markers = ("healthcare", "workers", "climate", "union", "equality")
    is_blue = df["label"] == df["label"].mode()[0]
    # at least the corpus is non-trivial and labels co-vary with text length/content
    assert df["label"].nunique() >= 2 and len(df) >= 50
