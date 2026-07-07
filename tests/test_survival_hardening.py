"""sv.tl.survival robustness — surfaced by reproducing the Rossi Cox model.

`duration=` is accepted as an alias for `time=` (intuitive), and a time column
equal to the event column no longer crashes with a cryptic pandas error.
"""
import warnings

import numpy as np
import pandas as pd

import socialverse as sv

warnings.simplefilter("ignore")


def _toy_survival(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = rng.exponential(np.exp(-0.4 * x)) * 8.0
    event = (t < 8.0).astype(int)
    t = np.minimum(t, 8.0)
    return pd.DataFrame({"t": t, "d": event, "x": x})


def test_survival_accepts_duration_alias():
    df = _toy_survival()
    s = sv.StudyState()
    sv.pp.ingest(s, data=df)
    s.write("variables", "outcome", "d")
    sv.tl.survival(s, duration="t", event="d", covariates=["x"])  # duration= (not time=)
    assert s.models["cox"]["n"] == len(df)      # ran via the alias, didn't fall through


def test_survival_time_equals_event_is_guarded_not_crash():
    df = _toy_survival()
    s = sv.StudyState()
    sv.pp.ingest(s, data=df)
    s.write("variables", "outcome", "d")
    # Passing the same column for time and event used to raise a cryptic
    # "cannot reindex on an axis with duplicate labels" — now it degrades cleanly.
    sv.tl.survival(s, time="d", event="d")
    m = s.models["cox"]
    assert m["n"] == 0 and "同一列" in m["note"]


def test_survival_reports_logrank_when_grouped():
    df = _toy_survival(n=120)
    df["g"] = (df["x"] > 0).astype(int)     # a binary grouping
    s = sv.StudyState()
    sv.pp.ingest(s, data=df)
    s.write("variables", "outcome", "d")
    sv.tl.survival(s, time="t", event="d", covariates=["x"], group="g")
    lr = s.models["km"]["logrank"]
    assert lr is not None and lr["df"] == 1 and 0.0 <= lr["p"] <= 1.0 and lr["chi2"] >= 0


def test_andersen_gill_single_interval_equals_standard_cox():
    """AG (start=) with one (0, t] interval per subject must equal standard Cox."""
    df = _toy_survival(n=120)
    long = df.rename(columns={"t": "stop", "d": "event"}).copy()
    long["start"] = 0.0

    s_ag = sv.StudyState()
    sv.pp.ingest(s_ag, data=long)
    s_ag.write("variables", "outcome", "event")
    sv.tl.survival(s_ag, time="stop", event="event", start="start", covariates=["x"])

    s_std = sv.StudyState()
    sv.pp.ingest(s_std, data=df)
    s_std.write("variables", "outcome", "d")
    sv.tl.survival(s_std, time="t", event="d", covariates=["x"])

    b_ag = s_ag.models["cox"]["log_hr"]["x"][0]
    b_std = s_std.models["cox"]["log_hr"]["x"][0]
    assert "Andersen-Gill" in s_ag.models["cox"]["estimator"]
    assert abs(b_ag - b_std) < 1e-6      # left-truncation at 0 ≡ ordinary Cox
    # (real-data recovery of a genuine time-varying effect — employment in the
    #  Rossi data, employed HR≈0.26 — is validated in notebook 18.)
