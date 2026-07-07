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
