"""Tiny deterministic toy datasets so the analysis chains (and the test-suite)
can actually run end-to-end without any external download. All generators are
seeded (``seed=0``) for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_did_panel", "load_survey", "load_corpus", "load_bib"]


def load_did_panel(n_units: int = 40, n_periods: int = 8, treat_period: int = 5,
                   att: float = -0.8, seed: int = 0) -> pd.DataFrame:
    """A staggered-free 2x2-ish DID panel with genuine *parallel pre-trends* and a
    real treatment effect ``att`` switched on at ``treat_period`` for the treated
    half. Columns: firm_id, year, treat, post, treat_post, first_treated, y, x1.
    """
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    treated = units < n_units // 2
    unit_fe = rng.normal(0, 1.0, n_units)
    time_fe = np.linspace(0, 1.2, n_periods)          # common trend (parallel)
    rows = []
    for i in units:
        for t in range(n_periods):
            post = int(t >= treat_period)
            tp = int(treated[i]) * post
            y = (2.0 + unit_fe[i] + time_fe[t] + att * tp
                 + rng.normal(0, 0.4))
            rows.append({
                "firm_id": int(i),
                "year": 2010 + t,
                "treat": int(treated[i]),
                "post": post,
                "treat_post": tp,
                "first_treated": (2010 + treat_period) if treated[i] else 0,
                "y": round(float(y), 4),
                "x1": round(float(rng.normal(0, 1)), 4),
            })
    return pd.DataFrame(rows)


def load_survey(n: int = 300, k_items: int = 6, seed: int = 0) -> pd.DataFrame:
    """A complex-survey-style table: ``k_items`` Likert items driven by one latent
    factor (so Cronbach's alpha is high), plus survey design columns
    (weight/strata/psu), an exposure and a binary outcome.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    items = {}
    for j in range(k_items):
        val = latent + rng.normal(0, 0.6, n)
        items[f"item{j + 1}"] = np.clip(np.round(val + 3), 1, 5).astype(int)
    df = pd.DataFrame(items)
    df["weight"] = np.round(rng.uniform(0.5, 3.0, n), 3)
    df["strata"] = rng.integers(1, 4, n)
    df["psu"] = rng.integers(1, 20, n)
    df["exposure"] = rng.integers(0, 2, n)
    lin = -0.3 + 0.9 * df["exposure"] + 0.4 * latent
    df["outcome"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-lin))).astype(int)
    return df


def load_corpus(seed: int = 0) -> dict[str, str]:
    """A handful of short 'interview' snippets with codeable themes and some PII
    to exercise the qualitative + redaction chain."""
    return {
        "int01": ("My name is Jane Doe and you can reach me at jane.doe@example.com. "
                  "Honestly the workload here is crushing; I feel burned out most weeks, "
                  "but my team supports me and that keeps me going."),
        "int02": ("I am Robert Smith. Call me on 415-555-0132. The pay is fine but "
                  "the lack of autonomy frustrates me. Recognition from managers is rare, "
                  "and that hurts morale across the whole department."),
        "int03": ("Contact: li.wei@example.org. What I value most is flexibility and "
                  "the support of colleagues. Burnout was real last year, yet the sense "
                  "of belonging pulled me back from quitting."),
    }


def load_bib(seed: int = 0) -> list[dict]:
    """A few reference records — some valid (with DOI), one suspicious (no DOI),
    one 'chimeric' (real-looking title, mismatched author) — for citation-verify."""
    return [
        {"id": "ok1", "title": "The Practice of Reflexivity in Qualitative Research",
         "authors": ["Finlay, L."], "year": 2002, "doi": "10.1177/104973202129120052"},
        {"id": "ok2", "title": "Using thematic analysis in psychology",
         "authors": ["Braun, V.", "Clarke, V."], "year": 2006, "doi": "10.1191/1478088706qp063oa"},
        {"id": "sus1", "title": "A Framework for Digital Distant Reading",
         "authors": ["Nobody, A."], "year": 2021, "doi": None},
        {"id": "chi1", "title": "Using thematic analysis in psychology",
         "authors": ["Foucault, M."], "year": 1975, "doi": "10.1191/1478088706qp063oa"},
    ]
