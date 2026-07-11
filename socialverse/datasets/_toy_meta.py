"""Meta-analysis example datasets.

``load_bcg`` is the **real, canonical** BCG-vaccine dataset (Colditz et al. 1994,
13 trials) used throughout metafor — so a socialverse pooling of it can be checked
digit-for-digit against published random-effects results. The remaining loaders
generate small synthetic study tables with a **known ground truth** (documented in
each docstring) so a tutorial can show the estimator recovering it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_bcg", "load_meta_prevalence", "load_meta_smd",
           "load_dta_accuracy", "load_network_trials", "load_dose_response"]


def load_bcg() -> pd.DataFrame:
    """BCG vaccine vs tuberculosis — 13 trials (Colditz et al. 1994; metafor ``dat.bcg``).

    Columns: ``trial``, ``author``, ``year``, ``tpos``/``tneg`` (vaccinated TB+/TB−),
    ``cpos``/``cneg`` (control TB+/TB−), ``ablat`` (absolute latitude, °).
    The canonical example: pool the log risk ratio; a random-effects model gives a
    protective effect with very high heterogeneity, explained largely by latitude.
    """
    rows = [
        ("Aronson", 1948, 4, 119, 11, 128, 44),
        ("Ferguson & Simes", 1949, 6, 300, 29, 274, 55),
        ("Rosenthal et al", 1960, 3, 228, 11, 209, 42),
        ("Hart & Sutherland", 1977, 62, 13536, 248, 12619, 52),
        ("Frimodt-Moller et al", 1973, 33, 5036, 47, 5761, 13),
        ("Stein & Aronson", 1953, 180, 1361, 372, 1079, 44),
        ("Vandiviere et al", 1973, 8, 2537, 10, 619, 19),
        ("TPT Madras", 1980, 505, 87886, 499, 87892, 13),
        ("Coetzee & Berjak", 1968, 29, 7470, 45, 7232, 27),
        ("Rosenthal et al", 1961, 17, 1699, 65, 1600, 42),
        ("Comstock et al", 1974, 186, 50448, 141, 27197, 18),
        ("Comstock & Webster", 1969, 5, 2493, 3, 2338, 33),
        ("Comstock et al", 1976, 27, 16886, 29, 17825, 33),
    ]
    df = pd.DataFrame(rows, columns=["author", "year", "tpos", "tneg", "cpos", "cneg", "ablat"])
    df.insert(0, "trial", [f"Trial {i+1}" for i in range(len(df))])
    return df


def load_meta_prevalence(seed: int = 0) -> pd.DataFrame:
    """Synthetic 3-level prevalence meta-analysis (ECR mental-health pattern).

    ~20 studies, each contributing 1–4 outcome estimates (so effects are nested in
    studies). Ground truth: overall logit-prevalence −0.6 (≈ 35%), between-study
    SD 0.30, within-study/between-outcome SD 0.20. Columns: ``study``, ``cases``,
    ``n``, ``year``, ``female_pct``, ``instrument``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(20):
        u3 = rng.normal(0, 0.30)
        for _ in range(rng.integers(1, 5)):
            u2 = rng.normal(0, 0.20)
            n = int(rng.integers(40, 400))
            p = 1 / (1 + np.exp(-(-0.6 + u3 + u2)))
            rows.append({"study": f"S{s:02d}", "cases": int(rng.binomial(n, p)), "n": n,
                         "year": 2008 + s % 14, "female_pct": round(float(rng.uniform(0.3, 0.85)), 2),
                         "instrument": rng.choice(["PHQ-9", "CES-D", "BDI"])})
    return pd.DataFrame(rows)


def load_meta_smd(seed: int = 0) -> pd.DataFrame:
    """Synthetic two-group SMD studies. Ground-truth Cohen's d = 0.45, τ = 0.18.

    Columns: ``study``, ``m1``,``sd1``,``n1`` (treatment), ``m2``,``sd2``,``n2`` (control),
    ``year``, ``dosage`` (a moderator).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(16):
        true_d = 0.45 + rng.normal(0, 0.18)
        n1 = int(rng.integers(20, 120)); n2 = int(rng.integers(20, 120))
        sd = float(rng.uniform(0.9, 1.2))
        m2 = float(rng.normal(10, 1)); m1 = m2 + true_d * sd
        rows.append({"study": f"Study {s+1}", "m1": round(m1 + rng.normal(0, sd / np.sqrt(n1)), 2),
                     "sd1": round(sd, 2), "n1": n1,
                     "m2": round(m2 + rng.normal(0, sd / np.sqrt(n2)), 2), "sd2": round(sd, 2), "n2": n2,
                     "year": 2005 + s, "dosage": int(rng.integers(1, 4))})
    return pd.DataFrame(rows)


def load_dta_accuracy(seed: int = 0) -> pd.DataFrame:
    """Synthetic diagnostic test accuracy. Ground truth sensitivity 0.85, specificity 0.80.

    Columns: ``study``, ``tp``, ``fp``, ``fn``, ``tn``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(18):
        se = 1 / (1 + np.exp(-(np.log(0.85 / 0.15) + rng.normal(0, 0.3))))
        sp = 1 / (1 + np.exp(-(np.log(0.80 / 0.20) + rng.normal(0, 0.3))))
        npos, nneg = int(rng.integers(30, 130)), int(rng.integers(30, 130))
        tp = int(rng.binomial(npos, se)); tn = int(rng.binomial(nneg, sp))
        rows.append({"study": f"Study {s+1}", "tp": tp, "fn": npos - tp, "tn": tn, "fp": nneg - tn})
    return pd.DataFrame(rows)


def load_network_trials(seed: int = 0) -> pd.DataFrame:
    """Synthetic 4-treatment network (arm-level, consistent). Ground-truth log-odds:
    A=−0.5 (reference), B=0.0, C=0.5, D=1.0 ⇒ e.g. D vs A = +1.5.

    Columns: ``study``, ``treat`` (A/B/C/D), ``events``, ``n``. Mix of two-arm trials
    across the comparisons that form a connected network.
    """
    rng = np.random.default_rng(seed)
    truth = {"A": -0.5, "B": 0.0, "C": 0.5, "D": 1.0}
    comparisons = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("A", "D")]
    rows = []; sid = 0
    for pair in comparisons:
        for _ in range(rng.integers(3, 6)):
            sid += 1
            for t in pair:
                n = int(rng.integers(80, 260)); p = 1 / (1 + np.exp(-truth[t]))
                rows.append({"study": f"T{sid:02d}", "treat": t, "events": int(rng.binomial(n, p)), "n": n})
    return pd.DataFrame(rows)


def load_dose_response(seed: int = 0) -> pd.DataFrame:
    """Synthetic dose-response (log-RR vs dose). Ground-truth slope 0.15 log-RR per unit.

    Columns: ``study``, ``dose``, ``logrr`` (reference dose row = 0), ``se``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(12):
        for dose in [0, 5, 10, 20]:
            lr = 0.15 * dose + rng.normal(0, 0.05)
            rows.append({"study": f"Study {s+1}", "dose": dose,
                         "logrr": round(lr if dose > 0 else 0.0, 4),
                         "se": (round(float(rng.uniform(0.06, 0.10)), 3) if dose > 0 else 0.01)})
    return pd.DataFrame(rows)
