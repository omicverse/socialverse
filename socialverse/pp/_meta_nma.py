"""``sv.pp._meta_nma`` — arm-level → pairwise contrasts for network meta-analysis.

Turn a long arm-level table (study × treatment with events/n or mean/sd/n) into
contrast-based data (each non-baseline arm vs the study baseline) with the
multi-arm within-study covariance recorded. Feeds ``sv.tl.netmeta``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._meta_es import _resolve_df


@register(
    name="nma_pairwise", aliases=["网络对比", "nma_contrasts", "pairwise_nma"],
    category="social_science_quant", tier="pro", skill="meta-analysis",
    languages=["Python"], key_tools=["numpy"],
    description="臂层数据(study×treatment)→ 网络 meta 的对比数据(各非基线臂 vs 基线),记录多臂协方差;供 sv.tl.netmeta",
    requires={"sources": ["datasets"]}, produces={"models": ["nma_contrasts"]},
)
def nma_pairwise(state: StudyState, **kwargs: Any) -> StudyState:
    """Arm-level → contrast-based NMA data.

    kwargs: ``study=``, ``treatment=``; binary: ``events=``,``n=`` (→ logOR);
    continuous: ``mean=``,``sd=``,``n=`` (→ MD). Baseline arm per study = the
    first, or lowest-order treatment. Stores contrasts + a within-study
    covariance flag for multi-arm trials.
    """
    df = _resolve_df(state, kwargs)
    if df is None:
        return state
    scol = kwargs.get("study"); tcol = kwargs.get("treatment")
    if scol not in df.columns or tcol not in df.columns:
        return state
    ev = kwargs.get("events"); n = kwargs.get("n"); mn = kwargs.get("mean"); sd = kwargs.get("sd")
    rows = []
    for s, sub in df.groupby(scol, sort=False):
        sub = sub.reset_index(drop=True)
        base = 0  # first arm as baseline
        tb = sub[tcol].iloc[base]
        multi = len(sub) > 2
        for j in range(len(sub)):
            if j == base:
                continue
            tj = sub[tcol].iloc[j]
            if ev is not None and n is not None:
                a, nb = float(sub[ev].iloc[j]), float(sub[n].iloc[j])
                c, nd = float(sub[ev].iloc[base]), float(sub[n].iloc[base])
                # continuity correction on zero cells
                if 0 in (a, nb - a, c, nd - c):
                    a, c = a + 0.5, c + 0.5; nb, nd = nb + 1, nd + 1
                TE = np.log((a / (nb - a)) / (c / (nd - c)))
                seTE = np.sqrt(1 / a + 1 / (nb - a) + 1 / c + 1 / (nd - c))
                vbase = 1 / c + 1 / (nd - c)   # baseline-arm variance (multi-arm cov)
                measure = "OR"
            elif mn is not None and sd is not None and n is not None:
                m1, s1, n1 = float(sub[mn].iloc[j]), float(sub[sd].iloc[j]), float(sub[n].iloc[j])
                m0, s0, n0 = float(sub[mn].iloc[base]), float(sub[sd].iloc[base]), float(sub[n].iloc[base])
                TE = m1 - m0
                seTE = np.sqrt(s1 ** 2 / n1 + s0 ** 2 / n0); vbase = s0 ** 2 / n0; measure = "MD"
            else:
                continue
            rows.append({"studlab": s, "treat1": tj, "treat2": tb, "TE": TE, "seTE": seTE,
                         "vbase": vbase, "multiarm": multi, "measure": measure})
    state.write("models", "nma_contrasts", pd.DataFrame(rows))
    return state
