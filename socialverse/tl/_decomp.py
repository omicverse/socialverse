"""``sv.tl._decomp`` — decomposition estimators the social sciences lean on:
the **Goodman-Bacon** diagnostic for staggered DiD and **Oaxaca-Blinder** for
between-group gaps.

- ``bacon_decompose`` — expresses a two-way-FE DiD estimate as the weighted average
  of every 2×2 sub-comparison (Goodman-Bacon 2021), and surfaces how much weight
  rides on the **forbidden** "already-treated as control" comparisons — the negative-
  weighting diagnostic you run before trusting a TWFE ATT.
- ``oaxaca`` — splits a mean outcome gap between two groups into an **explained**
  (endowments / composition) part and an **unexplained** (returns / structure) part
  (Oaxaca 1973; Blinder 1973), the workhorse for wage-gap / discrimination studies.

Both are native (numpy + statsmodels) and exact: the Bacon weights reconstruct the
TWFE estimate, and the Oaxaca components sum to the raw gap.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _cols, _get_datasets, _pick_outcome, _within_fit
from ._fect import _build_matrices


# ==================================================================== bacon_decompose
@register(
    name="bacon_decompose",
    aliases=["古德曼培根分解", "goodman_bacon", "bacon", "did_decomposition"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy"],
    description="Goodman-Bacon 分解:把 TWFE-DiD 拆成所有 2×2 比较的加权平均,量化'已处理作对照'的禁忌权重",
    requires={"design": ["panel_id", "time", "treatment", "first_treated"],
              "variables": ["outcome"]},
    produces={"diagnostics": ["bacon"]},
    auto_fix="escalate",
)
def bacon_decompose(state: StudyState, **kwargs: Any) -> StudyState:
    """Goodman-Bacon (2021) decomposition of a staggered TWFE DiD.

    Decomposes the two-way-FE estimate into 2×2 sub-comparisons of three types —
    treated-vs-never-treated, earlier-vs-later (clean), and later-vs-earlier
    (**forbidden**: already-treated units used as controls). Reports each type's
    weight and average 2×2 estimate, verifies the weighted sum reproduces the TWFE
    ATT, and flags the total forbidden weight (large ⇒ TWFE is untrustworthy; prefer
    ``fect`` / ``sun_abraham``). Assumes an (approximately) balanced panel.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)

    def _empty(note):
        state.write("diagnostics", "bacon", {"comparisons": [], "note": note})
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time")):
        return _empty("缺少面板数据或设计列(panel_id/time)")
    y_col = _pick_outcome(df, cols, exclude=[c for c in cols.values() if c])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    Y, D, E, units, times, onset = _build_matrices(df, cols, y_col)
    N, T = Y.shape
    balanced = bool(E.all())

    def ym(um, ts):
        sub = Y[np.ix_(np.where(um)[0], list(ts))]
        return float(np.nanmean(sub)) if sub.size else np.nan

    # TWFE target: within-OLS of Y on treat_post + unit/time FE over OBSERVED cells
    # only (never nan_to_num the rectangle — that would count absent cells as 0).
    ii, tt = np.where(E)
    fit = _within_fit(Y[ii, tt], D[ii, tt].reshape(-1, 1), units[ii], times[tt], units[ii])
    twfe = float(fit["beta"][0]) if fit is not None else float("nan")

    timing = sorted({int(o) for o in onset if o >= 0})  # onset index per cohort
    U = onset < 0

    def Dbar(o):
        return (T - o) / T  # share of periods a cohort treated from index o is treated

    comps = []
    for k in timing:
        mk = onset == k
        nk = mk.mean()
        post_k, pre_k = set(range(k, T)), set(range(0, k))
        if not pre_k or not post_k:
            continue
        if U.any():
            nU = U.mean()
            nkU = nk / (nk + nU)
            b = (ym(mk, post_k) - ym(mk, pre_k)) - (ym(U, post_k) - ym(U, pre_k))
            s = (nk + nU) ** 2 * nkU * (1 - nkU) * Dbar(k) * (1 - Dbar(k))
            if np.isfinite(b) and s > 0:
                comps.append({"type": "treated_vs_never", "early": k, "late": None,
                              "weight_raw": s, "estimate": b})
    for k, l in itertools.combinations(timing, 2):  # k earlier than l
        mk, ml = onset == k, onset == l
        nk, nl = mk.mean(), ml.mean()
        nkl = nk / (nk + nl)
        dk, dl = Dbar(k), Dbar(l)
        if not (0 < dl < dk < 1):
            continue
        # 2a earlier-as-treated vs later-not-yet (window before l is treated)
        prek, postk = set(range(0, k)), set(range(k, l))
        if prek and postk:
            b1 = (ym(mk, postk) - ym(mk, prek)) - (ym(ml, postk) - ym(ml, prek))
            s1 = ((nk + nl) * (1 - dl)) ** 2 * nkl * (1 - nkl) * (dk - dl) / (1 - dl) * (1 - dk) / (1 - dl)
            if np.isfinite(b1) and s1 > 0:
                comps.append({"type": "earlier_vs_later", "early": k, "late": l,
                              "weight_raw": s1, "estimate": b1})
        # 2b later-as-treated vs earlier-already-treated (FORBIDDEN)
        prel, postl = set(range(k, l)), set(range(l, T))
        if prel and postl:
            b2 = (ym(ml, postl) - ym(ml, prel)) - (ym(mk, postl) - ym(mk, prel))
            s2 = ((nk + nl) * dk) ** 2 * nkl * (1 - nkl) * dl / dk * (dk - dl) / dk
            if np.isfinite(b2) and s2 > 0:
                comps.append({"type": "later_vs_earlier_forbidden", "early": k, "late": l,
                              "weight_raw": s2, "estimate": b2})

    S = sum(c["weight_raw"] for c in comps)
    if S <= 0:
        return _empty("无有效 2×2 比较(可能无处理时点变异或无对照)")
    for c in comps:
        c["weight"] = c["weight_raw"] / S
        del c["weight_raw"]
    recon = float(sum(c["weight"] * c["estimate"] for c in comps))

    by_type = {}
    for c in comps:
        t = c["type"]
        by_type.setdefault(t, {"weight": 0.0, "wsum": 0.0})
        by_type[t]["weight"] += c["weight"]
        by_type[t]["wsum"] += c["weight"] * c["estimate"]
    for t in by_type:
        w = by_type[t]["weight"]
        by_type[t]["avg_estimate"] = by_type[t].pop("wsum") / w if w else None
    forbidden_w = by_type.get("later_vs_earlier_forbidden", {}).get("weight", 0.0)

    state.write("diagnostics", "bacon", {
        "twfe_att": twfe, "reconstructed": recon, "reconstruction_ok": abs(recon - twfe) < 1e-6,
        "forbidden_weight": forbidden_w, "by_type": by_type,
        "comparisons": sorted(comps, key=lambda c: -c["weight"]),
        "n_cohorts": len(timing), "has_never_treated": bool(U.any()), "balanced": balanced,
        "verdict": ("分解不完整(重构≠TWFE:可能含首期即处理 cohort 或面板不平衡)—— 勿据此判 TWFE"
                    if abs(recon - twfe) >= 1e-6
                    else "TWFE 可信(禁忌权重低)" if forbidden_w < 0.1
                    else "禁忌比较权重较高 — TWFE 可能有偏,改用 fect/sun_abraham"),
        "note": "Goodman-Bacon:TWFE = Σ 权重×2×2;forbidden=已处理单位当对照(负权重来源)"
                + ("" if balanced else "。⚠️面板不平衡,权重为近似"),
    })
    return state


# ============================================================================ oaxaca
@register(
    name="oaxaca",
    aliases=["oaxaca_blinder", "blinder_oaxaca", "gap_decomposition", "工资差距分解"],
    category="causal",
    tier="plus",
    skill="econometrics-replication",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="Oaxaca-Blinder 分解:把两组均值差拆成 explained(禀赋/构成)+ unexplained(回报/结构)",
    requires={"variables": ["outcome"]},
    produces={"models": ["oaxaca"]},
    auto_fix="escalate",
)
def oaxaca(state: StudyState, **kwargs: Any) -> StudyState:
    """Oaxaca-Blinder decomposition of a between-group mean gap.

    Fits ``Y ~ X`` separately in group A and group B and splits ``ȳ_A - ȳ_B`` into a
    threefold decomposition (endowments + coefficients + interaction) and a twofold
    decomposition (explained + unexplained) against a pooled reference. The
    unexplained/coefficients part is the classic "returns / discrimination" residual.

    Keyword arguments: ``group=`` binary group column (A = its larger value),
    ``predictors=`` covariate columns, ``outcome=`` (or from variables).
    """
    df = _get_datasets(state, kwargs)
    sm = __import__("statsmodels.api", fromlist=["api"])
    Y = kwargs.get("outcome") or state.variables.get("outcome")
    group = kwargs.get("group")
    preds = kwargs.get("predictors") or kwargs.get("X")
    if isinstance(preds, str):
        preds = [preds]

    def _empty(note):
        state.write("models", "oaxaca", {"gap": None, "note": note})
        return state

    if df is None or Y is None or group is None:
        return _empty("缺少 data / outcome / group(二分组列)")
    if group not in df.columns:
        return _empty(f"分组列 {group} 不在数据中")
    if Y not in df.columns:
        return _empty(f"结果列 {Y} 不在数据中")
    if not preds:
        preds = [c for c in df.columns if c not in (Y, group)
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not preds:
        return _empty("找不到预测变量(predictors)")

    gv = df[group]
    hi = gv.dropna().unique()
    if len(hi) != 2:
        return _empty("group 必须恰好二分类")
    A_val = max(hi)
    isA = (gv == A_val).to_numpy()
    Xmat = df[preds].apply(pd.to_numeric, errors="coerce")
    yv = pd.to_numeric(df[Y], errors="coerce")
    ok = Xmat.notna().all(axis=1) & yv.notna() & gv.notna()
    X = sm.add_constant(Xmat[ok], has_constant="add").to_numpy(float)
    y = yv[ok].to_numpy(float)
    a = isA[ok.to_numpy()]

    if a.sum() < len(preds) + 2 or (~a).sum() < len(preds) + 2:
        return _empty("某组样本量过小,无法分组回归")

    def fit(mask):
        r = sm.OLS(y[mask], X[mask]).fit()
        return np.asarray(r.params, float), X[mask].mean(0)

    bA, xA = fit(a)
    bB, xB = fit(~a)
    bstar = np.asarray(sm.OLS(y, X).fit().params, float)  # pooled reference
    gap = float(y[a].mean() - y[~a].mean())

    endow = float((xA - xB) @ bB)
    coeff = float(xB @ (bA - bB))
    inter = float((xA - xB) @ (bA - bB))
    explained = float((xA - xB) @ bstar)
    unexplained = float(xA @ (bA - bstar) + xB @ (bstar - bB))

    names = ["const"] + list(preds)
    contrib = {names[j]: {"endowment": float((xA[j] - xB[j]) * bB[j]),
                          "coefficient": float(xB[j] * (bA[j] - bB[j]))}
               for j in range(len(names))}

    state.write("models", "oaxaca", {
        "gap": gap, "group_high": _pyval(A_val),
        "threefold": {"endowments": endow, "coefficients": coeff, "interaction": inter,
                      "sum": endow + coeff + inter},
        "twofold": {"explained": explained, "unexplained": unexplained,
                    "sum": explained + unexplained},
        "per_predictor": contrib, "predictors": list(preds), "n": int(ok.sum()),
        "estimator": "oaxaca_blinder",
        "note": "Oaxaca-Blinder:explained=禀赋/构成差(可解释),unexplained=回报/结构差(常释为歧视/结构)",
    })
    return state


def _pyval(v):
    try:
        return v.item()
    except Exception:
        return v


__all__ = ["bacon_decompose", "oaxaca"]
