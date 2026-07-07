"""``sv.tl._did_robust`` — robustness/design estimators around DiD:
**synthetic difference-in-differences** and the **Honest-DiD** parallel-trends
sensitivity.

- ``synth_did`` — Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021): combines
  synthetic-control **unit weights** (match treated pre-trends) with DiD **time
  weights**, giving an estimator more robust than either classic SCM or TWFE-DiD.
  Jackknife SE over units.
- ``honest_did`` — Rambachan & Roth (2023): parallel trends is never exactly true, so
  report how large a violation would have to be to overturn the conclusion. Takes an
  event-study path and returns the treatment effect's robust confidence set under the
  relative-magnitudes restriction ΔRM(M), plus the **breakdown M** at which
  significance is lost. (Simplified ΔRM; the full FLCI/conditional inference is the
  HonestDiD package's domain — flagged in the note.)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _cols, _get_datasets, _pick_outcome
from ._fect import _build_matrices


# ========================================================================= synth_did
def _sdid_point(Ymat, tr, co, pre, post):
    """Synthetic-DiD point estimate given treated/control units and pre/post periods."""
    from scipy.optimize import nnls
    Ypre_co = Ymat[np.ix_(co, pre)]                       # |co| × |pre|
    ypre_tr = Ymat[np.ix_(tr, pre)].mean(0)               # |pre|
    w, _ = nnls(Ypre_co.T, ypre_tr)
    w = w / w.sum() if w.sum() > 1e-8 else np.full(len(co), 1.0 / len(co))
    yco_post = Ymat[np.ix_(co, post)].mean(1)             # |co|
    lam, _ = nnls(Ypre_co, yco_post)
    lam = lam / lam.sum() if lam.sum() > 1e-8 else np.full(len(pre), 1.0 / len(pre))
    tr_post = Ymat[np.ix_(tr, post)].mean(0).mean()
    tr_pre = lam @ Ymat[np.ix_(tr, pre)].mean(0)
    co_post = w @ Ymat[np.ix_(co, post)].mean(1)
    co_pre = w @ (Ymat[np.ix_(co, pre)] @ lam)
    return float((tr_post - tr_pre) - (co_post - co_pre))


@register(
    name="synth_did",
    aliases=["合成双重差分", "synthetic_did", "sdid"],
    category="causal",
    tier="pro",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "scipy"],
    description="合成双重差分 SDID:单位权重(合成控制)+时间权重(DiD)→比经典 SCM/TWFE 更稳健;jackknife SE",
    requires={"design": ["panel_id", "time", "treatment", "first_treated"],
              "variables": ["outcome"]},
    produces={"models": ["synth_did"]},
    auto_fix="escalate",
)
def synth_did(state: StudyState, **kwargs: Any) -> StudyState:
    """Synthetic difference-in-differences (block adoption).

    Uses never-treated units as the donor pool and the earliest treatment date as the
    block time; treated = ever-treated units, pre/post split at that date. Assumes
    approximately common adoption timing (staggered timing is collapsed to the earliest
    onset — flagged). SE by leave-one-unit-out jackknife.
    """
    df = _get_datasets(state, kwargs)
    cols = _cols(state, kwargs)

    def _empty(note):
        state.write("models", "synth_did", {"att": None, "note": note})
        return state

    if df is None or any(cols[k] is None for k in ("panel_id", "time")):
        return _empty("缺少面板数据或设计列(panel_id/time)")
    y_col = _pick_outcome(df, cols, exclude=[c for c in cols.values() if c])
    if y_col is None:
        return _empty("找不到结果变量(outcome)")

    Y, D, E, units, times, onset = _build_matrices(df, cols, y_col)
    N, T = Y.shape
    if not E.all():
        return _empty("SDID 需平衡面板(无缺失格)")
    tr = np.where(onset >= 0)[0]
    co = np.where(onset < 0)[0]
    if len(tr) < 1 or len(co) < 2:
        return _empty("SDID 需 ≥1 处理单位 + ≥2 never-treated 对照(donor pool)")
    T0 = int(onset[tr].min())
    staggered = bool(len(np.unique(onset[tr])) > 1)
    if T0 < 1 or T0 >= T:
        return _empty("无处理前期或无处理后期")
    pre, post = np.arange(T0), np.arange(T0, T)

    att = _sdid_point(Y, tr, co, pre, post)

    # jackknife over all units (leave-one-out) — needs >1 treated for the treated LOO
    jk = []
    allu = np.concatenate([tr, co])
    for u in allu:
        tr2 = tr[tr != u]
        co2 = co[co != u]
        if len(tr2) < 1 or len(co2) < 2:
            continue
        try:
            jk.append(_sdid_point(Y, tr2, co2, pre, post))
        except Exception:
            continue
    jk = np.array(jk)
    n_j = jk.size
    se = float(np.sqrt((n_j - 1) / n_j * np.sum((jk - jk.mean()) ** 2))) if n_j > 2 else None
    p = ci = None
    if se and se > 0:
        from scipy import stats
        p = float(2 * (1 - stats.norm.cdf(abs(att / se))))
        ci = [att - 1.96 * se, att + 1.96 * se]

    # classic DiD for contrast
    did = float((Y[np.ix_(tr, post)].mean() - Y[np.ix_(tr, pre)].mean())
                - (Y[np.ix_(co, post)].mean() - Y[np.ix_(co, pre)].mean()))
    state.write("models", "synth_did", {
        "att": att, "se": se, "ci": ci, "p": p, "did_contrast": did,
        "n_treated": int(len(tr)), "n_control": int(len(co)), "block_time_index": T0,
        "outcome": y_col, "estimator": "synthetic_did",
        "note": "合成双重差分(单位权重+时间权重);jackknife SE"
                + ("。⚠️处理时点交错,已折叠到最早采纳期(block 近似)" if staggered else ""),
    })
    return state


# ======================================================================== honest_did
def _event_path(state, kwargs):
    """Return sorted (rel_period, coef, se) from kwargs or an event-study-style model."""
    ec = kwargs.get("event_coefs") or kwargs.get("coefs")
    if isinstance(ec, dict):
        items = []
        for k, v in ec.items():
            se = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else None
            c = v[0] if isinstance(v, (list, tuple)) else v
            items.append((int(k), float(c), (float(se) if se is not None else None)))
        return sorted(items)
    for slot in ("sun_abraham", "event_study"):
        m = state.models.get(slot, {})
        cf = m.get("coefs")
        if isinstance(cf, dict):
            items = []
            for k, v in cf.items():
                try:
                    rk = int(k)
                except Exception:
                    continue
                if isinstance(v, (list, tuple)):
                    items.append((rk, float(v[0]), float(v[1]) if len(v) > 1 else None))
            if items:
                return sorted(items)
    return None


@register(
    name="honest_did",
    aliases=["诚实DiD", "honest_parallel_trends", "rambachan_roth", "pretrend_sensitivity"],
    category="causal",
    tier="pro",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy"],
    description="Honest-DiD 平行趋势敏感性(Rambachan-Roth ΔRM):事件研究→稳健 CIvsM + breakdown M(简化版)",
    requires={},  # consumes event_study/sun_abraham coefs or a coefs= kwarg (validated in body)
    produces={"diagnostics": ["honest_did"]},
    prerequisites={"optional_functions": ["event_study", "sun_abraham"]},
    auto_fix="escalate",
)
def honest_did(state: StudyState, **kwargs: Any) -> StudyState:
    """Honest-DiD sensitivity under the relative-magnitudes restriction ΔRM(M).

    Consumes an event-study path (from ``models['event_study']`` /
    ``models['sun_abraham']`` or a ``coefs={rel: (coef, se)}`` kwarg). For a target
    post-period it reports the robust confidence interval as a function of ``M`` (how
    many times the largest pre-period violation the post-trend may be) and the
    **breakdown M** where the effect stops being significant. Large breakdown M ⇒ the
    finding survives sizeable parallel-trends violations.
    """
    path = _event_path(state, kwargs)

    def _empty(note):
        state.write("diagnostics", "honest_did", {"robust_ci": {}, "note": note})
        return state

    if not path:
        return _empty("找不到事件研究系数(先跑 event_study/sun_abraham,或传 coefs=)")
    pre = [(r, c, s) for r, c, s in path if r < 0]
    post = [(r, c, s) for r, c, s in path if r >= 0]
    if len(pre) < 2 or not post:
        return _empty("需 ≥2 个处理前期 + ≥1 处理后期以量化违背幅度")

    # max pre-period "violation": largest consecutive first-difference (deviation from flat)
    pre_sorted = sorted(pre)
    diffs = [abs(pre_sorted[i + 1][1] - pre_sorted[i][1]) for i in range(len(pre_sorted) - 1)]
    # include the jump from the last pre period to the (normalized 0) base
    diffs.append(abs(pre_sorted[-1][1]))
    V = max(diffs) if diffs else 0.0

    # target post period: the specified one, else the last observed post effect
    tgt = kwargs.get("target_period")
    target = next((p for p in post if p[0] == tgt), None) if tgt is not None else post[-1]
    if target is None:
        return _empty(f"target_period={tgt} 不在处理后期({[p[0] for p in post]})")
    h = max(1, target[0] + 1)  # horizon (periods since treatment onset, base at rel=-1→h≥1)
    theta, se = target[1], (target[2] or 0.0)
    z = 1.96
    Mgrid = list(kwargs.get("M_grid", [0.0, 0.5, 1.0, 1.5, 2.0]))
    robust = {}
    for M in Mgrid:
        bias = M * V * h
        lo, hi = theta - bias - z * se, theta + bias + z * se
        robust[str(M)] = {"lo": float(lo), "hi": float(hi),
                          "significant": bool(lo > 0 or hi < 0)}
    # breakdown M: smallest M with 0 in the robust CI  ->  (|theta| - z*se) / (V*h).
    # A perfectly flat pre-trend (V*h==0) means NO finite violation can overturn a
    # significant effect — the most robust case, breakdown = +inf (not 0).
    sig_alone = abs(theta) > z * se
    if not sig_alone:
        breakdown = 0.0                       # effect not significant even at M=0
    elif V * h <= 0:
        breakdown = float("inf")              # flat pre-trend: unbreakable
    else:
        breakdown = float((abs(theta) - z * se) / (V * h))
    if breakdown == float("inf"):
        verdict = "极稳健:处理前趋势零违背,任意 M 都无法翻案"
    elif breakdown >= 1.0:
        verdict = f"稳健:需 M≈{breakdown:.2f}× 的处理前违背才翻案"
    elif sig_alone:
        verdict = f"脆弱:M≈{breakdown:.2f}(<1)的违背即可翻案 — 结论对平行趋势敏感"
    else:
        verdict = "本身不显著(M=0 的 CI 已含 0)"

    state.write("diagnostics", "honest_did", {
        "target_period": int(target[0]), "estimate": float(theta), "se": float(se),
        "pre_violation_scale": float(V), "robust_ci": robust,
        "breakdown_M": breakdown,
        "verdict": verdict,
        "note": "Rambachan-Roth ΔRM 相对幅度敏感性(简化版:以最大处理前一阶差为违背标尺;"
                "完整 FLCI/条件推断请用 HonestDiD 包)",
    })
    return state


__all__ = ["synth_did", "honest_did"]
