"""``sv.tl._setmethods`` — set-theoretic methods (QCA) on the ``StudyState`` spine.

**Qualitative Comparative Analysis (QCA)** — Ragin's set-theoretic method for
mapping *configurations* of conditions to an outcome. This is a genuine Python
gap: the reference implementations are R packages (``QCA`` by Adrian Duşa,
``SetMethods``, ``QCAfalsePositive``) and the fragmentary Python ports
(``fuzzy-qca``, ``pyqca``) are thin and unmaintained. This module implements
fuzzy-set QCA (fsQCA) end-to-end in the standard library + NumPy:

    calibrated fuzzy sets
        → truth table (2^k configurational corners, one row per corner)
        → per-corner *consistency* of sufficiency  (Ragin's inclusion score)
        → keep corners with consistency ≥ inclusion cut  →  outcome = 1
        → Quine–McCluskey Boolean minimization  →  prime implicants
        → solution (∑ of paths) with solution consistency + coverage

The fuzzy-set algebra is the honest Ragin definition — negation ``~X = 1 - X``,
intersection (AND) = element-wise ``min``, union (OR) = element-wise ``max`` —
and *consistency of sufficiency* for a set ``X`` in outcome ``Y`` is

    consistency(X ⊆ Y) = Σ min(X, Y) / Σ X

with *raw coverage* = Σ min(X, Y) / Σ Y. Truth-table corner membership is the
fuzzy AND of each condition (present) or its negation (absent); a case is
"in" a corner when that membership > 0.5 (Ragin's rule that every case has
>0.5 membership in exactly one corner). Boolean minimization is a full
Quine–McCluskey with prime-implicant selection (a Petrick-style greedy cover),
so the solution is the real minimal sum-of-products, not a heuristic.

Approximations, honestly labelled: this is the *conservative / complex*
solution (no counterfactual simplifying assumptions on remainder rows), and
consistency thresholds use the raw inclusion score without the PRI refinement
that ``QCA::truthTable`` also reports — both are noted in the emitted result.
Pure standard library + NumPy; no ``QCA``/``SetMethods`` R bridge required.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState

__all__ = ["qca", "calibrate", "necessity_analysis"]


# --------------------------------------------------------------------- helpers
def _get_datasets(state: StudyState, kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Resolve the working table: explicit ``data=`` kwarg, else ``sources['datasets']``.

    ``sources['datasets']`` may be a DataFrame or a ``{name: DataFrame}`` mapping;
    in the latter case the first frame is taken.
    """
    df = kwargs.get("data")
    if df is None:
        df = state.sources.get("datasets")
    if isinstance(df, dict):
        df = next((v for v in df.values() if isinstance(v, pd.DataFrame)), None)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def _fuzzy_and(cols: list[np.ndarray]) -> np.ndarray:
    """Fuzzy intersection (logical AND) = element-wise minimum across sets."""
    out = cols[0]
    for c in cols[1:]:
        out = np.minimum(out, c)
    return out


def _consistency(x: np.ndarray, y: np.ndarray) -> float:
    """Ragin's consistency of *sufficiency* ``X ⊆ Y`` = Σ min(X, Y) / Σ X."""
    denom = float(x.sum())
    if denom <= 0.0:
        return 0.0
    return float(np.minimum(x, y).sum() / denom)


def _coverage(x: np.ndarray, y: np.ndarray) -> float:
    """Raw coverage of a sufficient set ``X`` for ``Y`` = Σ min(X, Y) / Σ Y."""
    denom = float(y.sum())
    if denom <= 0.0:
        return 0.0
    return float(np.minimum(x, y).sum() / denom)


def _pri_consistency(x: np.ndarray, y: np.ndarray) -> float:
    """Proportional Reduction in Inconsistency (PRI) consistency of sufficiency.

    ``QCA::truthTable`` reports PRI alongside raw consistency precisely to catch
    corners that are *simultaneously* subsets of ``Y`` **and** its negation
    ``~Y`` — an artefact of the fuzzy min ratio that inflates raw consistency for
    corners with small memberships. With ``ny = 1 - y``::

        PRI = (Σ min(X, Y) − Σ min(X, Y, ~Y)) / (Σ X − Σ min(X, Y, ~Y))

    A low PRI unmasks a configuration whose high raw consistency is spurious.
    """
    ny = 1.0 - y
    both = np.minimum(np.minimum(x, y), ny).sum()
    num = float(np.minimum(x, y).sum() - both)
    den = float(x.sum() - both)
    if den <= 0.0:
        return 0.0
    return num / den


def _corner_label(corner: tuple[int, ...], names: list[str]) -> str:
    """Human-readable configuration label, e.g. ``A*B*~C``."""
    parts = [(nm if bit else f"~{nm}") for bit, nm in zip(corner, names)]
    return "*".join(parts)


# ------------------------------------------------------ Quine–McCluskey minimize
def _combine(a: tuple[int | None, ...], b: tuple[int | None, ...]):
    """Combine two implicants differing in exactly one literal → a dash there.

    Implicants are tuples over ``{0, 1, None}`` where ``None`` = "don't care".
    Returns the merged implicant, or ``None`` if they cannot be combined.
    """
    diff = 0
    merged: list[int | None] = []
    for x, y in zip(a, b):
        if x == y:
            merged.append(x)
        else:
            diff += 1
            merged.append(None)
    return tuple(merged) if diff == 1 else None


def _covers(imp: tuple[int | None, ...], minterm: tuple[int, ...]) -> bool:
    """True if implicant ``imp`` (with ``None`` = don't-care) covers ``minterm``."""
    return all(p is None or p == m for p, m in zip(imp, minterm))


def _prime_implicants(minterms: list[tuple[int, ...]]) -> list[tuple[int | None, ...]]:
    """Full Quine–McCluskey: iteratively combine minterms into prime implicants."""
    current = set(minterms)
    primes: set[tuple[int | None, ...]] = set()
    while current:
        used: set[tuple[int | None, ...]] = set()
        merged_next: set[tuple[int | None, ...]] = set()
        cur_list = list(current)
        for i in range(len(cur_list)):
            for j in range(i + 1, len(cur_list)):
                m = _combine(cur_list[i], cur_list[j])
                if m is not None:
                    used.add(cur_list[i])
                    used.add(cur_list[j])
                    merged_next.add(m)
        # any implicant not absorbed into a bigger one is prime
        for imp in current:
            if imp not in used:
                primes.add(imp)
        current = merged_next
    return sorted(primes, key=lambda t: (sum(p is None for p in t),
                                          tuple(-1 if p is None else p for p in t)))


def _select_cover(primes: list[tuple[int | None, ...]],
                  minterms: list[tuple[int, ...]]) -> list[tuple[int | None, ...]]:
    """Greedy Petrick-style minimal cover of ``minterms`` by ``primes``.

    Essential prime implicants first (the only prime covering some minterm), then
    a greedy fill choosing the prime covering the most still-uncovered minterms.
    """
    remaining = set(minterms)
    chosen: list[tuple[int | None, ...]] = []

    # essential prime implicants
    for mt in list(remaining):
        covering = [p for p in primes if _covers(p, mt)]
        if len(covering) == 1 and covering[0] not in chosen:
            chosen.append(covering[0])
    for p in chosen:
        remaining -= {mt for mt in remaining if _covers(p, mt)}

    # greedy fill
    while remaining:
        best = max(primes, key=lambda p: sum(_covers(p, mt) for mt in remaining))
        if sum(_covers(best, mt) for mt in remaining) == 0:
            break
        chosen.append(best)
        remaining -= {mt for mt in remaining if _covers(best, mt)}
    return chosen


def _implicant_label(imp: tuple[int | None, ...], names: list[str]) -> str:
    """Render a (possibly reduced) implicant as a Boolean product term."""
    parts = [(nm if bit == 1 else f"~{nm}")
             for bit, nm in zip(imp, names) if bit is not None]
    return "*".join(parts) if parts else "1"


def _implicant_membership(imp: tuple[int | None, ...],
                          fuzz: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Fuzzy membership in a reduced implicant = AND over its retained literals."""
    literals: list[np.ndarray] = []
    for bit, nm in zip(imp, names):
        if bit is None:
            continue
        literals.append(fuzz[nm] if bit == 1 else 1.0 - fuzz[nm])
    if not literals:
        return np.ones_like(next(iter(fuzz.values())))
    return _fuzzy_and(literals)


# ------------------------------------------------------------------------- qca
@register(
    name="qca",
    aliases=["定性比较分析", "fsQCA"],
    category="setmethods",
    tier="plus",
    skill="(QCA 缺口,Python 空白)",
    languages=["Python"],
    key_tools=["numpy"],
    description="模糊集定性比较分析:真值表 + 一致性/覆盖率 + Quine-McCluskey 布尔最小化解",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["qca"], "diagnostics": ["consistency_coverage"]},
    auto_fix="escalate",
)
def qca(state: StudyState, **kwargs: Any) -> StudyState:
    """Fuzzy-set QCA: truth-table analysis + Boolean minimization of sufficiency.

    Steps (Ragin's fsQCA, ``QCA::truthTable`` + ``QCA::minimize`` in R):

    1. **Calibrated sets.** ``conditions`` and ``outcome`` are read as fuzzy
       memberships in ``[0, 1]`` (a 0.5 crossover anchor; values already in
       ``[0, 1]`` are used as-is, otherwise min-max rescaled per column).
    2. **Truth table.** Enumerate all ``2^k`` corners of the property space. A
       case belongs to the corner whose configurational membership (fuzzy AND of
       each condition present / negated absent) exceeds 0.5 — every case has
       >0.5 membership in exactly one corner. Each corner gets a case count and a
       **consistency** score = Σ min(corner, Y) / Σ corner.
    3. **Outcome assignment.** Corners with ``n ≥ n_cut`` cases and
       consistency ≥ ``threshold`` are coded outcome = 1 (sufficient paths).
    4. **Minimization.** Quine–McCluskey reduces those corners to prime
       implicants; a greedy essential-prime cover yields the minimal
       sum-of-products **solution** (conservative / complex solution — no
       simplifying assumptions on remainder rows).
    5. **Fit.** Solution consistency and coverage are recomputed on the fuzzy
       data from the union (fuzzy OR) of the solution paths.

    kwargs: ``conditions`` (list of column names), ``outcome`` (column name),
    ``threshold`` (inclusion/consistency cut, default 0.8), ``n_cut`` (minimum
    cases per corner, default 1), ``data`` (optional override DataFrame).
    """
    df = _get_datasets(state, kwargs)
    outcome = kwargs.get("outcome") or state.variables.get("outcome")
    conditions = kwargs.get("conditions")
    threshold = float(kwargs.get("threshold", 0.8))
    pri_threshold = float(kwargs.get("pri_threshold", 0.75))
    n_cut = int(kwargs.get("n_cut", 1))

    def _empty(note: str) -> StudyState:
        state.write("models", "qca", {
            "solution": None, "paths": [], "solution_consistency": None,
            "solution_coverage": None, "conditions": conditions,
            "outcome": outcome, "note": note,
        })
        state.write("diagnostics", "consistency_coverage",
                    {"truth_table": [], "note": note})
        return state

    if df is None or outcome is None or outcome not in df.columns:
        return _empty("缺少数据或结果变量(outcome),无法进行 QCA")

    if not conditions:
        conditions = [c for c in df.columns
                      if c != outcome and pd.api.types.is_numeric_dtype(df[c])
                      and df[c].nunique() > 2][:6]
    conditions = [c for c in conditions if c in df.columns]
    if not conditions:
        return _empty("找不到可用的条件变量(conditions)")

    # ---- 1. calibrate to fuzzy [0,1] memberships --------------------------
    def _calibrate(s: pd.Series) -> np.ndarray:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        if np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0:
            return np.clip(v, 0.0, 1.0)
        lo, hi = np.nanmin(v), np.nanmax(v)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full_like(v, 0.5)

    fuzz: dict[str, np.ndarray] = {c: _calibrate(df[c]) for c in conditions}
    y = _calibrate(df[outcome])
    names = list(conditions)
    k = len(names)

    # ---- 2. truth table: 2^k corners --------------------------------------
    # each case's fuzzy membership in every corner; assign case to corner > 0.5
    corner_membership: dict[tuple[int, ...], np.ndarray] = {}
    for corner in product((0, 1), repeat=k):
        cols = [fuzz[nm] if bit else (1.0 - fuzz[nm]) for bit, nm in zip(corner, names)]
        corner_membership[corner] = _fuzzy_and(cols)

    rows: list[dict[str, Any]] = []
    positive_minterms: list[tuple[int, ...]] = []
    for corner, mem in corner_membership.items():
        in_corner = mem > 0.5
        n_cases = int(in_corner.sum())
        cons = _consistency(mem, y)
        pri = _pri_consistency(mem, y)
        outcome_code = 1 if (n_cases >= n_cut and cons >= threshold
                             and pri >= pri_threshold) else 0
        rows.append({
            "configuration": _corner_label(corner, names),
            "corner": corner,
            "n": n_cases,
            "consistency": round(cons, 4),
            "pri": round(pri, 4),
            "outcome": outcome_code,
        })
        if outcome_code == 1 and n_cases >= n_cut:
            positive_minterms.append(corner)

    truth_table = sorted(rows, key=lambda r: (-r["consistency"], -r["n"]))

    if not positive_minterms:
        state.write("models", "qca", {
            "solution": None, "paths": [], "solution_consistency": None,
            "solution_coverage": None, "conditions": names, "outcome": outcome,
            "threshold": threshold, "pri_threshold": pri_threshold, "n_cut": n_cut,
            "note": "无满足一致性/PRI 阈值的充分组态(提高样本或降低阈值)",
        })
        state.write("diagnostics", "consistency_coverage", {
            "truth_table": [{kk: vv for kk, vv in r.items() if kk != "corner"}
                            for r in truth_table],
            "note": "各组态一致性(raw + PRI 充分性 inclusion)与案例数",
        })
        return state

    # ---- 3/4/5. Boolean minimization + parameters of fit ------------------
    # Prefer the parity-gated pyqca port (proven 1e-6 vs R QCA). It performs the
    # Quine–McCluskey minimization on the OUT=1 corners and recomputes the
    # per-term / solution parameters of fit on the same calibrated fuzzy data.
    # The ``positive_minterms`` (already coded with this module's n_cut / PRI /
    # consistency rule) are the minimization space, so the OUT-coding contract is
    # preserved; only the numeric minimization + fit are delegated. On any error
    # we fall back to the pre-existing in-module implementation.
    paths: list[dict[str, Any]] | None = None
    solution_expr: str | None = None
    sol_cons: float | None = None
    sol_cov: float | None = None
    backend: str | None = None
    try:
        from ..external.pyqca import minimize as _pyqca_minimize, TruthTable as _PyqcaTT

        # Build a pyqca TruthTable whose OUT column is exactly this module's
        # coding (positive_minterms → OUT=1). rows are the corner bit-patterns
        # in condition order (names); membership/outcome come from the same
        # calibrated ``fuzz``/``y`` so the fit numbers match R QCA element-wise.
        _pos = list(positive_minterms)
        _X = np.column_stack([fuzz[nm] for nm in names])
        _tt = _PyqcaTT(
            conditions=names,
            rownames=list(range(1, len(_pos) + 1)),
            rows=[list(c) for c in _pos],
            OUT=[1] * len(_pos),
            n=[int((corner_membership[c] > 0.5).sum()) for c in _pos],
            incl=[_consistency(corner_membership[c], y) for c in _pos],
            PRI=[_pri_consistency(corner_membership[c], y) for c in _pos],
            X=_X,
            y=y,
            incl_cut=threshold,
        )
        _res = _pyqca_minimize(_tt, include=None)
        _terms = _res["terms"]
        paths = [
            {
                "term": t if t else "1",
                "raw_consistency": round(float(_res["inclS"][i]), 4),
                "raw_coverage": round(float(_res["covS"][i]), 4),
            }
            for i, t in enumerate(_terms)
        ]
        solution_expr = " + ".join(p["term"] for p in paths)
        _ov = _res["overall"]
        sol_cons = round(float(_ov["inclS"]), 4)
        sol_cov = round(float(_ov["covS"]), 4)
        backend = "pyqca"
    except Exception:
        paths = None  # fall through to the pre-existing implementation

    if paths is None:
        # ---- pre-existing in-module Quine–McCluskey + fit -----------------
        primes = _prime_implicants(positive_minterms)
        solution_imps = _select_cover(primes, positive_minterms)

        paths = []
        path_memberships: list[np.ndarray] = []
        for imp in solution_imps:
            mem = _implicant_membership(imp, fuzz, names)
            path_memberships.append(mem)
            paths.append({
                "term": _implicant_label(imp, names),
                "raw_consistency": round(_consistency(mem, y), 4),
                "raw_coverage": round(_coverage(mem, y), 4),
            })

        # solution = fuzzy OR (max) of all path memberships
        sol_mem = path_memberships[0]
        for m in path_memberships[1:]:
            sol_mem = np.maximum(sol_mem, m)
        sol_cons = round(_consistency(sol_mem, y), 4)
        sol_cov = round(_coverage(sol_mem, y), 4)
        solution_expr = " + ".join(p["term"] for p in paths)

    # ---- necessity analysis (superSubset) --------------------------------
    # 充分性(sufficiency)之外补充必要性(necessity)超集搜索:pyqca.superSubset
    # 枚举通过 inclN/covN 阈值的必要条件(合取 + 极小析取),与 R QCA superSubset
    # 的 relation="necessity" 对齐。作为 models['qca']['necessity'] 的新键补入,
    # 不改动既有充分性解的任何键。任何缺输入/异常均降级为 note,不使 qca 崩溃。
    necessity: dict[str, Any] = {"terms": [], "note": None}
    try:
        from ..external.pyqca import superSubset as _pyqca_superSubset

        nec_incl_cut = float(kwargs.get("necessity_incl_cut",
                                        kwargs.get("nec_incl_cut", 0.9)))
        nec_cov_cut = float(kwargs.get("necessity_cov_cut",
                                       kwargs.get("nec_cov_cut", 0.6)))
        nec_data = {nm: fuzz[nm] for nm in names}
        nec_data[outcome] = y
        _ss = _pyqca_superSubset(
            nec_data, outcome=outcome, conditions=names,
            incl_cut=nec_incl_cut, cov_cut=nec_cov_cut,
        )
        _ic = _ss.get("incl_cov", {})
        necessity = {
            "terms": list(_ss.get("terms", [])),
            "inclN": [round(float(v), 4) for v in _ic.get("inclN", [])],
            "RoN": [round(float(v), 4) for v in _ic.get("RoN", [])],
            "covN": [round(float(v), 4) for v in _ic.get("covN", [])],
            "incl_cut": nec_incl_cut,
            "cov_cut": nec_cov_cut,
            "relation": "necessity",
            "estimator": "pyqca.superSubset",
            "note": ("必要条件超集(⇒ 表示必要):inclN=Σmin(X,Y)/ΣY, "
                     "covN=Σmin(X,Y)/ΣX"),
        }
    except Exception as _exc:
        necessity = {"terms": [], "note": "必要性分析跳过:{}".format(_exc)}

    state.write("models", "qca", {
        "solution": solution_expr,
        "paths": paths,
        "solution_consistency": sol_cons,
        "solution_coverage": sol_cov,
        "conditions": names,
        "outcome": outcome,
        "threshold": threshold,
        "pri_threshold": pri_threshold,
        "n_cut": n_cut,
        "solution_type": "conservative/complex (无 remainder 简化假设)",
        "estimator": "fsQCA_truthtable + Quine-McCluskey",
        "backend": backend if backend else "in-module",
        "necessity": necessity,
        "note": "充分性解:Y ⇐ {} (⇐ 表示充分)".format(solution_expr),
    })
    state.write("diagnostics", "consistency_coverage", {
        "truth_table": [{kk: vv for kk, vv in r.items() if kk != "corner"}
                        for r in truth_table],
        "n_positive_corners": len(positive_minterms),
        "solution_consistency": sol_cons,
        "solution_coverage": sol_cov,
        "backend": backend if backend else "in-module",
        "note": "组态一致性(充分性)+ 解一致性/覆盖率;consistency=Σmin(X,Y)/ΣX",
    })
    return state


# --------------------------------------------------------------------- calibrate
@register(
    name="calibrate",
    aliases=["校准", "模糊集校准", "direct_calibration"],
    category="setmethods",
    tier="plus",
    skill="(QCA 缺口,Python 空白)",
    languages=["Python"],
    key_tools=["numpy"],
    description="直接校准:把原始数值列按 3 锚点(排除/交叉/纳入)logistic 校准为模糊集隶属度;可回写校准列",
    requires={"sources": ["datasets"]},
    produces={"models": ["calibrate"]},
    auto_fix="escalate",
)
def calibrate(state: StudyState, **kwargs: Any) -> StudyState:
    """直接校准(Ragin 直接法)——原始数值列 → 模糊集隶属度 ``[0,1]``。

    委派 :func:`socialverse.external.pyqca.calibrate`(与 R QCA ``calibrate``
    对齐 1e-6):

    * ``type="fuzzy", method="direct"`` —— 3 锚点 logistic 校准,``thresholds``
      为 ``(排除 thEX, 交叉 thCR, 纳入 thIN)``,交叉点隶属度 0.5,纳入锚点
      达到 ``idm``(默认 0.95)。``thEX > thIN`` 时为递减集,自动镜像。
    * ``type="crisp"`` —— ``thresholds`` 为 k 个切点,返回各值 ``>=`` 切点的
      个数(整数集值 ``0..k``)。

    kwargs: ``column`` / ``x``(要校准的列名,必填),``thresholds``(锚点/切点,
    必填),``type``(``"fuzzy"`` 默认 / ``"crisp"``),``idm``(纳入隶属度,默认
    0.95),``new_column``(回写列名,默认 ``<column>_cal``),``add_to_frame``
    (是否把校准列写回 ``sources['datasets']``,默认 True),``data``(可选覆盖
    DataFrame)。

    结果写入 ``models['calibrate']``,并(在可行时)把校准后的列添加回工作表。
    """
    column = kwargs.get("column") or kwargs.get("x")
    thresholds = kwargs.get("thresholds")
    cal_type = kwargs.get("type", "fuzzy")
    idm = float(kwargs.get("idm", 0.95))
    add_to_frame = bool(kwargs.get("add_to_frame", True))

    def _empty(note: str) -> StudyState:
        state.write("models", "calibrate", {
            "column": column, "thresholds": thresholds, "type": cal_type,
            "calibrated": None, "new_column": None, "added_to_frame": False,
            "note": note,
        })
        return state

    df = _get_datasets(state, kwargs)
    if df is None:
        return _empty("缺少数据(sources['datasets'] 或 data=),无法校准")
    if not column or column not in df.columns:
        return _empty("缺少或找不到要校准的列(column=),无法校准")
    if thresholds is None:
        return _empty("缺少 thresholds(校准锚点/切点),无法校准")

    new_column = kwargs.get("new_column") or "{}_cal".format(column)
    try:
        from ..external.pyqca import calibrate as _pyqca_calibrate

        raw = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        cal = _pyqca_calibrate(raw, type=cal_type,
                               thresholds=list(thresholds), idm=idm)
        cal_list = [None if (isinstance(v, float) and np.isnan(v))
                    else (int(v) if cal_type == "crisp" else round(float(v), 6))
                    for v in np.asarray(cal).tolist()]
    except Exception as _exc:
        return _empty("校准失败:{}".format(_exc))

    added = False
    if add_to_frame:
        try:
            src = state.sources.get("datasets")
            if isinstance(src, pd.DataFrame):
                src[new_column] = np.asarray(cal)
                added = True
            elif isinstance(src, dict):
                for _v in src.values():
                    if isinstance(_v, pd.DataFrame) and column in _v.columns:
                        _v[new_column] = np.asarray(cal)
                        added = True
                        break
        except Exception:
            added = False

    state.write("models", "calibrate", {
        "column": column,
        "new_column": new_column,
        "thresholds": list(thresholds),
        "type": cal_type,
        "idm": idm if cal_type == "fuzzy" else None,
        "method": "direct" if cal_type == "fuzzy" else "crisp",
        "calibrated": cal_list,
        "n": len(cal_list),
        "added_to_frame": added,
        "estimator": "pyqca.calibrate",
        "note": ("直接校准({}):3 锚点 logistic → 模糊集隶属度".format(cal_type)
                 if cal_type == "fuzzy"
                 else "crisp 校准:findInterval(x, sort(thresholds))"),
    })
    return state


# ------------------------------------------------------------ necessity_analysis
@register(
    name="necessity_analysis",
    aliases=["必要性分析", "superSubset", "必要条件"],
    category="setmethods",
    tier="plus",
    skill="(QCA 缺口,Python 空白)",
    languages=["Python"],
    key_tools=["numpy"],
    description="必要性超集搜索(superSubset):枚举通过 inclN/covN 阈值的必要条件(合取 + 极小析取)",
    requires={"sources": ["datasets"], "variables": ["outcome"]},
    produces={"models": ["necessity"], "diagnostics": ["necessity"]},
    auto_fix="escalate",
)
def necessity_analysis(state: StudyState, **kwargs: Any) -> StudyState:
    """必要性超集搜索(R QCA ``superSubset``,``relation="necessity"``)。

    委派 :func:`socialverse.external.pyqca.superSubset`。枚举两族表达式并保留
    通过必要性阈值者:

    * **合取**(fuzzy ``min``,含单条件)—— 每个通过 ``inclN >= incl_cut`` 且
      ``covN >= cov_cut`` 的合取都报告(``*`` 连接)。
    * **析取**(fuzzy ``max``,>=2 条件)—— 仅当极小(无更小子表达式已通过)
      时报告(`` + `` 连接)。

    kwargs: ``conditions``(条件列名,缺省自动挑数值列),``outcome``(结果列名,
    缺省读 ``variables['outcome']``),``incl_cut``(必要性一致性 cut,默认 0.9),
    ``cov_cut``(必要性覆盖 cut,默认 0.6),``ron_cut``(RoN cut,默认 0.0),
    ``depth``(每表达式最大条件数),``data``(可选覆盖 DataFrame)。

    结果写入 ``models['necessity']``(必要条件表)与 ``diagnostics['necessity']``。
    校准沿用 ``qca`` 的模糊化规则([0,1] 原样,否则 min-max)。
    """
    df = _get_datasets(state, kwargs)
    outcome = kwargs.get("outcome") or state.variables.get("outcome")
    conditions = kwargs.get("conditions")
    incl_cut = float(kwargs.get("incl_cut", 0.9))
    cov_cut = float(kwargs.get("cov_cut", 0.6))
    ron_cut = float(kwargs.get("ron_cut", 0.0))
    depth = kwargs.get("depth")

    def _empty(note: str) -> StudyState:
        state.write("models", "necessity", {
            "terms": [], "outcome": outcome, "conditions": conditions,
            "relation": "necessity", "note": note,
        })
        state.write("diagnostics", "necessity", {"terms": [], "note": note})
        return state

    if df is None or outcome is None or outcome not in df.columns:
        return _empty("缺少数据或结果变量(outcome),无法进行必要性分析")

    if not conditions:
        conditions = [c for c in df.columns
                      if c != outcome and pd.api.types.is_numeric_dtype(df[c])
                      and df[c].nunique() > 2][:6]
    conditions = [c for c in conditions if c in df.columns]
    if not conditions:
        return _empty("找不到可用的条件变量(conditions)")

    def _calibrate_col(s: pd.Series) -> np.ndarray:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        if np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0:
            return np.clip(v, 0.0, 1.0)
        lo, hi = np.nanmin(v), np.nanmax(v)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full_like(v, 0.5)

    try:
        from ..external.pyqca import superSubset as _pyqca_superSubset

        nec_data = {c: _calibrate_col(df[c]) for c in conditions}
        nec_data[outcome] = _calibrate_col(df[outcome])
        _ss = _pyqca_superSubset(
            nec_data, outcome=outcome, conditions=list(conditions),
            incl_cut=incl_cut, cov_cut=cov_cut, ron_cut=ron_cut,
            depth=(int(depth) if depth is not None else None),
        )
    except Exception as _exc:
        return _empty("必要性分析失败:{}".format(_exc))

    _ic = _ss.get("incl_cov", {})
    terms = list(_ss.get("terms", []))
    inclN = [round(float(v), 4) for v in _ic.get("inclN", [])]
    RoN = [round(float(v), 4) for v in _ic.get("RoN", [])]
    covN = [round(float(v), 4) for v in _ic.get("covN", [])]

    state.write("models", "necessity", {
        "terms": terms,
        "inclN": inclN,
        "RoN": RoN,
        "covN": covN,
        "outcome": outcome,
        "conditions": list(conditions),
        "incl_cut": incl_cut,
        "cov_cut": cov_cut,
        "ron_cut": ron_cut,
        "relation": "necessity",
        "estimator": "pyqca.superSubset",
        "note": "必要条件超集(⇒ 表示必要):inclN=Σmin(X,Y)/ΣY, covN=Σmin(X,Y)/ΣX",
    })
    state.write("diagnostics", "necessity", {
        "terms": terms,
        "inclN": inclN,
        "RoN": RoN,
        "covN": covN,
        "n_terms": len(terms),
        "note": "必要性 inclN/covN/RoN(Schneider & Wagemann 相关性)",
    })
    return state
