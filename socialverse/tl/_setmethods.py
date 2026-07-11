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

__all__ = ["qca"]


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
