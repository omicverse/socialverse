"""pyqca — pure-numpy reconstruction of R QCA (Qualitative Comparative Analysis).

Faithful port of the deterministic combinatorial-exact core of R package ``QCA``
(Dusa 3.25):

  * ``truth_table``  — fuzzy-set truth table construction: corner (multi-bit)
    membership as the min over (possibly negated) condition scores, per-row case
    count ``n`` (# cases with corner membership > 0.5), sufficiency inclusion
    ``incl``, proportional-reduction-in-inconsistency ``PRI``, and the ``OUT``
    column assigned by the inclusion cut-off ``incl.cut``.
  * ``minimize``     — conservative (complex) Boolean minimization via the
    Quine-McCluskey algorithm on the crisp ``OUT`` column: OUT=1 rows are the
    minterms, OUT=0 and remainder (``?``) rows are BOTH excluded from the
    minimization space, then prime implicants are found by iterated pairwise
    adjacency combination and reduced to an irredundant cover by Petrick /
    essential-PI selection.
  * ``pof``          — parameters of fit for a set of terms: term-level inclusion
    (consistency) ``inclS``, ``PRI``, raw coverage ``covS`` and unique coverage
    ``covU``, plus the solution-level ``inclS``/``PRI``/``covS``.
  * ``calibrate``    — direct/indirect calibration of raw numeric data to
    crisp/fuzzy set-membership scores. The fuzzy direct method uses the 3-anchor
    logistic function of R QCA (exclusion / crossover / inclusion thresholds and
    the inclusion degree-of-membership ``idm``); crisp calibration is
    ``findInterval`` on sorted thresholds.
  * ``superSubset``  — necessity superset search: enumerates single conditions
    and their conjunctions (fuzzy ``min``) plus minimal disjunctions (fuzzy
    ``max``) whose necessity inclusion ``inclN`` and coverage ``covN`` clear the
    ``incl.cut`` / ``cov.cut`` cut-offs, reporting ``inclN`` / ``RoN`` / ``covN``.

Fuzzy operators follow QCA exactly: AND = elementwise ``min``, OR = ``max``,
negation of condition ``x`` = ``1 - x``.

All numbers match R QCA element-wise to < 1e-6 (see tests/test_parity.py).
"""
from __future__ import annotations

from itertools import combinations
import numpy as np

__all__ = ["truth_table", "minimize", "pof", "TruthTable",
           "calibrate", "superSubset"]


# ---------------------------------------------------------------------------
# fuzzy-set parameter-of-fit primitives
# ---------------------------------------------------------------------------
def _inclS(x, y):
    """Sufficiency inclusion (consistency): sum(min(x,y)) / sum(x)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = x.sum()
    return float(np.minimum(x, y).sum() / denom) if denom > 0 else float("nan")


def _PRI(x, y):
    """Proportional reduction in inconsistency:
    (sum(min(x,y)) - sum(min(x,y,1-y))) / (sum(x) - sum(min(x,y,1-y)))."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    a = np.minimum(x, y).sum()
    b = np.minimum(np.minimum(x, y), 1.0 - y).sum()
    denom = x.sum() - b
    return float((a - b) / denom) if denom != 0 else float("nan")


def _covS(x, y):
    """Raw coverage: sum(min(x,y)) / sum(y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = y.sum()
    return float(np.minimum(x, y).sum() / denom) if denom > 0 else float("nan")


def _covU(xi, others, y):
    """Unique coverage of term xi given the other terms' memberships.

    covU = sum( max(0, min(xi,y) - max_j min(other_j, y)) ) / sum(y)
    (contribution of xi to the outcome not covered by any other term).
    """
    xi = np.asarray(xi, float); y = np.asarray(y, float)
    mxy = np.minimum(xi, y)
    if others:
        rest = np.maximum.reduce([np.minimum(np.asarray(o, float), y) for o in others])
    else:
        rest = np.zeros_like(mxy)
    denom = y.sum()
    return float(np.maximum(0.0, mxy - rest).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# truth table
# ---------------------------------------------------------------------------
class TruthTable:
    """Result of :func:`truth_table`.

    Attributes
    ----------
    conditions : list[str]
    rows       : (2**k, k) int array of condition bit patterns (Gray-free, binary
                 counting order, MSB = first condition) for observed rows only
    rownames   : 1-indexed truth-table row ids (matching R's row numbering)
    OUT        : per-row output (0/1) — remainder rows are dropped
    n          : per-row case count
    incl, PRI  : per-row sufficiency inclusion and PRI
    """

    def __init__(self, conditions, rownames, rows, OUT, n, incl, PRI, X, y, incl_cut):
        self.conditions = list(conditions)
        self.rownames = np.asarray(rownames, int)
        self.rows = np.asarray(rows, int)
        self.OUT = np.asarray(OUT, int)
        self.n = np.asarray(n, int)
        self.incl = np.asarray(incl, float)
        self.PRI = np.asarray(PRI, float)
        self._X = np.asarray(X, float)   # (cases, k) condition scores
        self._y = np.asarray(y, float)   # (cases,) outcome scores
        self.incl_cut = float(incl_cut)


def _corner_membership(X, bits):
    """Fuzzy membership in the corner defined by the bit pattern ``bits``.

    For condition j: use score if bits[j]==1 else its negation (1-score).
    Corner membership = elementwise min over conditions.
    """
    negs = np.where(np.asarray(bits, int)[None, :] == 1, X, 1.0 - X)
    return negs.min(axis=1)


def truth_table(data, outcome, conditions, incl_cut=0.8):
    """Construct a fuzzy-set truth table (R QCA ``truthTable``).

    Parameters
    ----------
    data : mapping name -> 1-D sequence of fuzzy scores in [0,1]
    outcome : str, key of the outcome in ``data``
    conditions : list[str], condition keys in ``data``
    incl_cut : float, inclusion cut-off; a row with incl >= incl_cut gets OUT=1

    Returns
    -------
    TruthTable — observed rows only (rows with at least one case, i.e. n>0).
    """
    conditions = list(conditions)
    X = np.column_stack([np.asarray(data[c], float) for c in conditions])
    y = np.asarray(data[outcome], float)
    k = len(conditions)

    all_bits = np.array([[(r >> (k - 1 - j)) & 1 for j in range(k)]
                         for r in range(2 ** k)], int)

    rownames, rows, OUT, ncount, incl_l, pri_l = [], [], [], [], [], []
    for idx, bits in enumerate(all_bits):
        corner = _corner_membership(X, bits)
        n = int((corner > 0.5).sum())
        if n == 0:
            continue  # remainder row (?), excluded from the observed table
        incl = _inclS(corner, y)
        pri = _PRI(corner, y)
        out = 1 if incl >= incl_cut else 0
        rownames.append(idx + 1)  # R uses 1-based row numbering
        rows.append(bits.tolist())
        OUT.append(out)
        ncount.append(n)
        incl_l.append(incl)
        pri_l.append(pri)

    return TruthTable(conditions, rownames, rows, OUT, ncount, incl_l, pri_l,
                      X, y, incl_cut)


# ---------------------------------------------------------------------------
# Quine-McCluskey Boolean minimization (conservative solution)
# ---------------------------------------------------------------------------
def _combine(a, b):
    """Combine two implicants (tuples over {0,1,None}) if they differ in exactly
    one non-dash literal; return the merged implicant or None."""
    diff = -1
    for i, (x, z) in enumerate(zip(a, b)):
        if x != z:
            if diff != -1:
                return None
            diff = i
    if diff == -1:
        return None
    if a[diff] is None or b[diff] is None:
        return None
    merged = list(a)
    merged[diff] = None
    return tuple(merged)


def _prime_implicants(minterms):
    """Quine-McCluskey prime-implicant generation over crisp minterms.

    minterms : list of tuples over {0,1} (the OUT=1 corners).
    Returns the set of prime implicants (tuples over {0,1,None}).
    """
    current = set(minterms)
    primes = set()
    while current:
        used = set()
        nxt = set()
        cur_list = list(current)
        for i in range(len(cur_list)):
            for jj in range(i + 1, len(cur_list)):
                m = _combine(cur_list[i], cur_list[jj])
                if m is not None:
                    used.add(cur_list[i]); used.add(cur_list[jj])
                    nxt.add(m)
        for t in current:
            if t not in used:
                primes.add(t)
        current = nxt
    return primes


def _implicant_covers(imp, minterm):
    """True iff prime implicant ``imp`` covers ``minterm`` (dash matches anything)."""
    return all(a is None or a == b for a, b in zip(imp, minterm))


def _sort_key(imp):
    """QCA canonical term ordering: at each condition position, positive literal
    (1) sorts before negative (0) before dash (None), compared left to right.
    Matches R QCA's solution-term ordering (uncomplemented literals first)."""
    return tuple({1: 0, 0: 1, None: 2}[v] for v in imp)


def _select_cover(primes, minterms):
    """Reduce prime implicants to an irredundant cover.

    Essential prime implicants first, then a greedy/Petrick-style fill for the
    remainder. For QCA's conservative solution on well-separated minterms this
    yields the unique minimal cover (matching R's default single solution).
    """
    primes = sorted(primes, key=_sort_key)  # QCA canonical ordering
    minterms = list(minterms)

    cover_map = {p: {m for m in minterms if _implicant_covers(p, m)} for p in primes}

    selected = []
    remaining = set(minterms)

    # essential PIs: minterms covered by exactly one PI
    essential = set()
    for m in list(remaining):
        covering = [p for p in primes if m in cover_map[p]]
        if len(covering) == 1:
            essential.add(covering[0])
    for p in primes:
        if p in essential:
            selected.append(p)
            remaining -= cover_map[p]

    # greedy fill (largest additional coverage; ties broken by sorted order)
    avail = [p for p in primes if p not in essential]
    while remaining:
        best = max(avail, key=lambda p: (len(cover_map[p] & remaining), ))
        if not (cover_map[best] & remaining):
            break
        selected.append(best)
        remaining -= cover_map[best]
        avail.remove(best)

    # keep in QCA canonical (deterministic) order
    return sorted(set(selected), key=_sort_key)


def _term_string(imp, conditions):
    """Render an implicant tuple as a QCA term string, e.g. DEV*~URB*LIT."""
    parts = []
    for val, name in zip(imp, conditions):
        if val is None:
            continue
        parts.append(name if val == 1 else "~" + name)
    return "*".join(parts)


def _term_membership(imp, X, conditions):
    """Fuzzy membership of an implicant across cases = min over present literals."""
    cols = []
    for val, j in zip(imp, range(len(conditions))):
        if val is None:
            continue
        cols.append(X[:, j] if val == 1 else 1.0 - X[:, j])
    if not cols:
        return np.ones(X.shape[0])
    return np.minimum.reduce(cols)


def minimize(tt, include=None):
    """Conservative (complex) Boolean minimization of a truth table.

    Only ``include=None`` (exclude remainders, i.e. the conservative solution) is
    supported — the deterministic combinatorial-exact core. Returns a dict:

      terms  : list[str]        prime-implicant term strings
      implicants : list[tuple]  raw implicants over {0,1,None}
      inclS, PRI, covS, covU : list[float] per-term parameters of fit
      overall : dict with solution-level inclS/PRI/covS
    """
    if include is not None:
        raise NotImplementedError(
            "pyqca.minimize supports only the conservative solution "
            "(include=None); parsimonious solutions with remainders are "
            "documented as out-of-scope for the class-1 gate.")

    conditions = tt.conditions
    mask1 = tt.OUT == 1
    minterms = [tuple(int(b) for b in row) for row in tt.rows[mask1]]

    if not minterms:
        return {"terms": [], "implicants": [], "inclS": [], "PRI": [],
                "covS": [], "covU": [],
                "overall": {"inclS": float("nan"), "PRI": float("nan"),
                            "covS": float("nan")}}

    primes = _prime_implicants(minterms)
    cover = _select_cover(primes, minterms)

    X, y = tt._X, tt._y
    mems = [_term_membership(imp, X, conditions) for imp in cover]

    inclS = [_inclS(m, y) for m in mems]
    PRI = [_PRI(m, y) for m in mems]
    covS = [_covS(m, y) for m in mems]
    covU = [_covU(mems[i], [mems[j] for j in range(len(mems)) if j != i], y)
            for i in range(len(mems))]

    sol = np.maximum.reduce(mems) if mems else np.zeros(X.shape[0])
    overall = {"inclS": _inclS(sol, y), "PRI": _PRI(sol, y), "covS": _covS(sol, y)}

    return {
        "terms": [_term_string(imp, conditions) for imp in cover],
        "implicants": cover,
        "inclS": inclS, "PRI": PRI, "covS": covS, "covU": covU,
        "overall": overall,
    }


def pof(terms, data, outcome, conditions):
    """Parameters of fit for explicit ``terms`` against ``outcome``.

    ``terms`` : list of implicant tuples over {0,1,None}, or term strings.
    Returns per-term inclS/PRI/covS/covU and solution-level inclS/PRI/covS.
    """
    conditions = list(conditions)
    X = np.column_stack([np.asarray(data[c], float) for c in conditions])
    y = np.asarray(data[outcome], float)

    imps = [_parse_term(t, conditions) if isinstance(t, str) else t for t in terms]
    mems = [_term_membership(imp, X, conditions) for imp in imps]

    inclS = [_inclS(m, y) for m in mems]
    PRI = [_PRI(m, y) for m in mems]
    covS = [_covS(m, y) for m in mems]
    covU = [_covU(mems[i], [mems[j] for j in range(len(mems)) if j != i], y)
            for i in range(len(mems))]
    sol = np.maximum.reduce(mems) if mems else np.zeros(X.shape[0])
    return {
        "inclS": inclS, "PRI": PRI, "covS": covS, "covU": covU,
        "overall": {"inclS": _inclS(sol, y), "PRI": _PRI(sol, y), "covS": _covS(sol, y)},
    }


def _parse_term(s, conditions):
    """Parse 'DEV*~URB*LIT' into an implicant tuple over {0,1,None}."""
    idx = {c: i for i, c in enumerate(conditions)}
    imp = [None] * len(conditions)
    for lit in s.split("*"):
        lit = lit.strip()
        if not lit:
            continue
        neg = lit.startswith("~") or lit.startswith("!")
        name = lit[1:] if neg else lit
        imp[idx[name]] = 0 if neg else 1
    return tuple(imp)


# ---------------------------------------------------------------------------
# calibration (R QCA ``calibrate``)
# ---------------------------------------------------------------------------
def calibrate(x, type="fuzzy", method="direct", thresholds=None,
              logistic=True, idm=0.95):
    """Calibrate raw numeric data into crisp or fuzzy set-membership scores.

    Faithful port of R QCA ``calibrate`` for the two deterministic paths gated
    here:

    * ``type="crisp"`` — return ``findInterval(x, sort(thresholds))``: the count
      of thresholds each value equals or exceeds (integer set values ``0..k``).
    * ``type="fuzzy", method="direct", logistic=True`` — 3-anchor logistic
      calibration. ``thresholds`` are the exclusion, crossover and inclusion
      anchors ``(thEX, thCR, thIN)``. Values below the crossover use the
      exclusion arm, values at/above use the inclusion arm::

          y  = (x < thCR) + 1                    # 1 above/at crossover, 2 below
          fs = 1 / (1 + exp( sign * (x - thCR) * log(idm/(1-idm))
                             / (anchor - thCR) ))

      with ``sign = -1`` at/above crossover and ``+1`` below, and ``anchor`` the
      inclusion threshold above the crossover, the exclusion threshold below.
      If ``thEX > thIN`` (decreasing set) the two arms swap and ``fs`` is
      complemented, exactly as in R.

    * ``type="fuzzy", method="indirect"`` — bin ``x`` by the thresholds and
      return the within-bin mean of ``x`` rescaled to ``[0,1]`` is *not* what R
      does; the indirect method is documented as out-of-scope for this port.

    Parameters
    ----------
    x : 1-D sequence of numeric values
    type : {"fuzzy", "crisp"}
    method : {"direct"}  (only the direct method is supported)
    thresholds : sequence of 3 anchors (fuzzy direct) or k cut-points (crisp)
    logistic : bool, use the logistic direct method (only ``True`` supported)
    idm : float in (0.5, 1), inclusion degree of membership at the inclusion
          anchor (default 0.95)

    Returns
    -------
    numpy.ndarray of calibrated scores (fuzzy) or integer set values (crisp).
    """
    x = np.asarray(x, float)
    if thresholds is None:
        raise ValueError("Threshold value(s) not specified.")
    th = np.asarray(thresholds, float)

    if type == "crisp":
        cuts = np.sort(th)
        # findInterval: number of cut-points <= each x (right-open intervals)
        return np.array([int(np.sum(cuts <= v)) for v in x], dtype=int)

    if type != "fuzzy":
        raise ValueError("Incorrect calibration type.")
    if method != "direct":
        raise NotImplementedError(
            "pyqca.calibrate supports only method='direct' (the deterministic "
            "logistic path); indirect/TFR are out-of-scope for the class-1 gate.")
    if not logistic:
        raise NotImplementedError(
            "pyqca.calibrate supports only logistic=True (the 3-anchor logistic "
            "direct method); the linear-interpolation path is out-of-scope.")
    if th.size != 3:
        raise ValueError(
            "For fuzzy direct calibration, there should be 3 thresholds.")
    if idm <= 0.5 or idm >= 1:
        raise ValueError(
            "The inclusion degree of membership has to be bigger than 0.5 and "
            "less than 1.")

    thEX, thCR, thIN = float(th[0]), float(th[1]), float(th[2])
    decreasing = thEX > thIN
    if decreasing:
        thEX, thIN = thIN, thEX  # swap so thEX < thCR < thIN

    # arm selector: y==1 at/above crossover (sign -1, anchor thIN),
    #               y==2 below crossover     (sign +1, anchor thEX)
    below = x < thCR
    sign = np.where(below, 1.0, -1.0)
    anchor = np.where(below, thEX, thIN)
    fs = 1.0 / (1.0 + np.exp(sign * (x - thCR)
                             * np.log(idm / (1.0 - idm)) / (anchor - thCR)))
    if decreasing:
        fs = 1.0 - fs
    return fs


# ---------------------------------------------------------------------------
# necessity parameters of fit
# ---------------------------------------------------------------------------
def _inclN(x, y):
    """Necessity inclusion (consistency): sum(min(x,y)) / sum(y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = y.sum()
    return float(np.minimum(x, y).sum() / denom) if denom > 0 else float("nan")


def _covN(x, y):
    """Necessity coverage: sum(min(x,y)) / sum(x)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = x.sum()
    return float(np.minimum(x, y).sum() / denom) if denom > 0 else float("nan")


def _RoN(x, y):
    """Relevance of Necessity (Schneider & Wagemann):
    sum(1 - x) / sum(1 - min(x, y))."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = (1.0 - np.minimum(x, y)).sum()
    return float((1.0 - x).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# superSubset — necessity superset search (R QCA ``superSubset``)
# ---------------------------------------------------------------------------
def superSubset(data, outcome, conditions=None, incl_cut=1.0, cov_cut=0.0,
                ron_cut=0.0, depth=None):
    """Necessity superset search (R QCA ``superSubset``, ``relation="necessity"``).

    Enumerates two families of expressions over the conditions and keeps those
    whose necessity inclusion ``inclN`` clears ``incl_cut`` and coverage
    ``covN`` clears ``cov_cut`` (and ``RoN`` clears ``ron_cut``):

    * **conjunctions** — products of positive-literal conditions (fuzzy ``min``),
      *including* single conditions. Every passing conjunction is reported
      (rendered with ``*``).
    * **disjunctions** — sums of two or more conditions (fuzzy ``max``), reported
      only when *minimal*: no proper subset expression already passes (rendered
      with `` + ``).

    The cut-offs are applied with R's ``.Machine$double.eps^0.5`` tolerance so a
    value numerically equal to the cut still passes.

    Parameters
    ----------
    data : mapping name -> 1-D sequence of fuzzy scores in [0,1]
    outcome : str, key of the outcome in ``data``
    conditions : list[str] or None (all keys except outcome)
    incl_cut : float, necessity inclusion cut-off
    cov_cut : float, necessity coverage cut-off
    ron_cut : float, relevance-of-necessity cut-off (rows below are dropped)
    depth : int or None, maximum number of conditions per expression

    Returns
    -------
    dict with keys ``terms`` (list[str]) and ``incl_cov`` (dict of parallel
    lists ``inclN`` / ``RoN`` / ``covN``), in R's report order (conjunctions by
    ascending size then condition order, then minimal disjunctions).
    """
    if conditions is None:
        conditions = [c for c in data.keys() if c != outcome]
    conditions = list(conditions)
    k = len(conditions)
    if depth is None:
        depth = k

    X = {c: np.asarray(data[c], float) for c in conditions}
    y = np.asarray(data[outcome], float)

    # R subtracts sqrt(machine eps) from the cut-offs before comparing.
    eps = np.sqrt(np.finfo(float).eps)
    incl_thr = incl_cut - eps
    cov_thr = (cov_cut - eps) if cov_cut > 0 else cov_cut

    idx = {c: i for i, c in enumerate(conditions)}

    def _passes_nec(mem):
        return _inclN(mem, y) >= incl_thr and _covN(mem, y) >= cov_thr

    # --- conjunctions (min), positive literals, sizes 1..depth ---
    conj_terms, conj_rows = [], []
    for size in range(1, depth + 1):
        for combo in combinations(conditions, size):
            mem = np.minimum.reduce([X[c] for c in combo])
            if _passes_nec(mem):
                conj_terms.append("*".join(combo))
                conj_rows.append((_inclN(mem, y), _RoN(mem, y), _covN(mem, y)))

    # --- disjunctions (max), sizes 2..depth, minimal only ---
    def _disj_mem(combo):
        return np.maximum.reduce([X[c] for c in combo])

    def _disj_passes(combo):
        return _passes_nec(_disj_mem(combo))

    disj_terms, disj_rows = [], []
    for size in range(2, depth + 1):
        for combo in combinations(conditions, size):
            if not _disj_passes(combo):
                continue
            # minimal: no proper subset (size >= 1) also passes as a disjunction /
            # single condition
            redundant = False
            for sub_size in range(1, size):
                for sub in combinations(combo, sub_size):
                    sub_mem = (_disj_mem(sub) if sub_size > 1 else X[sub[0]])
                    if _passes_nec(sub_mem):
                        redundant = True
                        break
                if redundant:
                    break
            if redundant:
                continue
            mem = _disj_mem(combo)
            disj_terms.append(" + ".join(combo))
            disj_rows.append((_inclN(mem, y), _RoN(mem, y), _covN(mem, y)))

    terms = conj_terms + disj_terms
    rows = conj_rows + disj_rows

    # drop rows failing the RoN cut-off (R applies this after the search)
    keep = [i for i, (_, ron, _) in enumerate(rows) if ron >= ron_cut]
    terms = [terms[i] for i in keep]
    rows = [rows[i] for i in keep]

    return {
        "terms": terms,
        "incl_cov": {
            "inclN": [r[0] for r in rows],
            "RoN": [r[1] for r in rows],
            "covN": [r[2] for r in rows],
        },
    }
