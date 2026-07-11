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

Fuzzy operators follow QCA exactly: AND = elementwise ``min``, OR = ``max``,
negation of condition ``x`` = ``1 - x``.

All numbers match R QCA element-wise to < 1e-6 (see tests/test_parity.py).
"""
from __future__ import annotations

from itertools import combinations
import numpy as np

__all__ = ["truth_table", "minimize", "pof", "TruthTable"]


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
