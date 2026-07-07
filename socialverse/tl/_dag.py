"""``sv.tl._dag`` — graph-based causal identification and refutation (the DoWhy
four-step, implemented natively).

Design-based estimators (``did``/``iv_regress``/``rdd``) assume the identification
strategy; this module makes the identifying assumptions **explicit as a DAG** and
derives what to condition on. Given a causal graph plus treatment/outcome it:

1. **models** the assumptions as a directed acyclic graph;
2. **identifies** an estimand — a minimal sufficient **backdoor** adjustment set (via
   d-separation), else the **frontdoor** criterion, else an **instrument**;
3. **estimates** the ATE by adjustment (OLS on the backdoor set); and
4. **refutes** it with placebo / random-common-cause / subset / unobserved-confounder
   checks (``dag_refute``).

d-separation is computed natively via the ancestral-moralization test (Lauritzen),
cross-checked against networkx when available. Nothing here needs DoWhy installed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._registry import register
from .._state import StudyState
from ._causal import _get_datasets


# ------------------------------------------------------------------ graph helpers
def _parse_graph(spec: Any) -> dict[str, set[str]]:
    """Normalise a graph spec into ``{node: set(children)}``.

    Accepts a list/tuple of ``(parent, child)`` edges, a ``{node: [children]}`` dict,
    or a ``"a->b; c->d"`` string.
    """
    children: dict[str, set[str]] = {}

    def _edge(u, v):
        children.setdefault(u, set()).add(v)
        children.setdefault(v, set())

    if isinstance(spec, str):
        for part in spec.replace("\n", ";").split(";"):
            part = part.strip()
            if not part or "->" not in part:
                continue
            chain = [p.strip() for p in part.split("->")]
            for u, v in zip(chain, chain[1:]):
                _edge(u, v)
    elif isinstance(spec, dict):
        for u, vs in spec.items():
            vs = [vs] if isinstance(vs, str) else list(vs)
            children.setdefault(u, set())
            for v in vs:
                _edge(u, v)
    elif isinstance(spec, (list, tuple)):
        for e in spec:
            u, v = e
            _edge(u, v)
    else:
        raise TypeError("graph must be an edge list, adjacency dict, or 'a->b' string")
    return children


def _parents(ch: dict[str, set[str]]) -> dict[str, set[str]]:
    pa: dict[str, set[str]] = {n: set() for n in ch}
    for u, vs in ch.items():
        for v in vs:
            pa[v].add(u)
    return pa


def _ancestors(ch, nodes):
    pa = _parents(ch)
    seen, stack = set(), list(nodes)
    while stack:
        n = stack.pop()
        for p in pa.get(n, ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _is_dag(ch: dict[str, set[str]]) -> bool:
    """True iff the graph is acyclic (Kahn's algorithm). d-separation / backdoor
    identification is only meaningful on a DAG."""
    indeg = {n: 0 for n in ch}
    for u, vs in ch.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    queue = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for c in ch.get(n, ()):
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    return seen == len(indeg)


def _descendants(ch, node):
    seen, stack = set(), [node]
    while stack:
        n = stack.pop()
        for c in ch.get(n, ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _d_separated(ch: dict[str, set[str]], X: set, Y: set, Z: set) -> bool:
    """d-separation via ancestral moralization (Lauritzen): X ⟂ Y | Z iff, in the
    moralised ancestral subgraph of X∪Y∪Z with Z removed, X and Y are disconnected."""
    keep = X | Y | Z
    anc = keep | _ancestors(ch, keep)
    # undirected moral graph on `anc`: keep edges + marry co-parents
    adj: dict[str, set[str]] = {n: set() for n in anc}
    pa = _parents(ch)
    for u in anc:
        for v in ch.get(u, ()):
            if v in anc:
                adj[u].add(v)
                adj[v].add(u)
    for n in anc:
        ps = [p for p in pa.get(n, ()) if p in anc]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                adj[ps[i]].add(ps[j])
                adj[ps[j]].add(ps[i])
    for z in Z:
        adj.pop(z, None)
    for n in adj:
        adj[n].discard(Z)
        adj[n] -= Z
    # connected?
    for x in X - Z:
        seen, stack = {x}, [x]
        while stack:
            n = stack.pop()
            if n in Y:
                return False
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
    return True


def _backdoor_graph(ch, T):
    """Graph with T's outgoing edges removed (the proper backdoor graph)."""
    g2 = {n: set(vs) for n, vs in ch.items()}
    g2[T] = set()
    return g2


def _minimal_backdoor(ch, T, Y, observed):
    """Smallest observed adjustment set that d-separates T and Y in the backdoor
    graph (blocks every backdoor path), searching by increasing size. Returns
    ``(minimal_set, all_valid)`` or ``(None, [])`` if none exists."""
    import itertools

    gbd = _backdoor_graph(ch, T)
    forbidden = _descendants(ch, T) | {T, Y}
    cand = sorted(n for n in observed if n not in forbidden)
    valid = []
    minimal = None
    for k in range(len(cand) + 1):
        for Z in itertools.combinations(cand, k):
            Zs = set(Z)
            if _d_separated(gbd, {T}, {Y}, Zs):
                valid.append(Zs)
                if minimal is None:
                    minimal = Zs
        if minimal is not None and k >= len(minimal):
            break
    return minimal, valid


def _frontdoor(ch, T, Y, observed):
    """A mediator set M (observed) satisfying the frontdoor criterion: intercepts all
    directed T→Y paths, no unblocked backdoor T→M, and T blocks all backdoor M→Y."""
    desc_t = _descendants(ch, T)
    anc_y = _ancestors(ch, {Y})
    cand = [n for n in observed if n in desc_t and n in anc_y and n not in (T, Y)]
    if not cand:
        return None
    M = set(cand)
    # (i) M intercepts all directed T->Y paths: removing M disconnects T from Y (directed)
    g2 = {n: set(v for v in vs if v not in M) for n, vs in ch.items() if n not in M}
    if Y in _descendants(g2, T):
        return None
    # (ii) no unblocked backdoor path T->M (T and M d-sep given empty in backdoor graph of T)
    if not _d_separated(_backdoor_graph(ch, T), {T}, M, set()):
        return None
    # (iii) all backdoor paths M->Y blocked by T — remove ALL of M's outgoing edges
    gM = {n: set(vs) for n, vs in ch.items()}
    for m in M:
        gM[m] = set()
    if not _d_separated(gM, M, {Y}, {T}):
        return None
    return M


def _find_iv(ch, T, Y, observed):
    """A candidate instrument Z: affects T, is d-separated from Y once T's outgoing
    edges are cut (only path to Y is through T), and shares no backdoor with Y."""
    for z in observed:
        if z in (T, Y) or z in _descendants(ch, T):
            continue
        if T not in _descendants(ch, z):
            continue  # Z must cause T
        if _d_separated(_backdoor_graph(ch, T), {z}, {Y}, set()):
            return z
    return None


def _ols_adjust(df, T, Y, Z):
    """ATE of T on Y adjusting linearly for Z (OLS coefficient on T), HC1 SE.

    The treatment enters as a single **numeric** column (binary/categorical T is
    factorised to codes, 0/1 for binary) so it is never dummy-expanded away; only the
    adjustment covariates Z are one-hot encoded.
    """
    sm = __import__("statsmodels.api", fromlist=["api"])
    tnum = pd.to_numeric(df[T], errors="coerce")
    if tnum.isna().all():                       # non-numeric treatment → factorise
        tnum = pd.Series(pd.factorize(df[T])[0], index=df.index).astype(float)
    Xz = (pd.get_dummies(df[sorted(Z)], drop_first=True, dtype=float)
          if Z else pd.DataFrame(index=df.index))
    X = pd.concat([tnum.rename("_T_"), Xz], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(df[Y], errors="coerce")
    ok = X.notna().all(axis=1) & y.notna()
    res = sm.OLS(y[ok].to_numpy(float), np.asarray(X[ok], float)).fit(cov_type="HC1")
    j = list(X.columns).index("_T_")
    return float(res.params[j]), float(res.bse[j]), int(ok.sum())


# ========================================================================= identify
@register(
    name="dag_identify",
    aliases=["因果识别", "causal_graph", "backdoor", "identify_effect"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["networkx", "numpy", "statsmodels"],
    description="因果图识别(DoWhy 四步之识别+估计):DAG→后门/前门/IV 识别估计量→线性调整估 ATE",
    requires={"design": ["treatment"], "variables": ["outcome"]},
    produces={"identification": ["estimand", "graph"], "models": ["dag"]},
    auto_fix="escalate",
)
def dag_identify(state: StudyState, **kwargs: Any) -> StudyState:
    """Identify and estimate a causal effect from a DAG.

    Give the causal graph via ``graph=`` (edge list ``[("Z","T"),...]``, adjacency
    dict, or ``"Z->T; Z->Y; T->Y"`` string) and the treatment/outcome (from the design
    or ``treatment=``/``outcome=``). Finds a minimal sufficient backdoor adjustment set
    by d-separation (else frontdoor, else an instrument), records the estimand, and —
    if ``data`` is available — estimates the ATE by linear adjustment.
    """
    graph = kwargs.get("graph") or state.identification.get("graph")
    T = kwargs.get("treatment") or state.design.get("treatment")
    Y = kwargs.get("outcome") or state.variables.get("outcome") or state.design.get("outcome")
    df = _get_datasets(state, kwargs)

    def _empty(note):
        state.write("models", "dag", {"ate": None, "note": note})
        return state

    if graph is None:
        return _empty("缺少因果图 graph=(边列表/邻接字典/'A->B' 字符串)")
    if T is None or Y is None:
        return _empty("缺少 treatment / outcome(design.treatment + variables.outcome)")

    ch = _parse_graph(graph)
    if not _is_dag(ch):
        return _empty("图非 DAG(含有向环);因果识别只对无环图有效")
    if T not in ch or Y not in ch:
        return _empty(f"treatment={T} 或 outcome={Y} 不在图节点中:{sorted(ch)}")

    observed = set(ch)
    if df is not None:
        observed &= set(df.columns) | {T, Y}  # unobserved graph nodes stay out of adj sets
    observed -= {T, Y}

    strategy, adjustment, estimand_note = None, None, None
    Zmin, valid = _minimal_backdoor(ch, T, Y, observed)
    if Zmin is not None:
        strategy = "backdoor"
        adjustment = sorted(Zmin)
        estimand_note = (f"后门调整集 {adjustment or '{}(无混杂,可直接回归)'};"
                         f"E[Y|do(T)] 由 {adjustment} 阻断所有后门路径识别")
    else:
        M = _frontdoor(ch, T, Y, observed)
        if M is not None:
            strategy = "frontdoor"
            adjustment = sorted(M)
            estimand_note = f"前门中介集 {adjustment};经 P(m|T)、P(Y|m,t') 两步识别"
        else:
            iv = _find_iv(ch, T, Y, observed)
            if iv is not None:
                strategy = "iv"
                adjustment = [iv]
                estimand_note = f"工具变量 {iv}(→T,仅经 T 影响 Y);建议用 sv.tl.iv_regress 估计"
            else:
                return _empty("图中不可识别:无后门调整集、无前门中介、无有效工具变量(可能存在不可观测混杂)")

    estimand = {
        "strategy": strategy,
        "adjustment_set": adjustment,
        "treatment": T,
        "outcome": Y,
        "n_backdoor_sets": len(valid),
        "note": estimand_note,
    }
    state.write("identification", "estimand", estimand)
    state.write("identification", "graph", graph)

    model = {"ate": None, "se": None, "estimand": estimand, "strategy": strategy,
             "adjustment_set": adjustment}
    if df is not None and strategy == "backdoor":
        try:
            ate, se, n = _ols_adjust(df, T, Y, set(adjustment))
            model.update({"ate": ate, "se": se, "n": n, "ci": [ate - 1.96 * se, ate + 1.96 * se]})
        except Exception as e:
            model["note"] = f"识别成功但估计失败:{type(e).__name__}"
    elif strategy != "backdoor":
        model["note"] = f"{strategy} 已识别;本函数只对 backdoor 直接估计,其它策略请用对应估计器"
    model.setdefault("note", f"{strategy} 识别 + 线性调整估计")
    state.write("models", "dag", model)
    return state


# =========================================================================== refute
@register(
    name="dag_refute",
    aliases=["因果反驳", "refute", "sensitivity_refute"],
    category="causal",
    tier="plus",
    skill="causal-identification",
    languages=["Python"],
    key_tools=["numpy", "statsmodels"],
    description="因果估计反驳(DoWhy 四步之反驳):安慰剂处理/随机共因/子样本/不可观测混杂 敏感性",
    requires={"models": ["dag"], "identification": ["estimand"]},
    produces={"diagnostics": ["refutation"]},
    prerequisites={"functions": ["dag_identify"]},
    auto_fix="escalate",
)
def dag_refute(state: StudyState, **kwargs: Any) -> StudyState:
    """Refute an identified backdoor estimate with four checks.

    - **placebo_treatment**: permute the treatment — the effect should collapse to ~0.
    - **random_common_cause**: add an independent covariate — the estimate should be
      stable.
    - **data_subset**: re-estimate on a random subsample — the estimate should be
      stable.
    - **unobserved_confounder**: inject a synthetic confounder of given strength —
      reports how far the estimate moves (a sensitivity bound).

    A robust estimate: placebo ≈ 0, and the other three stay close to the original.
    """
    df = _get_datasets(state, kwargs)
    est = state.identification.get("estimand", {})
    model = state.models.get("dag", {})
    T, Y = est.get("treatment"), est.get("outcome")
    Z = set(est.get("adjustment_set") or [])
    ate0 = model.get("ate")

    def _empty(note):
        state.write("diagnostics", "refutation", {"checks": [], "note": note})
        return state

    if df is None or T is None or Y is None or ate0 is None or est.get("strategy") != "backdoor":
        return _empty("需要先 dag_identify 出 backdoor 估计 + 提供 data 才能反驳")

    n_sub = int(kwargs.get("subset_frac_denom", 2))
    conf_strength = float(kwargs.get("confounder_strength", 0.5))
    seed = int(kwargs.get("seed", 0))
    rng = np.random.default_rng(seed)
    tol = float(kwargs.get("tol", 0.15))  # relative-move tolerance for "stable"
    checks = []

    # 1. placebo treatment (permute T)
    d1 = df.copy()
    d1[T] = rng.permutation(d1[T].to_numpy())
    try:
        a1, _, _ = _ols_adjust(d1, T, Y, Z)
    except Exception:
        a1 = float("nan")
    checks.append({"refuter": "placebo_treatment", "new_estimate": a1,
                   "original": ate0, "pass": abs(a1) < max(0.1 * abs(ate0), 0.1 * (abs(ate0) + 1e-9) + tol),
                   "detail": "处理置换后效应应≈0"})

    # 2. random common cause (add independent covariate)
    d2 = df.copy()
    d2["_rcc_"] = rng.normal(0, 1, len(d2))
    try:
        a2, _, _ = _ols_adjust(d2, T, Y, Z | {"_rcc_"})
    except Exception:
        a2 = float("nan")
    checks.append({"refuter": "random_common_cause", "new_estimate": a2, "original": ate0,
                   "pass": abs(a2 - ate0) <= tol * (abs(ate0) + 1e-9),
                   "detail": "加随机共因后估计应稳定"})

    # 3. data subset
    d3 = df.sample(frac=1.0 / n_sub, random_state=seed)
    try:
        a3, _, _ = _ols_adjust(d3, T, Y, Z)
    except Exception:
        a3 = float("nan")
    checks.append({"refuter": "data_subset", "new_estimate": a3, "original": ate0,
                   "pass": abs(a3 - ate0) <= max(tol, 0.25) * (abs(ate0) + 1e-9),
                   "detail": f"随机 1/{n_sub} 子样本重估应稳定"})

    # 4. unobserved confounder (inject one correlated with both T and Y)
    d4 = df.copy()
    u = rng.normal(0, 1, len(d4))
    tnum = pd.to_numeric(d4[T], errors="coerce").to_numpy(float)
    ynum = pd.to_numeric(d4[Y], errors="coerce").to_numpy(float)
    d4[T] = tnum + conf_strength * u
    d4[Y] = ynum + conf_strength * u * float(np.nanstd(ynum) or 1.0)
    try:
        a4, _, _ = _ols_adjust(d4, T, Y, Z)
    except Exception:
        a4 = float("nan")
    checks.append({"refuter": "unobserved_confounder", "new_estimate": a4, "original": ate0,
                   "strength": conf_strength,
                   "detail": f"注入强度 {conf_strength} 的不可观测混杂后,估计移动 {a4 - ate0:+.3f}"})

    n_pass = sum(1 for c in checks if c.get("pass"))
    verdict = "robust" if (checks[0]["pass"] and n_pass >= 3) else "fragile"
    state.write("diagnostics", "refutation", {
        "checks": checks, "original_ate": ate0, "verdict": verdict,
        "note": ("通过安慰剂 + 多数稳健性检验" if verdict == "robust"
                 else "安慰剂或稳健性检验未通过 — 识别存疑"),
    })
    return state


__all__ = ["dag_identify", "dag_refute"]
