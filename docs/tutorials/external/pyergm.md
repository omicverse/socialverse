# pyergm — ergm in Python

> Exponential Random Graph Models (ERGM) for social networks, callable from Python at 1e-6 parity with R `ergm`, no R runtime required.

## What `ergm` does

`ergm` (part of the `statnet` suite) is the standard R toolkit for fitting Exponential Random Graph Models — statistical models of network formation in which the probability of an observed graph is proportional to `exp(θ · g(y))` for a vector of sufficient statistics `g(y)` (edge count, triangles, degree distribution, homophily terms, reciprocity, …). Social scientists reach for it to test hypotheses like "does homophily on wealth explain this marriage network better than chance" or "is reciprocity a significant structural driver of this advice network," while properly accounting for the dependence between ties that ordinary logistic regression ignores. `ergm` also exposes `summary(net ~ terms)` for the *observed* sufficient statistics of a network (no estimation involved) and, via companion package `sna`, the Holland–Leinhardt directed triad census — both frequently used as descriptive network summaries or as goodness-of-fit targets.

## The port

Public functions exposed by `socialverse.external.pyergm` (see `__init__.py` `__all__`):

- `dyads(n, directed=False)` — enumerates all dyad index pairs of an n-node network (upper-triangle for undirected, all ordered `i != j` for directed).
- `change_stats_edges(pairs)` — the `edges` term's change statistic (constant 1 for every dyad).
- `change_stats_nodecov(pairs, attr)` — the `nodecov(attr)` change statistic, `attr[i] + attr[j]`.
- `change_stats_nodematch(pairs, attr)` — the `nodematch(attr)` change statistic, `1{attr[i] == attr[j]}`.
- `build_design(adjacency, terms, directed=False)` — assembles the dyadic design matrix `X`, tie-indicator response `y`, and column labels for a set of dyad-independent terms.
- `ergm_mple(adjacency, terms, directed=False, max_iter=100, tol=1e-12)` — fits a dyad-independent ERGM by maximum pseudo-likelihood (IRLS logistic regression on change statistics); returns an `MPLEResult`.
- `MPLEResult` — dataclass holding `terms`, `coef`, `se`, `vcov`, `n_iter`, `loglik`, with a `.summary()` pretty-printer.
- `summary_formula(adjacency, terms, directed=False, attr_name=None)` — observed sufficient statistics `g(y)` for a requested term set (`edges`, `triangle`, `degree`/`idegree`/`odegree`, `kstar`/`istar`/`ostar`, `mutual`, `nodecov`, `nodematch`); exact counts, no estimation.
- `triad_census(adjacency)` — the Holland–Leinhardt 16-type directed triad census (`sna::triad.census`), with column order given by `TRIAD_CENSUS_LABELS`.
- `TRIAD_CENSUS_LABELS` — the 16 MAN-code labels (`"003"`, `"012"`, …, `"300"`) in the order `triad_census` returns them.

The port is pure `numpy`/`scipy` (IRLS via `scipy.linalg.solve`/`inv`) — no R runtime, no `rpy2`. It is scoped to **dyad-independent** terms (`edges`, `nodecov`, `nodematch`): for these the ERGM pseudo-likelihood factorizes exactly into a logistic regression on change statistics, so the fit is deterministic and convex. Dyad-*dependent* terms (`triangle`, `gwesp`, k-stars as an estimation target, MCMC-MLE generally) are out of scope for `ergm_mple` — they require stochastic MCMC-MLE in real `ergm` and are not reproduced here.

It is wired into socialverse as two registered functions in `socialverse/tl/_network2.py`:

- `sv.tl.ergm` — fits an ERGM by MPLE on a directed edgelist. It delegates to `ergm_mple` whenever the requested `terms` are a subset of `{edges, nodecov, nodematch}` (the port's exact scope); for `mutual`/`transitive` terms it falls back to an in-module change-statistics + logistic-regression engine, since those are dyad-dependent and outside `ergm_mple`'s guarantee.
- `sv.tl.network_statistics` — delegates directly to `summary_formula` (observed sufficient statistics) and `triad_census` (16-type MAN census) for a directed edgelist; always uses the port when a usable edge table is present.

:::{admonition} Parity gate
:class: note

The port is pinned to R `ergm`/`sna` at `max_abs_err < 1e-6` on 8 deterministic parity tests (`socialverse/external/pyergm/tests/test_parity.py`), run against the canonical Padgett Florentine marriage network (`ergm::flomarriage`, 16 nodes, undirected) plus a small 5-node directed fixture for the directed-only terms.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pyergm import (
    ergm_mple,
    summary_formula,
    triad_census,
    TRIAD_CENSUS_LABELS,
)

# --- a tiny undirected network with a numeric vertex covariate ("wealth") ---
adjacency = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=float)
wealth = np.array([10.0, 25.0, 40.0, 15.0, 5.0])

# 1) MPLE fit: edges + nodecov(wealth) -- dyad-independent, deterministic
fit = ergm_mple(adjacency, ["edges", ("nodecov", wealth)], directed=False)
print(fit.summary())        # per-term coefficient + model-based SE
print("log pseudo-likelihood:", fit.loglik)

# 2) observed sufficient statistics -- summary(net ~ edges + triangle + degree(0:3))
stats, labels = summary_formula(
    adjacency,
    terms=["edges", "triangle", ("degree", [0, 1, 2, 3])],
    directed=False,
)
for label, value in zip(labels, stats):
    print(f"{label}: {value:.0f}")

# 3) directed triad census on a small directed network
dir_adjacency = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
], dtype=float)
census = triad_census(dir_adjacency)
print(dict(zip(TRIAD_CENSUS_LABELS, census.astype(int))))

# --- equivalently, via the registered socialverse functions on a StudyState ---
# sv.tl.ergm(state, edges=edge_df, terms=["edges", "nodecov"], wealth=wealth_map)
# sv.tl.network_statistics(state, edges=edge_df, terms=["edges", "mutual"])
```

## R ↔ Python dictionary

| R (`ergm`) | socialverse | notes |
|---|---|---|
| `ergm(net ~ edges + nodecov("wealth"), estimate="MPLE")` | `ergm_mple(adjacency, ["edges", ("nodecov", wealth)])` or `sv.tl.ergm(state, edges=..., terms=["edges","nodecov"])` | dyad-independent only; MPLE = logistic regression on change statistics |
| `ergm(net ~ edges + nodematch("attr"))` | `ergm_mple(adjacency, ["edges", ("nodematch", attr)])` | homophily term, dyad-independent |
| `summary(net ~ edges + triangle + degree(0:6) + kstar(2))` | `summary_formula(adjacency, ["edges", "triangle", ("degree", list(range(7))), ("kstar", 2)])` | exact observed sufficient statistics, no estimation |
| `summary(net ~ mutual + istar(2) + ostar(2))` (directed) | `summary_formula(adjacency, ["mutual", ("istar", 2), ("ostar", 2)], directed=True)` | directed dyad-independent + simple dyad-dependent counts |
| `sna::triad.census(net)` | `triad_census(adjacency)` | Holland–Leinhardt 16-type MAN census; labels in `TRIAD_CENSUS_LABELS` |
| `ergm(net ~ edges + mutual + gwesp(...), estimate="MCMC-MLE")` | not ported (stochastic; falls back to socialverse's internal change-stats logistic engine inside `sv.tl.ergm`) | dyad-dependent terms need intractable-normalizing-constant MCMC-MLE |

## Parity evidence

8 deterministic parity tests in `socialverse/external/pyergm/tests/test_parity.py`, gated against R `ergm`/`sna` output in `reference.json`:

- `test_mple_coef`, `test_mple_se` — MPLE coefficients and model-based standard errors for `edges + nodecov(wealth)` on `flomarriage`, gated at `max_abs_err < 1e-6`.
- `test_design_dimensions` — dyadic design shape (120 dyads × 2 predictors for the 16-node undirected network) and response/label sanity checks.
- `test_summary_undirected_stats`, `test_summary_undirected_labels` — observed sufficient statistics (`edges`, `triangle`, `degree0`..`degree6`, `kstar2`, `nodecov`, `nodematch`) on `flomarriage`, exact-equality (0 tolerance — these are integer/real counts, not estimates).
- `test_summary_directed_stats` — observed statistics (`edges`, `mutual`, `istar2`, `ostar2`, `idegree1/2`, `odegree1/2`) on a 5-node directed fixture, exact-equality.
- `test_triad_census_counts`, `test_triad_census_labels_and_total` — the 16-class directed triad census on the same directed fixture, exact-equality plus a total-count sanity check (`sum == C(n,3)`).

:::{admonition} Stochastic terms out of scope
:class: warning

`ergm_mple` only covers **dyad-independent** terms (`edges`, `nodecov`, `nodematch`), for which the pseudo-likelihood factorizes exactly and MPLE is a convex logistic regression — genuinely deterministic and therefore parity-gated at 1e-6. Dyad-dependent terms (`triangle`, `gwesp`, k-star estimation, general MCMC-MLE) and dynamic SAOM models (RSiena) are stochastic in R and are **not** reproduced by this port; `sv.tl.ergm` falls back to its own approximate change-statistics logistic engine for those terms rather than claiming 1e-6 parity.
:::

To reproduce the gate locally:

```bash
Rscript socialverse/external/pyergm/tests/r_reference_driver.R
pytest socialverse/external/pyergm/tests/
```

## In the socialverse workflow

Day-to-day, call `sv.tl.ergm` to fit an ERGM on a directed edgelist (it silently prefers the parity-gated `ergm_mple` path for `edges`/`nodecov`/`nodematch` terms and falls back otherwise), or `sv.tl.network_statistics` for observed sufficient statistics plus the triad census. Both are registered in the `net` category with `requires={"sources": ["datasets"]}` — the registry enforces this contract, and `registry_lookup("ergm")` / `sv.list_functions()` will confirm the live signature and `requires`/`produces` before you build a workflow around it.
