# pynetmeta — netmeta in Python

> Frequentist graph-theoretical network (mixed-treatment-comparison) meta-analysis, ported from R's `netmeta` to pure numpy/scipy at 1e-6 parity — no R runtime required.

## What `netmeta` does

`netmeta` is the standard R package for frequentist network meta-analysis (NMA): it pools direct and indirect evidence across a network of three or more treatments using the graph-theoretical / electrical-network approach of Rücker (2012), which is algebraically equivalent to multivariate weighted least squares. Social scientists and health-policy researchers reach for it when a literature contains multiple pairwise comparisons among competing interventions (drugs, programs, policies) that were never all measured head-to-head in the same study, and they need a single coherent ranking plus consistent effect estimates that borrow strength across the whole comparison graph. Beyond pooled estimates it also delivers heterogeneity (Q, τ²), P-score/SUCRA-style ranking, and diagnostics for inconsistency between direct and indirect evidence — node-splitting and net heat / design-based decomposition.

## The port

`socialverse.external.pynetmeta` exposes:

- `netmeta(TE, seTE, treat1, treat2, studlab, reference_group=None, level=0.95, method_tau="DL")` — fits the network model via `prepare()` + the Rücker Laplacian-pseudoinverse machinery (`invmat`, `multiarm`, `nma_ruecker`), for both the common (fixed, τ=0) and random-effects (DerSimonian-Laird τ²) models in one call. Returns a `NetMeta` object.
- `NetMeta` — result container exposing `.trts` (sorted treatment names), `.TE_fixed`/`.seTE_fixed` and `.TE_random`/`.seTE_random` (treatment × treatment pooled-effect and SE matrices), `.Q`/`.df_Q`/`.pval_Q`, `.tau2`/`.tau`, and a `.comparison(treat, reference, random=False)` convenience method that returns `(TE, seTE)` for one treatment pair.
- `netmeasures(net, random=False, tau_preset=None, sep=":")` — ports `netmeta::netmeasures`: per-comparison network measures derived from the Krahn (2013) design-based hat matrix — `proportion` (proportion of direct evidence), `meanpath` (mean path length), `minpar` and `minpar_study` (minimal parallelism, network- and study-level).

It is pure numpy/scipy — no `rpy2`, no R installation, no MCMC sampler anywhere in the call path. It is wired into socialverse as the registered Tier-3 function **`sv.tl.netmeta`** (module `socialverse/tl/_meta_nma.py`), which calls `external.pynetmeta.netmeta` internally to compute the pooled treatment-vs-reference β/covariance contract and league table, and `external.pynetmeta.netmeasures` to attach per-comparison network measures to the result. `sv.tl.netmeta` requires `models["nma_contrasts"]` (produced upstream by `sv.pp.nma_pairwise`) and produces `models["nma"]`.

:::{admonition} Parity gate
:class: note

This port is pinned to R `netmeta` (reference version 3.6-1) to `max_abs_err < 1e-6` across 12 deterministic parity tests, covering the fixed- and random-effects pooled TE/SE matrices, heterogeneity (Q, df, p-value, τ², τ), the reference-treatment column, and all four `netmeasures` outputs in both fixed and random modes.
:::

## Quickstart

```python
import numpy as np
from socialverse.external.pynetmeta import netmeta, netmeasures

# A small 4-treatment network of two-arm studies, connected with one loop.
# TE/seTE are log-odds-ratio contrasts (treat1 vs treat2) per study.
TE       = [-0.30, -0.10,  0.20, -0.22, -0.15]
seTE     = [ 0.20,  0.18,  0.25,  0.22,  0.19]
treat1   = ["metf", "rosi", "sulf", "metf", "metf"]
treat2   = ["plac", "plac", "plac", "rosi", "sulf"]
studlab  = ["S1",    "S2",   "S3",   "S4",   "S5"]
# S4 (metf vs rosi) closes a loop, so the design carries both direct and
# indirect evidence — that is what drives the Q inconsistency test below.

net = netmeta(TE, seTE, treat1, treat2, studlab, reference_group="plac")

print(net.trts)                       # sorted treatment names: ['metf', 'plac', 'rosi', 'sulf']
print(net.TE_fixed, net.seTE_fixed)   # common-effect pooled TE/SE matrices (treatment x treatment)
print(net.TE_random, net.seTE_random) # random-effects (DerSimonian-Laird) pooled matrices
print(net.Q, net.df_Q, net.pval_Q)    # global heterogeneity/inconsistency test
print(net.tau2, net.tau)              # DL between-study variance

te, se = net.comparison("metf", "plac", random=True)  # one pairwise contrast
print("metf vs plac (random):", te, se)

# Per-comparison network measures: proportion of direct evidence, mean path
# length, and minimal parallelism (network- and study-level).
nm = netmeasures(net, random=False)
print(nm["proportion"])   # e.g. {'metf:plac': 0.7..., 'rosi:plac': 1.0, ...}
```

## R ↔ Python dictionary

| R (`netmeta`) | socialverse | notes |
|---|---|---|
| `netmeta(TE, seTE, treat1, treat2, studlab, reference.group=...)` | `socialverse.external.pynetmeta.netmeta(TE, seTE, treat1, treat2, studlab, reference_group=...)` | same DerSimonian-Laird default (`method_tau="DL"` is the only supported method); returns a `NetMeta` object instead of an R list |
| `net$TE.fixed` / `net$seTE.fixed` | `net.TE_fixed` / `net.seTE_fixed` | treatment × treatment matrices, indexed by `net.trts` |
| `net$TE.random` / `net$seTE.random` | `net.TE_random` / `net.seTE_random` | DerSimonian-Laird random-effects matrices |
| `net$Q`, `net$df.Q`, `net$pval.Q`, `net$tau2`, `net$tau` | `net.Q`, `net.df_Q`, `net.pval_Q`, `net.tau2`, `net.tau` | global heterogeneity/inconsistency statistics |
| `netmeasures(net, random=TRUE)` | `socialverse.external.pynetmeta.netmeasures(net, random=True)` | returns a dict of `{comparison_label: value}` per measure instead of an R data frame |
| high-level analyst call | `sv.tl.netmeta(state, reference=..., comb="random"\|"fixed")` | the registered socialverse function; consumes `models["nma_contrasts"]` (from `sv.pp.nma_pairwise`), writes `models["nma"]` with the league table, Q/τ², and `netmeasures` |

## Parity evidence

12 deterministic parity tests (`socialverse/external/pynetmeta/tests/test_parity.py`) gate the port against an R `netmeta` reference fit (`reference.json`, generated by `r_reference_driver.R`) at `max_abs_err < 1e-6`. Gated quantities: the treatment ordering (`trts`), the full fixed-effect and random-effects TE/SE matrices, the global heterogeneity trio (Q, df, p-value), the DerSimonian-Laird τ²/τ, one named pairwise contrast against the reference treatment, and all four `netmeasures` outputs (`proportion`, `meanpath`, `minpar`, `minpar_study`) in both fixed and random mode, plus a sanity bound that `proportion` lies in [0, 1] and equals 1 for a comparison with only direct two-arm evidence.

:::{admonition} No stochastic slack in this port
:class: warning

Every quantity gated here — the Laplacian pseudoinverse, DerSimonian-Laird τ², and the Krahn hat-matrix network measures — is a closed-form linear-algebra computation with no sampling or iterative optimizer, so the 1e-6 tolerance is not loosened anywhere in this port (unlike ports that wrap bootstrap SEs or MCMC-based estimators).
:::

Reproduce locally:

```bash
Rscript socialverse/external/pynetmeta/tests/r_reference_driver.R
pytest socialverse/external/pynetmeta/tests/
```

## In the socialverse workflow

Day to day, call `sv.tl.netmeta` after building the contrast table with `sv.pp.nma_pairwise` — it is the registered entry point that wraps this port and writes `models["nma"]` (league table, heterogeneity, `netmeasures`) for downstream ranking (`sv.tl.netrank`), rankogram/SUCRA (`sv.tl.nma_rankogram`), and inconsistency checks (`sv.tl.nma_inconsistency`, `sv.tl.netsplit`). Use `registry_lookup("netmeta")` or `sv.list_functions()` to confirm the live signature, `requires`/`produces` contract, and tier before scripting against it, since the registry — not this page — is the source of truth for the deployed API.
