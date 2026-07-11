# R packages, ported

The methods a social scientist reaches for mostly live in **R**. socialverse
re-implements the numerical core of the most-used packages in pure
`numpy`/`scipy`, so you can run them from Python with no R runtime — and pins each
one to its R reference to within `max_abs_err < 1e-6` on a deterministic core.

Every port lives under `socialverse.external.<port>` and is wired into the
high-level `sv.tl.*` / `sv.pp.*` API, so in day-to-day use you call the
registered function and the parity-gated engine runs underneath.

## The rebuildr protocol

Each port is built with the
[omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr) protocol:

1. The **R source is the executable specification** — not prose, the actual
   package.
2. A **parity gate threshold is committed before the port is written**
   (`1e-6` for a deterministic, class-1 quantity).
3. The gate is **never widened to hide a wrong number**. Where a quantity is
   genuinely stochastic (a bootstrap SE, an MCMC-MLE estimate) or the reference
   itself only converges to a looser tolerance, that limit is documented rather
   than papered over.

Concretely, every port ships:

```
socialverse/external/<port>/
├── <port>.py                     # the pure numpy/scipy port
├── __init__.py
└── tests/
    ├── r_reference_driver.R       # runs the real R package → reference.json
    ├── reference.json             # committed reference values (run tests without R)
    └── test_parity.py             # asserts port == reference at 1e-6
```

The committed `reference.json` means **you can run the parity tests without R**;
R is only needed to regenerate the references from scratch.

## The 14 ports

| Port | R package | What it does | Domain | Parity tests |
|---|---|---|---|---|
| [pymetafor](pymetafor.md) | `metafor::rma` | random / mixed-effects meta-analysis (REML/ML/DL/EE, Knapp–Hartung, I²/H², meta-regression, BLUP) | Meta-analysis | 9 |
| [pynetmeta](pynetmeta.md) | `netmeta` | frequentist network meta-analysis (graph-theoretical, SUCRA, net heat) | Meta-analysis | 12 |
| [pyrobumeta](pyrobumeta.md) | `robumeta` | robust variance meta-regression (RVE, CR2, Satterthwaite df) | Meta-analysis | 4 |
| [pymada](pymada.md) | `mada` | bivariate diagnostic-accuracy meta-analysis (Reitsma, SROC/AUC) | Meta-analysis | 8 |
| [pysurvey](pysurvey.md) | `survey` | design-based complex-survey estimation (svydesign/svymean/svytotal/svyglm, svyby/ratio/ciprop) | Survey & causal | 8 |
| [pyfixest](pyfixest.md) | `fixest` | high-dimensional fixed-effects regression + clustered SE + Poisson PMLE | Survey & causal | 6 |
| [pydid](pydid.md) | `did` | Callaway–Sant'Anna staggered difference-in-differences (att_gt / aggte) | Survey & causal | 7 |
| [pymatchit](pymatchit.md) | `MatchIt` | propensity-score matching + balance diagnostics | Survey & causal | 12 |
| [pysurvival](pysurvival.md) | `survival` | Kaplan–Meier, Cox PH (Efron/Breslow), conditional logit, parametric AFT | Survival | 9 |
| [pypsych](pypsych.md) | `psych` | reliability (α/ω), ICC, correlation tests, factor analysis | Psychometrics | 8 |
| [pylavaan](pylavaan.md) | `lavaan` | confirmatory factor analysis / SEM (ML, full fit-index battery, modification indices) | Psychometrics | 8 |
| [pyqca](pyqca.md) | `QCA` | qualitative comparative analysis (calibration, truth tables, Quine–McCluskey minimization) | Configurational | 10 |
| [pyergm](pyergm.md) | `ergm` | exponential random graph models (sufficient statistics, dyad-independent MPLE, triad census) | Networks | 8 |
| [pydemography](pydemography.md) | `demography` | life tables + Kitagawa / Oaxaca decomposition | Demography | 6 |

**115 parity tests in total, all green at `max_abs_err < 1e-6`** on the
deterministic core of each package.

Pick a package from the sidebar. Each tutorial covers: what the R package is, the
port and how it's wired into socialverse, a runnable Python example, the R↔Python
function dictionary, and the parity evidence.
