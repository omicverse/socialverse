# socialverse

**socialverse** is a Python framework for **quantitative and computational social
science** — survey epidemiology, meta-analysis, causal inference, psychometrics,
demography, and configurational / network methods — behind a single,
contract-checked registry that grounds every claim in an actual estimation.

socialverse is the social-science sibling of
[omicverse](https://github.com/Starlitnightly/omicverse): the same registry-driven
design, a different domain.

```{button-link} tutorials/index.html
:color: primary
:shadow:
Browse the tutorials →
```

## A pure-Python home for R's social-science stack

Most of the methods a social scientist reaches for live in **R** (`metafor`,
`survey`, `survival`, `lavaan`, `MatchIt`, `did`, …), Stata, or SPSS. socialverse
**re-implements the numerical core of those packages in pure `numpy` / `scipy`**
and puts them behind one Python API — no `rpy2`, no R runtime, no Stata licence.

Each re-implementation is a **parity-gated port** built with the
[omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr) protocol: the
R source is the executable specification, and every port ships a test that pins the
Python result to the R result to within `max_abs_err < 1e-6` on a deterministic
core.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 📊 Meta-analysis
`pymetafor` · `pynetmeta` · `pyrobumeta` · `pymada`

Random / mixed-effects, network, robust-variance, and diagnostic-accuracy
meta-analysis.
:::

:::{grid-item-card} 🎯 Survey & causal
`pysurvey` · `pyfixest` · `pydid` · `pymatchit`

Design-based survey estimation, fixed-effects regression, staggered DiD, and
propensity-score matching.
:::

:::{grid-item-card} ⏱ Survival & longitudinal
`pysurvival`

Kaplan–Meier, Cox PH, conditional logit, and parametric AFT models.
:::

:::{grid-item-card} 🧭 Psychometrics
`pypsych` · `pylavaan`

Reliability, ICC, factor analysis, and confirmatory factor analysis / SEM.
:::

:::{grid-item-card} 🕸 Configurational & networks
`pyqca` · `pyergm`

Qualitative comparative analysis and exponential random graph models.
:::

:::{grid-item-card} 👥 Demography
`pydemography`

Life tables and Kitagawa / Oaxaca decomposition.
:::

::::

:::{note}
**The registry is the spine.** Every analysis is a registered function with an
explicit contract — `requires` (which study slots must be populated first),
`produces` (what it writes back), and a `tier`. A `StudyState` carries the data and
the evidence chain from step to step, so a downstream estimator refuses to run on an
undeclared design rather than silently returning a wrong number. Call
`sv.list_functions()` to see everything available.
:::

## Quick start

```bash
pip install socialverse
```

```python
import socialverse as sv

sv.__version__          # '0.7.2'
sv.list_functions()     # every registered analysis, grouped by category

# a StudyState carries data + the evidence chain between steps
st = sv.StudyState()
```

Each tutorial in the sidebar takes one R package, explains what it does, shows the
Python port, the R↔Python function dictionary, a runnable example, and the parity
evidence that pins the port to the reference implementation.

```{toctree}
:hidden:
:maxdepth: 2

Installation <Installation.md>
Tutorials <tutorials/index.md>
Release notes <Release_notes.md>
```
