# Installation

socialverse is a pure-Python package. Its numerical core depends only on
`numpy`, `scipy`, and `pandas` — **no R, Stata, or SPSS runtime is required** to
use any of the ported methods.

## From PyPI

```bash
pip install socialverse
```

```python
import socialverse as sv
print(sv.__version__)     # 0.7.2
```

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.9 | runtime |
| `numpy`, `scipy` | the numerical core of every R-package port |
| `pandas` | tabular study data / `StudyState` frames |
| `matplotlib` | plotting (`sv.pl.*`) |

Optional accelerators are auto-detected when present (e.g. `factor_analyzer` for
psychometrics) and the pure-`numpy`/`scipy` implementation is always the fallback,
so results are identical whether or not the optional package is installed.

## Verifying the install

```python
import socialverse as sv

# list every registered analysis, grouped by category
for category, fns in sv.list_functions().items():
    print(category, "→", len(fns), "functions")

# the parity-gated R-package ports live under socialverse.external
from socialverse.external import pymetafor, pysurvey, pysurvival   # etc.
```

## For contributors — reproducing the parity gates

Every port ships an R reference driver and a parity test. Reproducing the gates
requires **R** (only for regenerating the reference values — never at runtime):

```bash
# R 4.5.x with the reference packages installed, e.g.
Rscript -e 'install.packages(c("metafor","survey","survival","lavaan"))'

# regenerate a reference and run the parity test for one port
Rscript socialverse/external/pymetafor/tests/r_reference_driver.R
pytest  socialverse/external/pymetafor/tests/ -q
```

The committed `reference.json` files let anyone run the parity tests **without R**;
R is only needed to regenerate them from scratch. See any
[tutorial](tutorials/external/index.md) for the parity evidence of a specific
package.

## Building these docs locally

```bash
pip install mkdocs-material mkdocs-glightbox
mkdocs serve            # http://127.0.0.1:8000
```
