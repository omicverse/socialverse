"""socialverse — a structured, dependency-annotated function registry for the
social sciences and humanities.

The design thesis (see README): what makes an analysis agent reliable is not a
unified data container but a **queryable registry of functions annotated with
explicit context dependencies**. omicverse proved this for AI-for-biology; social
data is too heterogeneous for a single AnnData, so ``socialverse`` keeps the
mechanism — the registry — and replaces the data container with a light 12-slot
:class:`~socialverse._state.StudyState` vocabulary that the contracts speak in.

Namespaces (two axes, like omicverse's ``pp/tl/pl`` × ``bulk/single/space``):

- phase:  ``sv.pp`` (prepare) · ``sv.tl`` (analyze) · ``sv.pl`` (plot)
- domain: causal · survey · qual · text · net · lens (inside ``tl``)
- plus the social-science-specific axes ``sv.gov`` (governance) and ``sv.lit``.

Query the registry instead of guessing::

    import socialverse as sv
    sv.registry.find("did")                 # what functions exist for DID?
    sv.registry.get_prerequisites("did")    # what does it need / produce?
    sv.registry.resolve_plan("did")         # order the chain to get there
"""
from __future__ import annotations

from ._registry import (  # noqa: F401
    FunctionRegistry,
    RegistryError,
    export_registry,
    find_function,
    get_prerequisites,
    list_functions,
    register,
    registry,
)
from ._slots import SLOTS, VALID_SLOTS  # noqa: F401
from ._state import StudyState  # noqa: F401

__version__ = "0.2.5"

# Import submodules for their side effect: each module's @register calls populate
# the singleton registry. Wrapped in a guard so a missing optional dep in one
# module never blocks the rest (fail-soft, like omicverse's lazy imports).
_SUBMODULES = ["pp", "tl", "pl", "gov", "lit"]


def _load_submodules() -> None:
    import importlib

    for name in _SUBMODULES:
        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # pragma: no cover - defensive
            import warnings

            warnings.warn(f"socialverse.{name} failed to load: {exc}")


_load_submodules()

# Stata/R/SPSS command-name compatibility aliases (py-lmer, py-stcox, py-svyglm …)
# so researchers can find methods by the command name they already know. Applied
# after all @register calls so every function that loaded gets its aliases.
try:
    from ._compat_aliases import apply as _apply_compat_aliases

    _apply_compat_aliases(registry)
except Exception:  # pragma: no cover - never let the compat layer break import
    pass

# the OmicOS-facing query surface (sv.utils.registry_lookup / registry_summary) —
# imported after the analysis modules so the registry is fully populated.
from . import utils  # noqa: E402,F401

# expose the phase namespaces if they imported
for _n in _SUBMODULES:
    try:
        import importlib as _il

        globals()[_n] = _il.import_module(f"{__name__}.{_n}")
    except Exception:  # pragma: no cover
        pass

__all__ = [
    "registry", "register", "StudyState", "SLOTS", "VALID_SLOTS",
    "FunctionRegistry", "RegistryError", "find_function", "get_prerequisites",
    "list_functions", "export_registry", "utils", "__version__",
]
