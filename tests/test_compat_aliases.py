"""The Stata/R/SPSS command-name compatibility layer resolves to the right function."""
from collections import Counter

import pytest

import socialverse as sv
from socialverse._compat_aliases import PY_ALIASES


@pytest.mark.parametrize("alias,expected", [
    ("py-lmer", "multilevel"),        # R lme4::lmer
    ("py-mixed", "multilevel"),       # Stata mixed
    ("py-stcox", "survival"),         # Stata stcox
    ("py-coxph", "survival"),         # R survival::coxph
    ("py-COXREG", "survival"),        # SPSS
    ("py-svyglm", "survey_estimate"), # R survey::svyglm
    ("py-svyset", "declare_design"),  # Stata design declaration
    ("py-rdrobust", "rdd"),           # rdrobust
    ("py-lavaan", "cfa"),             # R lavaan
    ("py-gsem", "sem"),               # Stata gsem
    ("py-mirt", "irt"),               # R mirt
    ("py-ergm", "ergm"),
    ("py-siena07", "saom"),           # R RSiena
    ("py-truthTable", "qca"),         # R QCA
    ("py-lagsarlm", "spatial_regression"),
    ("py-localmoran", "spatial_autocorr"),
    ("py-oaxaca", "decomposition"),
    ("py-stylo", "stylometry"),
])
def test_exact_alias_resolves(alias, expected):
    entry = sv.registry.get(alias)
    assert entry is not None, f"{alias} did not resolve"
    assert entry.name == expected


@pytest.mark.parametrize("bare,expected", [("stcox", "survival"), ("lmer", "multilevel"),
                                           ("svyglm", "survey_estimate")])
def test_bare_command_fuzzy_matches(bare, expected):
    """A user typing the bare command (no py-) still finds it via fuzzy search."""
    hits = sv.registry.find(bare)
    assert hits and hits[0]["name"] == expected


def test_no_alias_binds_to_two_functions():
    """Every py- alias (case-insensitive) belongs to exactly one function."""
    counts = Counter(a.lower() for aliases in PY_ALIASES.values() for a in aliases)
    dups = {a: n for a, n in counts.items() if n > 1}
    assert not dups, f"duplicate aliases across functions: {dups}"


def test_every_target_function_exists():
    """Every function named in the compat map is actually registered."""
    for name in PY_ALIASES:
        assert sv.registry.get(name) is not None, f"unknown target function: {name}"


def test_registry_lookup_surface():
    """The OmicOS-facing registry_lookup consumes the aliases too."""
    assert "sv.tl.survival" in sv.utils.registry_lookup("py-coxph", 1)
