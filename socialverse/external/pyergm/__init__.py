"""pyergm — pure-Python reconstruction of R **ergm** (dyad-independent MPLE).

Exposes the maximum pseudo-likelihood estimator for dyad-independent Exponential
Random Graph Models, which is exactly a logistic regression on dyads with change
statistics as predictors. MCMC-MLE (stochastic) is intentionally out of scope.
"""
from .pyergm import (
    MPLEResult,
    TRIAD_CENSUS_LABELS,
    build_design,
    change_stats_edges,
    change_stats_nodecov,
    change_stats_nodematch,
    dyads,
    ergm_mple,
    summary_formula,
    triad_census,
)

__all__ = [
    "ergm_mple",
    "build_design",
    "dyads",
    "change_stats_edges",
    "change_stats_nodecov",
    "change_stats_nodematch",
    "MPLEResult",
    "summary_formula",
    "triad_census",
    "TRIAD_CENSUS_LABELS",
]
__ergm_reference_version__ = "4.x"
