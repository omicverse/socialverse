"""socialverse.datasets — tiny, deterministic toy datasets for demos and tests."""
from ._toy import load_bib, load_corpus, load_did_panel, load_survey  # noqa: F401
from ._toy_methods import (  # noqa: F401
    load_demography, load_irt, load_multilevel, load_network, load_qca,
    load_rdd, load_spatial, load_stylometry, load_survival,
)
from ._toy_p0 import (  # noqa: F401
    load_iv, load_mediation, load_ratings, load_regression, load_treatment,
)

__all__ = [
    "load_did_panel", "load_survey", "load_corpus", "load_bib",
    "load_rdd", "load_survival", "load_spatial", "load_irt", "load_qca",
    "load_demography", "load_multilevel", "load_stylometry", "load_network",
    "load_regression", "load_iv", "load_treatment", "load_mediation", "load_ratings",
]
