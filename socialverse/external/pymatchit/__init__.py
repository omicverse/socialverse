"""pymatchit — pure-Python reconstruction of R MatchIt (nearest / glm)."""
from .pymatchit import (
    glm_logit_ps,
    nearest_match,
    smd,
    matchit,
    MatchItResult,
    get_w_from_ps,
    mahalanobis_dist,
    balance_table,
)

__all__ = [
    "glm_logit_ps",
    "nearest_match",
    "smd",
    "matchit",
    "MatchItResult",
    "get_w_from_ps",
    "mahalanobis_dist",
    "balance_table",
]
__matchit_reference_version__ = "4.7.2"
