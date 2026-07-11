"""pymatchit — pure-Python reconstruction of R MatchIt (nearest / glm)."""
from .pymatchit import (
    glm_logit_ps,
    nearest_match,
    smd,
    matchit,
    MatchItResult,
)

__all__ = ["glm_logit_ps", "nearest_match", "smd", "matchit", "MatchItResult"]
__matchit_reference_version__ = "4.7.2"
