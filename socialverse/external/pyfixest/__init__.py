"""pyfixest — pure-Python reconstruction of R fixest (feols within estimator)."""
from .pyfixest import feols, demean, fepois, newey_west
__all__ = ["feols", "demean", "fepois", "newey_west"]
__fixest_reference_version__ = "0.14.2"
