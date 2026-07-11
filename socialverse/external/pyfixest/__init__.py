"""pyfixest — pure-Python reconstruction of R fixest (feols within estimator)."""
from .pyfixest import feols, demean
__all__ = ["feols", "demean"]
__fixest_reference_version__ = "0.14.2"
