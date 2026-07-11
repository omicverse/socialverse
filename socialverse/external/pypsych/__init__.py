"""pypsych — pure-Python reconstruction of R psych (alpha / omega / fa PA)."""
from .pypsych import smc, cronbach_alpha, fa_pa, omega_total
__all__ = ["smc", "cronbach_alpha", "fa_pa", "omega_total"]
__psych_reference_version__ = "2.6.5"
