"""pypsych — pure-Python reconstruction of R psych (alpha / omega / fa PA / ICC / corr.test)."""
from .pypsych import smc, cronbach_alpha, fa_pa, omega_total, ICC, corr_test
__all__ = ["smc", "cronbach_alpha", "fa_pa", "omega_total", "ICC", "corr_test"]
__psych_reference_version__ = "2.6.5"
