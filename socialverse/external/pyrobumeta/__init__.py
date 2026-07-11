"""pyrobumeta — pure-Python reconstruction of R robumeta (RVE for dependent effects)."""
from .pyrobumeta import robu, impute_covariance_matrix, coef_test
__all__ = ["robu", "impute_covariance_matrix", "coef_test"]
__robumeta_reference_version__ = "2.1"
__clubSandwich_reference_version__ = "0.7.0"
