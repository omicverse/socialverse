"""pyqca — pure-Python reconstruction of R QCA (Qualitative Comparative Analysis).

Exposes truth-table construction, Quine-McCluskey Boolean minimization
(conservative solution), and parameters of fit (consistency/coverage/PRI).
"""
from .pyqca import truth_table, minimize, pof, TruthTable
__all__ = ["truth_table", "minimize", "pof", "TruthTable"]
__qca_reference_version__ = "3.25"
