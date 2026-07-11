"""pyqca — pure-Python reconstruction of R QCA (Qualitative Comparative Analysis).

Exposes truth-table construction, Quine-McCluskey Boolean minimization
(conservative solution), parameters of fit (consistency/coverage/PRI),
direct/crisp data calibration, and necessity superset search.
"""
from .pyqca import (truth_table, minimize, pof, TruthTable,
                    calibrate, superSubset)
__all__ = ["truth_table", "minimize", "pof", "TruthTable",
           "calibrate", "superSubset"]
__qca_reference_version__ = "3.25"
