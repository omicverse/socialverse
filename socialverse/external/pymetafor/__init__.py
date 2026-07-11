"""pymetafor — pure-Python reconstruction of metafor::rma (parity-gated).

Reconstructed under the Omicverse-RebuildR protocol; see tests/test_parity.py
for the class-1 numerical parity gate against metafor 5.0.1.
"""
from .rma import rma, RMAResult, blup, BLUPResult
__all__ = ["rma", "RMAResult", "blup", "BLUPResult"]
__metafor_reference_version__ = "5.0.1"
