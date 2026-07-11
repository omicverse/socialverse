"""pynetmeta -- pure-Python reconstruction of R netmeta (frequentist
graph-theoretical network meta-analysis, Ruecker 2012)."""
from .pynetmeta import netmeta, NetMeta, netmeasures
__all__ = ["netmeta", "NetMeta", "netmeasures"]
__netmeta_reference_version__ = "3.6-1"
