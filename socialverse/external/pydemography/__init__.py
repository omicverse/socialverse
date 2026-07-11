"""pydemography -- pure-Python reconstruction of R demography (life tables)
plus Kitagawa and Oaxaca-Blinder decompositions."""
from .pydemography import (
    life_table, life_expectancy, kitagawa, oaxaca,
)
__all__ = ["life_table", "life_expectancy", "kitagawa", "oaxaca"]
__demography_reference_version__ = "2.0.1"
