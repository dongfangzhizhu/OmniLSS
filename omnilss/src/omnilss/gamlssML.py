"""R-aligned maximum likelihood fitting interfaces.

R source reference:
- file: `gamlss/R/gamlssML.R`
- functions: `gamlssML`
"""

from .fitting import gamlss_ml

gamlssML = gamlss_ml

__all__ = [
    "gamlssML",
    "gamlss_ml",
]
