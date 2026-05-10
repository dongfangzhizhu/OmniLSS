"""R-aligned deviance increment interfaces.

R source reference:
- file: `gamlss/R/DevianceIncr.R`
- functions: `devianceIncr`
"""

from .operations import deviance_increment

devianceIncr = deviance_increment

__all__ = [
    "devianceIncr",
    "deviance_increment",
]
