"""R-aligned leverage interfaces.

R source reference:
- file: `gamlss/R/hatvalues.R`
- functions: `hatvalues.gamlss`
"""

from .operations import hat_wx

hatvalues = hat_wx

__all__ = [
    "hatvalues",
    "hat_wx",
]
