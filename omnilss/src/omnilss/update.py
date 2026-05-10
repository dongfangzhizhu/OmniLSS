"""R-aligned update interfaces.

R source reference:
- file: `gamlss/R/update.R`
- functions: `update.gamlss`
"""

from .methods import update_model

update = update_model

__all__ = [
    "update",
    "update_model",
]
