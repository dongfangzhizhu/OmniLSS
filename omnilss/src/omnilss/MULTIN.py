"""Multinomial family constructor aligned with R `gamlss.dist::MULTIN` intent.

R reference: `gamlss.dist/R/MULTIN.R::MULTIN`.
"""

from __future__ import annotations

from .distributions_b7 import MN3, MN4, MN5
from .families import FamilyDefinition


def MULTIN(n_categories: int = 3) -> FamilyDefinition:
    """Return the categorical-multinomial family matching category count.

    Parameters
    ----------
    n_categories : int, default=3
        Number of response categories. Currently supports 3, 4, or 5,
        mapped to `MN3`, `MN4`, and `MN5` respectively.
    """
    if n_categories == 3:
        return MN3()
    if n_categories == 4:
        return MN4()
    if n_categories == 5:
        return MN5()
    raise ValueError(
        "MULTIN currently supports n_categories in {3, 4, 5}; "
        f"got {n_categories}."
    )
