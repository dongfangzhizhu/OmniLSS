"""Multinomial family constructor aligned with R `gamlss.dist::MULTIN` intent.

R reference: `gamlss.dist/R/MULTIN.R::MULTIN`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b7 import MN3, MN4, MN5
from .families import FamilyDefinition


@dataclass(frozen=True)
class MULTINFamily(FamilyDefinition):
    """Compatibility shell for MULTIN constructor."""


def _clone_with_name(base: FamilyDefinition, name: str) -> FamilyDefinition:
    return MULTINFamily(
        name=name,
        parameters=base.parameters,
        g_dev_inc=base.g_dev_inc,
        type=base.type,
        links=base.links,
        link_functions=base.link_functions,
        link_inverses=base.link_inverses,
        link_derivatives=base.link_derivatives,
        score_functions=base.score_functions,
        hessian_functions=base.hessian_functions,
        fixed_parameters=base.fixed_parameters,
        d=base.d,
        p=base.p,
        q=base.q,
        r=base.r,
    )


def MULTIN(n_categories: int = 3) -> FamilyDefinition:
    """Return the categorical-multinomial family matching category count.

    Parameters
    ----------
    n_categories : int, default=3
        Number of response categories. Currently supports 3, 4, or 5,
        mapped to `MN3`, `MN4`, and `MN5` respectively.
    """
    if n_categories == 3:
        return _clone_with_name(MN3(), "MULTIN")
    if n_categories == 4:
        return _clone_with_name(MN4(), "MULTIN")
    if n_categories == 5:
        return _clone_with_name(MN5(), "MULTIN")
    raise ValueError(
        "MULTIN currently supports n_categories in {3, 4, 5}; "
        f"got {n_categories}."
    )
