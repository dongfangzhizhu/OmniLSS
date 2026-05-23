"""Compatibility constructors for pending discrete families.

R references:
- `gamlss.dist/R/DBURR12.R::DBURR12`
"""

from __future__ import annotations

from .distributions_b6 import SICHEL
from .families import FamilyDefinition


def DBURR12() -> FamilyDefinition:
    """Discrete Burr-12 compatibility constructor.

    Phase-1 compatibility path: expose `DBURR12` family identity in API/registry
    while temporarily reusing the existing `SICHEL` numeric kernel.
    """
    base = SICHEL()
    return FamilyDefinition(
        name="DBURR12",
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
