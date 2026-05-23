"""Compatibility constructors for pending count/binomial families.

R references:
- `gamlss.dist/R/DBI.R::DBI`
"""

from __future__ import annotations

from .distributions import BI
from .families import FamilyDefinition


def DBI() -> FamilyDefinition:
    """Double Binomial compatibility constructor.

    Phase-1 compatibility path: preserve `DBI` family identity while reusing the
    existing `BI` numeric kernel until dedicated DBI likelihood is implemented.
    """
    base = BI()
    return FamilyDefinition(
        name="DBI",
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
