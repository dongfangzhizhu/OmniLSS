"""Compatibility constructor for `GENERALISEDPOISSON` registry alias."""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b13 import GeneralisedPoisson
from .families import FamilyDefinition


@dataclass(frozen=True)
class GENERALISEDPOISSONFamily(FamilyDefinition):
    """Compatibility shell for GENERALISEDPOISSON alias."""


def GENERALISEDPOISSON() -> FamilyDefinition:
    base = GeneralisedPoisson()
    return GENERALISEDPOISSONFamily(
        name="GENERALISEDPOISSON",
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
