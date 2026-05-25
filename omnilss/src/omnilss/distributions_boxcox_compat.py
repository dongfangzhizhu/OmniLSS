"""Compatibility constructors for pending untruncated Box-Cox families.

R references:
- `gamlss.dist/R/BCCGuntr.R::BCCGuntr`
- `gamlss.dist/R/BCPEuntr.R::BCPEuntr`
- `gamlss.dist/R/BCTuntr.R::BCTuntr`
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions import BCCG, BCPE, BCT
from .families import FamilyDefinition


@dataclass(frozen=True)
class BCCGuntrFamily(FamilyDefinition):
    """Compatibility shell for BCCGUNTR family."""


@dataclass(frozen=True)
class BCPEuntrFamily(FamilyDefinition):
    """Compatibility shell for BCPEUNTR family."""


@dataclass(frozen=True)
class BCTuntrFamily(FamilyDefinition):
    """Compatibility shell for BCTUNTR family."""


def _clone(base: FamilyDefinition, name: str, family_class) -> FamilyDefinition:
    return family_class(
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


def BCCGuntr() -> FamilyDefinition:
    return _clone(BCCG(), "BCCGUNTR", BCCGuntrFamily)


def BCPEuntr() -> FamilyDefinition:
    return _clone(BCPE(), "BCPEUNTR", BCPEuntrFamily)


def BCTuntr() -> FamilyDefinition:
    return _clone(BCT(), "BCTUNTR", BCTuntrFamily)
