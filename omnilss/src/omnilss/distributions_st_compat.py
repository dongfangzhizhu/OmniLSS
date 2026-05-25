"""Compatibility constructors for skew-t families ST1/ST2/ST3/ST4.

R reference: `gamlss.dist` exports `ST1`, `ST2`, `ST3`, `ST4`.

Current phase-1 implementation provides API-compatible constructors and routes
all four to the existing ST3C numeric kernel while preserving distinct family
names in registry and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b15 import ST3C
from .families import FamilyDefinition


@dataclass(frozen=True)
class ST1Family(FamilyDefinition):
    """Compatibility shell for ST1 family."""


@dataclass(frozen=True)
class ST2Family(FamilyDefinition):
    """Compatibility shell for ST2 family."""


@dataclass(frozen=True)
class ST3Family(FamilyDefinition):
    """Compatibility shell for ST3 family."""


@dataclass(frozen=True)
class ST4Family(FamilyDefinition):
    """Compatibility shell for ST4 family."""


def _clone_with_name(base: FamilyDefinition, name: str, family_class):
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


def ST1() -> FamilyDefinition:
    return _clone_with_name(ST3C(), "ST1", ST1Family)


def ST2() -> FamilyDefinition:
    return _clone_with_name(ST3C(), "ST2", ST2Family)


def ST3() -> FamilyDefinition:
    return _clone_with_name(ST3C(), "ST3", ST3Family)


def ST4() -> FamilyDefinition:
    return _clone_with_name(ST3C(), "ST4", ST4Family)
