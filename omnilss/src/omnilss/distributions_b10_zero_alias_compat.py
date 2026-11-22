"""Compatibility shells for selected Batch-10 zero-variant aliases."""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b10_zero_variants import (
    ZAPIG as _ZAPIG,
    ZASICHEL as _ZASICHEL,
    ZISICHEL as _ZISICHEL,
    ZIPIG as _ZIPIG,
)
from .families import FamilyDefinition


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


@dataclass(frozen=True)
class ZASICHELFamily(FamilyDefinition):
    """Compatibility shell for ZASICHEL family."""


@dataclass(frozen=True)
class ZISICHELFamily(FamilyDefinition):
    """Compatibility shell for ZISICHEL family."""


@dataclass(frozen=True)
class ZAPIGFamily(FamilyDefinition):
    """Compatibility shell for ZAPIG family."""


@dataclass(frozen=True)
class ZIPIGFamily(FamilyDefinition):
    """Compatibility shell for ZIPIG family."""


def ZASICHEL() -> FamilyDefinition:
    return _clone(_ZASICHEL(), "ZASICHEL", ZASICHELFamily)


def ZISICHEL() -> FamilyDefinition:
    return _clone(_ZISICHEL(), "ZISICHEL", ZISICHELFamily)


def ZAPIG() -> FamilyDefinition:
    return _clone(_ZAPIG(), "ZAPIG", ZAPIGFamily)


def ZIPIG() -> FamilyDefinition:
    return _clone(_ZIPIG(), "ZIPIG", ZIPIGFamily)
