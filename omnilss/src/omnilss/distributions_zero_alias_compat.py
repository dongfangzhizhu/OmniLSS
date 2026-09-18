"""Compatibility shells for selected zero-inflated alias families."""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b10_zero_variants import ZABB as _ZABB, ZABNB as _ZABNB, ZABI as _ZABI, ZANBI as _ZANBI, ZAZIPF as _ZAZIPF, ZIBB as _ZIBB, ZIBI as _ZIBI, ZIBNB as _ZIBNB
from .distributions_b5 import ZINBI as _ZINBI
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
class ZABBFamily(FamilyDefinition):
    """Compatibility shell for ZABB family."""


@dataclass(frozen=True)
class ZIBBFamily(FamilyDefinition):
    """Compatibility shell for ZIBB family."""


@dataclass(frozen=True)
class ZAZIPFFamily(FamilyDefinition):
    """Compatibility shell for ZAZIPF family."""


@dataclass(frozen=True)
class ZABNBFamily(FamilyDefinition):
    """Compatibility shell for ZABNB family."""


@dataclass(frozen=True)
class ZIBNBFamily(FamilyDefinition):
    """Compatibility shell for ZIBNB family."""


@dataclass(frozen=True)
class ZABIFamily(FamilyDefinition):
    """Compatibility shell for ZABI family."""


@dataclass(frozen=True)
class ZANBIFamily(FamilyDefinition):
    """Compatibility shell for ZANBI family."""


@dataclass(frozen=True)
class ZIBIFamily(FamilyDefinition):
    """Compatibility shell for ZIBI family."""


@dataclass(frozen=True)
class ZINBIFamily(FamilyDefinition):
    """Compatibility shell for ZINBI family."""




def ZABB(bd=10) -> FamilyDefinition:
    return _clone(_ZABB(bd=bd), "ZABB", ZABBFamily)


def ZIBB(bd=10) -> FamilyDefinition:
    return _clone(_ZIBB(bd=bd), "ZIBB", ZIBBFamily)


def ZAZIPF() -> FamilyDefinition:
    return _clone(_ZAZIPF(), "ZAZIPF", ZAZIPFFamily)


def ZABNB() -> FamilyDefinition:
    return _clone(_ZABNB(), "ZABNB", ZABNBFamily)


def ZIBNB() -> FamilyDefinition:
    return _clone(_ZIBNB(), "ZIBNB", ZIBNBFamily)

def ZABI() -> FamilyDefinition:
    return _clone(_ZABI(), "ZABI", ZABIFamily)


def ZANBI() -> FamilyDefinition:
    return _clone(_ZANBI(), "ZANBI", ZANBIFamily)


def ZIBI() -> FamilyDefinition:
    return _clone(_ZIBI(), "ZIBI", ZIBIFamily)


def ZINBI() -> FamilyDefinition:
    return _clone(_ZINBI(), "ZINBI", ZINBIFamily)
