"""Compatibility shells for selected zero-altered mixed/count aliases."""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b5 import ZAGA as _ZAGA, ZAIG as _ZAIG, ZAP as _ZAP
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
class ZAGAFamily(FamilyDefinition):
    """Compatibility shell for ZAGA family."""


@dataclass(frozen=True)
class ZAIGFamily(FamilyDefinition):
    """Compatibility shell for ZAIG family."""


@dataclass(frozen=True)
class ZAPFamily(FamilyDefinition):
    """Compatibility shell for ZAP family."""


def ZAGA() -> FamilyDefinition:
    return _clone(_ZAGA(), "ZAGA", ZAGAFamily)


def ZAIG() -> FamilyDefinition:
    return _clone(_ZAIG(), "ZAIG", ZAIGFamily)


def ZAP() -> FamilyDefinition:
    return _clone(_ZAP(), "ZAP", ZAPFamily)
