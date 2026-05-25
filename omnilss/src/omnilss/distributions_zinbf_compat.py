"""Compatibility shell for ZINBF alias family."""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b15 import ZINBF as _ZINBF
from .families import FamilyDefinition


@dataclass(frozen=True)
class ZINBFFamilyCompat(FamilyDefinition):
    """Compatibility shell for ZINBF family."""


def ZINBF() -> FamilyDefinition:
    base = _ZINBF()
    return ZINBFFamilyCompat(
        name="ZINBF",
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
