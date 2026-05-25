"""Compatibility shells for uppercase `*O` alias families."""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b14 import BCPEo, BCTo, JSUo, SHASHo2
from .distributions_b3 import SHASHo
from .distributions_b16 import BCCGo, LOGSHASHo, PARETO1o
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
class SHASHOFamily(FamilyDefinition): pass
@dataclass(frozen=True)
class JSUOFamily(FamilyDefinition): pass
@dataclass(frozen=True)
class SHASHO2Family(FamilyDefinition): pass
@dataclass(frozen=True)
class BCPEOFamily(FamilyDefinition): pass
@dataclass(frozen=True)
class BCTOFamily(FamilyDefinition): pass
@dataclass(frozen=True)
class PARETO1OFamily(FamilyDefinition): pass
@dataclass(frozen=True)
class BCCGOFamily(FamilyDefinition): pass
@dataclass(frozen=True)
class LOGSHASHOFamily(FamilyDefinition): pass


def SHASHO() -> FamilyDefinition: return _clone(SHASHo(), "SHASHO", SHASHOFamily)
def JSUO() -> FamilyDefinition: return _clone(JSUo(), "JSUO", JSUOFamily)
def SHASHO2() -> FamilyDefinition: return _clone(SHASHo2(), "SHASHO2", SHASHO2Family)
def BCPEO() -> FamilyDefinition: return _clone(BCPEo(), "BCPEO", BCPEOFamily)
def BCTO() -> FamilyDefinition: return _clone(BCTo(), "BCTO", BCTOFamily)
def PARETO1O() -> FamilyDefinition: return _clone(PARETO1o(), "PARETO1O", PARETO1OFamily)
def BCCGO() -> FamilyDefinition: return _clone(BCCGo(), "BCCGO", BCCGOFamily)
def LOGSHASHO() -> FamilyDefinition: return _clone(LOGSHASHo(), "LOGSHASHO", LOGSHASHOFamily)
