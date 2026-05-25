"""Concrete compatibility constructors for common `gen.Family` skeleton names."""

from __future__ import annotations

from dataclasses import dataclass

from .gen_family_builder import build_gen_family_skeleton
from .families import FamilyDefinition


@dataclass(frozen=True)
class GenSkeletonCompatFamily(FamilyDefinition):
    """Compatibility shell for `gen.Family` skeleton aliases."""


def _make(name: str) -> FamilyDefinition:
    base = build_gen_family_skeleton(name)
    return GenSkeletonCompatFamily(
        name=base.name,
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


def count_1_31() -> FamilyDefinition: return _make("count_1_31")
def count_1_22() -> FamilyDefinition: return _make("count_1_22")
def count_2_32() -> FamilyDefinition: return _make("count_2_32")
def count_2_32R() -> FamilyDefinition: return _make("count_2_32R")
def count_2_33() -> FamilyDefinition: return _make("count_2_33")
def count_3_32() -> FamilyDefinition: return _make("count_3_32")
def count_3_33() -> FamilyDefinition: return _make("count_3_33")
def binom_1_31() -> FamilyDefinition: return _make("binom_1_31")
def binom_2_33() -> FamilyDefinition: return _make("binom_2_33")
def binom_3_33() -> FamilyDefinition: return _make("binom_3_33")
def contR_2_12() -> FamilyDefinition: return _make("contR_2_12")
def contR_3_11() -> FamilyDefinition: return _make("contR_3_11")
def contR_4_13() -> FamilyDefinition: return _make("contR_4_13")
def contRplus_1_1() -> FamilyDefinition: return _make("contRplus_1_1")
def contRplus_2_11() -> FamilyDefinition: return _make("contRplus_2_11")
def contRplus_3_13() -> FamilyDefinition: return _make("contRplus_3_13")
def contRplus_4_33() -> FamilyDefinition: return _make("contRplus_4_33")
def contR01_2_13() -> FamilyDefinition: return _make("contR01_2_13")
def contR01_4_33() -> FamilyDefinition: return _make("contR01_4_33")
