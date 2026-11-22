"""Multinomial family constructor aligned with R `gamlss.dist::MULTIN` intent.

R reference: `gamlss.dist/R/MULTIN.R::MULTIN`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .distributions_b7 import MN3, MN4, MN5
from .families import FamilyDefinition


@dataclass(frozen=True)
class MULTINFamily(FamilyDefinition):
    """Compatibility shell for MULTIN constructor."""


def _clone_with_name(base: FamilyDefinition, name: str) -> FamilyDefinition:
    """Copy every field of base into a MULTINFamily shell with a new name.

    Built from dataclasses.fields rather than a hand-written list: the earlier
    hand-written copy silently dropped constructor_parameters and
    fixed_parameter_defaults, so a MULTIN shell could disagree with the MN*
    family it wraps the moment either field mattered (D2.142).
    """
    payload = {
        field.name: getattr(base, field.name) for field in fields(base) if field.init
    }
    payload["name"] = name
    return MULTINFamily(**payload)


def MULTIN(n_categories: int = 3) -> FamilyDefinition:
    """Return the categorical-multinomial family matching category count.

    Parameters
    ----------
    n_categories : int, default=3
        Number of response categories. Currently supports 3, 4, or 5,
        mapped to `MN3`, `MN4`, and `MN5` respectively.
    """
    if n_categories == 3:
        return _clone_with_name(MN3(), "MULTIN")
    if n_categories == 4:
        return _clone_with_name(MN4(), "MULTIN")
    if n_categories == 5:
        return _clone_with_name(MN5(), "MULTIN")
    raise ValueError(
        "MULTIN currently supports n_categories in {3, 4, 5}; " f"got {n_categories}."
    )
