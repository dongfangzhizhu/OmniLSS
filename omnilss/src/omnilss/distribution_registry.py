"""Phase 1 distribution registry.

Avoids hard-coded family branching by registering resolvers in one place.
"""

from __future__ import annotations

from collections.abc import Callable

from .distributions import resolve_family
from .families import FamilyDefinition

Resolver = Callable[[], FamilyDefinition]


class DistributionRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Resolver] = {}

    def register(self, name: str, builder: Resolver) -> None:
        key = str(name).upper()
        self._builders[key] = builder

    def get(self, name: str) -> FamilyDefinition:
        key = str(name).upper()
        if key not in self._builders:
            raise KeyError(f"Distribution '{name}' is not registered")
        return self._builders[key]()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


def create_default_registry() -> DistributionRegistry:
    reg = DistributionRegistry()
    for fam in ("NO", "GA", "TF", "BE", "ZIP", "ZINBI"):
        reg.register(fam, lambda fam=fam: resolve_family(fam))
    return reg
