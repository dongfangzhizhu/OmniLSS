"""Distribution registry as the preferred family-resolution entry point."""

from __future__ import annotations

from collections.abc import Callable

from .families import FamilyDefinition

Resolver = Callable[[], FamilyDefinition]


class DistributionRegistry:
    """Name->family resolver map."""

    def __init__(self) -> None:
        self._builders: dict[str, Resolver] = {}

    def register(self, name: str, builder: Resolver) -> None:
        self._builders[str(name).upper()] = builder

    def get(self, name: str) -> FamilyDefinition:
        key = str(name).upper()
        if key not in self._builders:
            raise KeyError(f"Distribution '{name}' is not registered")
        return self._builders[key]()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


_REGISTERED_FAMILIES: tuple[str, ...] = ('BB', 'BCCG', 'BCPE', 'BCPEO', 'BCT', 'BCTO', 'BE', 'BEINF', 'BEINF0', 'BEINF1', 'BEOI', 'BEZI', 'BI', 'BNB', 'DEL', 'DPO', 'EXGAUS', 'EXP', 'GA', 'GB2', 'GEOM', 'GG', 'GT', 'GU', 'IG', 'IGAMMA', 'JSU', 'JSUO', 'LNO', 'LO', 'LOGNO', 'LOGNO2', 'MN3', 'MN4', 'MN5', 'NBI', 'NBII', 'NET', 'NO', 'NO2', 'PARETO', 'PARETO2', 'PE', 'PIG', 'PO', 'RG', 'SHASH', 'SHASHO', 'SHASHO2', 'SI', 'SICHEL', 'SIMPLEX', 'SN1', 'SN2', 'ST5', 'TF', 'WARING', 'WEI', 'YULE', 'ZAGA', 'ZAIG', 'ZAP', 'ZIBI', 'ZINBI', 'ZIP', 'ZIP2')


def _build_from_legacy(name: str) -> FamilyDefinition:
    from .distributions import _resolve_family_legacy

    return _resolve_family_legacy(name)


def create_default_registry() -> DistributionRegistry:
    """Create the default registry covering all currently implemented families."""
    reg = DistributionRegistry()
    for name in _REGISTERED_FAMILIES:
        reg.register(name, lambda name=name: _build_from_legacy(name))
    return reg


_DEFAULT_REGISTRY: DistributionRegistry | None = None


def get_default_registry() -> DistributionRegistry:
    """Return a process-global default distribution registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_registry()
    return _DEFAULT_REGISTRY
