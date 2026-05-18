"""Authoritative distribution-family registry.

The registry is the single public lookup surface for distribution families.
Factories are stored by uppercase family name and instantiate a fresh
``FamilyDefinition`` when resolved.  The initial registry is populated from the
legacy resolver table to avoid changing individual distribution modules all at
once; new families should call :func:`register` directly after defining their
factory.
"""

from __future__ import annotations

from collections.abc import Callable
from .families import FamilyDefinition

Resolver = Callable[[], FamilyDefinition]


_REGISTERED_FAMILIES: tuple[str, ...] = (
    "BB", "BCCG", "BCPE", "BCPEO", "BCT", "BCTO", "BE", "BEINF", "BEINF0",
    "BEINF1", "BEOI", "BEZI", "BI", "BNB", "DEL", "DPO", "EXGAUS", "EXP",
    "GA", "GB2", "GEOM", "GG", "GT", "GU", "IG", "IGAMMA", "JSU", "JSUO",
    "LNO", "LO", "LOGNO", "LOGNO2", "MN3", "MN4", "MN5", "NBI", "NBII",
    "NET", "NO", "NO2", "PARETO", "PARETO2", "PE", "PIG", "PO", "RG",
    "SHASH", "SHASHO", "SHASHO2", "SI", "SICHEL", "SIMPLEX", "SN1", "SN2",
    "ST5", "TF", "WARING", "WEI", "YULE", "ZAGA", "ZAIG", "ZAP", "ZIBI",
    "ZINBI", "ZIP", "ZIP2",
)

_REGISTRY: dict[str, Resolver] = {}
_BOOTSTRAPPED = False


def _build_from_legacy(name: str) -> FamilyDefinition:
    """Instantiate a family through the legacy resolver implementation."""
    from .distributions import _resolve_family_legacy

    return _resolve_family_legacy(name)


def register(name: str, factory: Resolver) -> None:
    """Register a distribution family factory.

    Parameters
    ----------
    name : str
        Family name.  Names are normalized to uppercase, e.g. ``"NO"``.
    factory : Callable[[], FamilyDefinition]
        Zero-argument factory returning a ``FamilyDefinition``.
    """
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTRY[str(name).upper()] = factory


def _ensure_bootstrapped() -> None:
    """Populate the registry with the built-in legacy families once."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for family_name in _REGISTERED_FAMILIES:
        register(
            family_name,
            lambda family_name=family_name: _build_from_legacy(family_name),
        )
    _BOOTSTRAPPED = True


def resolve(name: str | FamilyDefinition) -> FamilyDefinition:
    """Resolve a distribution family instance by name or return an instance.

    String lookup is case-insensitive.  Already-instantiated
    ``FamilyDefinition`` objects are returned unchanged.
    """
    if isinstance(name, FamilyDefinition):
        return name
    _ensure_bootstrapped()
    key = str(name).upper()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown family {name!r}. Available: {available}")
    return _REGISTRY[key]()


def list_families() -> list[str]:
    """Return registered family names sorted alphabetically."""
    _ensure_bootstrapped()
    return sorted(_REGISTRY.keys())


class DistributionRegistry:
    """Compatibility wrapper around a name->family resolver map."""

    def __init__(self, builders: dict[str, Resolver] | None = None) -> None:
        self._builders: dict[str, Resolver] = dict(builders or {})

    def register(self, name: str, builder: Resolver) -> None:
        if not callable(builder):
            raise TypeError("builder must be callable")
        self._builders[str(name).upper()] = builder

    def get(self, name: str) -> FamilyDefinition:
        key = str(name).upper()
        if key not in self._builders:
            raise KeyError(f"Distribution '{name}' is not registered")
        return self._builders[key]()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


def create_default_registry() -> DistributionRegistry:
    """Create a snapshot registry covering all currently registered families."""
    _ensure_bootstrapped()
    return DistributionRegistry(dict(_REGISTRY))


_DEFAULT_REGISTRY: DistributionRegistry | None = None


def get_default_registry() -> DistributionRegistry:
    """Return a process-global default distribution registry snapshot."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_registry()
    return _DEFAULT_REGISTRY


__all__ = [
    "DistributionRegistry",
    "_REGISTERED_FAMILIES",
    "create_default_registry",
    "get_default_registry",
    "list_families",
    "register",
    "resolve",
]
