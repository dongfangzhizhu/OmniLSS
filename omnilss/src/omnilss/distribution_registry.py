"""Authoritative distribution-family registry.

The registry is the single public lookup surface for distribution families.
Factories are stored by uppercase family name and instantiate a fresh
``FamilyDefinition`` when resolved.  The initial registry is populated from a
single built-in dictionary that maps family names to zero-argument factory
locations; new families should call :func:`register` directly after defining
their factory.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from .families import FamilyDefinition

Resolver = Callable[[], FamilyDefinition]


_BUILTIN_FAMILY_FACTORIES: dict[str, tuple[str, str]] = {
    # Core families from distributions.py
    "NO": ("omnilss.distributions", "NO"),
    "PO": ("omnilss.distributions", "PO"),
    "BI": ("omnilss.distributions", "BI"),
    "GA": ("omnilss.distributions", "GA"),
    "EXP": ("omnilss.distributions", "EXP"),
    "LOGNO": ("omnilss.distributions", "LOGNO"),
    "NBI": ("omnilss.distributions", "NBI"),
    "IG": ("omnilss.distributions", "IG"),
    "LO": ("omnilss.distributions", "LO"),
    "BE": ("omnilss.distributions", "BE"),
    "WEI": ("omnilss.distributions", "WEI"),
    "GEOM": ("omnilss.distributions", "GEOM"),
    "ZIP": ("omnilss.distributions", "ZIP"),
    "TF": ("omnilss.distributions", "TF"),
    "TF2": ("omnilss.distributions", "TF2"),
    "JSU": ("omnilss.distributions", "JSU"),
    "BCCG": ("omnilss.distributions", "BCCG"),
    "BCT": ("omnilss.distributions", "BCT"),
    "BCCGUNTR": ("omnilss.distributions_boxcox_compat", "BCCGuntr"),
    "BCPEUNTR": ("omnilss.distributions_boxcox_compat", "BCPEuntr"),
    "BCTUNTR": ("omnilss.distributions_boxcox_compat", "BCTuntr"),
    "COUNT_1_31": ("omnilss.distributions_gen_skeleton_compat", "count_1_31"),
    "COUNT_1_22": ("omnilss.distributions_gen_skeleton_compat", "count_1_22"),
    "COUNT_2_32": ("omnilss.distributions_gen_skeleton_compat", "count_2_32"),
    "COUNT_2_32R": ("omnilss.distributions_gen_skeleton_compat", "count_2_32R"),
    "COUNT_2_33": ("omnilss.distributions_gen_skeleton_compat", "count_2_33"),
    "COUNT_3_32": ("omnilss.distributions_gen_skeleton_compat", "count_3_32"),
    "COUNT_3_33": ("omnilss.distributions_gen_skeleton_compat", "count_3_33"),
    "BINOM_1_31": ("omnilss.distributions_gen_skeleton_compat", "binom_1_31"),
    "BINOM_2_33": ("omnilss.distributions_gen_skeleton_compat", "binom_2_33"),
    "BINOM_3_33": ("omnilss.distributions_gen_skeleton_compat", "binom_3_33"),
    "CONTR_2_12": ("omnilss.distributions_gen_skeleton_compat", "contR_2_12"),
    "CONTR_3_11": ("omnilss.distributions_gen_skeleton_compat", "contR_3_11"),
    "CONTR_4_13": ("omnilss.distributions_gen_skeleton_compat", "contR_4_13"),
    "CONTRPLUS_1_1": ("omnilss.distributions_gen_skeleton_compat", "contRplus_1_1"),
    "CONTRPLUS_2_11": ("omnilss.distributions_gen_skeleton_compat", "contRplus_2_11"),
    "CONTRPLUS_3_13": ("omnilss.distributions_gen_skeleton_compat", "contRplus_3_13"),
    "CONTRPLUS_4_33": ("omnilss.distributions_gen_skeleton_compat", "contRplus_4_33"),
    "CONTR01_2_13": ("omnilss.distributions_gen_skeleton_compat", "contR01_2_13"),
    "CONTR01_4_33": ("omnilss.distributions_gen_skeleton_compat", "contR01_4_33"),
    "BCPE": ("omnilss.distributions", "BCPE"),
    # AD/batch extension families.
    "GU": ("omnilss.distributions_b1", "GU"),
    "RG": ("omnilss.distributions_b1", "RG"),
    "IGAMMA": ("omnilss.distributions_b1", "IGAMMA"),
    "PARETO2": ("omnilss.distributions_b1", "PARETO2"),
    "NBII": ("omnilss.distributions_b1", "NBII"),
    "NO2": ("omnilss.distributions_b2", "NO2"),
    "LOGNO2": ("omnilss.distributions_b2", "LOGNO2"),
    "PE": ("omnilss.distributions_b2", "PE"),
    "SIMPLEX": ("omnilss.distributions_b2", "SIMPLEX"),
    "EXGAUS": ("omnilss.distributions_b2", "exGAUS"),
    "SHASH": ("omnilss.distributions_b3", "SHASH"),
    "SHASHO": ("omnilss.distributions_b3", "SHASHo"),
    "SN1": ("omnilss.distributions_b3", "SN1"),
    "SN2": ("omnilss.distributions_b3", "SN2"),
    "GT": ("omnilss.distributions_b3", "GT"),
    "BEINF": ("omnilss.distributions_b4", "BEINF"),
    "BEINF0": ("omnilss.distributions_b4", "BEINF0"),
    "BEINF1": ("omnilss.distributions_b4", "BEINF1"),
    "BEZI": ("omnilss.distributions_b4", "BEZI"),
    "BEOI": ("omnilss.distributions_b4", "BEOI"),
    "ZAGA": ("omnilss.distributions_b5", "ZAGA"),
    "ZAIG": ("omnilss.distributions_b5", "ZAIG"),
    "ZIP2": ("omnilss.distributions_b5", "ZIP2"),
    "ZINBI": ("omnilss.distributions_b5", "ZINBI"),
    "ZAP": ("omnilss.distributions_b5", "ZAP"),
    "ZIBI": ("omnilss.distributions_b10_zero_variants", "ZIBI"),
    "PIG": ("omnilss.distributions_b6", "PIG"),
    "SICHEL": ("omnilss.distributions_b6", "SICHEL"),
    "SI": ("omnilss.distributions_b6", "SI"),
    "DPO": ("omnilss.distributions_b6", "DPO"),
    "DEL": ("omnilss.distributions_b6", "DEL"),
    "YULE": ("omnilss.distributions_b6", "YULE"),
    "WARING": ("omnilss.distributions_b6", "WARING"),
    "BB": ("omnilss.distributions_b7", "BB"),
    "DBI": ("omnilss.distributions_count_compat", "DBI"),
    "DBURR12": ("omnilss.distributions_discrete_compat", "DBURR12"),
    "BNB": ("omnilss.distributions_b7", "BNB"),
    "MN3": ("omnilss.distributions_b7", "MN3"),
    "MN4": ("omnilss.distributions_b7", "MN4"),
    "MN5": ("omnilss.distributions_b7", "MN5"),
    "MULTIN": ("omnilss.MULTIN", "MULTIN"),
    "GG": ("omnilss.distributions_b8", "GG"),
    "GB2": ("omnilss.distributions_b8", "GB2"),
    "PARETO": ("omnilss.distributions_b8", "PARETO"),
    "NET": ("omnilss.distributions_b8", "NET"),
    "LNO": ("omnilss.distributions_b8", "LNO"),
    "SHASHO2": ("omnilss.distributions_o_alias_compat", "SHASHO2"),
    "JSUO": ("omnilss.distributions_o_alias_compat", "JSUO"),
    "ST5": ("omnilss.distributions_b14", "ST5"),
    "ST1": ("omnilss.distributions_st_compat", "ST1"),
    "ST2": ("omnilss.distributions_st_compat", "ST2"),
    "ST3": ("omnilss.distributions_st_compat", "ST3"),
    "ST4": ("omnilss.distributions_st_compat", "ST4"),
    "BCPEO": ("omnilss.distributions_o_alias_compat", "BCPEO"),
    "BCTO": ("omnilss.distributions_o_alias_compat", "BCTO"),
    "DPO1": ("omnilss.distributions_b16", "DPO1"),
    "PARETO1": ("omnilss.distributions_b16", "PARETO1"),
    "PARETO1O": ("omnilss.distributions_o_alias_compat", "PARETO1O"),
    "SEP": ("omnilss.distributions_b16", "SEP"),
    "BCCGO": ("omnilss.distributions_o_alias_compat", "BCCGO"),
    "LOGSHASHO": ("omnilss.distributions_o_alias_compat", "LOGSHASHO"),
    "LG": ("omnilss.distributions_b9", "LG"),
    "ZIPF": ("omnilss.distributions_b9", "ZIPF"),
    "ZALG": ("omnilss.distributions_b9", "ZALG"),
    "IGA": ("omnilss.distributions_b13", "IGA"),
    "PIG2": ("omnilss.distributions_b13", "PIG2"),
    "PE2": ("omnilss.distributions_b13", "PE2"),
    "LOGITNO": ("omnilss.distributions_b13", "LOGITNO"),
    "DELAPORT": ("omnilss.distributions_b13", "DELAPORT"),
    "GENERALISEDPOISSON": ("omnilss.distributions_generalisedpoisson_compat", "GENERALISEDPOISSON"),
    "GP": ("omnilss.distributions_gp_compat", "GP"),
    "ZASICHEL": ("omnilss.distributions_b10_zero_variants", "ZASICHEL"),
    "ZISICHEL": ("omnilss.distributions_b10_zero_variants", "ZISICHEL"),
    "ZAPIG": ("omnilss.distributions_b10_zero_variants", "ZAPIG"),
    "ZIPIG": ("omnilss.distributions_b10_zero_variants", "ZIPIG"),
    "ZABB": ("omnilss.distributions_b10_zero_variants", "ZABB"),
    "ZIBB": ("omnilss.distributions_b10_zero_variants", "ZIBB"),
    "ZAZIPF": ("omnilss.distributions_b10_zero_variants", "ZAZIPF"),
    "ZABNB": ("omnilss.distributions_b10_zero_variants", "ZABNB"),
    "ZIBNB": ("omnilss.distributions_b10_zero_variants", "ZIBNB"),
    "ZINBF": ("omnilss.distributions_b15", "ZINBF"),
    "GIG": ("omnilss.distributions_b11", "GIG"),
    "GB1": ("omnilss.distributions_b11", "GB1"),
    "WEI2": ("omnilss.distributions_b12", "WEI2"),
    "WEI3": ("omnilss.distributions_b12", "WEI3"),
    "NOF": ("omnilss.distributions_b12", "NOF"),
    "GAF": ("omnilss.distributions_b12", "GAF"),
    "RGE": ("omnilss.distributions_b12", "RGE"),
    "SEP1": ("omnilss.distributions_b15", "SEP1"),
    "SEP2": ("omnilss.distributions_b15", "SEP2"),
    "SEP3": ("omnilss.distributions_b15", "SEP3"),
    "SEP4": ("omnilss.distributions_b15", "SEP4"),
    "SST": ("omnilss.distributions_b15", "SST"),
    "ST3C": ("omnilss.distributions_b15", "ST3C"),
    "EGB2": ("omnilss.distributions_b15", "EGB2"),
    "LQNO": ("omnilss.distributions_b15", "LQNO"),
    "NBF": ("omnilss.distributions_b15", "NBF"),
    "LOGSHASH": ("omnilss.distributions_b15", "LOGSHASH"),
    "GPO": ("omnilss.distributions_b15", "GPO"),
    "ZABI": ("omnilss.distributions_b10_zero_variants", "ZABI"),
    "ZANBI": ("omnilss.distributions_b10_zero_variants", "ZANBI"),
}

_REGISTERED_FAMILIES: tuple[str, ...] = tuple(sorted(_BUILTIN_FAMILY_FACTORIES))

_REGISTRY: dict[str, Resolver] = {}
_BOOTSTRAPPED = False


def _build_builtin(name: str) -> FamilyDefinition:
    """Instantiate a built-in family from the authoritative registry table."""
    module_name, attr_name = _BUILTIN_FAMILY_FACTORIES[name]
    factory = getattr(import_module(module_name), attr_name)
    return factory()


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
    """Populate the registry with the built-in family dictionary once."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for family_name in _REGISTERED_FAMILIES:
        register(
            family_name,
            lambda family_name=family_name: _build_builtin(family_name),
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
    "_BUILTIN_FAMILY_FACTORIES",
    "_REGISTERED_FAMILIES",
    "create_default_registry",
    "get_default_registry",
    "list_families",
    "register",
    "resolve",
]
