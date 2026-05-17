"""Runtime family capability registry.

This module is the first implementation step for the six-month execution plan's
family capability registry.  It intentionally separates *registration* from
*validation*: a distribution may be importable and usable in experiments while
still lacking enough evidence to be treated as production-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .distribution_registry import _REGISTERED_FAMILIES


class CapabilityStatus(str, Enum):
    """Evidence level for a family feature."""

    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class FamilyCapabilityError(ValueError):
    """Raised when a requested family feature is not available at the requested tier."""


FEATURES: tuple[str, ...] = (
    "rs_fit",
    "rs_jax_fit",
    "cg_fit",
    "prediction",
    "sampling",
    "smooth_terms",
    "r_consistency",
    "ad_hessian",
    "production_safe",
)

# Families with explicit R-consistency test modules or batch consistency suites in
# the repository.  These are marked as evidence-backed for the *R consistency*
# feature only; they are not automatically production-safe.
_R_CONSISTENCY_EVIDENCE = frozenset(
    {
        "BCCG",
        "BCPE",
        "BCT",
        "BE",
        "BEINF",
        "BI",
        "EXP",
        "GA",
        "GEOM",
        "GT",
        "GU",
        "IG",
        "IGAMMA",
        "LO",
        "LOGNO",
        "LOGNO2",
        "NBI",
        "NBII",
        "NO",
        "NO2",
        "PARETO2",
        "PE",
        "PO",
        "RG",
        "SHASH",
        "SIMPLEX",
        "SN1",
        "TF",
        "WEI",
        "ZAGA",
        "ZIP",
    }
)

# JAX RS support is intentionally narrower than normal RS support.  The fitting
# documentation describes this route as available for these core families but not
# yet a universal production default.
_RS_JAX_SUPPORTED = frozenset({"NO", "GA", "PO", "BI", "WEI", "TF"})

# Keep this set small.  A family is production-safe only when the current codebase
# has strong evidence across fit, prediction, artifact, and R-consistency paths.
_PRODUCTION_SAFE = frozenset({"NO"})


@dataclass(frozen=True)
class FamilyCapability:
    """Capability statuses for one distribution family."""

    name: str
    features: Mapping[str, CapabilityStatus]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def status(self, feature: str) -> CapabilityStatus:
        """Return the status for ``feature`` or raise for unknown feature names."""

        if feature not in FEATURES:
            raise KeyError(f"unknown family capability feature {feature!r}")
        return self.features[feature]

    @property
    def is_production_safe(self) -> bool:
        """Whether the family is currently marked as production-safe."""

        return self.status("production_safe") is CapabilityStatus.VALIDATED

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "name": self.name,
            "features": {k: v.value for k, v in self.features.items()},
            "notes": list(self.notes),
        }


def _build_capability(name: str) -> FamilyCapability:
    key = name.upper()
    features = {feature: CapabilityStatus.EXPERIMENTAL for feature in FEATURES}

    # Normal RS is the broadest runtime path, but most families still need more
    # validation evidence before they can be called production-safe.
    features["rs_fit"] = CapabilityStatus.EXPERIMENTAL
    features["prediction"] = CapabilityStatus.EXPERIMENTAL
    features["sampling"] = CapabilityStatus.EXPERIMENTAL
    features["smooth_terms"] = CapabilityStatus.EXPERIMENTAL
    features["cg_fit"] = CapabilityStatus.EXPERIMENTAL

    features["rs_jax_fit"] = (
        CapabilityStatus.EXPERIMENTAL
        if key in _RS_JAX_SUPPORTED
        else CapabilityStatus.UNSUPPORTED
    )
    features["r_consistency"] = (
        CapabilityStatus.VALIDATED
        if key in _R_CONSISTENCY_EVIDENCE
        else CapabilityStatus.UNSUPPORTED
    )
    features["ad_hessian"] = (
        CapabilityStatus.EXPERIMENTAL
        if key in {"NO", "GA", "PO", "BI", "WEI", "TF"}
        else CapabilityStatus.UNSUPPORTED
    )
    features["production_safe"] = (
        CapabilityStatus.VALIDATED
        if key in _PRODUCTION_SAFE
        else CapabilityStatus.EXPERIMENTAL
    )

    notes: list[str] = []
    if key in _PRODUCTION_SAFE:
        notes.append("production-safe baseline family with the strongest current evidence")
    elif key in _R_CONSISTENCY_EVIDENCE:
        notes.append(
            "has repository R-consistency test coverage; still requires feature-specific gates"
        )
    else:
        notes.append("registered family without enough validation evidence for production use")
    if features["rs_jax_fit"] is CapabilityStatus.UNSUPPORTED:
        notes.append("RS_JAX route is not advertised for this family")

    return FamilyCapability(name=key, features=features, notes=tuple(notes))


_DEFAULT_CAPABILITIES: dict[str, FamilyCapability] = {
    name: _build_capability(name) for name in _REGISTERED_FAMILIES
}


def list_family_capabilities() -> tuple[FamilyCapability, ...]:
    """Return all registered family capabilities sorted by family name."""

    return tuple(_DEFAULT_CAPABILITIES[name] for name in sorted(_DEFAULT_CAPABILITIES))


def family_capability_names() -> tuple[str, ...]:
    """Return registered family names covered by the capability registry."""

    return tuple(sorted(_DEFAULT_CAPABILITIES))


def get_family_capability(name: str) -> FamilyCapability:
    """Return capability metadata for a family name."""

    key = str(name).upper()
    try:
        return _DEFAULT_CAPABILITIES[key]
    except KeyError as exc:
        raise KeyError(f"family {name!r} is not present in the capability registry") from exc


def family_supports(
    name: str,
    feature: str,
    *,
    include_experimental: bool = True,
) -> bool:
    """Return whether a family supports a feature at the requested evidence tier."""

    status = get_family_capability(name).status(feature)
    if status is CapabilityStatus.VALIDATED:
        return True
    if include_experimental and status is CapabilityStatus.EXPERIMENTAL:
        return True
    return False


def require_family_capability(
    name: str,
    feature: str,
    *,
    allow_experimental: bool = False,
) -> FamilyCapability:
    """Return capability metadata or raise if the feature is not sufficiently supported.

    Parameters
    ----------
    name:
        Family name, case-insensitive.
    feature:
        One of :data:`FEATURES`.
    allow_experimental:
        If ``False`` only validated features pass.  If ``True``, both validated
        and experimental features pass, while unsupported features still fail.
    """

    capability = get_family_capability(name)
    status = capability.status(feature)
    if status is CapabilityStatus.VALIDATED:
        return capability
    if allow_experimental and status is CapabilityStatus.EXPERIMENTAL:
        return capability
    if status is CapabilityStatus.EXPERIMENTAL:
        raise FamilyCapabilityError(
            f"family {capability.name!r} feature {feature!r} is experimental; "
            "pass allow_experimental=True to opt in"
        )
    raise FamilyCapabilityError(
        f"family {capability.name!r} does not support feature {feature!r}"
    )


__all__ = [
    "CapabilityStatus",
    "FEATURES",
    "FamilyCapability",
    "FamilyCapabilityError",
    "family_capability_names",
    "family_supports",
    "get_family_capability",
    "list_family_capabilities",
    "require_family_capability",
]
