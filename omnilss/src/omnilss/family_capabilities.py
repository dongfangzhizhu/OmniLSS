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


CAPABILITY_MATRIX_VERSION = 3

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

METHOD_CAPABILITY_FEATURES: tuple[tuple[str, str], ...] = (
    ("RS", "rs_fit"),
    ("RS_JAX", "rs_jax_fit"),
    ("CG", "cg_fit"),
    ("MIXED", "cg_fit"),
    ("JOINT", "cg_fit"),
    ("LBFGS", "cg_fit"),
)

# Backward-compatible public alias used by earlier Month 1 capability-gate
# tests and documents.  Keep this bound to the same tuple so generated
# matrices, top-level APIs, and service route admission cannot drift.
METHOD_ROUTE_FEATURES = METHOD_CAPABILITY_FEATURES

_CAPABILITY_MATRIX_STRICT_POLICY: dict[str, bool] = {
    "default_allow_experimental": True,
    "strict_capabilities_allow_experimental": False,
    "unsupported_routes_fail_fast": True,
}


def _capability_matrix_issue(
    code: str,
    message: str,
    *,
    path: str,
    severity: str = "error",
) -> dict[str, str]:
    """Return a JSON-friendly capability matrix validation issue."""

    return {
        "severity": severity,
        "code": code,
        "path": path,
        "message": message,
    }


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
        for validated_feature in (
            "rs_fit",
            "prediction",
            "r_consistency",
            "production_safe",
        ):
            features[validated_feature] = CapabilityStatus.VALIDATED
        notes.append(
            "production-safe baseline family with the strongest current evidence"
        )
    elif key in _R_CONSISTENCY_EVIDENCE:
        notes.append(
            "has repository R-consistency test coverage; still requires feature-specific gates"
        )
    else:
        notes.append(
            "registered family without enough validation evidence for production use"
        )
    if features["rs_jax_fit"] is CapabilityStatus.UNSUPPORTED:
        notes.append("RS_JAX route is not advertised for this family")

    return FamilyCapability(name=key, features=features, notes=tuple(notes))


_DEFAULT_CAPABILITIES: dict[str, FamilyCapability] = {
    name: _build_capability(name) for name in _REGISTERED_FAMILIES
}


def list_family_capabilities() -> tuple[FamilyCapability, ...]:
    """Return all registered family capabilities sorted by family name."""

    return tuple(_DEFAULT_CAPABILITIES[name] for name in sorted(_DEFAULT_CAPABILITIES))


def method_capability_features() -> dict[str, str]:
    """Return the public fitting-method to family-capability feature map."""

    return dict(METHOD_CAPABILITY_FEATURES)


def method_route_feature(method_name: str) -> str:
    """Return the capability feature used to gate a fitting method.

    Method names are case-insensitive.  A clear :class:`KeyError` is raised for
    unknown methods so callers do not silently bypass the shared route map.
    """

    method = str(method_name).upper()
    try:
        return method_capability_features()[method]
    except KeyError as exc:
        raise KeyError(
            f"method {method_name!r} is not present in the capability routing map"
        ) from exc


def capability_matrix() -> dict[str, object]:
    """Return a machine-readable snapshot of the runtime capability matrix."""

    capabilities = [capability.as_dict() for capability in list_family_capabilities()]
    return {
        "version": CAPABILITY_MATRIX_VERSION,
        "features": list(FEATURES),
        "method_capability_features": method_capability_features(),
        # ``method_routes`` is retained as a compatibility alias for clients and
        # documents created during the first capability-gate iteration.
        "method_routes": method_capability_features(),
        "strict_capability_policy": dict(_CAPABILITY_MATRIX_STRICT_POLICY),
        "families": {item["name"]: item for item in capabilities},
    }


def validate_capability_matrix_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a serialized capability matrix against the current runtime schema.

    The validator is intentionally strict for generated release artifacts: it
    verifies the schema version, feature list, method routing aliases, policy
    flags, family coverage, and per-family feature status values.  It returns a
    stable JSON-friendly report instead of raising so CLI tools and service
    checks can surface every drift issue in one response.
    """

    issues: list[dict[str, str]] = []
    if not isinstance(payload, Mapping):
        return {
            "ok": False,
            "version": None,
            "expected_version": CAPABILITY_MATRIX_VERSION,
            "issues": [
                _capability_matrix_issue(
                    "matrix_not_mapping",
                    "capability matrix payload must be a JSON object",
                    path="$",
                )
            ],
        }

    version = payload.get("version")
    if version != CAPABILITY_MATRIX_VERSION:
        issues.append(
            _capability_matrix_issue(
                "version_mismatch",
                (
                    "expected capability matrix version "
                    f"{CAPABILITY_MATRIX_VERSION}, got {version!r}"
                ),
                path="$.version",
            )
        )

    features = payload.get("features")
    if features != list(FEATURES):
        issues.append(
            _capability_matrix_issue(
                "features_mismatch",
                "capability matrix features do not match the runtime registry",
                path="$.features",
            )
        )

    expected_method_features = method_capability_features()
    method_features = payload.get("method_capability_features")
    if method_features != expected_method_features:
        issues.append(
            _capability_matrix_issue(
                "method_capability_features_mismatch",
                "method_capability_features does not match runtime routing",
                path="$.method_capability_features",
            )
        )

    method_routes = payload.get("method_routes")
    if method_routes != expected_method_features:
        issues.append(
            _capability_matrix_issue(
                "method_routes_mismatch",
                "method_routes compatibility alias must mirror method_capability_features",
                path="$.method_routes",
            )
        )

    policy = payload.get("strict_capability_policy")
    if policy != _CAPABILITY_MATRIX_STRICT_POLICY:
        issues.append(
            _capability_matrix_issue(
                "strict_policy_mismatch",
                "strict_capability_policy does not match the runtime policy",
                path="$.strict_capability_policy",
            )
        )

    families = payload.get("families")
    if not isinstance(families, Mapping):
        issues.append(
            _capability_matrix_issue(
                "families_not_mapping",
                "families must be a mapping keyed by family name",
                path="$.families",
            )
        )
    else:
        expected_names = set(family_capability_names())
        actual_names = set(map(str, families.keys()))
        missing_names = sorted(expected_names - actual_names)
        extra_names = sorted(actual_names - expected_names)
        if missing_names:
            issues.append(
                _capability_matrix_issue(
                    "missing_families",
                    (
                        "capability matrix is missing families: "
                        f"{', '.join(missing_names)}"
                    ),
                    path="$.families",
                )
            )
        if extra_names:
            issues.append(
                _capability_matrix_issue(
                    "unknown_families",
                    (
                        "capability matrix contains unknown families: "
                        f"{', '.join(extra_names)}"
                    ),
                    path="$.families",
                )
            )

        valid_statuses = {status.value for status in CapabilityStatus}
        for family_name, family_payload in families.items():
            family_path = f"$.families.{family_name}"
            if not isinstance(family_payload, Mapping):
                issues.append(
                    _capability_matrix_issue(
                        "family_not_mapping",
                        f"family {family_name!r} entry must be an object",
                        path=family_path,
                    )
                )
                continue
            if family_payload.get("name") != family_name:
                issues.append(
                    _capability_matrix_issue(
                        "family_name_mismatch",
                        "family entry name must match its map key",
                        path=f"{family_path}.name",
                    )
                )
            family_features = family_payload.get("features")
            if not isinstance(family_features, Mapping):
                issues.append(
                    _capability_matrix_issue(
                        "family_features_not_mapping",
                        "family features must be a mapping",
                        path=f"{family_path}.features",
                    )
                )
                continue
            feature_names = set(map(str, family_features.keys()))
            missing_features = sorted(set(FEATURES) - feature_names)
            extra_features = sorted(feature_names - set(FEATURES))
            if missing_features:
                issues.append(
                    _capability_matrix_issue(
                        "family_missing_features",
                        (
                            f"family {family_name!r} is missing features: "
                            f"{', '.join(missing_features)}"
                        ),
                        path=f"{family_path}.features",
                    )
                )
            if extra_features:
                issues.append(
                    _capability_matrix_issue(
                        "family_unknown_features",
                        (
                            f"family {family_name!r} has unknown features: "
                            f"{', '.join(extra_features)}"
                        ),
                        path=f"{family_path}.features",
                    )
                )
            for feature, status in family_features.items():
                if status not in valid_statuses:
                    issues.append(
                        _capability_matrix_issue(
                            "invalid_feature_status",
                            (
                                f"family {family_name!r} feature {feature!r} "
                                f"has invalid status {status!r}"
                            ),
                            path=f"{family_path}.features.{feature}",
                        )
                    )
            notes = family_payload.get("notes")
            if not isinstance(notes, list) or not all(
                isinstance(note, str) for note in notes
            ):
                issues.append(
                    _capability_matrix_issue(
                        "family_notes_not_string_list",
                        "family notes must be a list of strings",
                        path=f"{family_path}.notes",
                    )
                )

    return {
        "ok": not issues,
        "version": version,
        "expected_version": CAPABILITY_MATRIX_VERSION,
        "issues": issues,
    }


def method_route_capability_report(
    family_name: str,
    method_name: str,
    *,
    strict: bool = False,
) -> dict[str, object]:
    """Return a JSON-friendly route-admission report for fitting methods.

    The report is intended for service boundaries that need to decide whether a
    fit job can be admitted before scheduling backend work.  Runtime fitting
    uses the same helper so public services, generated capability matrices, and
    ``gamlss()`` share one method/family routing contract.
    """

    method = str(method_name).upper()
    feature = method_capability_features().get(method)
    allow_experimental = not strict
    if feature is None:
        return {
            "ok": False,
            "family": str(family_name).upper(),
            "method": method,
            "feature": None,
            "status": None,
            "strict": bool(strict),
            "allow_experimental": allow_experimental,
            "code": "unknown_method",
            "message": (
                f"method {method_name!r} is not present in the capability "
                "routing map"
            ),
        }

    family = str(family_name).upper()
    if family not in _DEFAULT_CAPABILITIES:
        return {
            "ok": False,
            "family": family,
            "method": method,
            "feature": feature,
            "status": None,
            "strict": bool(strict),
            "allow_experimental": allow_experimental,
            "code": "unknown_family",
            "message": (
                f"family {family_name!r} is not present in the capability "
                "registry"
            ),
        }

    capability = get_family_capability(family)
    status = capability.status(feature)
    if status is CapabilityStatus.VALIDATED:
        ok = True
        code = "validated"
        message = (
            f"family {capability.name!r} method {method!r} is validated "
            f"through capability feature {feature!r}"
        )
    elif status is CapabilityStatus.EXPERIMENTAL and allow_experimental:
        ok = True
        code = "experimental_allowed"
        message = (
            f"family {capability.name!r} method {method!r} is experimental "
            "and allowed by the current policy"
        )
    elif status is CapabilityStatus.EXPERIMENTAL:
        ok = False
        code = "experimental_requires_opt_in"
        message = (
            f"family {capability.name!r} method {method!r} is experimental; "
            "disable strict capability mode or explicitly opt in before using it"
        )
    else:
        ok = False
        code = "unsupported_route"
        message = (
            f"family {capability.name!r} method {method!r} is not supported "
            f"because capability feature {feature!r} is unsupported"
        )

    return {
        "ok": ok,
        "family": capability.name,
        "method": method,
        "feature": feature,
        "status": status.value,
        "strict": bool(strict),
        "allow_experimental": allow_experimental,
        "code": code,
        "message": message,
    }


def require_method_route(
    family_name: str,
    method_name: str,
    *,
    allow_experimental: bool = False,
) -> FamilyCapability:
    """Return capability metadata or raise for a blocked method/family route.

    This helper is the Python API counterpart to
    :func:`method_route_capability_report`; it uses the same method-to-feature
    map that is emitted in capability matrices and exposed through service
    metadata endpoints.
    """

    feature = method_route_feature(method_name)
    return require_family_capability(
        family_name, feature, allow_experimental=allow_experimental
    )


def family_capability_names() -> tuple[str, ...]:
    """Return registered family names covered by the capability registry."""

    return tuple(sorted(_DEFAULT_CAPABILITIES))


def get_family_capability(name: str) -> FamilyCapability:
    """Return capability metadata for a family name."""

    key = str(name).upper()
    try:
        return _DEFAULT_CAPABILITIES[key]
    except KeyError as exc:
        raise KeyError(
            f"family {name!r} is not present in the capability registry"
        ) from exc


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
    "CAPABILITY_MATRIX_VERSION",
    "FEATURES",
    "METHOD_ROUTE_FEATURES",
    "capability_matrix",
    "FamilyCapability",
    "FamilyCapabilityError",
    "family_capability_names",
    "family_supports",
    "get_family_capability",
    "list_family_capabilities",
    "method_capability_features",
    "method_route_capability_report",
    "method_route_feature",
    "METHOD_CAPABILITY_FEATURES",
    "validate_capability_matrix_payload",
    "require_family_capability",
    "require_method_route",
]
