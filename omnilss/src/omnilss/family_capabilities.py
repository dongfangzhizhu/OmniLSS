"""Runtime family capability registry.

This module is the first implementation step for the six-month execution plan's
family capability registry.  It intentionally separates *registration* from
*validation*: a distribution may be importable and usable in experiments while
still lacking enough evidence to be treated as production-safe.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping

from .distribution_registry import _BUILTIN_FAMILY_FACTORIES, _REGISTERED_FAMILIES


class CapabilityStatus(str, Enum):
    """Evidence level for a family feature."""

    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class ImplementationStatus(str, Enum):
    """Implementation truth for a registered R distribution family."""

    IMPLEMENTED = "implemented"
    EXPERIMENTAL = "experimental"
    COMPAT_ALIAS = "compat_alias"
    PLACEHOLDER = "placeholder"
    MISSING = "missing"
    UNSUPPORTED_WITH_RATIONALE = "unsupported_with_rationale"


class FamilyCapabilityError(ValueError):
    """Raised when a requested family feature is not available at the requested tier."""


class ProductionFamilyRejectedError(FamilyCapabilityError):
    """Fail-closed rejection for an unsafe family on a production route.

    ``report`` is deliberately retained as structured diagnostic data so direct
    fit, batch, benchmark, and service callers cannot lose the status or
    reason that caused admission to fail.
    """

    def __init__(self, report: Mapping[str, object]) -> None:
        self.report = dict(report)
        super().__init__(str(self.report.get("message", "production family rejected")))

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


CAPABILITY_MATRIX_VERSION = 4

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
        "DEL",
        "DELAPORT",
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
        "ZIPF",
        "ZALG",
    }
)

# JAX RS support is intentionally narrower than normal RS support.  The fitting
# documentation describes this route as available for these core families but not
# yet a universal production default.
_RS_JAX_SUPPORTED = frozenset({"NO", "GA", "PO", "BI", "WEI", "TF"})

# Fixed R regression coverage for the ordinary RS fitting route.  This is a
# feature-level admission signal only; it does not make the family production
# safe or claim CG/MIXED parity.
_RS_FIT_EVIDENCE = frozenset({"NO", "ZIP"})

# Keep this set small.  A family is production-safe only when the current codebase
# has strong evidence across fit, prediction, artifact, and R-consistency paths.
_PRODUCTION_SAFE = frozenset({"NO"})

_KNOWN_COMPAT_ALIAS_FAMILIES = frozenset(
    {
        "BCCGUNTR",
        "BCPEUNTR",
        "BCTUNTR",
        "DBI",
        "DBURR12",
        "MULTIN",
        "COUNT_1_31",
        "COUNT_1_22",
        "COUNT_2_32",
        "COUNT_2_32R",
        "COUNT_2_33",
        "COUNT_3_32",
        "COUNT_3_33",
        "BINOM_1_31",
        "BINOM_2_33",
        "BINOM_3_33",
        "CONTR_2_12",
        "CONTR_3_11",
        "CONTR_4_13",
        "CONTRPLUS_1_1",
        "CONTRPLUS_2_11",
        "CONTRPLUS_3_13",
        "CONTRPLUS_4_33",
        "CONTR01_2_13",
        "CONTR01_4_33",
    }
)

_COMPAT_FACTORY_MODULES = frozenset(
    {
        "omnilss.distributions_boxcox_compat",
        "omnilss.distributions_count_compat",
        "omnilss.distributions_discrete_compat",
        "omnilss.distributions_gen_skeleton_compat",
    }
)


@lru_cache(maxsize=None)
def _module_source(module_name: str) -> str:
    """Return source for ``module_name`` without importing the module."""

    try:
        spec = find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return ""
    if spec is None or spec.origin is None:
        return ""
    try:
        return Path(spec.origin).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


@lru_cache(maxsize=None)
def _module_ast(module_name: str) -> ast.Module | None:
    """Parse module source without importing it."""

    source = _module_source(module_name)
    if not source:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


@lru_cache(maxsize=None)
def _module_function_defs(module_name: str) -> dict[str, ast.FunctionDef]:
    """Return function definitions for a registry factory module without importing it."""

    module = _module_ast(module_name)
    if module is None:
        return {}
    return {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _build_ad_family_call_has_placeholder(call: ast.Call) -> bool:
    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    for dpqr_name in ("p", "q", "r"):
        value = keywords.get(dpqr_name)
        if value is None:
            return True
        if isinstance(value, ast.Constant) and value.value is None:
            return True
    return False


def _call_has_complete_dpqr_keywords(call: ast.Call) -> bool:
    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    for dpqr_name in ("p", "q", "r"):
        value = keywords.get(dpqr_name)
        if value is None:
            return False
        if isinstance(value, ast.Constant) and value.value is None:
            return False
    return True


def _function_has_complete_dpqr_keywords(node: ast.FunctionDef) -> bool:
    """Return whether a factory explicitly wires all p/q/r family methods."""

    return any(
        isinstance(child, ast.Call) and _call_has_complete_dpqr_keywords(child)
        for child in ast.walk(node)
    )


def _function_uses_build_ad_placeholder(
    function_name: str,
    defs: Mapping[str, ast.FunctionDef],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if function_name in seen:
        return False
    node = defs.get(function_name)
    if node is None:
        return False
    if _function_has_complete_dpqr_keywords(node):
        return False
    next_seen = seen | {function_name}
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        call_name = _call_name(child.func)
        if call_name == "build_ad_family":
            if _build_ad_family_call_has_placeholder(child):
                return True
        elif call_name in defs and _function_uses_build_ad_placeholder(
            call_name, defs, next_seen
        ):
            return True
    return False


@lru_cache(maxsize=None)
def _factory_uses_placeholder_dpqr(module_name: str, attr_name: str) -> bool:
    defs = _module_function_defs(module_name)
    return _function_uses_build_ad_placeholder(attr_name, defs)


_DPQR_PLACEHOLDER_SOURCE_MARKERS = (
    "placeholder",
    "not implemented",
    "normal approximation",
    "normal cdf as fallback",
    "normal as fallback",
    "return nan",
    "jnp.nan",
)


def _function_source(module_name: str, node: ast.FunctionDef) -> str:
    source = _module_source(module_name)
    if not source:
        return ""
    return ast.get_source_segment(source, node) or ""


@lru_cache(maxsize=None)
def _dpqr_function_uses_placeholder(function_name: str) -> bool:
    """Return whether a dpqr helper has obvious placeholder behavior."""

    defs = _module_function_defs("omnilss.dpqr_functions")
    node = defs.get(function_name)
    if node is None:
        return False
    source = _function_source("omnilss.dpqr_functions", node).lower()
    # NaN is a valid result for invalid inputs.  SICHEL's three public
    # functions perform real numerical work and use conditional NaN returns
    # only for those boundaries; do not classify that handling as a stub.
    if function_name in {"pSICHEL", "qSICHEL", "rSICHEL"}:
        source = source.replace("jnp.nan", "")
    return any(marker in source for marker in _DPQR_PLACEHOLDER_SOURCE_MARKERS)
    return any(marker in source for marker in _DPQR_PLACEHOLDER_SOURCE_MARKERS)


def _factory_dpqr_keyword_names(
    function_name: str,
    defs: Mapping[str, ast.FunctionDef],
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return p/q/r keyword function names referenced by a family factory."""

    if function_name in seen:
        return frozenset()
    node = defs.get(function_name)
    if node is None:
        return frozenset()
    next_seen = seen | {function_name}
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        for keyword in child.keywords:
            if keyword.arg not in {"p", "q", "r"}:
                continue
            if isinstance(keyword.value, ast.Name):
                names.add(keyword.value.id)
        call_name = _call_name(child.func)
        if call_name in defs:
            names.update(_factory_dpqr_keyword_names(call_name, defs, next_seen))
    return frozenset(names)


@lru_cache(maxsize=None)
def _factory_uses_placeholder_dpqr_functions(module_name: str, attr_name: str) -> bool:
    """Detect factories wired to dpqr_functions placeholder/approximation helpers."""

    defs = _module_function_defs(module_name)
    for function_name in _factory_dpqr_keyword_names(attr_name, defs):
        if _dpqr_function_uses_placeholder(function_name):
            return True
    return False


@dataclass(frozen=True)
class FamilyCapability:
    """Capability statuses for one distribution family."""

    name: str
    implementation_status: ImplementationStatus
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
            "implementation_status": self.implementation_status.value,
            "features": {k: v.value for k, v in self.features.items()},
            "notes": list(self.notes),
        }


def implementation_status_for_family(name: str) -> ImplementationStatus:
    """Return the code-derived implementation truth for a registered family.

    This status is intentionally separate from feature evidence.  For example,
    a compatibility alias may have smoke or R-consistency tests that prove the
    wrapper is callable, but it still must not be counted as a migrated
    statistical implementation.
    """

    key = str(name).upper()
    factory = _BUILTIN_FAMILY_FACTORIES.get(key)
    if factory is None:
        return ImplementationStatus.MISSING

    module_name, _attr_name = factory
    if key in _KNOWN_COMPAT_ALIAS_FAMILIES or module_name in _COMPAT_FACTORY_MODULES:
        return ImplementationStatus.COMPAT_ALIAS
    if _factory_uses_placeholder_dpqr(module_name, _attr_name):
        return ImplementationStatus.PLACEHOLDER
    if key == "ZIP":
        return ImplementationStatus.EXPERIMENTAL
    if _factory_uses_placeholder_dpqr_functions(module_name, _attr_name):
        return ImplementationStatus.PLACEHOLDER
    return ImplementationStatus.EXPERIMENTAL


def _build_capability(name: str) -> FamilyCapability:
    key = name.upper()
    implementation_status = implementation_status_for_family(key)
    features = {feature: CapabilityStatus.EXPERIMENTAL for feature in FEATURES}

    # Normal RS is the broadest runtime path, but most families still need more
    # validation evidence before they can be called production-safe.
    features["rs_fit"] = (
        CapabilityStatus.VALIDATED
        if key in _RS_FIT_EVIDENCE
        else CapabilityStatus.EXPERIMENTAL
    )
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
        implementation_status = ImplementationStatus.IMPLEMENTED
        notes.append(
            "production-safe baseline family with the strongest current evidence"
        )
    elif implementation_status is ImplementationStatus.COMPAT_ALIAS:
        notes.append(
            "public compatibility constructor reuses another numeric kernel; not counted as migrated R behavior"
        )
    elif implementation_status is ImplementationStatus.PLACEHOLDER:
        notes.append(
            "public surface contains placeholder numerical behavior and requires replacement"
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

    return FamilyCapability(
        name=key,
        implementation_status=implementation_status,
        features=features,
        notes=tuple(notes),
    )


_DEFAULT_CAPABILITIES: dict[str, FamilyCapability] = {
    name.upper(): _build_capability(name) for name in _REGISTERED_FAMILIES
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
        "implementation_statuses": [status.value for status in ImplementationStatus],
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

    implementation_statuses = payload.get("implementation_statuses")
    expected_implementation_statuses = [
        status.value for status in ImplementationStatus
    ]
    if implementation_statuses != expected_implementation_statuses:
        issues.append(
            _capability_matrix_issue(
                "implementation_statuses_mismatch",
                "implementation statuses do not match the runtime registry",
                path="$.implementation_statuses",
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
            implementation_status = family_payload.get("implementation_status")
            valid_implementation_statuses = {
                status.value for status in ImplementationStatus
            }
            if implementation_status not in valid_implementation_statuses:
                issues.append(
                    _capability_matrix_issue(
                        "invalid_implementation_status",
                        (
                            f"family {family_name!r} has invalid implementation "
                            f"status {implementation_status!r}"
                        ),
                        path=f"{family_path}.implementation_status",
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
    allow_compat_alias: bool = False,
    allow_placeholder: bool = False,
    formulas: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a JSON-friendly route-admission report for fitting methods.

    The report is intended for service boundaries that need to decide whether a
    fit job can be admitted before scheduling backend work.  Runtime fitting
    uses the same helper so public services, generated capability matrices, and
    ``gamlss()`` share one method/family routing contract.
    """

    method = str(method_name).upper()
    if method == "CG" and formulas:
        from .formula_parser import has_smooth_terms

        smooth_parameters = [
            parameter for parameter, formula in formulas.items()
            if has_smooth_terms(formula)
        ]
        if smooth_parameters:
            return {
                "ok": False,
                "family": str(family_name).upper(),
                "method": method,
                "feature": "smooth_terms",
                "status": "unsupported",
                "strict": bool(strict),
                "allow_experimental": not strict,
                "allow_compat_alias": bool(allow_compat_alias),
                "allow_placeholder": bool(allow_placeholder),
                "code": "smooth_terms_unsupported_for_method",
                "smooth_parameters": smooth_parameters,
                "message": (
                    "method='CG' does not support penalized smooth terms; "
                    "use method='RS' or method='auto'."
                ),
            }
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
    implementation_status = capability.implementation_status
    if (
        implementation_status is ImplementationStatus.COMPAT_ALIAS
        and not allow_compat_alias
    ):
        return {
            "ok": False,
            "family": capability.name,
            "method": method,
            "feature": feature,
            "status": None,
            "implementation_status": implementation_status.value,
            "strict": bool(strict),
            "allow_experimental": allow_experimental,
            "allow_compat_alias": bool(allow_compat_alias),
            "allow_placeholder": bool(allow_placeholder),
            "code": "compat_alias_requires_opt_in",
            "message": (
                f"family {capability.name!r} method {method!r} is a "
                "compatibility alias; pass allow_compat_alias=True only for "
                "explicit migration or smoke-test work"
            ),
        }
    if (
        implementation_status is ImplementationStatus.PLACEHOLDER
        and not allow_placeholder
    ):
        return {
            "ok": False,
            "family": capability.name,
            "method": method,
            "feature": feature,
            "status": None,
            "implementation_status": implementation_status.value,
            "strict": bool(strict),
            "allow_experimental": allow_experimental,
            "allow_compat_alias": bool(allow_compat_alias),
            "allow_placeholder": bool(allow_placeholder),
            "code": "placeholder_requires_opt_in",
            "message": (
                f"family {capability.name!r} method {method!r} has placeholder "
                "numeric behavior; pass allow_placeholder=True only for explicit "
                "migration or smoke-test work"
            ),
        }
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
        "implementation_status": implementation_status.value,
        "strict": bool(strict),
        "allow_experimental": allow_experimental,
        "allow_compat_alias": bool(allow_compat_alias),
        "allow_placeholder": bool(allow_placeholder),
        "code": code,
        "message": message,
    }


def require_production_family_route(
    family_name: str,
    method_name: str,
    *,
    route: str,
) -> dict[str, object]:
    """Admit a public production route or raise with structured evidence.

    Migration-only opt-ins are intentionally unavailable here: placeholder and
    compatibility-alias kernels must never be admitted by public fitting,
    batch, benchmark, or service routes.
    """

    report = method_route_capability_report(family_name, method_name, strict=False)
    report = {**report, "route": route}
    if report["ok"]:
        return report
    if report.get("implementation_status") in {
        ImplementationStatus.PLACEHOLDER.value,
        ImplementationStatus.COMPAT_ALIAS.value,
    }:
        raise ProductionFamilyRejectedError(report)
    return report


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
    "ImplementationStatus",
    "METHOD_ROUTE_FEATURES",
    "capability_matrix",
    "FamilyCapability",
    "FamilyCapabilityError",
    "ProductionFamilyRejectedError",
    "family_capability_names",
    "family_supports",
    "get_family_capability",
    "implementation_status_for_family",
    "list_family_capabilities",
    "method_capability_features",
    "method_route_capability_report",
    "method_route_feature",
    "METHOD_CAPABILITY_FEATURES",
    "validate_capability_matrix_payload",
    "require_family_capability",
    "require_production_family_route",
    "require_method_route",
]
