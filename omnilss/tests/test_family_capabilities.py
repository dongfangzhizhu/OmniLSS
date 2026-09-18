from __future__ import annotations

import pytest

from omnilss.distribution_registry import _REGISTERED_FAMILIES
from omnilss.family_capabilities import (
    CapabilityStatus,
    CAPABILITY_MATRIX_VERSION,
    FEATURES,
    ImplementationStatus,
    METHOD_ROUTE_FEATURES,
    FamilyCapabilityError,
    capability_matrix,
    method_capability_features,
    method_route_capability_report,
    family_capability_names,
    family_supports,
    get_family_capability,
    _dpqr_function_uses_placeholder,
    implementation_status_for_family,
    method_route_feature,
    list_family_capabilities,
    require_family_capability,
    require_method_route,
)


def test_capability_registry_covers_all_registered_families():
    assert set(family_capability_names()) == set(_REGISTERED_FAMILIES)
    assert len(list_family_capabilities()) == len(_REGISTERED_FAMILIES)


def test_every_family_has_every_feature_status():
    for capability in list_family_capabilities():
        assert set(capability.features) == set(FEATURES)
        assert isinstance(capability.implementation_status, ImplementationStatus)
        assert all(
            isinstance(status, CapabilityStatus)
            for status in capability.features.values()
        )
        assert capability.as_dict()["name"] == capability.name
        assert (
            capability.as_dict()["implementation_status"]
            == capability.implementation_status.value
        )


def test_known_core_family_statuses_are_case_insensitive():
    no_capability = get_family_capability("no")
    assert no_capability.implementation_status is ImplementationStatus.IMPLEMENTED
    assert no_capability.status("r_consistency") is CapabilityStatus.VALIDATED
    assert no_capability.status("production_safe") is CapabilityStatus.VALIDATED
    assert family_supports("NO", "rs_jax_fit")


def test_known_compat_alias_families_are_marked_separately_from_evidence():
    for family_name in ("DBI", "DBURR12", "BCCGUNTR", "COUNT_1_31"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is ImplementationStatus.COMPAT_ALIAS
        assert implementation_status_for_family(family_name) is (
            ImplementationStatus.COMPAT_ALIAS
        )
        assert any("compatibility constructor" in note for note in capability.notes)


def test_ad_families_without_explicit_dpqr_are_marked_placeholder():
    for family_name in ("GB1", "GIG"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is ImplementationStatus.PLACEHOLDER
        assert implementation_status_for_family(family_name) is (
            ImplementationStatus.PLACEHOLDER
        )
        assert any("placeholder numerical behavior" in note for note in capability.notes)


def test_dpqr_placeholder_helpers_are_marked_placeholder():
    for helper_name in (
        "generic_p_not_implemented",
    ):
        assert _dpqr_function_uses_placeholder(helper_name) is True

def test_ad_families_with_explicit_dpqr_are_not_false_placeholder():
    for family_name in ("BE", "WEI", "IG", "NBI", "NBII", "SEP1", "ST1", "TF", "GB2"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is ImplementationStatus.EXPERIMENTAL
        assert implementation_status_for_family(family_name) is (
            ImplementationStatus.EXPERIMENTAL
        )


def test_sichel_boundary_nan_handling_is_not_placeholder_behavior():
    """Invalid-input NaN returns must not mask a complete SICHEL dpqr wiring."""
    capability = get_family_capability("SICHEL")
    assert capability.implementation_status is ImplementationStatus.EXPERIMENTAL
    assert implementation_status_for_family("SICHEL") is ImplementationStatus.EXPERIMENTAL


def test_del_explicit_dpqr_is_experimental_not_placeholder():
    """DEL has a real finite-support implementation, but remains experimental."""
    for family_name in ("DEL", "DELAPORT"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is ImplementationStatus.EXPERIMENTAL
        assert implementation_status_for_family(family_name) is ImplementationStatus.EXPERIMENTAL
        assert capability.status("r_consistency") is CapabilityStatus.VALIDATED
        assert capability.status("production_safe") is CapabilityStatus.EXPERIMENTAL


def test_unsupported_jax_route_is_reported():
    capability = get_family_capability("GB2")
    assert capability.status("rs_jax_fit") is CapabilityStatus.UNSUPPORTED
    assert not family_supports("GB2", "rs_jax_fit")
    with pytest.raises(FamilyCapabilityError, match="does not support"):
        require_family_capability("GB2", "rs_jax_fit", allow_experimental=True)


def test_experimental_features_require_explicit_opt_in():
    with pytest.raises(FamilyCapabilityError, match="experimental"):
        require_family_capability("GA", "rs_fit")
    assert (
        require_family_capability("GA", "rs_fit", allow_experimental=True).name == "GA"
    )


def test_unknown_feature_and_family_fail_clearly():
    with pytest.raises(KeyError, match="unknown family capability feature"):
        get_family_capability("NO").status("not_a_feature")
    with pytest.raises(KeyError, match="not present"):
        get_family_capability("NOT_A_FAMILY")


def test_method_capability_features_match_runtime_routing_contract():
    expected = {
        "RS": "rs_fit",
        "RS_JAX": "rs_jax_fit",
        "CG": "cg_fit",
        "MIXED": "cg_fit",
        "JOINT": "cg_fit",
        "LBFGS": "cg_fit",
    }
    assert method_capability_features() == expected
    assert dict(METHOD_ROUTE_FEATURES) == expected


def test_method_route_capability_report_supports_service_admission():
    validated = method_route_capability_report("NO", "RS", strict=True)
    assert validated == {
        "ok": True,
        "family": "NO",
        "method": "RS",
        "feature": "rs_fit",
        "status": "validated",
        "implementation_status": "implemented",
        "strict": True,
        "allow_experimental": False,
        "allow_compat_alias": False,
        "allow_placeholder": False,
        "code": "validated",
        "message": "family 'NO' method 'RS' is validated through capability feature 'rs_fit'",
    }

    experimental = method_route_capability_report("GA", "RS", strict=True)
    assert experimental["ok"] is False
    assert experimental["code"] == "experimental_requires_opt_in"
    assert experimental["status"] == "experimental"

    unsupported = method_route_capability_report("GIG", "RS_JAX")
    assert unsupported["ok"] is False
    assert unsupported["code"] == "placeholder_requires_opt_in"
    assert unsupported["feature"] == "rs_jax_fit"
    assert unsupported["implementation_status"] == "placeholder"

    unsupported_with_placeholder_opt_in = method_route_capability_report(
        "GIG", "RS_JAX", allow_placeholder=True
    )
    assert unsupported_with_placeholder_opt_in["ok"] is False
    assert unsupported_with_placeholder_opt_in["code"] == "unsupported_route"

    unknown = method_route_capability_report("NO", "NOT_A_METHOD")
    assert unknown["ok"] is False
    assert unknown["code"] == "unknown_method"
    assert unknown["feature"] is None

    unknown_family = method_route_capability_report("NOT_A_FAMILY", "RS")
    assert unknown_family["ok"] is False
    assert unknown_family["code"] == "unknown_family"
    assert unknown_family["feature"] == "rs_fit"

    compat = method_route_capability_report("DBI", "RS")
    assert compat["ok"] is False
    assert compat["code"] == "compat_alias_requires_opt_in"
    assert compat["implementation_status"] == "compat_alias"

    compat_opt_in = method_route_capability_report(
        "DBI", "RS", allow_compat_alias=True
    )
    assert compat_opt_in["ok"] is True
    assert compat_opt_in["implementation_status"] == "compat_alias"


def test_capability_matrix_is_machine_readable():
    matrix = capability_matrix()
    assert matrix["version"] == CAPABILITY_MATRIX_VERSION
    assert matrix["version"] == 4
    assert matrix["features"] == list(FEATURES)
    assert matrix["implementation_statuses"] == [
        status.value for status in ImplementationStatus
    ]
    assert matrix["method_capability_features"] == method_capability_features()
    assert matrix["method_capability_features"]["RS_JAX"] == "rs_jax_fit"
    assert matrix["strict_capability_policy"] == {
        "default_allow_experimental": True,
        "strict_capabilities_allow_experimental": False,
        "unsupported_routes_fail_fast": True,
    }
    assert "NO" in matrix["families"]
    assert matrix["families"]["NO"]["implementation_status"] == "implemented"
    assert matrix["families"]["ST1"]["implementation_status"] == "experimental"
    assert matrix["families"]["GB2"]["implementation_status"] == "experimental"
    assert matrix["families"]["NO"]["features"]["production_safe"] == "validated"


def test_production_safe_family_has_validated_core_route():
    capability = get_family_capability("NO")
    assert capability.status("rs_fit") is CapabilityStatus.VALIDATED
    assert capability.status("prediction") is CapabilityStatus.VALIDATED
    assert require_family_capability("NO", "rs_fit").name == "NO"


def test_capability_matrix_exposes_method_route_feature_map():
    matrix = capability_matrix()
    assert matrix["method_routes"] == matrix["method_capability_features"]
    assert matrix["method_routes"]["RS"] == "rs_fit"
    assert matrix["method_routes"]["RS_JAX"] == "rs_jax_fit"
    assert method_route_feature("lbfgs") == "cg_fit"
    with pytest.raises(KeyError, match="capability routing map"):
        method_route_feature("NOT_A_METHOD")


def test_require_method_route_uses_family_capability_tiers():
    assert require_method_route("NO", "RS").name == "NO"
    with pytest.raises(FamilyCapabilityError, match="experimental"):
        require_method_route("GA", "RS")
    assert require_method_route("GA", "RS", allow_experimental=True).name == "GA"
    with pytest.raises(FamilyCapabilityError, match="does not support"):
        require_method_route("GB2", "RS_JAX", allow_experimental=True)
    with pytest.raises(KeyError, match="capability routing map"):
        require_method_route("NO", "NOT_A_METHOD")
