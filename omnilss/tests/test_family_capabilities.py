from __future__ import annotations

import pytest

from omnilss.distribution_registry import _REGISTERED_FAMILIES
from omnilss.family_capabilities import (
    CapabilityStatus,
    FEATURES,
    FamilyCapabilityError,
    capability_matrix,
    family_capability_names,
    family_supports,
    get_family_capability,
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
        assert all(
            isinstance(status, CapabilityStatus)
            for status in capability.features.values()
        )
        assert capability.as_dict()["name"] == capability.name


def test_known_core_family_statuses_are_case_insensitive():
    no_capability = get_family_capability("no")
    assert no_capability.status("r_consistency") is CapabilityStatus.VALIDATED
    assert no_capability.status("production_safe") is CapabilityStatus.VALIDATED
    assert family_supports("NO", "rs_jax_fit")


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


def test_capability_matrix_is_machine_readable():
    matrix = capability_matrix()
    assert matrix["version"] == 1
    assert matrix["features"] == list(FEATURES)
    assert "NO" in matrix["families"]
    assert matrix["families"]["NO"]["features"]["production_safe"] == "validated"


def test_production_safe_family_has_validated_core_route():
    capability = get_family_capability("NO")
    assert capability.status("rs_fit") is CapabilityStatus.VALIDATED
    assert capability.status("prediction") is CapabilityStatus.VALIDATED
    assert require_family_capability("NO", "rs_fit").name == "NO"


def test_capability_matrix_exposes_method_route_feature_map():
    matrix = capability_matrix()
    assert matrix["method_routes"]["RS"] == "rs_fit"
    assert matrix["method_routes"]["RS_JAX"] == "rs_jax_fit"
    assert method_route_feature("lbfgs") == "cg_fit"


def test_require_method_route_uses_family_capability_tiers():
    assert require_method_route("NO", "RS").name == "NO"
    with pytest.raises(FamilyCapabilityError, match="experimental"):
        require_method_route("GA", "RS")
    assert require_method_route("GA", "RS", allow_experimental=True).name == "GA"
    with pytest.raises(FamilyCapabilityError, match="does not support"):
        require_method_route("GB2", "RS_JAX", allow_experimental=True)
