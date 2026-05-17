from __future__ import annotations

import pytest

from omnilss.distribution_registry import _REGISTERED_FAMILIES
from omnilss.family_capabilities import (
    CapabilityStatus,
    FEATURES,
    FamilyCapabilityError,
    family_capability_names,
    family_supports,
    get_family_capability,
    list_family_capabilities,
    require_family_capability,
)


def test_capability_registry_covers_all_registered_families():
    assert set(family_capability_names()) == set(_REGISTERED_FAMILIES)
    assert len(list_family_capabilities()) == len(_REGISTERED_FAMILIES)


def test_every_family_has_every_feature_status():
    for capability in list_family_capabilities():
        assert set(capability.features) == set(FEATURES)
        assert all(isinstance(status, CapabilityStatus) for status in capability.features.values())
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
    assert require_family_capability("GA", "rs_fit", allow_experimental=True).name == "GA"


def test_unknown_feature_and_family_fail_clearly():
    with pytest.raises(KeyError, match="unknown family capability feature"):
        get_family_capability("NO").status("not_a_feature")
    with pytest.raises(KeyError, match="not present"):
        get_family_capability("NOT_A_FAMILY")
