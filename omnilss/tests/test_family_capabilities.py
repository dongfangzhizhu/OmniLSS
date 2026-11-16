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
    # D2.16 used BCCGUNTR as the sample here; it is a full implementation now
    # (its own untruncated density and R's analytic derivatives), so it left
    # the compat-alias set.  D2.23 did the same for DBI, which stopped being a
    # BI forwarder and became a real two-parameter likelihood -- its coverage
    # ratchet moved DBI/dDBI/pDBI/qDBI/rDBI out of compat_alias in the same
    # commit.  D2.46 did it for DBURR12, whose module moved to
    # omnilss.distributions_dburr12.  The remaining samples still hold.
    for family_name in ("COUNT_1_22", "COUNT_1_31"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is ImplementationStatus.COMPAT_ALIAS
        assert implementation_status_for_family(family_name) is (
            ImplementationStatus.COMPAT_ALIAS
        )
        assert any("compatibility constructor" in note for note in capability.notes)


def test_no_family_is_left_in_the_placeholder_branch():
    # This test used to assert PLACEHOLDER for one sample family, and the
    # sample kept going stale (GIG left in D2.48, LOGSHASH afterwards, IGA in
    # D2.77, GENERALISEDPOISSON in D2.79, DPO1 in D2.80).  There are no
    # placeholder families left, so the assertion is now about the set -- and
    # the detector itself is still exercised by
    # test_dpqr_placeholder_helpers_are_marked_placeholder below.
    placeholders = [
        capability.name
        for capability in list_family_capabilities()
        if capability.implementation_status is ImplementationStatus.PLACEHOLDER
    ]
    assert placeholders == [], placeholders
    assert all(
        implementation_status_for_family(name) is not ImplementationStatus.PLACEHOLDER
        for name in family_capability_names()
    )


def test_dpqr_placeholder_helpers_are_marked_placeholder():
    for helper_name in (
        "generic_p_not_implemented",
    ):
        assert _dpqr_function_uses_placeholder(helper_name) is True

def test_ad_families_with_explicit_dpqr_are_not_false_placeholder():
    # The invariant is "not misclassified as a placeholder", not "stays
    # experimental": D2.241 admitted BE/WEI/IG/NBI/NBII, which legitimately
    # promotes their implementation status to IMPLEMENTED.  The code-derived
    # status (which ignores admission) stays EXPERIMENTAL for all of them.
    for family_name in ("BB", "BE", "WEI", "IG", "NBI", "NBII", "SEP1", "ST1", "TF", "GB2", "GB1"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is not ImplementationStatus.PLACEHOLDER
        assert implementation_status_for_family(family_name) is (
            ImplementationStatus.EXPERIMENTAL
        )


def test_zipf_and_zalg_r_consistency_evidence_is_registered():
    for family_name in ("ZIPF", "ZALG"):
        capability = get_family_capability(family_name)
        assert capability.implementation_status is ImplementationStatus.EXPERIMENTAL
        assert capability.status("r_consistency") is CapabilityStatus.VALIDATED


def test_zip_rs_fit_evidence_is_admitted_in_strict_route():
    capability = get_family_capability("ZIP")
    assert capability.status("rs_fit") is CapabilityStatus.VALIDATED
    report = method_route_capability_report("ZIP", "RS", strict=True)
    assert report["ok"] is True
    assert report["code"] == "validated"


def test_zip_prediction_evidence_is_registered():
    capability = get_family_capability("ZIP")
    assert capability.status("prediction") is CapabilityStatus.VALIDATED


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
    # GA became production-safe in D2.241, so the sample is DEL, which still has
    # no RS fit evidence and therefore an experimental rs_fit feature.
    with pytest.raises(FamilyCapabilityError, match="experimental"):
        require_family_capability("DEL", "rs_fit")
    assert (
        require_family_capability("DEL", "rs_fit", allow_experimental=True).name == "DEL"
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
        "HYBRID": "cg_fit",
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

    experimental = method_route_capability_report("DEL", "RS", strict=True)
    assert experimental["ok"] is False
    assert experimental["code"] == "experimental_requires_opt_in"
    assert experimental["status"] == "experimental"

    # No family holds the placeholder branch any more (DPO1 was the last,
    # D2.80), so the placeholder-route assertions moved to the detector test
    # above.  What still matters here is that a route the family does not
    # support is refused rather than silently admitted: DPO1 does not
    # advertise RS_JAX.
    unsupported = method_route_capability_report("DPO1", "RS_JAX")
    assert unsupported["ok"] is False
    assert unsupported["code"] == "unsupported_route"
    assert unsupported["feature"] == "rs_jax_fit"

    unknown = method_route_capability_report("NO", "NOT_A_METHOD")
    assert unknown["ok"] is False
    assert unknown["code"] == "unknown_method"
    assert unknown["feature"] is None

    unknown_family = method_route_capability_report("NOT_A_FAMILY", "RS")
    assert unknown_family["ok"] is False
    assert unknown_family["code"] == "unknown_family"
    assert unknown_family["feature"] == "rs_fit"

    # DBI was the sample here until D2.23, when it stopped forwarding to BI and
    # became a real two-parameter likelihood -- so it is now `experimental`
    # and requires its own opt-in, covered just above.  DBURR12 followed in
    # D2.46; COUNT_1_22 keeps this branch exercised.
    compat = method_route_capability_report("COUNT_1_22", "RS")
    assert compat["ok"] is False
    assert compat["code"] == "compat_alias_requires_opt_in"
    assert compat["implementation_status"] == "compat_alias"

    compat_opt_in = method_route_capability_report(
        "COUNT_1_22", "RS", allow_compat_alias=True
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
        require_method_route("DEL", "RS")
    assert require_method_route("DEL", "RS", allow_experimental=True).name == "DEL"
    with pytest.raises(FamilyCapabilityError, match="does not support"):
        require_method_route("GB2", "RS_JAX", allow_experimental=True)
    with pytest.raises(KeyError, match="capability routing map"):
        require_method_route("NO", "NOT_A_METHOD")
