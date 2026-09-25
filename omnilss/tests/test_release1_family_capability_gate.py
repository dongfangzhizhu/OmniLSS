from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnilss import gamlss
from omnilss.benchmarks.runner import BenchmarkRunner, BenchmarkSpec
from omnilss.family_capabilities import (
    FamilyCapabilityError,
    ImplementationStatus,
    ProductionFamilyRejectedError,
    get_family_capability,
    require_production_family_route,
)
from tools.generate_family_capability_gate_evidence import build_case, write_artifacts


# Validates: Requirements 1.6
REPO_ROOT = Path(__file__).resolve().parents[2]
UNSAFE = tuple(
    capability.name
    for capability in (
        get_family_capability(name)
        for name in ("GIG", "DBI", "NBI")
    )
    if capability.implementation_status
    in {ImplementationStatus.PLACEHOLDER, ImplementationStatus.COMPAT_ALIAS}
)


def test_all_known_unsafe_states_are_rejected_with_structured_reason():
    assert {get_family_capability(name).implementation_status for name in UNSAFE} == {
        ImplementationStatus.PLACEHOLDER,
        ImplementationStatus.COMPAT_ALIAS,
    }
    for family in UNSAFE:
        with pytest.raises(ProductionFamilyRejectedError) as caught:
            require_production_family_route(family, "RS", route="public_fit")
        report = caught.value.to_dict()
        assert report["family"] == family
        assert report["route"] == "public_fit"
        assert report["implementation_status"] in {"placeholder", "compat_alias"}
        assert report["code"] in {"placeholder_requires_opt_in", "compat_alias_requires_opt_in"}


def test_implemented_family_is_admitted_and_unsafe_exploratory_fit_is_blocked():
    assert require_production_family_route("NO", "RS", route="public_fit")["ok"] is True
    with pytest.raises(FamilyCapabilityError, match="capability feature 'rs_fit'"):
        gamlss("y ~ x", family="GIG", data={"y": [1.0, 2.0], "x": [0.0, 1.0]})


def test_experimental_family_is_fail_closed_only_on_production_route():
    with pytest.raises(ProductionFamilyRejectedError) as caught:
        require_production_family_route("GA", "RS", route="service")
    report = caught.value.to_dict()
    assert report["implementation_status"] == "experimental"
    assert report["code"] == "experimental_requires_opt_in"


def test_benchmark_route_uses_the_shared_gate_before_workload():
    runner = BenchmarkRunner()
    with pytest.raises(ProductionFamilyRejectedError) as caught:
        runner.run(BenchmarkSpec("correctness", "DBI", "RS", "normal", n=4))
    assert caught.value.to_dict()["route"] == "benchmark"


def test_property_unsafe_statuses_and_all_production_routes_are_consistent_100_runs():
    """Property 2: every unsafe family is rejected identically at every route."""
    for _ in range(100):
        for family in UNSAFE:
            for route in ("public_fit", "batch", "benchmark", "service"):
                with pytest.raises(ProductionFamilyRejectedError) as caught:
                    require_production_family_route(family, "RS", route=route)
                report = caught.value.to_dict()
                assert report["route"] == route
                assert report["ok"] is False
                assert report["implementation_status"] in {"placeholder", "compat_alias"}
        for route in ("public_fit", "batch", "benchmark", "service"):
            report = require_production_family_route("NO", "RS", route=route)
            assert report["ok"] is True
            assert report["implementation_status"] == "implemented"


def test_fixed_r_source_case_records_rejection_and_never_claims_r_success(tmp_path: Path):
    case = build_case(REPO_ROOT)
    assert case["fixed_r_sources"]
    assert case["family_status"] == "placeholder"
    assert case["admission"]["ok"] is False
    assert case["comparison"] == {
        "status": "excluded",
        "r_consistency_success": False,
        "reason": "production capability rejection occurred before R comparison",
    }
    json_path, markdown_path = write_artifacts(tmp_path, REPO_ROOT)
    assert json.loads(json_path.read_text(encoding="utf-8"))["comparison"]["r_consistency_success"] is False
    assert "not a success result" in markdown_path.read_text(encoding="utf-8")
