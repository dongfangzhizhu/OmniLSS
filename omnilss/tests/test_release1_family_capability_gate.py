from __future__ import annotations

import inspect
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
        for name in ("LOGSHASH", "DBI", "NBI")
    )
    if capability.implementation_status
    in {
        ImplementationStatus.PLACEHOLDER,
        ImplementationStatus.COMPAT_ALIAS,
        # D1 cleared the placeholder queue: EXPERIMENTAL is the state the
        # production gate still refuses (see the docstring above), so it has to
        # be part of "known unsafe" or this collector returns nothing.
        ImplementationStatus.EXPERIMENTAL,
    }
)


def test_all_known_unsafe_states_are_rejected_with_structured_reason():
    # The set membership is derived above (any family that is still a
    # placeholder or a compat alias), so this asserts the *invariant* -- every
    # unsafe family is in one of the two fail-closed states -- rather than
    # asserting that both states are represented.  Those are different claims,
    # and only the first survives a family leaving the compat-alias list:
    # DBI did so in D2.23, becoming a real two-parameter likelihood, and GIG
    # did so in D2.48, gaining d/p/q/r plus R's derivative slots.  The stronger
    # assertion then failed with "extra items in the right set" while the gate
    # itself was working exactly as designed.
    assert UNSAFE, "no family is fail-closed any more -- is that intended?"
    for family in UNSAFE:
        assert get_family_capability(family).implementation_status in {
            ImplementationStatus.PLACEHOLDER,
            ImplementationStatus.COMPAT_ALIAS,
            ImplementationStatus.EXPERIMENTAL,
        }, (
            f"{family} is in UNSAFE but its status is "
            f"{get_family_capability(family).implementation_status}"
        )
    for family in UNSAFE:
        with pytest.raises(ProductionFamilyRejectedError) as caught:
            require_production_family_route(family, "RS", route="public_fit")
        report = caught.value.to_dict()
        assert report["family"] == family
        assert report["route"] == "public_fit"
        assert report["implementation_status"] in {
            "placeholder", "compat_alias", "experimental",
        }
        assert report["code"] in {
            "placeholder_requires_opt_in",
            "compat_alias_requires_opt_in",
            "experimental_requires_opt_in",
        }


@pytest.mark.parametrize("route", ("public_fit", "service", "benchmark"))
def test_dbi_stays_fail_closed_after_leaving_the_compat_alias_list(route):
    """DBI became a real family in D2.23 -- it must not have become *admitted*.

    DBI dropped out of the compat-alias set (and out of ``UNSAFE`` above) when
    it stopped forwarding to BI and gained a real two-parameter likelihood.
    That is the intended direction of travel, but "no longer a compat alias"
    and "safe for production" are different claims, and only one is true:
    ``EXPERIMENTAL`` is still rejected on every production route.

    Asserted per route rather than once, because the routes have separate
    gates and a family can be blocked on one and admitted on another --
    ``test_experimental_family_is_fail_closed_only_on_production_route``
    already distinguishes that shape for GA.

    There is deliberately **no opt-in** on this function: its docstring rules
    out migration flags "so that placeholder and compatibility-alias kernels
    must never be admitted by public fitting, batch, benchmark, or service
    routes", and ``EXPERIMENTAL`` inherits that.  Asserted by inspecting the
    signature rather than by calling it -- a test that discovers the flag is
    missing by catching ``TypeError`` passes for the wrong reason if the flag
    is ever added on purpose.
    """
    with pytest.raises(ProductionFamilyRejectedError) as caught:
        require_production_family_route("DBI", "RS", route=route)
    report = caught.value.to_dict()
    assert report["ok"] is False
    assert report["implementation_status"] == "experimental"
    assert report["code"] == "experimental_requires_opt_in"
    assert report["route"] == route

    assert "allow_experimental" not in inspect.signature(
        require_production_family_route
    ).parameters, (
        "a production route gained an experimental opt-in; if that was "
        "deliberate, this test's premise needs revisiting with the same care "
        "as the gate itself"
    )


def test_implemented_family_is_admitted_and_unsafe_exploratory_fit_is_blocked():
    assert require_production_family_route("NO", "RS", route="public_fit")["ok"] is True
    # GIG was this sample until D2.48, then LOGSHASH, and D1 has since cleared
    # the placeholder queue entirely: LOGSHASH is EXPERIMENTAL now, so a plain
    # gamlss() call is *allowed* and the fail-closed behaviour lives in strict
    # capability mode.  The message must name the experimental state.
    gamlss("y ~ x", family="LOGSHASH", data={"y": [1.0, 2.0], "x": [0.0, 1.0]})
    with pytest.raises(FamilyCapabilityError, match="experimental"):
        gamlss(
            "y ~ x", family="LOGSHASH", data={"y": [1.0, 2.0], "x": [0.0, 1.0]},
            strict_capabilities=True,
        )


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
                assert report["implementation_status"] in {
                    "placeholder", "compat_alias", "experimental",
                }
        for route in ("public_fit", "batch", "benchmark", "service"):
            report = require_production_family_route("NO", "RS", route=route)
            assert report["ok"] is True
            assert report["implementation_status"] == "implemented"


def test_fixed_r_source_case_records_rejection_and_never_claims_r_success(tmp_path: Path):
    case = build_case(REPO_ROOT)
    assert case["fixed_r_sources"]
    # D1 cleared the placeholder queue; the fixed-R-source case family is
    # experimental now, and the rejection reason follows the matrix.
    assert case["family_status"] == "experimental"
    assert case["admission"]["ok"] is False
    assert case["comparison"] == {
        "status": "excluded",
        "r_consistency_success": False,
        "reason": "production capability rejection occurred before R comparison",
    }
    json_path, markdown_path = write_artifacts(tmp_path, REPO_ROOT)
    assert json.loads(json_path.read_text(encoding="utf-8"))["comparison"]["r_consistency_success"] is False
    assert "not a success result" in markdown_path.read_text(encoding="utf-8")
