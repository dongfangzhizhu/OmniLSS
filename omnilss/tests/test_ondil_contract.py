"""Contract tests for the ondil-shaped distribution view (P5-1b).

The written contract is docs/development/p5-ondil-distribution-contract-2026-11-15.md.
These tests pin the executable half: theta ordering, per-observation
derivatives, link direction, the boundary rule (with a ratchet for the known
violations) and the generated artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import omnilss.ondil_contract as contract

ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "development"
    / "p5-ondil-contract-2026-11-15.json"
)
FAMILIES = tuple(sorted(contract.ONDIL_PARAMETER_ORDER))


def _theta(name: str, rows: int = 3) -> np.ndarray:
    view = contract.as_ondil(name)
    return np.tile(np.asarray(contract._probe_theta(view)), (rows, 1))


@pytest.mark.parametrize("name", FAMILIES)
def test_theta_round_trip_matches_declared_order(name: str) -> None:
    view = contract.as_ondil(name)
    params = {
        parameter: np.full(3, 0.5 + index)
        for index, parameter in enumerate(view.estimable)
    }
    theta = contract.theta_from_params(name, params)

    assert theta.shape == (3, view.n_params)
    for index, parameter in enumerate(view.estimable):
        np.testing.assert_allclose(theta[:, index], params[parameter])
    restored = contract.params_from_theta(name, theta)
    for parameter in view.estimable:
        np.testing.assert_allclose(restored[parameter], params[parameter])


@pytest.mark.parametrize("name", FAMILIES)
def test_contract_report_has_no_mismatches(name: str) -> None:
    report = contract.contract_report(name)

    assert report["order_matches_ondil"] is True
    assert report["mismatches"] == []
    assert report["theta_order"] == {
        str(index): parameter for index, parameter in enumerate(report["estimable"])
    }
    for link in report["links"].values():
        assert link["ondil"] != "unsupported", link


@pytest.mark.parametrize("name", ("NO", "GA", "PO"))
def test_view_agrees_with_the_family_dpqr(name: str) -> None:
    view = contract.as_ondil(name)
    theta = _theta(name)
    y = np.array([0.4, 1.2, 2.5])
    params = contract.params_from_theta(name, theta)

    expected = np.asarray(view.family.d(y, **params, log=True))
    np.testing.assert_allclose(view.logpdf(y, theta), expected, rtol=1e-12)
    expected_cdf = np.asarray(view.family.p(y, **params))
    np.testing.assert_allclose(view.cdf(y, theta), expected_cdf, rtol=1e-10)
    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = view.ppf(probabilities, theta)
    if view.is_discrete:
        # ppf is the smallest integer with F(k) >= p, so the round trip is an
        # inequality and the previous integer must fall short of p.
        assert np.all(view.cdf(quantiles, theta) >= probabilities - 1e-12)
        assert np.all(view.cdf(quantiles - 1.0, theta) < probabilities)
    else:
        np.testing.assert_allclose(view.cdf(quantiles, theta), probabilities, atol=1e-8)


@pytest.mark.parametrize("name", ("NO", "GA", "BE"))
def test_score_is_per_observation_and_matches_finite_difference(name: str) -> None:
    view = contract.as_ondil(name)
    theta = _theta(name)
    y = np.array([0.2, 0.5, 0.8])

    for param in range(view.n_params):
        analytic = view.dl1_dp1(y, theta, param)
        assert analytic.shape == y.shape
        step = 1e-6
        plus = theta.copy()
        minus = theta.copy()
        plus[:, param] += step
        minus[:, param] -= step
        numeric = (view.logpdf(y, plus) - view.logpdf(y, minus)) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_cross_derivative_is_symmetric_and_matches_finite_difference() -> None:
    view = contract.as_ondil("NO")
    theta = _theta("NO")
    y = np.array([0.2, 0.5, 0.8])

    cross = view.dl2_dpp(y, theta, (0, 1))
    np.testing.assert_allclose(cross, view.dl2_dpp(y, theta, (1, 0)), rtol=1e-10)

    step = 1e-6
    plus = theta.copy()
    minus = theta.copy()
    plus[:, 1] += step
    minus[:, 1] -= step
    numeric = (view.dl1_dp1(y, plus, 0) - view.dl1_dp1(y, minus, 0)) / (2 * step)
    np.testing.assert_allclose(cross, numeric, rtol=1e-4, atol=1e-6)

    with pytest.raises(ValueError, match="different parameters"):
        view.dl2_dpp(y, theta, (0, 0))


@pytest.mark.parametrize("name", FAMILIES)
def test_boundary_rule_ratchet(name: str) -> None:
    probe = contract.probe_support(name)

    assert probe["probe_valid"], probe
    assert probe["boundary_ok"] or name in contract.KNOWN_BOUNDARY_VIOLATIONS, probe


def test_known_boundary_violations_are_exactly_the_frozen_set() -> None:
    observed = {
        name for name in FAMILIES if not contract.probe_support(name)["boundary_ok"]
    }
    assert observed == set(contract.KNOWN_BOUNDARY_VIOLATIONS)


def test_log_link_direction_matches_the_contract() -> None:
    view = contract.as_ondil("GA")
    link = view.links[1]  # sigma uses the log link

    assert link.name == "Log"
    np.testing.assert_allclose(link.inverse_derivative(0.0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(link.link_derivative(2.0), 0.5, rtol=1e-12)
    mu = np.array([0.5, 1.0, 2.0])
    eta = link.link(mu)
    product = link.link_derivative(mu) * link.inverse_derivative(eta)
    np.testing.assert_allclose(product, 1.0, rtol=1e-12)


def test_mixed_family_keeps_atoms_in_the_log_density() -> None:
    view = contract.as_ondil("BEINF")
    theta = _theta("BEINF")

    assert view.is_discrete is False
    atom = view.logpdf(np.array([0.0]), theta[:1])
    assert np.isfinite(atom[0])
    below = view.logpdf(np.array([-0.001]), theta[:1])
    assert below[0] == -np.inf
    report = contract.contract_report("BEINF")
    assert report["methods"]["logpmf"].startswith("unsupported")


def test_discrete_family_reports_pmf_and_no_cross_derivative() -> None:
    view = contract.as_ondil("PO")
    report = contract.contract_report("PO")

    assert view.is_discrete is True
    assert report["methods"]["logpmf"] == "supported"
    assert report["methods"]["logpdf"].startswith("unsupported")
    assert report["methods"]["dl2_dpp"].startswith("n/a")


def test_generated_artifact_matches_the_live_scan() -> None:
    if not ARTIFACT.exists():
        pytest.skip("run tools/ondil_contract_scan.py --write first")

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    summary = payload["summary"]

    assert summary["families"] == len(FAMILIES)
    assert summary["order_mismatches"] == 0
    assert summary["support_mismatches"] == 0
    assert summary["boundary_violations"] == sorted(contract.KNOWN_BOUNDARY_VIOLATIONS)
    assert [row["family"] for row in payload["families"]] == sorted(FAMILIES)
    assert payload["contract"].endswith("p5-ondil-distribution-contract-2026-11-15.md")
