# SPDX-License-Identifier: GPL-3.0-or-later
"""Live-R parity for R-compatible MIXED (P0-3).

Golden reference: ``omnilss/tests/fixtures/r_golden/mixed_2026-11-14.json``,
written by ``generate_mixed_golden.R`` against live R 4.6.0 / gamlss 5.5.0 /
gamlss.dist 6.1.1 with ``set.seed(1)``.

R's ``mixed(n1, n2)`` is a *local* function inside ``gamlss()``
(``R/gamlss/R/gamlss-5.R:163``)::

    mixed <- function(n1=1, n2=20) { RS(n.cyc=n1); CG(n.cyc=n2) }

Our ``mixed_fit(n1, n2)`` mirrors it with ``rs_fit(max_iter=n1)`` ->
``cg_fit_v2(max_iter=n2, initial_model=...)``.  Before P0-3 the only MIXED
tests asserted *internal* self-consistency (mixed == our own RS-to-CG
pipeline); this file pins the actual R numbers, which is the P0-3 acceptance
criterion ("与 R mixed 数值/语义对照通过").

The old "75% damped RS" inline semantics (``_apply_method_step``'s MIXED
branch and both docstrings) were removed in P0-3: public
``method="MIXED"`` returns ``mixed_fit`` early, so that branch was dead code
and its "experimental damped RS" description contradicted shipped behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss
from omnilss._fitting_utils import _apply_method_step
from omnilss.algorithms.mixed_algorithm import mixed_fit

GOLDEN = Path(__file__).parent / "fixtures" / "r_golden" / "mixed_2026-11-14.json"

# (n1, n2) pairs R's mixed() was run with, plus their golden key.
CASES = ((1, 4), (2, 3), (1, 0))


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(golden) -> dict[str, np.ndarray]:
    return {
        "y": np.asarray(golden["data"]["y"], dtype=np.float64),
        "x": np.asarray(golden["data"]["x"], dtype=np.float64),
    }


def _fit(n1: int, n2: int, data: dict):
    return mixed_fit(
        "y ~ x", family="NO", data=data, sigma_formula="~ x", n1=n1, n2=n2
    )


@pytest.mark.parametrize(("n1", "n2"), CASES)
def test_mixed_coefficients_match_live_r(golden, data, n1, n2):
    """mu/sigma coefficients reproduce R's mixed(n1,n2) to optimizer tolerance."""
    ref = golden[f"mixed_{n1}_{n2}"]
    model = _fit(n1, n2, data)
    mu = np.ravel(model.coefficients["mu"])
    sigma = np.ravel(model.coefficients["sigma"])
    # CG/RS optimiser paths differ in their last WLS step, so fit parity uses
    # looser tolerances than the bit-for-bit density(dpqr) gates.  The measured
    # gaps are mu<=2.2e-5, sigma<=1.6e-4 -- well inside these bounds.
    np.testing.assert_allclose(mu, np.asarray(ref["mu"], float),
                               rtol=1e-4, atol=1e-4, err_msg="mu coefficients")
    np.testing.assert_allclose(sigma, np.asarray(ref["sigma"], float),
                               rtol=1e-3, atol=1e-3, err_msg="sigma coefficients")


@pytest.mark.parametrize(("n1", "n2"), CASES)
def test_mixed_deviance_matches_live_r(golden, data, n1, n2):
    ref = golden[f"mixed_{n1}_{n2}"]
    model = _fit(n1, n2, data)
    np.testing.assert_allclose(float(model.g_dev), float(ref["G.dev"]),
                               rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(("n1", "n2"), CASES)
def test_mixed_fitted_values_match_live_r(golden, data, n1, n2):
    ref = golden[f"mixed_{n1}_{n2}"]
    model = _fit(n1, n2, data)
    for param in ("mu", "sigma"):
        np.testing.assert_allclose(
            np.ravel(model.fitted_values[param]),
            np.asarray(ref[f"{param}.fv"], float),
            rtol=1e-3, atol=1e-3, err_msg=f"{param} fitted values",
        )


def test_mixed_stage_sequence_converges_to_rs_solution(golden, data):
    """R's mixed(2,3) lands on the same optimum as a full RS() fit."""
    mixed = _fit(2, 3, data)
    np.testing.assert_allclose(
        np.ravel(mixed.coefficients["mu"]), np.asarray(golden["RS"]["mu"], float),
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(float(mixed.g_dev), float(golden["RS"]["G.dev"]),
                               rtol=1e-5, atol=1e-4)


def test_public_mixed_uses_r_compatible_stages(data):
    """method='MIXED' must dispatch to mixed_fit, not the old inline damping."""
    model = gamlss("y ~ x", family="NO", data=data, sigma_formula="~ x",
                   method="MIXED", mixed_n1=2, mixed_n2=3)
    assert model.additional_slots["method"] == "MIXED"
    assert model.additional_slots["mixed_rs_n1"] == 2
    assert model.additional_slots["mixed_cg_n2"] == 3
    assert model.call["mixed_n1"] == 2 and model.call["mixed_n2"] == 3


def test_inline_step_no_longer_claims_a_damped_mixed_semantics():
    """P0-3 removed the dead 75%-damped MIXED branch (防回潮).

    MIXED is handled by mixed_fit; the inline IRLS helper must treat it like a
    full step (RS), never the old 0.75 damping that the shipped MIXED never used.
    """
    previous = np.array([0.0, 0.0])
    proposed = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        _apply_method_step(previous, proposed, "MIXED"), proposed
    )
