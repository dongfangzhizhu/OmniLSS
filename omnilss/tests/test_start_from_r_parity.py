# SPDX-License-Identifier: GPL-3.0-or-later
"""R's `start.from` warm start (D2.132).

R reference: `omnilss/tests/fixtures/r_golden/start_from_2026-11-01.json`
(written by `tmp/d2132_parity.R` against live R 4.6.0 / gamlss 5.5.0).

R semantics (`gamlss-5.R:965-991`): a fitted gamlss object replaces the
`*.start` arguments — every parameter the previous model estimated seeds the
next fit with its *fitted values* (`mu.start <- start.from$mu.fv`), and any
explicitly passed start is discarded.  The fixture proves that identity on the
R side: `start.from` and the equivalent explicit starts land on the same
deviance to the last digit.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import GA, gamlss

GOLDEN = (
    Path(__file__).parent / "fixtures" / "r_golden" / "start_from_2026-11-01.json"
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(golden) -> dict:
    return {
        "y": np.asarray(golden["data"]["y"], dtype=np.float64),
        "x": np.asarray(golden["data"]["x"], dtype=np.float64),
    }


def _values(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def test_fixture_proves_r_identity(golden):
    """The R side must show start.from == explicit fitted-value starts."""
    assert golden["start_from"]["deviance"] == pytest.approx(
        golden["explicit_starts"]["deviance"], rel=0, abs=0
    )


def test_start_from_matches_live_r(golden, data):
    first = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=2)
    warm = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=2, start_from=first)

    assert float(warm.g_dev) == pytest.approx(
        float(golden["start_from"]["deviance"]), rel=1e-6
    )
    # Deviance agrees to 1e-7 while the coefficients sit ~2e-5 apart on a very
    # flat surface: R stops at its 2-cycle point (its own n.cyc=20 run returns
    # the same number), we keep going, so the coefficient tolerance is loose.
    np.testing.assert_allclose(
        _values(warm.coefficients["mu"]),
        _values(golden["start_from"]["mu"]),
        rtol=1e-4,
    )


def test_start_from_is_the_same_as_explicit_fitted_value_starts(golden, data):
    """The R identity (fv-as-start) must hold on our side too, bit for bit."""
    first = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=2)
    warm = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=2, start_from=first)
    explicit = gamlss(
        "y ~ x", family=GA(), data=data, method="RS", n_cyc=2,
        mu_start=first.fitted_values["mu"],
        sigma_start=first.fitted_values["sigma"],
    )
    assert float(warm.g_dev) == float(explicit.g_dev)
    np.testing.assert_array_equal(
        np.asarray(warm.coefficients["mu"]), np.asarray(explicit.coefficients["mu"])
    )


def test_start_from_overrides_explicit_starts(golden, data):
    """R sets mu.start <- NULL when start.from is given; so do we."""
    first = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=2)
    with_override = gamlss(
        "y ~ x", family=GA(), data=data, method="RS", n_cyc=2,
        start_from=first, mu_start=99.0, sigma_start=9.0,
    )
    plain = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=2, start_from=first)
    assert float(with_override.g_dev) == float(plain.g_dev)


def test_warm_start_costs_fewer_cycles(golden, data):
    """The point of the feature: a refit of a converged model is cheap."""
    converged = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=20)
    warm = gamlss("y ~ x", family=GA(), data=data, method="RS", n_cyc=1, start_from=converged)
    assert int(warm.iter) == 1
    assert float(warm.g_dev) == pytest.approx(float(converged.g_dev), rel=1e-10)


def test_non_model_object_is_rejected(golden, data):
    """R: stop("The object in start.from is not a gamlss object")."""
    with pytest.raises(ValueError, match="must be a fitted model"):
        gamlss("y ~ x", family=GA(), data=data, method="RS", start_from={"mu": 1.0})


def test_model_without_matching_parameters_is_rejected(golden, data):
    class EmptyModel:
        fitted_values = {}
        family = "NO"

    with pytest.raises(ValueError, match="no fitted values for family"):
        gamlss("y ~ x", family=GA(), data=data, method="RS", start_from=EmptyModel())
