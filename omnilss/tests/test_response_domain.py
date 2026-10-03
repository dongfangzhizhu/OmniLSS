# SPDX-License-Identifier: GPL-3.0-or-later
"""Family response-domain validation — R parity for ``family$y.valid``.

R source: ``gamlss/R/gamlss-5.R:960`` inside ``gamlss()``::

    ##checking the permissible y values
    if (!family$y.valid(y))
        stop( "response variable out of range")

The check runs *before* any IRLS step, so R never produces a model for
out-of-support data.  Before this module OmniLSS only warned about eta clipping
and returned a model whose deviance was meaningless.
"""

import numpy as np
import pytest

from omnilss import GA, NO, gamlss
from omnilss.distributions import resolve_family
from omnilss.response_domain import (
    ResponseDomainError,
    response_violation_count,
    validate_response,
)

from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge

# Live-R cases only; the pure-Python predicate tests above always run.
_r_live = pytest.mark.skipif(not R_AVAILABLE, reason="R bridge not available")


# ---------------------------------------------------------------------------
# Predicate-level behaviour (no fitting involved)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("family", "value", "valid"),
    [
        # GA: R `all(y > 0)`
        ("GA", 1.0, True),
        ("GA", 0.0, False),
        ("GA", -0.5, False),
        # ZIP: R `all(y >= 0)` — zero allowed
        ("ZIP", 0.0, True),
        ("ZIP", -1.0, False),
        # BEINF: R `all(y >= 0 & y <= 1)` — both endpoints included
        ("BEINF", 0.0, True),
        ("BEINF", 1.0, True),
        ("BEINF", 1.5, False),
        # BEZI: R `all(y >= 0 & y < 1)` — 1 excluded
        ("BEZI", 0.0, True),
        ("BEZI", 1.0, False),
        # BEOI: R `all(y > 0 & y <= 1)` — 0 excluded
        ("BEOI", 0.0, False),
        ("BEOI", 1.0, True),
        # PARETO: R `all(y > 1)`
        ("PARETO", 1.0, False),
        ("PARETO", 1.5, True),
        # MN4 / MN5: shifted integer supports
        ("MN4", 0.9, False),
        ("MN4", 1.0, True),
        ("MN4", 4.0, True),
        ("MN4", 4.1, False),
        ("MN5", 5.5, False),
        # ZIPF: R `all(y >= 1)`
        ("ZIPF", 0.5, False),
        ("ZIPF", 1.0, True),
    ],
)
def test_predicate_matches_r_y_valid(family, value, valid):
    """Each entry mirrors the R ``y.valid`` body for that family."""
    assert response_violation_count(family, np.array([value])) == (0 if valid else 1)


def test_unbounded_family_accepts_negative_response():
    """NO declares ``y.valid = function(y) TRUE`` — negatives are legitimate."""
    assert response_violation_count("NO", np.array([-10.0, 0.0, 5.0])) == 0


def test_families_without_r_y_valid_stay_permissive():
    """R declares no ``y.valid`` for these, so Python must not invent one.

    BETA, PO, ZINB, ALPHA and DAGUM carry no ``y.valid`` in ``gamlss.dist``.
    Adding a restriction R does not enforce would reject data R fits happily.
    """
    for family in ("BETA", "PO", "ZINB", "ALPHA", "DAGUM"):
        assert response_violation_count(family, np.array([-1.0, 0.0, 2.0])) == 0


def test_nan_and_inf_rejected_even_for_unbounded_family():
    """No family can fit a non-finite response, including the permissive ones."""
    for family in ("NO", "BETA", "GA"):
        assert response_violation_count(family, np.array([1.0, np.nan])) == 1
        assert response_violation_count(family, np.array([1.0, np.inf])) == 1


def test_error_message_reproduces_r_text():
    with pytest.raises(ResponseDomainError) as excinfo:
        validate_response("GA", np.array([1.0, -2.0, 3.0]))
    message = str(excinfo.value)
    assert "response variable out of range" in message
    assert "GA" in message
    assert excinfo.value.offending == 1
    assert excinfo.value.sample == (-2.0,)


def test_violation_count_reports_all_offenders():
    assert response_violation_count("GA", np.array([0.0, -1.0, -2.0, 1.0])) == 3


def test_empty_response_is_vacuously_valid():
    assert validate_response("GA", np.array([])) == 0


# ---------------------------------------------------------------------------
# Integration with the fitting entry point
# ---------------------------------------------------------------------------

def test_gamlss_rejects_out_of_support_response():
    """GA cannot fit a negative response — R stops, so must we."""
    data = {"y": [1.0, 2.0, 3.0, -1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(ResponseDomainError, match="response variable out of range"):
        gamlss("y ~ x", family=GA(), data=data, method="RS")


@pytest.mark.parametrize("method", ["RS", "CG", "MIXED"])
def test_every_method_route_rejects_out_of_support_response(method):
    """The guard sits in ``gamlss()``, so no method can bypass it.

    Checking only the NumPy RS path would leave CG / MIXED / RS_JAX fitting
    data R refuses — the same silent-divergence class D2.2 closed.
    """
    data = {"y": [1.0, 2.0, 3.0, -1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(ResponseDomainError):
        gamlss("y ~ x", family=GA(), data=data, method=method)


def test_in_support_response_still_fits():
    """The guard must not reject legitimate data."""
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    model = gamlss("y ~ x", family=GA(), data=data, method="RS", max_iter=20)
    assert np.isfinite(model.deviance)


def test_negative_response_fine_for_no_family():
    """NO is unbounded on the response; a negative response is valid data."""
    data = {"y": [-3.0, -1.0, 2.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    model = gamlss("y ~ x", family=NO(), data=data, method="RS", max_iter=20)
    assert np.isfinite(model.deviance)


def test_zero_is_valid_for_zip_but_not_for_ga():
    """The two positive-family rules differ exactly at zero."""
    zero_data = {"y": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]}
    assert gamlss("y ~ x", family=resolve_family("ZIP"), data=zero_data,
                  method="RS", max_iter=5) is not None
    with pytest.raises(ResponseDomainError):
        gamlss("y ~ x", family=GA(), data=zero_data, method="RS", max_iter=5)


# ---------------------------------------------------------------------------
# Live R bridge — proves the parity claim rather than asserting it
# ---------------------------------------------------------------------------

@_r_live
def test_r_gamlss_also_rejects_ga_with_negative_response():
    """R must stop on the same data, with the same message.

    The bridge surfaces R's ``stop()`` as a ``RuntimeError`` rather than a
    ``{"success": False}`` dict, so the assertion is on the raised message.
    """
    data = {"y": [1.0, 2.0, 3.0, -1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(RuntimeError, match="response variable out of range"):
        RBridge().call_r_gamlss(
            data=data, formula="y ~ x", family="GA", sigma_formula="~1", method="RS"
        )


@_r_live
def test_r_gamlss_accepts_the_same_in_support_data_python_accepts():
    """Both sides fit the positive-response GA data — no over-rejection."""
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    result = RBridge().call_r_gamlss(
        data=data, formula="y ~ x", family="GA", sigma_formula="~1", method="RS"
    )
    assert result["success"], result.get("error", "R failed")

    py = gamlss("y ~ x", family=GA(), data=data, method="RS", max_iter=20)
    np.testing.assert_allclose(
        float(py.deviance), float(result["deviance"]), rtol=1e-3, atol=1e-3
    )
