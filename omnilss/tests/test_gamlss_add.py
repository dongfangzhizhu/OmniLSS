# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add additive terms: ga() / ba() and gamlss_ga / gamlss_ba (D2.248).

R's ga()/ba() build an mgcv basis and gamlss.ga()/gamlss.ba() refit it for the
working response.  The port keeps the interface and delegates the basis to the
existing smooth machinery, so the invariant is:

    gamlss(y ~ ga(~ pb(x)))  ==  gamlss(y ~ pb(x))

mgcv's REML penalty is a separate estimator and is deliberately not claimed as
parity; the callback itself is pinned as a weighted least-squares refit.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import ba, ga, gamlss, gamlss_ba, gamlss_ga
from omnilss.formula_parser import parse_formula


def _data(seed: int = 3, n: int = 120) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.0 + 0.5 * np.sin(x) + 0.3 * z + rng.normal(scale=0.3, size=n)
    return {"y": y, "x": x, "z": z}


@pytest.mark.parametrize("constructor", (ga, ba))
def test_additive_terms_require_an_inner_formula(constructor):
    with pytest.raises(ValueError, match="starting with ~"):
        constructor("y ~ x")
    with pytest.raises(ValueError, match="starting with ~"):
        constructor(None)


@pytest.mark.parametrize("constructor", (ga, ba))
def test_additive_terms_record_the_inner_terms(constructor):
    term = constructor("~ pb(x) + z")
    assert term.formula == "~ pb(x) + z"
    assert [smoother.variable for smoother in term.parsed.smooth_terms] == ["x"]
    assert [linear.variable for linear in term.parsed.linear_terms] == ["z"]
    assert term.control["method"] == "REML"


@pytest.mark.parametrize("name", ("ga", "ba"))
def test_formula_expansion_matches_the_explicit_smooth(name):
    expanded = parse_formula(f"y ~ {name}(~ pb(x) + z)")
    explicit = parse_formula("y ~ pb(x) + z")
    assert [(s.smoother, s.variable) for s in expanded.smooth_terms] == [
        (s.smoother, s.variable) for s in explicit.smooth_terms
    ]
    assert [t.variable for t in expanded.linear_terms] == [
        t.variable for t in explicit.linear_terms
    ]
    assert expanded.has_intercept == explicit.has_intercept


@pytest.mark.parametrize("name", ("ga", "ba"))
def test_gamlss_additive_delegates_to_the_explicit_smooth(name):
    data = _data()
    explicit = gamlss("y ~ pb(x)", family="NO", data=data, max_iter=60)
    additive = gamlss(f"y ~ {name}(~ pb(x))", family="NO", data=data, max_iter=60)
    assert float(additive.g_dev) == pytest.approx(float(explicit.g_dev), rel=1e-12)
    np.testing.assert_allclose(
        np.ravel(np.asarray(additive.coefficients["mu"], dtype=float)),
        np.ravel(np.asarray(explicit.coefficients["mu"], dtype=float)),
        rtol=0.0,
        atol=0.0,
    )


def test_ba_with_two_smooths_matches_the_explicit_sum():
    data = _data()
    explicit = gamlss("y ~ pb(x) + pb(z)", family="NO", data=data, max_iter=60)
    additive = gamlss("y ~ ba(~ pb(x) + pb(z))", family="NO", data=data, max_iter=60)
    assert float(additive.g_dev) == pytest.approx(float(explicit.g_dev), rel=1e-12)


@pytest.mark.parametrize("callback, kind", ((gamlss_ga, "ga"), (gamlss_ba, "ba")))
def test_additive_callback_is_a_weighted_least_squares_refit(callback, kind):
    data = _data()
    working = data["y"]
    result = callback(data, working, None, formula="~ pb(x)")
    assert result.kind == kind
    assert result.fitted.shape == working.shape
    np.testing.assert_allclose(result.fitted, result.design @ result.coefficients, rtol=0, atol=0)
    # Normal equations: residuals are orthogonal to every design column.
    residual = working - result.fitted
    assert float(np.max(np.abs(result.design.T @ residual))) < 1e-8

    weights = np.linspace(0.5, 2.0, working.size)
    weighted = callback(data, working, weights, formula="~ pb(x)")
    np.testing.assert_allclose(
        weighted.fitted, weighted.design @ weighted.coefficients, rtol=0, atol=0
    )
    assert float(np.max(np.abs(weighted.design.T @ (weights * (working - weighted.fitted))))) < 1e-8


def test_additive_callback_validates_its_inputs():
    data = _data()
    with pytest.raises(ValueError, match="starting with ~"):
        gamlss_ga(data, data["y"], None, formula="pb(x)")
    with pytest.raises(ValueError, match="same length"):
        gamlss_ga(data, data["y"], np.ones(3), formula="~ pb(x)")
