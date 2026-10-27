"""GENERALISEDPOISSON had a density that did not honour its own contract (D2.79).

The family documented "GP(mu, sigma) with E[Y] = mu" and implemented

    p(y) = lambda (lambda + sigma y)^(y-1) exp(-(lambda + sigma y)) / y!,
    lambda = mu / (1 + sigma),

which is a valid distribution but is *not* mean-parameterized: with mu = 2,
sigma = 0.5 the mean is 2.667, and with mu = 5, sigma = 0.2 it is 5.208.  R
GPO delivers E[Y] = mu exactly on the same grid, and the family had no CDF at
all (pIGA-style placeholder).

It now shares GPO kernels -- which are the parity-verified ones -- and the
scanner probes it as R GPO.  These tests pin the contract that was broken:
the PMF sums to one *and* its mean is mu, the CDF is real, and a fitted model
produces finite residuals.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402


@pytest.mark.parametrize("mu,sigma", [(2.0, 0.5), (5.0, 0.2), (1.0, 0.9)])
def test_the_pmf_is_normalized_and_its_mean_is_mu(mu, sigma):
    """The old density summed to 1 but had the wrong mean -- 2.667 for mu=2."""
    family = resolve("GENERALISEDPOISSON")
    y = np.arange(0.0, 400.0)
    pmf = np.asarray(
        family.d(y, np.full_like(y, mu), np.full_like(y, sigma)), dtype=float
    )
    assert np.isfinite(pmf).all()
    np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-10)
    np.testing.assert_allclose(float((y * pmf).sum()), mu, rtol=1e-4)


def test_it_shares_the_gpo_kernels():
    alias = resolve("GENERALISEDPOISSON")
    reference = resolve("GPO")
    assert alias.p.__name__ == reference.p.__name__
    assert alias.d.__name__ == reference.d.__name__
    assert alias.links == reference.links
    y = np.array([0.0, 1.0, 3.0, 8.0])
    mu = np.full(4, 2.5)
    sigma = np.full(4, 0.4)
    np.testing.assert_allclose(
        np.asarray(alias.d(y, mu, sigma), dtype=float),
        np.asarray(reference.d(y, mu, sigma), dtype=float),
        rtol=1e-15,
    )


def test_the_cdf_is_real_and_inverts():
    family = resolve("GENERALISEDPOISSON")
    assert "placeholder" not in family.p.__name__
    cdf = np.asarray(family.p(np.arange(0.0, 60.0), 2.5, 0.4), dtype=float)
    assert np.isfinite(cdf).all()
    assert np.all(np.diff(cdf) >= -1e-12)
    # The tail of a GP with sigma = 0.4 is fat: P(Y > 11) is still ~2%, so the
    # "reaches 1" check has to be made where the mass actually is.
    np.testing.assert_allclose(cdf[-1], 1.0, atol=1e-6)
    quantiles = np.asarray(family.q([0.1, 0.5, 0.9], 2.5, 0.4), dtype=float)
    assert np.all(np.diff(quantiles) >= 0)


def test_a_fit_produces_finite_residuals():
    from omnilss import gamlss

    rng = np.random.default_rng(6)
    x = rng.normal(size=60)
    y = rng.poisson(3.0, 60).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss(
            "y ~ x",
            family="GENERALISEDPOISSON",
            data={"y": y, "x": x},
            sigma_formula="~1",
            method="RS",
            max_iter=20,
            allow_compat_alias=True,
        )
    residuals = np.asarray(model.residuals, dtype=float)
    assert residuals.shape == (60,)
    assert np.isfinite(residuals).all()