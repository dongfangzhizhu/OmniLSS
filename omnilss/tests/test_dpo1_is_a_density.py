"""DPO1 was not a density: it summed to +inf and spiked at zero (D2.80).

The old body added 0.5*log(2*pi*sigma*y) as an "approximate normalizing
constant" and then subtracted exactly the same term in the same expression,
leaving

    log f(y) = (y log mu - mu - log y!) / sigma,

a scaled Poisson -- normalised only at sigma = 1, plus a +inf spike at y = 0
from the subtracted log(0).  Measured before the fix, with mu = 3: the PMF
summed to +inf for sigma = 1, 0.5 and 2, with means 3.000 / 0.456 / 10.171.
DPO, the R-verified double Poisson, sums to 1.000 with mean 3.013 / 2.999 on
the same grid.  DPO1 also had no CDF and was typed Continuous.

These tests pin the three properties that were broken: it is a probability
mass function (sums to one, finite everywhere), it is mean-parameterized, and
it is Discrete with a real CDF.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402


@pytest.mark.parametrize("mu,sigma", [(3.0, 1.0), (3.0, 0.5), (3.0, 2.0)])
def test_the_pmf_is_finite_and_sums_to_one(mu, sigma):
    """Before D2.80 the sum was +inf at every sigma."""
    family = resolve("DPO1")
    y = np.arange(0.0, 400.0)
    pmf = np.asarray(
        family.d(y, np.full_like(y, mu), np.full_like(y, sigma)), dtype=float
    )
    assert np.isfinite(pmf).all()
    np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-9)


@pytest.mark.parametrize("mu,sigma", [(3.0, 1.0), (3.0, 0.5), (3.0, 2.0)])
def test_the_mean_is_mu(mu, sigma):
    family = resolve("DPO1")
    y = np.arange(0.0, 400.0)
    pmf = np.asarray(
        family.d(y, np.full_like(y, mu), np.full_like(y, sigma)), dtype=float
    )
    # rel=1e-2, not 1e-6: the double Poisson normalizing constant in gamlss is
    # an *approximation*, so even R's dDPO has a mean 0.4% off mu at
    # sigma = 0.5 (3.0129 for mu = 3).  DPO1 inherits exactly that number; the
    # point of the test is the order of magnitude the broken body missed by
    # (mean 0.456 at sigma = 0.5).
    assert float((y * pmf).sum()) == pytest.approx(mu, rel=1e-2)


def test_it_is_discrete_with_a_real_cdf():
    family = resolve("DPO1")
    assert family.type == "Discrete"
    assert "placeholder" not in family.p.__name__
    cdf = np.asarray(family.p(np.arange(0.0, 40.0), 3.0, 0.5), dtype=float)
    assert np.isfinite(cdf).all()
    assert np.all(np.diff(cdf) >= -1e-12)
    quantiles = np.asarray(family.q([0.1, 0.5, 0.9], 3.0, 0.5), dtype=float)
    assert np.all(np.diff(quantiles) >= 0)


def test_it_shares_the_dpo_kernels():
    alias = resolve("DPO1")
    reference = resolve("DPO")
    assert alias.d.__name__ == reference.d.__name__
    assert alias.p.__name__ == reference.p.__name__
    y = np.array([0.0, 1.0, 4.0, 9.0])
    mu = np.full(4, 3.0)
    sigma = np.full(4, 0.7)
    np.testing.assert_allclose(
        np.asarray(alias.d(y, mu, sigma), dtype=float),
        np.asarray(reference.d(y, mu, sigma), dtype=float),
        rtol=1e-15,
    )


def test_a_fit_produces_finite_residuals():
    from omnilss import gamlss

    rng = np.random.default_rng(8)
    x = rng.normal(size=60)
    y = rng.poisson(3.0, 60).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss(
            "y ~ x",
            family="DPO1",
            data={"y": y, "x": x},
            sigma_formula="~1",
            method="RS",
            max_iter=20,
            allow_compat_alias=True,
        )
    residuals = np.asarray(model.residuals, dtype=float)
    assert residuals.shape == (60,)
    assert np.isfinite(residuals).all()