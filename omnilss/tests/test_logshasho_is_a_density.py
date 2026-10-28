"""LOGSHASHo: R ships a density and a CDF that disagree (D2.81).

In the vendored gamlss.dist/R/LOGSHASHo.R the two halves do different things:

    dLOGSHASHo(x, ...)   z <- (x - mu)/sigma          # SHASH density on x
                         ft[x <= 0] <- 0
    pLOGSHASHo(q, ...)   cdf <- pSHASHo(log(q), ...)  # SHASH CDF on log q

So the density is a SHASH truncated to (0, inf) without renormalising, while
the CDF is that of exp(SHASH).  They cannot both be right, and the density
half is not a density: with mu = 0, sigma = 1, nu = 0, tau = 1 it integrates
to 0.500; with mu = -1 to 0.0023 (measured, D2.81).

p, q and r all agree with the log-transform reading, so that is what the
family implements here: f(y) = f_SHASHo(log y) / y.  These tests pin the
consistency R does not have -- the density is the derivative of the CDF, the
CDF is R pSHASHo at log y, and the family integrates to one.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402

_PARAMS = {
    "neutral": dict(mu=0.0, sigma=1.0, nu=0.5, tau=1.2),
    "skewed": dict(mu=1.0, sigma=0.5, nu=1.5, tau=0.8),
}


def _args(y, params):
    return [np.full(np.size(y), float(value)) for value in params.values()]


@pytest.mark.parametrize("case", sorted(_PARAMS))
def test_the_density_integrates_to_the_cdf_increment(case):
    """R dLOGSHASHo integrates to ~0.5 at mu = 0 -- it is a truncated SHASH.

    The check is the fundamental theorem of calculus on a finite window, not
    "the integral reaches 1": the log-SHASH tail is heavy enough that a fixed
    upper limit (1e4, say) legitimately misses 4.6% at nu = 1.5, tau = 0.8.
    Total mass is a separate test below.
    """
    family = resolve("LOGSHASHO")
    params = _PARAMS[case]
    lo, hi = 1e-3, 1e6
    y = np.geomspace(lo, hi, 400000)
    density = np.asarray(family.d(y, *_args(y, params)), dtype=float)
    assert np.isfinite(density).all()
    integral = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(y)))
    window = float(np.asarray(family.p(hi, **params)) - np.asarray(family.p(lo, **params)))
    assert integral == pytest.approx(window, abs=2e-6)


@pytest.mark.parametrize("case", sorted(_PARAMS))
def test_the_total_mass_is_one(case):
    """Before D2.81 the density carried 0.500 / 0.0023 of the mass."""
    family = resolve("LOGSHASHO")
    params = _PARAMS[case]
    # 1e30, not 1e10: with tau < 1 the sinh-arcsinh tails are heavier than
    # normal, and F(1e10) is still 0.999961 at nu = 1.5, tau = 0.8.  That is a
    # property of the family, not a defect -- the mass does converge.
    total = float(np.asarray(family.p(1e30, **params)))
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("case", sorted(_PARAMS))
def test_the_density_is_the_derivative_of_the_cdf(case):
    """The check R itself would fail: dLOGSHASHo is not d/dy pLOGSHASHo."""
    family = resolve("LOGSHASHO")
    params = _PARAMS[case]
    y = np.linspace(0.05, 40.0, 40)
    args = _args(y, params)
    density = np.asarray(family.d(y, *args), dtype=float)
    step = 1e-6
    upper = np.asarray(family.p(y + step, *args), dtype=float)
    lower = np.asarray(family.p(y - step, *args), dtype=float)
    finite_difference = (upper - lower) / (2.0 * step)
    relative = np.abs(finite_difference - density) / np.maximum(density, 1e-300)
    assert relative.max() < 1e-5, relative.max()


def test_the_cdf_is_r_p_shasho_at_log_y():
    """This is exactly what R pLOGSHASHo computes; SHASHo is R-verified."""
    logshasho = resolve("LOGSHASHO")
    shasho = resolve("SHASHO")
    params = _PARAMS["neutral"]
    y = np.linspace(0.05, 40.0, 40)
    ours = np.asarray(logshasho.p(y, *_args(y, params)), dtype=float)
    reference = np.asarray(shasho.p(np.log(y), *_args(y, params)), dtype=float)
    np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-15)


def test_the_cdf_limits_and_the_quantile_inverse():
    family = resolve("LOGSHASHO")
    params = _PARAMS["neutral"]
    assert float(np.asarray(family.p(0.0, **params))) == 0.0
    assert float(np.asarray(family.p(1e9, **params))) == pytest.approx(1.0)
    probabilities = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles = np.asarray(family.q(probabilities, **params), dtype=float)
    np.testing.assert_allclose(
        np.asarray(family.p(quantiles, **params), dtype=float),
        probabilities,
        rtol=1e-12,
    )


def test_a_fit_produces_finite_residuals():
    from omnilss import gamlss

    rng = np.random.default_rng(9)
    x = rng.normal(size=60)
    y = np.exp(1.0 + 0.5 * x + rng.normal(size=60) * 0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss(
            "y ~ x",
            family="LOGSHASHo",
            data={"y": y, "x": x},
            sigma_formula="~1",
            nu_formula="~1",
            tau_formula="~1",
            method="RS",
            max_iter=20,
            allow_compat_alias=True,
        )
    residuals = np.asarray(model.residuals, dtype=float)
    assert residuals.shape == (60,)
    assert np.isfinite(residuals).all()