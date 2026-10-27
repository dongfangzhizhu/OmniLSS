"""IGA is IG: shared kernels, real CDF, measured gates (D2.77).

The registry carries IGA as an "inverse Gaussian alternative" and the
log-density it shipped with is R dIG term for term once lambda = 1/sigma^2 --
the two kernels agree to 4.4e-16.  What it did *not* have was d/p/q/r: the
family was built from the auto-density only, so pIGA/qIGA answered NaN, no
quantile residual existed, and gamlss.dist exports no dIGA/pIGA/qIGA, so every
parity probe came back "no d/p/q counterpart in R".

It now shares IG builder (kernels, expected-information hessians, links) and
the scanner maps IGA to R IG when it builds the R jobs.  The tests here pin
the three things that makes true: the density is IG density, the CDF and
quantiles are R pIG/qIG, and a fitted IGA model behaves like a fitted IG.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402

_Y = np.array([0.3, 1.0, 2.5, 5.0])
_MU = np.array([1.0, 1.5, 2.0, 2.5])
_SIGMA = np.array([0.5, 0.75, 1.0, 1.25])

#: live R 4.6.0 + gamlss.dist 6.1.1, dIG/pIG at (y, mu, sigma) above.
_R_LOG_DENSITY = [
    -1.6864988128224891, -0.73002189285165719,
    -2.3058746310159055, -3.6202389531700327,
]
_R_CDF = [
    0.0083718337617715993, 0.46891379914085318,
    0.77311487994222139, 0.87596224655332122,
]


class TestTheSharedKernels:
    def test_the_density_is_ig_density(self):
        iga = resolve("IGA")
        ig = resolve("IG")
        ours = np.asarray(iga.d(_Y, _MU, _SIGMA, log=True), dtype=float)
        reference = np.asarray(ig.d(_Y, _MU, _SIGMA, log=True), dtype=float)
        np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-15)

    def test_the_cdf_and_quantiles_are_not_placeholders_any_more(self):
        iga = resolve("IGA")
        assert "placeholder" not in getattr(iga.p, "__name__", ""), iga.p.__name__
        assert "placeholder" not in getattr(iga.q, "__name__", ""), iga.q.__name__
        cdf = np.asarray(iga.p(_Y, _MU, _SIGMA), dtype=float)
        assert np.isfinite(cdf).all()
        quantiles = np.asarray(iga.q([0.1, 0.5, 0.9], 1.5, 0.75), dtype=float)
        assert np.isfinite(quantiles).all()
        assert np.all(np.diff(quantiles) > 0)

    def test_the_hessians_are_the_expected_information_ones(self):
        """Plain AD hessians made an IGA fit diverge to 8.07e+21."""
        iga = resolve("IGA")
        ig = resolve("IG")
        assert iga.hessian_functions["mu"].__name__ == ig.hessian_functions["mu"].__name__
        mu = np.array([1.5])
        sigma = np.array([0.75])
        ours = np.asarray(
            iga.hessian_functions["mu"](y=np.array([1.0]), mu=mu, sigma=sigma),
            dtype=float,
        )
        expected = -1.0 / (sigma ** 2 * mu ** 3)
        np.testing.assert_allclose(ours, expected, rtol=1e-15)


class TestAgainstLiveR:
    def test_log_density_matches_r(self):
        iga = resolve("IGA")
        ours = np.asarray(iga.d(_Y, _MU, _SIGMA, log=True), dtype=float)
        np.testing.assert_allclose(ours, _R_LOG_DENSITY, rtol=1e-13)

    def test_cdf_matches_r(self):
        iga = resolve("IGA")
        ours = np.asarray(iga.p(_Y, _MU, _SIGMA), dtype=float)
        np.testing.assert_allclose(ours, _R_CDF, rtol=1e-13)


class TestFittedModel:
    def test_a_fit_produces_finite_residuals(self):
        from omnilss import gamlss

        rng = np.random.default_rng(4)
        x = rng.normal(size=60)
        y = rng.wald(1.5, 3.0, 60) + 0.05
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss(
                "y ~ x",
                family="IGA",
                data={"y": y, "x": x},
                sigma_formula="~1",
                method="RS",
                max_iter=20,
                allow_compat_alias=True,
            )
        residuals = np.asarray(model.residuals, dtype=float)
        assert residuals.shape == (60,)
        assert np.isfinite(residuals).all()
        assert np.isfinite(float(model.deviance))