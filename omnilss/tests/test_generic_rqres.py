"""Quantile residuals for every family with a CDF (D2.74).

``_build_rqres_callable`` used to hand back a kernel for exactly five
families -- PO, BI, GEOM, NBI, ZIP -- and None for the other 141.  A model
without a kernel falls back to ``_compute_residuals``, i.e. raw ``y - mu``
(or a Pearson-style ``(y-mu)/sqrt(mu)``), and those are not comparable across
families.  That matters because the quantile residual is the thing GAMLSS
diagnostics are built on (worm plots, centile curves): a residual that is not
approximately N(0, 1) under a correct model is not usable for either.

The generic path evaluates the family's own CDF, so it is exactly as good as
the CDF is -- which is why this file also pins the three properties that can
be checked without R:

* it reproduces the five hand-written kernels (they stay as the reference),
* continuous families give standard-normal residuals under the correct model,
* every fitted parameter reaches the CDF (the ZIBI regression below: with
  only mu/sigma the CDF fell back to bd = 1 defaults and every residual came
  back as qnorm(1 - eps) = 8.126).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss._fitting_residuals import (  # noqa: E402
    _HAND_CODED_RQRES,
    _build_rqres_callable,
    _discrete_midpoint_rqres,
    _generic_rqres,
)
from omnilss.distribution_registry import list_families, resolve  # noqa: E402


def _cdf_placeholder_families() -> set[str]:
    """Families whose CDF is the build_ad_family stub (no residuals)."""
    return {
        name for name in list_families()
        if "placeholder" in getattr(resolve(name).p, "__name__", "p")
    }


class TestCoverage:
    def test_every_family_with_a_real_cdf_gets_a_kernel(self):
        """The point of the change: coverage, not one more special case."""
        placeholder = _cdf_placeholder_families()
        covered = [
            name for name in list_families()
            if name not in placeholder
            and _build_rqres_callable(resolve(name)) is not None
        ]
        assert len(covered) >= 140, len(covered)
        assert set(covered) | placeholder == set(list_families())

    def test_the_four_placeholder_families_still_return_none(self):
        """No CDF means no quantile residual -- not an invented one."""
        placeholder = _cdf_placeholder_families()
        # IGA left this set in D2.77 (it shares IG's kernels).  The three that
        # remain have no closed-form CDF in gamlss.dist either, so there is no
        # reference to implement against.
        assert placeholder == {
            "DPO1",
            "GENERALISEDPOISSON",
            "LOGSHASHO",
        }, sorted(placeholder)
        for name in placeholder:
            assert _build_rqres_callable(resolve(name)) is None


class TestReproducesTheHandWrittenKernels:
    """The five hand-written kernels are the reference, not a legacy path."""

    _CASES = {
        "PO": (np.array([0.0, 1.0, 2.0, 5.0]), {"mu": np.full(4, 2.5)}),
        "BI": (np.array([0.0, 1.0, 1.0, 0.0]), {"mu": np.full(4, 0.4)}),
        "GEOM": (np.array([0.0, 1.0, 3.0, 7.0]), {"mu": np.full(4, 1.7)}),
        "NBI": (
            np.array([0.0, 1.0, 3.0, 7.0]),
            {"mu": np.full(4, 2.0), "sigma": np.full(4, 0.4)},
        ),
        "ZIP": (
            np.array([0.0, 0.0, 2.0, 5.0]),
            {"mu": np.full(4, 2.0), "sigma": np.full(4, 0.3)},
        ),
    }

    @pytest.mark.parametrize("family", sorted(_CASES))
    def test_the_generic_path_matches_the_kernel(self, family):
        assert family in _HAND_CODED_RQRES
        y, params = self._CASES[family]
        fam = resolve(family)
        hand = _discrete_midpoint_rqres(
            fam, y=y, mu=params["mu"], sigma=params.get("sigma")
        )
        generic = _generic_rqres(fam, {"y": y, **params})
        np.testing.assert_allclose(generic, np.asarray(hand), rtol=0, atol=1e-12)


class TestStandardNormalResiduals:
    """Under the correct model a continuous family's residuals are N(0, 1)."""

    @pytest.mark.parametrize(
        "family,params",
        [
            ("NO", {"mu": np.zeros(4000), "sigma": np.ones(4000)}),
            ("GA", {"mu": np.full(4000, 2.0), "sigma": np.full(4000, 0.5)}),
        ],
    )
    def test_kolmogorov_smirnov_against_the_standard_normal(self, family, params):
        from scipy.stats import kstest

        rng = np.random.default_rng(7)
        fam = resolve(family)
        u = rng.uniform(size=4000)
        args = [params[name] for name in fam.parameters]
        y = np.asarray(fam.q(u, *args), dtype=float)
        residuals = _generic_rqres(fam, {"y": y, **params})
        result = kstest(residuals, "norm")
        assert result.pvalue > 0.01, result


class TestFittedModels:
    def _fit(self, family, data):
        from omnilss import gamlss

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gamlss(
                "y ~ x",
                family=family,
                data=data,
                sigma_formula="~1",
                method="RS",
                max_iter=20,
                allow_compat_alias=True,
            )

    def test_continuous_and_discrete_models_get_finite_residuals(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=60)
        cases = {
            "NO": {"y": 1.0 + 0.5 * x + rng.normal(size=60), "x": x},
            "GA": {"y": rng.gamma(2.0, 0.5, 60) + 0.1, "x": x},
            "PO": {"y": rng.poisson(2.0, 60).astype(float), "x": x},
        }
        for family, data in cases.items():
            model = self._fit(family, data)
            residuals = np.asarray(model.residuals, dtype=float)
            assert residuals.shape == (60,), (family, residuals.shape)
            assert np.isfinite(residuals).all(), family
            assert abs(float(np.mean(residuals))) < 0.5, (family, residuals.mean())

    def test_every_fitted_parameter_reaches_the_cdf(self):
        """The ZIBI regression: bd = 10 must reach the CDF, not bd = 1.

        With only mu/sigma passed, dZIBI-style CDFs default to bd = 1, so every
        y > 1 saturates at p = 1 and all residuals come back as the same
        qnorm(1 - eps) = 8.126.
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=60)
        y = rng.binomial(10, 0.4, 60).astype(float)
        model = self._fit(
            "ZIBI", {"y": y, "x": x, "bd": np.full(60, 10.0)}
        )
        residuals = np.asarray(model.residuals, dtype=float)
        assert np.isfinite(residuals).all()
        assert len(np.unique(np.round(residuals, 6))) > 30, residuals
        assert not np.allclose(residuals, 8.12589066)