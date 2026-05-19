"""Unit tests for Week 1 cross-derivative infrastructure."""

from __future__ import annotations

import numpy as np

from omnilss.derivatives.cross_derivatives import cross_hessian_from_family
from omnilss.distributions import GA, NO, WEI


def _run_family_case(family, y, param_values):
    result = cross_hessian_from_family(y=y, param_values=param_values, family=family)
    params = tuple(family.estimable_parameters)
    n = len(y)
    for pi in params:
        for pj in params:
            arr = np.asarray(result[(pi, pj)])
            assert arr.shape == (n,)
            assert np.all(np.isfinite(arr))
            mirror = np.asarray(result[(pj, pi)])
            np.testing.assert_allclose(arr, mirror, rtol=1e-6, atol=1e-6)


def test_cross_hessian_no_shapes_and_symmetry():
    rng = np.random.default_rng(0)
    n = 24
    y = rng.normal(loc=1.0, scale=0.5, size=n)
    _run_family_case(
        NO(),
        y=y,
        param_values={"mu": np.full(n, 1.0), "sigma": np.full(n, 0.7)},
    )


def test_cross_hessian_ga_shapes_and_symmetry():
    rng = np.random.default_rng(1)
    n = 20
    y = rng.gamma(shape=2.0, scale=1.0, size=n)
    _run_family_case(
        GA(),
        y=y,
        param_values={"mu": np.full(n, 2.0), "sigma": np.full(n, 0.5)},
    )


def test_cross_hessian_wei_shapes_and_symmetry():
    rng = np.random.default_rng(2)
    n = 18
    y = rng.weibull(a=1.8, size=n) + 0.05
    _run_family_case(
        WEI(),
        y=y,
        param_values={"mu": np.full(n, 1.2), "sigma": np.full(n, 1.1)},
    )
