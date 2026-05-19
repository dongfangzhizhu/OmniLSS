"""Unit tests for Week 1 cross-derivative infrastructure."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from omnilss.derivatives.cross_derivatives import (
    cross_hessian,
    cross_hessian_from_family,
)
from omnilss.distributions import GA, NO, WEI


jax.config.update("jax_enable_x64", True)


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


def test_cross_hessian_generic_matches_analytic_quadratic():
    n = 7
    a = jnp.linspace(-1.0, 1.0, n)
    b = jnp.linspace(0.5, 2.5, n)

    def ll_fn(params):
        x = params["x"]
        y = params["y"]
        # elementwise ll_i = x_i^2 + 3*x_i*y_i + 2*y_i^2
        return x * x + 3.0 * x * y + 2.0 * y * y

    out = cross_hessian(ll_fn, {"x": a, "y": b})
    np.testing.assert_allclose(np.asarray(out[("x", "x")]), 2.0 * np.eye(n), atol=1e-10)
    np.testing.assert_allclose(np.asarray(out[("y", "y")]), 4.0 * np.eye(n), atol=1e-10)
    np.testing.assert_allclose(np.asarray(out[("x", "y")]), 3.0 * np.eye(n), atol=1e-10)
    np.testing.assert_allclose(np.asarray(out[("y", "x")]), 3.0 * np.eye(n), atol=1e-10)


def test_cross_hessian_generic_finite_difference_check():
    n = 5
    x0 = jnp.linspace(-0.3, 0.4, n)
    y0 = jnp.linspace(0.6, 1.0, n)

    def ll_fn(params):
        x = params["x"]
        y = params["y"]
        return jnp.sin(x) * jnp.exp(y) - 0.5 * x * y

    def scalar_obj(x, y):
        return jnp.sum(ll_fn({"x": x, "y": y}))

    check_grads(scalar_obj, (x0, y0), order=2, modes=("fwd", "rev"), atol=1e-5, rtol=1e-5)

    out = cross_hessian(ll_fn, {"x": x0, "y": y0})
    np.testing.assert_allclose(
        np.asarray(out[("x", "y")]),
        np.asarray(out[("y", "x")]).T,
        rtol=1e-8,
        atol=1e-8,
    )
