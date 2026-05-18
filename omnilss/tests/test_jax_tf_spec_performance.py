# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for handwritten TF JAX score/Hessian functions."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from omnilss.algorithms.jax_family_specs import make_tf_spec


def _tf_loglik_scalar(y_s, mu_s, sigma_s, nu_s):
    sigma_s = jnp.maximum(sigma_s, jnp.finfo(jnp.float64).eps)
    nu_s = jnp.maximum(nu_s, jnp.finfo(jnp.float64).eps)
    z = (y_s - mu_s) / sigma_s
    return (
        jax.scipy.special.gammaln((nu_s + 1.0) / 2.0)
        - jax.scipy.special.gammaln(nu_s / 2.0)
        - 0.5 * jnp.log(nu_s * jnp.pi)
        - jnp.log(sigma_s)
        - ((nu_s + 1.0) / 2.0) * jnp.log1p(jnp.square(z) / nu_s)
    )


def test_tf_handwritten_scores_match_autodiff_reference():
    spec = make_tf_spec()
    y = jnp.asarray([-0.7, 0.2, 1.4, 2.1], dtype=jnp.float64)
    mu = jnp.asarray([-0.5, 0.0, 1.0, 1.7], dtype=jnp.float64)
    sigma = jnp.asarray([0.8, 1.1, 1.3, 0.9], dtype=jnp.float64)
    nu = jnp.asarray([4.0, 5.0, 8.0, 10.0], dtype=jnp.float64)

    grad_mu = jax.vmap(jax.grad(_tf_loglik_scalar, argnums=1))(y, mu, sigma, nu)
    grad_sigma = jax.vmap(jax.grad(_tf_loglik_scalar, argnums=2))(y, mu, sigma, nu)
    grad_nu = jax.vmap(jax.grad(_tf_loglik_scalar, argnums=3))(y, mu, sigma, nu)

    np.testing.assert_allclose(spec.score_fns[0](y, mu, sigma, nu), grad_mu, rtol=1e-10)
    np.testing.assert_allclose(spec.score_fns[1](y, mu, sigma, nu), grad_sigma, rtol=1e-10)
    np.testing.assert_allclose(spec.score_fns[2](y, mu, sigma, nu), grad_nu, rtol=1e-10)


def test_tf_fisher_hessians_are_negative_and_finite():
    spec = make_tf_spec()
    y = jnp.linspace(-2.0, 2.0, 25, dtype=jnp.float64)
    mu = jnp.zeros_like(y)
    sigma = jnp.ones_like(y)
    nu = jnp.ones_like(y) * 7.0

    for hess_fn in spec.hessian_fns:
        hess = hess_fn(y, mu, sigma, nu)
        assert jnp.all(jnp.isfinite(hess))
        assert jnp.all(hess < 0.0)
