"""Pure JAX contract tests for the Generalised Pareto compatibility family."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import quad

import jax
import jax.numpy as jnp

from omnilss.distributions_gp_compat import dGP, pGP, qGP, rGP
from omnilss.distributions_gp_compat import GP
from omnilss.algorithms.rs_algorithm import JaxSolverBackend


class TestGPUnit(unittest.TestCase):
    def test_family_derivatives_and_log_link_chain_rule(self):
        family = GP()
        y, mu, sigma = jnp.array([0.2, 1.0, 4.0]), 2.0, 0.7
        score_mu = family.score_functions["mu"](y, mu, sigma)
        score_sigma = family.score_functions["sigma"](y, mu, sigma)
        hess_mu = family.hessian_functions["mu"](y, mu, sigma)
        hess_sigma = family.hessian_functions["sigma"](y, mu, sigma)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.stack((score_mu, score_sigma, hess_mu, hess_sigma))))))
        eta_mu, eta_sigma = jnp.log(mu), jnp.log(sigma)
        link_mu = family.link_derivatives["mu"](eta_mu)
        link_sigma = family.link_derivatives["sigma"](eta_sigma)
        self.assertTrue(bool(jnp.allclose(link_mu, mu)))
        self.assertTrue(bool(jnp.allclose(link_sigma, sigma)))
        eta_score_mu = jax.grad(lambda e: jnp.sum(dGP(y, jnp.exp(e), sigma, log=True)))(eta_mu)
        eta_score_sigma = jax.grad(lambda e: jnp.sum(dGP(y, mu, jnp.exp(e), log=True)))(eta_sigma)
        np.testing.assert_allclose(eta_score_mu, jnp.sum(score_mu) * mu, rtol=1e-6)
        np.testing.assert_allclose(eta_score_sigma, jnp.sum(score_sigma) * sigma, rtol=1e-6)

    def test_jax_solver_backend_consumes_gp_derivatives(self):
        family = GP()
        score, hessian = JaxSolverBackend(family).compute_score_hessian(
            parameter="sigma", y=np.array([0.2, 1.0]), fitted=np.array([0.7, 0.8]),
            context={"mu": np.array([2.0, 2.0])})
        self.assertEqual(score.shape, (2,))
        self.assertEqual(hessian.shape, (2,))
        self.assertTrue(np.isfinite(score).all() and np.isfinite(hessian).all())
    def test_density_is_cdf_derivative_and_normalized(self):
        for mu, sigma in ((2.0, 0.7), (0.4, 3.0), (5.0, 1.0)):
            y = jnp.array([0.01, 0.5, 3.0, 20.0])
            derivative = jax.vmap(jax.grad(lambda x: pGP(x, mu, sigma)))(y)
            np.testing.assert_allclose(dGP(y, mu, sigma), derivative, rtol=1e-10, atol=1e-15)
            density = jax.jit(lambda x: dGP(x, mu, sigma))
            integral, _ = quad(lambda x: float(density(x)), 0, np.inf)
            self.assertAlmostEqual(integral, 1.0, places=8)

    def test_density_log_density_and_support(self):
        y = jnp.array([0.1, 1.0, 10.0])
        log_density = dGP(y, 2.0, 0.7, log=True)
        self.assertTrue(bool(jnp.allclose(dGP(y, 2.0, 0.7), jnp.exp(log_density))))
        self.assertTrue(bool(jnp.all(dGP(jnp.array([-1.0, 0.0]), 2.0, 0.7) == 0)))

    def test_probability_quantile_and_log_tail(self):
        probabilities = jnp.array([0.01, 0.1, 0.5, 0.9, 0.99])
        quantiles = qGP(probabilities, 2.0, 0.7)
        self.assertTrue(bool(jnp.allclose(pGP(quantiles, 2.0, 0.7), probabilities, rtol=1e-6)))
        self.assertTrue(bool(jnp.allclose(
            pGP(quantiles, 2.0, 0.7, lower_tail=False), 1.0 - probabilities, rtol=1e-6
        )))
        self.assertTrue(bool(jnp.isneginf(pGP(0.0, 2.0, 0.7, log_p=True))))
        self.assertTrue(bool(jnp.isposinf(qGP(1.0, 2.0, 0.7))))

    def test_broadcast_jit_and_ad(self):
        y = jnp.array([0.1, 1.0, 3.0])
        result = jax.jit(dGP)(y, 2.0, jnp.array([0.5, 0.7, 1.0]))
        self.assertEqual(result.shape, y.shape)
        derivative = jax.grad(lambda mu: jnp.sum(dGP(y, mu, 0.7, log=True)))(2.0)
        self.assertTrue(bool(jnp.isfinite(derivative)))

    def test_invalid_parameters_and_random_generation(self):
        self.assertTrue(bool(jnp.all(jnp.isnan(pGP(1.0, 0.0, 1.0)))))
        self.assertTrue(bool(jnp.all(jnp.isnan(qGP(0.5, 1.0, -1.0)))))
        sample = rGP(jax.random.key(42), 128, 2.0, 0.7)
        self.assertEqual(sample.shape, (128,))
        self.assertTrue(bool(jnp.all(sample >= 0)))


if __name__ == "__main__":
    unittest.main()