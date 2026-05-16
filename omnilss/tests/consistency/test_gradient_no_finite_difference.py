import numpy as np
import jax
import jax.numpy as jnp

from omnilss.algorithms.jax_family_specs import get_jax_spec


def _finite_diff_grad_mu(loglik_fn, y, mu, sigma, eps=1e-5):
    lp = loglik_fn(y, mu + eps, sigma)
    lm = loglik_fn(y, mu - eps, sigma)
    return (lp - lm) / (2.0 * eps)


def test_no_gradient_matches_finite_difference():
    spec = get_jax_spec("NO")
    y = jnp.array([0.2, -0.1, 1.5, 0.0], dtype=jnp.float64)
    mu = jnp.array([0.0, 0.1, 1.0, -0.2], dtype=jnp.float64)
    sigma = jnp.array([1.2, 0.8, 1.5, 2.0], dtype=jnp.float64)

    loglik_fn = spec.loglik_fn
    score_mu_fn = spec.score_fns[0]

    grad_fd = _finite_diff_grad_mu(loglik_fn, y, mu, sigma)
    grad_ad = score_mu_fn(y, mu, sigma)

    np.testing.assert_allclose(np.asarray(grad_ad), np.asarray(grad_fd), rtol=6e-2, atol=3e-3)
