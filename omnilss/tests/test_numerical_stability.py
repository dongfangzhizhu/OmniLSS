import numpy as np
import jax.numpy as jnp

from omnilss.numerical_stability import regularize_hessian, sanitize_gradient, step_halving


def test_sanitize_gradient_replaces_non_finite_and_clips():
    g = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 2.0])
    out = sanitize_gradient(g, clip_value=10.0)
    np.testing.assert_allclose(np.asarray(out), np.array([0.0, 10.0, -10.0, 2.0]))


def test_regularize_hessian_adds_diagonal_lambda():
    h = jnp.array([[1.0, 0.2], [0.2, 3.0]])
    out = regularize_hessian(h, lam=1e-3)
    expected = np.array([[1.001, 0.2], [0.2, 3.001]])
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-7)


def test_step_halving_returns_damped_step():
    cur = jnp.array([0.0, 2.0])
    cand = jnp.array([2.0, -2.0])
    out = step_halving(cur, cand, factor=0.5)
    np.testing.assert_allclose(np.asarray(out), np.array([1.0, 0.0]))
