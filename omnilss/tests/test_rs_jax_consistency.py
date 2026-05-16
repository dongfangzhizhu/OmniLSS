import numpy as np
import jax.numpy as jnp

from omnilss.rs_jax import _jax_wls, _jax_penalized_wls


def test_jax_wls_matches_numpy_lstsq():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    z = rng.normal(size=200)
    w = np.exp(rng.normal(size=200))

    sqrt_w = np.sqrt(w)
    beta_np, *_ = np.linalg.lstsq(X * sqrt_w[:, None], z * sqrt_w, rcond=None)
    beta_jax = np.asarray(_jax_wls(jnp.asarray(X), jnp.asarray(z), jnp.asarray(w)))

    np.testing.assert_allclose(beta_jax, beta_np, rtol=1e-10, atol=1e-10)


def test_jax_penalized_wls_solves_normal_equation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4))
    z = rng.normal(size=150)
    w = np.exp(rng.normal(size=150))
    penalty = np.diag([0.0, 0.1, 0.2, 0.3])

    beta = np.asarray(
        _jax_penalized_wls(jnp.asarray(X), jnp.asarray(z), jnp.asarray(w), jnp.asarray(penalty))
    )

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    zw = z * sqrt_w
    lhs = Xw.T @ Xw + penalty
    rhs = Xw.T @ zw
    np.testing.assert_allclose(lhs @ beta, rhs, rtol=1e-6, atol=1e-6)
