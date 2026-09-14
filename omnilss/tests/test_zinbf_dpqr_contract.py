import jax
import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve


def test_zinbf_r_compatible_contract():
    fam = resolve("ZINBF")
    x = jnp.arange(6.0)
    args = (2.0, 0.7, 1.5, 0.2)
    d = fam.d(x, *args)
    p = fam.p(x, *args)
    assert d.shape == x.shape
    np.testing.assert_allclose(fam.d(x, *args, log=True), jnp.log(d), equal_nan=True)
    np.testing.assert_allclose(fam.p(x, *args, log_p=True), jnp.log(p))
    np.testing.assert_allclose(fam.p(x, *args), jnp.cumsum(d), rtol=1e-6)

    probs = jnp.array([0.01, 0.2, 0.5, 0.99])
    quantiles = fam.q(probs, *args)
    assert jnp.all(jnp.isfinite(quantiles))
    assert jnp.all(fam.p(quantiles, *args) >= probs)

    samples = fam.r(jax.random.key(0), 128, *args)
    assert samples.shape == (128,)
    assert jnp.all((samples >= 0) & (samples == jnp.floor(samples)))


def test_zinbf_broadcasting_and_tails():
    fam = resolve("ZINBF")
    x = jnp.arange(4.0)[:, None]
    mu = jnp.array([1.0, 3.0, 5.0])[None, :]
    values = fam.p(x, mu, 0.8, 1.7, 0.15)
    assert values.shape == (4, 3)
    np.testing.assert_allclose(
        fam.p(x, mu, 0.8, 1.7, 0.15, lower_tail=False), 1.0 - values
    )