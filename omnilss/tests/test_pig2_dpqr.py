import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from omnilss.distribution_registry import resolve
from omnilss.dpqr_functions import dPIG, dPIG2, pPIG, pPIG2, qPIG, qPIG2, rPIG2


def test_pig2_matches_r_parameter_conversion():
    x = jnp.array([0, 1])
    mu = jnp.array([0.8, 2.0])
    sigma = jnp.array([0.3, 1.4])
    alpha = 1.0 / (jnp.sqrt(mu * mu + sigma * sigma) - mu)
    np.testing.assert_allclose(dPIG2(x, mu, sigma), dPIG(x, mu, alpha))
    np.testing.assert_allclose(pPIG2(x, mu, sigma), pPIG(x, mu, alpha))
    probs = jnp.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(qPIG2(probs, mu, sigma), qPIG(probs, mu, alpha))


def test_pig2_family_exposes_complete_api_and_boundaries():
    family = resolve("PIG2")
    assert callable(family.d) and callable(family.p)
    assert callable(family.q) and callable(family.r)
    assert family.p(jnp.array([-1, 0]), 1.0, 1.0)[0] == 0
    assert family.q(jnp.array([0.0, 1.0]), 1.0, 1.0)[0] == 0
    assert jnp.isnan(family.q(jnp.array([-0.1]), 1.0, 1.0)[0])


def test_pig2_random_generation_is_nonnegative():
    samples = rPIG2(jrandom.PRNGKey(42), 128, 2.0, 0.5)
    assert samples.shape == (128,)
    assert jnp.all(samples >= 0)