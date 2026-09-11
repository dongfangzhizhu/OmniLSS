import jax
import jax.numpy as jnp
import numpy as np

from omnilss.dpqr_functions import dSICHEL
from omnilss.distributions_b6 import SICHEL


def test_density_broadcast_jit():
    x = jnp.arange(3.)[:, None]
    mu = jnp.array([1., 3.])
    nu = jnp.array([-.5, .2])
    actual = jax.jit(dSICHEL)(x, mu, .5, nu)
    expected = [[dSICHEL(float(i), float(m), .5, float(n))
                 for m, n in zip(mu, nu)] for i in range(3)]
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (3, 2)


def test_density_family_boundaries():
    x = jnp.array([-1., .5, jnp.inf, -jnp.inf, jnp.nan, 10001., 0., 2.])
    actual = dSICHEL(x, 1., .5, .2)
    np.testing.assert_array_equal(actual[:4], 0.)
    assert np.isnan(actual[4:6]).all()
    np.testing.assert_allclose(actual, SICHEL().d(x, 1., .5, .2))
    assert np.isnan(dSICHEL(0., sigma=-1.))
    assert np.isnan(dSICHEL(0., nu=np.inf))
    assert np.isneginf(dSICHEL(-1., log=True))


def test_density_normalization_and_gradient():
    mass = dSICHEL(jnp.arange(101.), 1., .5, .2)
    np.testing.assert_allclose(mass.sum(), 1., atol=1e-6)
    gradient = jax.grad(lambda m: dSICHEL(2., m, .5, .2, log=True))(1.)
    assert np.isfinite(gradient)