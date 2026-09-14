import numpy as np
import jax

from omnilss.distribution_registry import resolve


def test_zilg_zero_inflated_logarithmic_contract() -> None:
    family = resolve("ZILG")
    assert family.parameters == ("mu", "nu")
    x = np.arange(0.0, 4.0)
    expected = [0.2, 0.8 * 0.4 / -np.log1p(-0.4), 0.8 * 0.4**2 / (2 * -np.log1p(-0.4)), 0.8 * 0.4**3 / (3 * -np.log1p(-0.4))]
    np.testing.assert_allclose(family.d(x, 0.4, 0.2), expected, rtol=1e-6)
    np.testing.assert_allclose(family.p([0, 1], 0.4, 0.2), [0.2, 0.82643686], rtol=1e-6)
    np.testing.assert_array_equal(family.q([0.1, 0.2, 0.5, 0.9], 0.4, 0.2), [0, 0, 1, 2])
    samples = np.asarray(family.r(jax.random.PRNGKey(7), 32, 0.4, 0.2))
    assert samples.shape == (32,)
    assert np.all(samples >= 0)