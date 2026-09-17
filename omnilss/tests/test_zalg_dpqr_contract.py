import numpy as np

from omnilss.distribution_registry import resolve


def test_zalg_r_parameterization_and_mass() -> None:
    family = resolve("ZALG")
    assert family.parameters == ("mu", "sigma")
    x = np.arange(0.0, 5.0)
    actual = np.asarray(family.d(x, 0.4, 0.2))
    expected = np.array([
        0.2,
        0.8 * 0.4 / (-np.log1p(-0.4)),
        0.8 * 0.4**2 / (2.0 * -np.log1p(-0.4)),
        0.8 * 0.4**3 / (3.0 * -np.log1p(-0.4)),
        0.8 * 0.4**4 / (4.0 * -np.log1p(-0.4)),
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_zalg_cdf_quantile_and_edges() -> None:
    family = resolve("ZALG")
    np.testing.assert_allclose(family.p(np.array([-1.0, 0.0, 1.0]), 0.4, 0.2), [0.0, 0.2, 0.82643686], rtol=1e-6)
    np.testing.assert_array_equal(family.q(np.array([0.1, 0.2, 0.5, 0.9]), 0.4, 0.2), [0.0, 0.0, 1.0, 2.0])
    assert np.isinf(float(family.q(1.0, 0.4, 0.2)))


def test_zalg_log_probability_broadcast_and_sampling() -> None:
    family = resolve("ZALG")
    log_cdf = np.asarray(family.p(np.array([0.0, 1.0]), np.array([0.2, 0.8]), 0.1, log_p=True))
    np.testing.assert_allclose(np.exp(log_cdf), [0.1, 0.1 + 0.9 * 0.8 / -np.log1p(-0.8)], rtol=1e-6)
    samples = np.asarray(family.r(__import__("jax").random.PRNGKey(4), 100, 0.4, 0.2))
    assert samples.shape == (100,)
    assert np.all(samples >= 0)


def test_zalg_negative_density_and_vectorized_parameter_broadcast() -> None:
    family = resolve("ZALG")
    np.testing.assert_allclose(family.d([-1.0, 0.0, 1.0], 0.4, 0.2), [0.0, 0.2, 0.8 * 0.4 / -np.log1p(-0.4)])
    values = np.asarray(family.p([0.0, 1.0, 2.0], [0.2, 0.4, 0.8], [0.1, 0.2, 0.3]))
    assert values.shape == (3,)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_zalg_upper_tail_and_quantile_zero_boundary() -> None:
    family = resolve("ZALG")
    lower = family.p([0.0, 1.0, 2.0], 0.4, 0.2)
    upper = family.p([0.0, 1.0, 2.0], 0.4, 0.2, lower_tail=False)
    np.testing.assert_allclose(lower + upper, 1.0)
    assert family.q(0.0, 0.4, 0.2) == 0.0
