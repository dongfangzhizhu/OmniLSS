"""Search limits must not change the distribution's upper tail."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.dpqr_functions import qSICHEL
from omnilss.distribution_registry import resolve


@pytest.mark.parametrize("name", ["SICHEL", "ZASICHEL", "ZISICHEL"])
def test_search_cap_preserves_upper_tail(name):
    quantile = qSICHEL if name == "SICHEL" else resolve(name).q
    for log_p in (False, True):
        probabilities = jnp.array([.9, .1, 1e-12])
        values = jnp.log(probabilities) if log_p else probabilities
        full = quantile(values, lower_tail=False, log_p=log_p)
        capped = jax.jit(lambda p: quantile(
            p, lower_tail=False, log_p=log_p, max_value=0))(values)
        np.testing.assert_allclose(capped, jnp.where(full == 0, 0., jnp.nan))
        assert np.isnan(capped[-1])
    for invalid in (-1, 10001, 1000000, 1.5, True):
        with pytest.raises(ValueError, match="max_value"):
            quantile(.5, max_value=invalid)


@pytest.mark.parametrize("name", ["ZASICHEL", "ZISICHEL"])
def test_point_mass_ignores_search_cap(name):
    quantile = resolve(name).q
    for lower_tail in (False, True):
        np.testing.assert_array_equal(quantile(
            jnp.array([0., .5, 1.]), tau=1., lower_tail=lower_tail,
            max_value=0), 0.)