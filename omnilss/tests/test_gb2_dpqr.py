"""Contract tests for GB2 (Generalized Beta Type 2)."""

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jrandom
from scipy.special import betaln

from omnilss.distribution_registry import resolve
from omnilss.dpqr_functions import dGB2, pGB2, qGB2, rGB2


MU, SIGMA, NU, TAU = 1.0, 1.0, 2.0, 3.0


@pytest.mark.parametrize("x", [0.2, 1.0, 3.0])
def test_gb2_density_matches_r_formula(x):
    expected = abs(SIGMA) * x ** (SIGMA * NU - 1) / (
        MU ** (SIGMA * NU) * np.exp(betaln(NU, TAU))
        * (1 + (x / MU) ** SIGMA) ** (NU + TAU)
    )
    assert float(dGB2(x, MU, SIGMA, NU, TAU)) == pytest.approx(expected, rel=1e-12)


def test_gb2_family_wires_real_dpqr():
    family = resolve("GB2")
    assert all(getattr(family, name) is not None for name in ("d", "p", "q", "r"))
    assert family.parameters == ("mu", "sigma", "nu", "tau")


def test_gb2_cdf_quantile_and_boundary_contract():
    x = jnp.linspace(0.3, 3.0, 20)
    probabilities = pGB2(x, MU, SIGMA, NU, TAU)
    np.testing.assert_allclose(qGB2(probabilities, MU, SIGMA, NU, TAU), x, rtol=1e-10)
    assert pGB2(0.0, MU, SIGMA, NU, TAU) == 0.0
    assert pGB2(jnp.inf, MU, SIGMA, NU, TAU) == 1.0
    assert qGB2(0.0, MU, SIGMA, NU, TAU) == 0.0
    assert jnp.isinf(qGB2(1.0, MU, SIGMA, NU, TAU))
    assert jnp.isnan(qGB2(-0.1, MU, SIGMA, NU, TAU))


def test_gb2_broadcast_and_tails():
    x = jnp.array([0.5, 1.0, 2.0])
    mu = jnp.array([0.8, 1.0, 1.2])
    values = pGB2(x, mu, SIGMA, NU, TAU)
    assert values.shape == x.shape
    np.testing.assert_allclose(
        pGB2(x, mu, SIGMA, NU, TAU, lower_tail=False) + values, 1.0
    )
    np.testing.assert_allclose(
        pGB2(x, mu, SIGMA, NU, TAU, log_p=True), jnp.log(values)
    )


def test_gb2_random_generation_respects_support_and_varies():
    samples = rGB2(jrandom.PRNGKey(42), 1000, MU, SIGMA, NU, TAU)
    assert samples.shape == (1000,)
    assert jnp.all(samples > 0)
    assert float(jnp.std(samples)) > 0.0
