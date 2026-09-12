import jax
import jax.numpy as jnp
import pytest

from omnilss import meanZAGA


class Model:
    family = "ZAGA"
    fitted_values = {"mu": jnp.array([2.0, 4.0]), "nu": jnp.array([0.2, 0.75])}


def test_mean_zaga_matches_r_formula_and_broadcasts():
    assert jnp.allclose(meanZAGA(Model()), jnp.array([1.6, 1.0]))


def test_mean_zaga_supports_jit_and_gradient():
    assert jnp.allclose(jax.jit(lambda mu: (1.0 - 0.25) * mu)(2.0), 1.5)
    assert jax.grad(lambda mu: meanZAGA(type("M", (), {
        "family": "ZAGA", "fitted_values": {"mu": mu, "nu": 0.25}
    })()))(2.0) == pytest.approx(0.75)


def test_mean_zaga_rejects_other_family():
    model = Model()
    model.family = "ZAIG"
    with pytest.raises(ValueError, match="does not have a ZAGA"):
        meanZAGA(model)