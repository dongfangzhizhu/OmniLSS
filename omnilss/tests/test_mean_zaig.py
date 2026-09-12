import jax
import jax.numpy as jnp
import pytest

from omnilss import meanZAIG


class Model:
    family = "ZAIG"
    fitted_values = {"mu": jnp.array([2.0, 4.0]), "nu": jnp.array([0.2, 0.75])}


def test_mean_zaig_matches_r_formula():
    assert jnp.allclose(meanZAIG(Model()), jnp.array([1.6, 1.0]))


def test_mean_zaig_supports_jit_and_gradient():
    def evaluate(mu):
        return (1.0 - 0.25) * mu

    assert jnp.allclose(jax.jit(evaluate)(2.0), 1.5)
    assert jax.grad(evaluate)(2.0) == pytest.approx(0.75)


def test_mean_zaig_rejects_other_family():
    model = Model()
    model.family = "ZAGA"
    with pytest.raises(ValueError, match="does not have a ZAIG"):
        meanZAIG(model)