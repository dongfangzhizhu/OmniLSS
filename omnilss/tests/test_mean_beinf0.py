import jax
import jax.numpy as jnp
import pytest

from omnilss import meanBEINF0


class Model:
    family = "BEINF0"
    fitted_values = {"mu": jnp.array([0.2, 0.6]), "nu": jnp.array([0.1, 0.5])}


def test_mean_beinf0_matches_r_formula():
    assert jnp.allclose(meanBEINF0(Model()), jnp.array([0.2 / 1.1, 0.4]))


def test_mean_beinf0_gradient_and_family_validation():
    assert jax.grad(lambda mu: mu / (1.0 + 0.5))(0.2) == pytest.approx(2 / 3)
    model = Model()
    model.family = "BEINF1"
    with pytest.raises(ValueError, match="does not have a BEINF0"):
        meanBEINF0(model)