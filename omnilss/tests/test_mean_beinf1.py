import jax
import jax.numpy as jnp
import pytest

from omnilss import meanBEINF1


class Model:
    family = "BEINF1"
    fitted_values = {"mu": jnp.array([0.2, 0.6]), "nu": jnp.array([0.1, 0.5])}


def test_mean_beinf1_matches_r_formula():
    assert jnp.allclose(meanBEINF1(Model()), jnp.array([0.3 / 1.1, 1.1 / 1.5]))


def test_mean_beinf1_gradient_and_family_validation():
    assert jax.grad(lambda mu: (0.5 + mu) / 1.5)(0.2) == pytest.approx(2 / 3)
    model = Model()
    model.family = "BEINF0"
    with pytest.raises(ValueError, match="does not have a BEINF1"):
        meanBEINF1(model)