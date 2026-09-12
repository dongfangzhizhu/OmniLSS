import jax.numpy as jnp
import pytest

from omnilss import meanBEINF


class Model:
    family = "BEINF"
    fitted_values = {
        "mu": jnp.array([0.2, 0.6]),
        "nu": jnp.array([0.1, 0.3]),
        "tau": jnp.array([0.4, 0.2]),
    }


def test_mean_beinf_matches_r_formula_and_broadcasts():
    expected = (jnp.array([0.4, 0.2]) + jnp.array([0.2, 0.6])) / (1 + jnp.array([0.1, 0.3]) + jnp.array([0.4, 0.2]))
    assert jnp.allclose(meanBEINF(Model()), expected)


def test_mean_beinf_rejects_other_family_and_missing_parameter():
    model = Model()
    model.family = "BEINF0"
    with pytest.raises(ValueError, match="does not have a BEINF"):
        meanBEINF(model)
    model.family = "BEINF"
    model.fitted_values = {"mu": 0.5, "nu": 0.2}
    with pytest.raises(ValueError, match="must contain"):
        meanBEINF(model)