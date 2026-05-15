from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from omnilss.core.constraints import Positive
from omnilss.core.distributions import (
    REQUIRED_DISTRIBUTION_METHODS,
    as_distribution_protocol,
    assert_distribution_protocol,
)
from omnilss.core.links import LogLink
from omnilss.core.optimization import OptaxOptimizer
from omnilss.core.params import Parameter, parameters_from_names
from omnilss.distributions import NO


def test_parameter_transform_inverse_validate_and_pytree_roundtrip() -> None:
    parameter = Parameter(name="sigma", link=LogLink(), constraint=Positive())
    value = jnp.array([0.5, 2.0, 4.0])

    eta = parameter.transform(value)
    restored = parameter.inverse_transform(eta)

    assert jnp.allclose(restored, value)
    assert bool(jnp.all(parameter.validate(restored)))
    leaves, treedef = jax.tree_util.tree_flatten(parameter)
    assert leaves == []
    assert jax.tree_util.tree_unflatten(treedef, leaves) == parameter


def test_canonical_parameters_cover_standard_gamlss_names() -> None:
    parameters = parameters_from_names(("mu", "sigma", "nu", "tau"))

    assert tuple(parameter.name for parameter in parameters) == ("mu", "sigma", "nu", "tau")
    assert isinstance(parameters[1].constraint, Positive)


def test_family_distribution_adapter_exposes_protocol_methods() -> None:
    distribution = as_distribution_protocol(NO())
    params = {"mu": jnp.array([0.0, 0.0]), "sigma": jnp.array([1.0, 1.0])}
    y = jnp.array([0.0, 1.0])

    assert_distribution_protocol(distribution)
    assert all(callable(getattr(distribution, name)) for name in REQUIRED_DISTRIBUTION_METHODS)
    assert jnp.allclose(
        distribution.logpdf(y, params),
        jnp.array([-0.91893853, -1.41893853]),
        atol=1e-6,
    )
    assert set(distribution.parameter_constraints()) == {"mu", "sigma"}
    assert set(distribution.links()) == {"mu", "sigma"}


def test_optax_optimizer_protocol_updates_without_model_mutation() -> None:
    optimizer = OptaxOptimizer(optax.sgd(learning_rate=0.1))
    params = {"x": jnp.array(1.0)}

    def loss_fn(candidate):
        return jnp.square(candidate["x"] - 3.0)

    state = optimizer.init(params)
    new_params, new_state = optimizer.step(loss_fn, params, state)

    assert new_state is not state
    assert params["x"] == 1.0
    assert new_params["x"] > params["x"]
