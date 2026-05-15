"""Canonical distribution protocol for the OmniLSS architecture freeze.

Distributions should be stateless, pure-functional, JAX-compatible, and
PyTree-compatible.  This module does not add new families; it defines the shared
contract that existing family implementations must converge toward.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp

from ..constraints import Constraint
from ..links import Link
from ..params import Parameter

Params = Mapping[str, Any]
Data = Any
RandomKey = Any


@runtime_checkable
class DistributionProtocol(Protocol):
    """Stateless functional contract for all OmniLSS distributions."""

    name: str

    def logpdf(self, y: Data, params: Params) -> jnp.ndarray:
        """Return log-density/log-mass values for observations under ``params``."""
        ...

    def cdf(self, y: Data, params: Params) -> jnp.ndarray:
        """Return cumulative probabilities under ``params``."""
        ...

    def ppf(self, q: Data, params: Params) -> jnp.ndarray:
        """Return quantiles for probabilities ``q`` under ``params``."""
        ...

    def sample(
        self, key: RandomKey, params: Params, shape: tuple[int, ...] = ()
    ) -> jnp.ndarray:
        """Draw samples from ``params`` using an explicit JAX PRNG key."""
        ...

    def score(self, y: Data, params: Params) -> Params:
        """Return first derivatives of log-likelihood with respect to parameters."""
        ...

    def hessian(self, y: Data, params: Params) -> Mapping[str, Params]:
        """Return second derivatives of log-likelihood with respect to parameters."""
        ...

    def init_params(self, y: Data) -> Params:
        """Return initial parameter values inferred from data without side effects."""
        ...

    def parameter_constraints(self) -> Mapping[str, Constraint]:
        """Return parameter-domain constraints keyed by parameter name."""
        ...

    def links(self) -> Mapping[str, Link]:
        """Return canonical link objects keyed by parameter name."""
        ...

    def parameters(self) -> tuple[Parameter, ...]:
        """Return canonical parameter definitions in distribution order."""
        ...


REQUIRED_DISTRIBUTION_METHODS = (
    "logpdf",
    "cdf",
    "ppf",
    "sample",
    "score",
    "hessian",
    "init_params",
    "parameter_constraints",
    "links",
    "parameters",
)


def assert_distribution_protocol(distribution: Any) -> None:
    """Raise ``TypeError`` if an object does not expose the canonical methods."""
    missing = [
        name
        for name in REQUIRED_DISTRIBUTION_METHODS
        if not callable(getattr(distribution, name, None))
    ]
    if missing:
        raise TypeError(
            f"{distribution!r} does not implement DistributionProtocol; missing {missing}"
        )


__all__ = [
    "Data",
    "DistributionProtocol",
    "Params",
    "REQUIRED_DISTRIBUTION_METHODS",
    "RandomKey",
    "assert_distribution_protocol",
]
