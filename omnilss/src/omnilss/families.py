"""Family protocol definitions for migrated GAMLSS helpers.

R source references:
- file: `gamlss/R/DevianceIncr.R`
- function: `devianceIncr`
- file: `gamlss/R/gamlss-5.R`
- family objects used by `gamlss()` and `glim.fit`
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class FamilyDefinition:
    """Python representation of a GAMLSS family.

    R source references:
    - file: `gamlss/R/DevianceIncr.R`
    - function: `devianceIncr`

    Standard distribution functions (matching R GAMLSS):
    - d: density/probability mass function
    - p: cumulative distribution function (CDF)
    - q: quantile function (inverse CDF)
    - r: random number generation

    Architecture note
    -----------------
    ``FamilyDefinition`` is the legacy/runtime-facing family contract used by
    existing fitting paths (``fitting.py``, ``operations.py`` and most
    distribution implementations). During the architecture-freeze transition,
    ``core.distributions.DistributionProtocol`` defines the canonical protocol
    boundary, and ``core.distributions.FamilyDistributionAdapter`` provides the
    bridge from ``FamilyDefinition`` to that protocol.
    """

    name: str
    parameters: tuple[str, ...]
    g_dev_inc: Callable[..., Any]
    type: str = "Continuous"
    links: Mapping[str, str] | None = None
    link_functions: Mapping[str, Callable[..., Any]] | None = None
    link_inverses: Mapping[str, Callable[..., Any]] | None = None
    link_derivatives: Mapping[str, Callable[..., Any]] | None = None
    score_functions: Mapping[str, Callable[..., Any]] | None = None
    hessian_functions: Mapping[str, Callable[..., Any]] | None = None
    fixed_parameters: tuple[str, ...] | None = (
        None  # Parameters that should not be estimated
    )

    # Standard distribution functions (d/p/q/r)
    d: Callable[..., Any] | None = None  # Density/PMF function
    p: Callable[..., Any] | None = None  # CDF function
    q: Callable[..., Any] | None = None  # Quantile function
    r: Callable[..., Any] | None = None  # Random generation function

    def pdf(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the density/PMF through the standard ``d`` function."""
        if self.d is None:
            raise NotImplementedError(f"Family {self.name!r} does not define a density function")
        result = self.d(*args, **kwargs)
        if args and jnp.asarray(args[0]).ndim > 0 and jnp.asarray(result).ndim == 0:
            return jnp.reshape(jnp.asarray(result), (1,))
        return result

    @property
    def nopar(self) -> int:
        """Return the number of active family parameters."""
        return len(self.parameters)

    @property
    def estimable_parameters(self) -> tuple[str, ...]:
        """Return parameters that should be estimated (not fixed)."""
        if self.fixed_parameters is None:
            return self.parameters
        return tuple(p for p in self.parameters if p not in self.fixed_parameters)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FamilyDefinition":
        """Build a family object from mapping-style test fixtures."""

        return cls(
            name=str(mapping["name"]),
            parameters=tuple(mapping["parameters"]),
            g_dev_inc=mapping["g_dev_inc"],
            type=str(mapping.get("type", "Continuous")),
            links=mapping.get("links"),
            link_functions=mapping.get("link_functions"),
            link_inverses=mapping.get("link_inverses"),
            link_derivatives=mapping.get("link_derivatives"),
            score_functions=mapping.get("score_functions"),
            hessian_functions=mapping.get("hessian_functions"),
            fixed_parameters=mapping.get("fixed_parameters"),
            # 标准分布函数（d/p/q/r）——兼容旧 mapping 不含这些字段的情况
            d=mapping.get("d"),
            p=mapping.get("p"),
            q=mapping.get("q"),
            r=mapping.get("r"),
        )
