"""R gamlss.dist inflated-distribution density plot helpers.

R sources: ``R/gamlss.dist/R/BEOI.R``, ``BEZI.R``, ``ZAGA.R``, ``ZAIG.R``.

``plotBEOI`` / ``plotBEZI`` / ``plotZAGA`` / ``plotZAIG`` draw the density of a
zero- or one-inflated family over a grid, plus the point mass at the inflated
boundary.  The Python port returns the *plot data* (grid, density, point mass,
ylim, title) and does not draw, following the rest of OmniLSS's plot ports.

Two R behaviours are reproduced deliberately:

* the ``from`` argument is dead: every function hard-codes
  ``seq(from=0.001, to=to, length.out=n)`` and never reads the caller's
  ``from``.  ``from_`` is accepted for call-site compatibility and ignored;
* ``plotBEOI`` prints ``pr1`` as a side effect (a leftover debug print).  The
  value is exposed as ``point_density`` instead; nothing is printed.

The grid density and the boundary point mass are evaluated through the
family's own ``d`` with positional parameters, so both the AD-generated
densities (BEOI/BEZI) and the dpqr densities (ZAGA/ZAIG) work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distribution_registry import resolve

#: R hard-codes 0.001 as the grid start regardless of the ``from`` argument.
_GRID_START = 0.001


@dataclass(frozen=True)
class InflatedDensityPlot:
    """The data R's ``plot*()`` hands to ``plot()`` / ``points()``."""

    family: str
    y: np.ndarray
    density: np.ndarray
    point: float
    point_density: float
    ylim: tuple[float, float]
    main: str
    parameters: dict[str, float]

    @property
    def point_mass(self) -> float:
        """R's ``pr0`` / ``pr1``: the density at the inflated boundary."""
        return self.point_density


def _inflated_plot(
    *,
    family: str,
    mu: float,
    sigma: float,
    nu: float,
    to: float,
    n: int,
    point: float,
    main: str,
    from_: float = 0.001,  # noqa: ARG001 - R's dead argument, deliberately ignored
) -> InflatedDensityPlot:
    definition = resolve(family)
    count = int(n)
    if count < 1:
        raise ValueError("n must be a positive integer")
    y = np.linspace(_GRID_START, float(to), count)

    def density_at(values: np.ndarray) -> np.ndarray:
        params = (
            np.full(values.shape, float(mu)),
            np.full(values.shape, float(sigma)),
            np.full(values.shape, float(nu)),
        )
        return np.atleast_1d(
            np.asarray(definition.d(values, *params), dtype=np.float64)
        )

    density = density_at(y)
    point_density = float(density_at(np.array([point], dtype=np.float64))[0])
    ylim = (0.0, float(max(float(np.max(density)), point_density)))
    return InflatedDensityPlot(
        family=family,
        y=y,
        density=density,
        point=float(point),
        point_density=point_density,
        ylim=ylim,
        main=main,
        parameters={"mu": float(mu), "sigma": float(sigma), "nu": float(nu)},
    )


def plotBEOI(
    mu: float = 0.5,
    sigma: float = 1.0,
    nu: float = 0.1,
    from_: float = 0.001,
    to: float = 1.0,
    n: int = 101,
    **kwargs: Any,
) -> InflatedDensityPlot:
    """R plotBEOI: one-inflated beta density grid plus the mass at y=1."""

    return _inflated_plot(
        family="BEOI", mu=mu, sigma=sigma, nu=nu, from_=from_, to=to, n=n,
        point=1.0, main="One Inflated Beta",
    )


def plotBEZI(
    mu: float = 0.5,
    sigma: float = 1.0,
    nu: float = 0.1,
    from_: float = 0.0,
    to: float = 0.999,
    n: int = 101,
    **kwargs: Any,
) -> InflatedDensityPlot:
    """R plotBEZI: zero-inflated beta density grid plus the mass at y=0."""

    return _inflated_plot(
        family="BEZI", mu=mu, sigma=sigma, nu=nu, from_=from_, to=to, n=n,
        point=0.0, main="Zero Inflated Beta",
    )


def plotZAGA(
    mu: float = 5.0,
    sigma: float = 1.0,
    nu: float = 0.1,
    from_: float = 0.0,
    to: float = 10.0,
    n: int = 101,
    main: str | None = None,
    **kwargs: Any,
) -> InflatedDensityPlot:
    """R plotZAGA: zero-adjusted gamma density grid plus the mass at y=0."""

    return _inflated_plot(
        family="ZAGA", mu=mu, sigma=sigma, nu=nu, from_=from_, to=to, n=n,
        point=0.0, main="Zero Adj. Gamma" if main is None else str(main),
    )


def plotZAIG(
    mu: float = 5.0,
    sigma: float = 1.0,
    nu: float = 0.1,
    from_: float = 0.0,
    to: float = 10.0,
    n: int = 101,
    main: str | None = None,
    **kwargs: Any,
) -> InflatedDensityPlot:
    """R plotZAIG: zero-adjusted inverse Gaussian density grid plus y=0 mass."""

    return _inflated_plot(
        family="ZAIG", mu=mu, sigma=sigma, nu=nu, from_=from_, to=to, n=n,
        point=0.0, main="Zero adjusted IG" if main is None else str(main),
    )


__all__ = [
    "InflatedDensityPlot",
    "plotBEOI",
    "plotBEZI",
    "plotZAGA",
    "plotZAIG",
]
