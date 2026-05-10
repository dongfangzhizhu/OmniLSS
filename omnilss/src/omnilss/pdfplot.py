"""R-aligned density plot interfaces.

R source reference:
- file: `gamlss/R/pdfplot.R`
- functions: `pdf.plot`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from .distributions import resolve_family
from .model import GAMLSSModel
from .operations import fitted, is_gamlss


@dataclass(frozen=True)
class PDFPlotEntry:
    """One staged `pdf.plot` density-curve payload."""

    index: int
    y: np.ndarray
    density: np.ndarray
    parameters: dict[str, float]
    observed_value: float | None


@dataclass(frozen=True)
class PDFPlotResult:
    """Structured staged `pdf.plot` payload."""

    family: str
    distribution_type: str
    source: str
    entries: tuple[PDFPlotEntry, ...]
    approximation: str


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _staged_density_curve(
    family: Any,
    y_values: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    """Approximate a density/pmf curve from the staged family deviance."""

    ordered_parameters = tuple(str(name) for name in getattr(family, "parameters", ()))
    if not ordered_parameters:
        raise ValueError("family does not define any parameters")
    args = [jnp.asarray(y_values, dtype=jnp.float64)]
    for parameter in ordered_parameters:
        if parameter not in parameters:
            raise ValueError(f"missing parameter {parameter!r} for density curve")
        args.append(jnp.asarray(parameters[parameter], dtype=jnp.float64))
    deviance = np.asarray(family.g_dev_inc(*args), dtype=np.float64)
    density = np.exp(-0.5 * deviance)
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    if str(getattr(family, "type", "Continuous")).lower() == "discrete":
        total = float(np.sum(density))
    else:
        total = float(np.trapezoid(density, y_values))
    if np.isfinite(total) and total > 0.0:
        density = density / total
    return density.astype(np.float64, copy=False)


def pdf_plot_data(
    obj: GAMLSSModel | None = None,
    obs: Sequence[int] = (1,),
    family: str | Any | None = None,
    mu: Sequence[float] | float | None = None,
    sigma: Sequence[float] | float | None = None,
    nu: Sequence[float] | float | None = None,
    tau: Sequence[float] | float | None = None,
    from_value: float = 0.0,
    to_value: float = 10.0,
    min_value: float | None = None,
    max_value: float | None = None,
    no_points: int = 201,
) -> PDFPlotResult:
    """R reference: `gamlss/R/pdfplot.R::pdf.plot`.

    Staged behavior:
    - Returns density/pmf curve data instead of drawing panels.
    - Supports model-based observation snapshots and direct family/parameter
      combinations.
    - Uses the staged family deviance increment to build numerically normalised
      density curves over a requested support grid.
    """

    if obj is None and family is None:
        family = "NO"
    resolved_family = resolve_family(obj.family if obj is not None else family)
    distribution_type = str(getattr(resolved_family, "type", "Continuous"))
    lower = float(from_value if min_value is None else min_value)
    upper = float(to_value if max_value is None else max_value)
    if no_points < 2:
        raise ValueError("no_points must be at least 2")
    if upper <= lower:
        raise ValueError("max_value/to_value must be greater than min_value/from_value")
    if distribution_type.lower() == "discrete":
        y_grid = np.arange(math.ceil(lower), math.floor(upper) + 1, dtype=np.float64)
        if y_grid.size == 0:
            raise ValueError("discrete support grid is empty")
    else:
        y_grid = np.linspace(lower, upper, int(no_points), dtype=np.float64)

    entries: list[PDFPlotEntry] = []
    source = "model" if obj is not None else "parameters"
    if obj is not None:
        _require_gamlss(obj)
        observed = np.asarray(obj.y, dtype=np.float64).ravel()
        parameter_names = tuple(str(name) for name in getattr(resolved_family, "parameters", obj.par))
        for panel_index, obs_index in enumerate(obs, start=1):
            zero_index = int(obs_index) - 1
            if zero_index < 0 or zero_index >= observed.size:
                raise IndexError(f"obs index {obs_index} is out of range")
            parameter_values: dict[str, float] = {}
            for parameter in parameter_names:
                fitted_array = np.asarray(fitted(obj, parameter), dtype=np.float64).ravel()
                parameter_values[parameter] = float(fitted_array[zero_index])
            entries.append(
                PDFPlotEntry(
                    index=panel_index,
                    y=y_grid,
                    density=_staged_density_curve(resolved_family, y_grid, parameter_values),
                    parameters=parameter_values,
                    observed_value=float(observed[zero_index]),
                )
            )
    else:
        parameter_vectors = {
            "mu": None if mu is None else np.asarray(mu, dtype=np.float64).ravel(),
            "sigma": None if sigma is None else np.asarray(sigma, dtype=np.float64).ravel(),
            "nu": None if nu is None else np.asarray(nu, dtype=np.float64).ravel(),
            "tau": None if tau is None else np.asarray(tau, dtype=np.float64).ravel(),
        }
        parameter_names = tuple(str(name) for name in getattr(resolved_family, "parameters", ()))
        missing = [name for name in parameter_names if parameter_vectors.get(name) is None]
        if missing:
            raise ValueError(f"missing parameter values for family {resolved_family.name}: {', '.join(missing)}")
        panel_count = max(int(parameter_vectors[name].size) for name in parameter_names)
        for panel_index in range(panel_count):
            parameter_values = {
                name: float(parameter_vectors[name][panel_index % parameter_vectors[name].size])
                for name in parameter_names
            }
            entries.append(
                PDFPlotEntry(
                    index=panel_index + 1,
                    y=y_grid,
                    density=_staged_density_curve(resolved_family, y_grid, parameter_values),
                    parameters=parameter_values,
                    observed_value=None,
                )
            )

    return PDFPlotResult(
        family=str(resolved_family.name),
        distribution_type=distribution_type,
        source=source,
        entries=tuple(entries),
        approximation="staged density from family deviance with numerical normalization",
    )


pdf_plot = pdf_plot_data

__all__ = [
    "PDFPlotEntry",
    "PDFPlotResult",
    "pdf_plot",
    "pdf_plot_data",
]
