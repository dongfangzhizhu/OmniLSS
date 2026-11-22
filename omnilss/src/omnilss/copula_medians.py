"""R gamlss.ggplots::Copula_ralated.R -- bivariate medians and 2-D ECDF.

R source: ``R/gamlss.ggplots/R/Copula_ralated.R``.

Five small numeric helpers used by the copula/bucket plots:

* ``median_cw``  component-wise (marginal) median;
* ``median_pc``  principal-component median -- but R's result is overwritten
  by ``out <- c(median(x), median(y))`` at the end, so all the ``getSVD`` /
  ``beta_coef`` / ``fv`` work (and the ``++beta_coef`` double unary plus) is
  dead code.  It returns the component-wise median too;
* ``median_sp``  spatial (geometric) median: minimise the sum of Euclidean
  distances with Nelder-Mead from the marginal medians;
* ``xy_median``  dispatch -- with an R bug: ``method="cw"`` calls
  ``median_pc`` (not ``median_cw``).  Since both return the marginal medians
  the numbers coincide, but the call is reproduced here;
* ``ECDF_2D``    ``mean(data.x <= value.x & data.y <= value.y)``.

Two R details are reproduced deliberately:

* ``y`` has no default in R, so the "2-column matrix" branch is unreachable
  unless the caller explicitly passes ``y=NULL``; R errors with
  ``argument "y" is missing, with no default`` otherwise.  Python accepts
  ``y=None`` (the intended behaviour) and documents the R quirk instead;
* ``ECDF_2D`` is scalar-only in spirit: vector ``value`` arguments are
  recycled (with a warning) and ``mean`` still returns one scalar.  ``np.resize``
  emulates that recycling so the port matches R rather than vectorising.

``median_sp`` note: R's ``optim`` default ``reltol`` (~1.49e-8) stops early,
about 3e-4 away from the true geometric median.  This port uses a tighter
Nelder-Mead tolerance and therefore lands on the actual minimum; the live-R
tests pin both R's default value (loose tolerance, recorded as R stopping
early) and R's ``reltol=1e-14`` value (tight agreement), and check that the
port's objective value is no worse than R's default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

_METHODS = ("cw", "pc", "sp")


def _as_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """R's data handling: (x, y) or one 2-column matrix/data.frame."""
    if y is None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Provide either (x, y) or a 2-column matrix/data.frame")
        return arr[:, 0], arr[:, 1]
    return (
        np.asarray(x, dtype=np.float64).ravel(),
        np.asarray(y, dtype=np.float64).ravel(),
    )


def median_cw(x: Any, y: Any = None) -> np.ndarray:
    """R median_cw: component-wise median ``c(median(x), median(y))``."""

    a, b = _as_pair(x, y)
    return np.array([np.median(a), np.median(b)], dtype=np.float64)


def median_pc(x: Any, y: Any = None) -> np.ndarray:
    """R median_pc: returns the component-wise median (R's dead-code overwrite)."""

    a, b = _as_pair(x, y)
    return np.array([np.median(a), np.median(b)], dtype=np.float64)


def median_sp(x: Any, y: Any = None) -> np.ndarray:
    """R median_sp: geometric median, Nelder-Mead from the marginal medians."""

    a, b = _as_pair(x, y)

    def objective(par: np.ndarray) -> float:
        return float(np.sum(np.sqrt((a - par[0]) ** 2 + (b - par[1]) ** 2)))

    result = minimize(
        objective,
        np.array([np.median(a), np.median(b)], dtype=np.float64),
        method="Nelder-Mead",
        options={"xatol": 1.0e-12, "fatol": 1.0e-12, "maxiter": 10000},
    )
    return np.asarray(result.x, dtype=np.float64)


def xy_median(x: Any, y: Any = None, method: str = "cw") -> np.ndarray:
    """R xy_median: ``match.arg`` dispatch (with the ``cw`` -> ``median_pc`` bug)."""

    if not isinstance(method, str):
        raise ValueError("method must be one of 'cw', 'pc', 'sp'")
    lowered = method.lower()
    matches = [name for name in _METHODS if name.startswith(lowered)]
    if len(matches) != 1:
        raise ValueError("method must be one of 'cw', 'pc', 'sp'")
    selected = matches[0]
    if selected == "sp":
        return median_sp(x, y)
    # R calls median_pc for BOTH "cw" and "pc" -- kept verbatim.
    return median_pc(x, y)


def ECDF_2D(value_x: Any, value_y: Any, data_x: Any, data_y: Any) -> float:
    """R ECDF_2D: ``mean(data.x <= value.x & data.y <= value.y)``."""

    dx = np.asarray(data_x, dtype=np.float64).ravel()
    dy = np.asarray(data_y, dtype=np.float64).ravel()
    vx = np.asarray(value_x, dtype=np.float64).ravel()
    vy = np.asarray(value_y, dtype=np.float64).ravel()
    # R recycles the shorter operand (warning for non-multiple lengths);
    # np.resize gives the same values.  mean() then yields one scalar.
    if vx.size != dx.size:
        vx = np.resize(vx, dx.size)
    if vy.size != dy.size:
        vy = np.resize(vy, dy.size)
    return float(np.mean((dx <= vx) & (dy <= vy)))


__all__ = ["ECDF_2D", "median_cw", "median_pc", "median_sp", "xy_median"]
