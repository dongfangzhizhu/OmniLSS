"""Bivariate ECDF / function grid data from ``gamlss.ggplots`` (base-R plots).

R sources: ``R/gamlss.ggplots/R/Copula_ralated_not_ggplot2.R``.  The R
functions draw base-graphics surfaces (``persp`` / ``image`` / ``contour``);
these ports return the numerical grid that would have been plotted instead.

Two numeric kernels, both fully deterministic and live-R verified in
``omnilss/tests/test_bivar_grid.py``:

* ``xy.ECDF.*``: a 40x40 grid ``z = outer(xg, yg, Vectorize(ECDF_2D))`` where
  ``ECDF_2D(vx, vy, x, y) = mean(x <= vx & y <= vy)`` (the D2.215 port).
  ``xg``/``yg`` each span ``[min, max]`` with ``length=40``.
* ``bivar.fun.*``: an ``n`` x ``n`` grid ``Z = matrix(f(grid$x, grid$y), n, n)``
  over ``expand.grid`` of ``seq(xlim, length.out=n)`` x ``seq(ylim, length.out=n)``.
  R's ``expand.grid`` varies x fastest and ``matrix(..., nrow=n)`` fills
  column-major, so ``Z[i, j] = f(gx[i], gy[j])`` (row=x, col=y).

R quirks reproduced: the three ``xy.ECDF.*`` variants compute the *identical*
grid (only the drawing call differs); ``xy.ECDF.persp`` hard-codes
``xlab="x"``/``ylab="y"``/``main="Bivariate CDF"`` instead of using its own
``xlab``/``main`` arguments (dead args); the three ``bivar.fun.*`` likewise
compute the identical grid.  ``n`` defaults to 100 for ``bivar.fun.*``.

The sibling ``xy.density.*`` (``MASS::kde2d``), ``bivar_pdf`` / ``bivar_fun``
(ggplot2) and ``bivar_fun_plotly`` are intentionally NOT ported here -- they
need external KDE / plotting stacks; see the module tests for the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


__all__ = [
    "BivarGrid",
    "xy_ECDF_persp",
    "xy_ECDF_image",
    "xy_ECDF_contour",
    "bivar_fun_contour",
    "bivar_fun_image",
    "bivar_fun_persp",
]


@dataclass(frozen=True)
class BivarGrid:
    """Numerical grid that an R surface plot would have drawn.

    ``x`` / ``y`` are the axis grid vectors; ``z`` is a 2-D array with
    ``z[i, j]`` = value at ``(x[i], y[j])`` (row=x, col=y), matching R's
    ``outer`` / column-major ``matrix`` layout flattened by ``as.numeric``.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray  # shape (len(x), len(y))


def _as_float_array(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _ecdf_2d(vx: float, vy: float, data_x: np.ndarray, data_y: np.ndarray) -> float:
    """R ``ECDF_2D``: mean(data.x <= value.x & data.y <= value.y)."""
    return float(np.mean((data_x <= vx) & (data_y <= vy)))


def _ecdf_grid(x: Any, y: Any) -> BivarGrid:
    data_x = _as_float_array(x)
    data_y = _as_float_array(y)
    # R: seq(min(x), max(x), length = 40) -- n=40 is hard-coded in all three
    gx = np.linspace(data_x.min(), data_x.max(), 40)
    gy = np.linspace(data_y.min(), data_y.max(), 40)
    # R: outer(xg, yg, Vectorize(...)) -> z[i, j] = ECDF(gx[i], gy[j])
    z = np.array(
        [[_ecdf_2d(float(a), float(b), data_x, data_y) for b in gy] for a in gx],
        dtype=np.float64,
    )
    return BivarGrid(x=gx, y=gy, z=z)


def xy_ECDF_persp(x: Any, y: Any, **_: Any) -> BivarGrid:
    """R ``xy.ECDF.persp`` grid data (``xlab``/``main`` are dead args)."""
    return _ecdf_grid(x, y)


def xy_ECDF_image(x: Any, y: Any, **_: Any) -> BivarGrid:
    """R ``xy.ECDF.image`` grid data."""
    return _ecdf_grid(x, y)


def xy_ECDF_contour(x: Any, y: Any, **_: Any) -> BivarGrid:
    """R ``xy.ECDF.contour`` grid data."""
    return _ecdf_grid(x, y)


def _bivar_fun_grid(
    f: Callable[..., Any],
    xlim: Any,
    ylim: Any,
    n: int,
) -> BivarGrid:
    # R: seq(xlim[1], xlim[2], length.out = n)
    gx = np.linspace(float(xlim[0]), float(xlim[1]), int(n))
    gy = np.linspace(float(ylim[0]), float(ylim[1]), int(n))
    # R expand.grid(x=gx, y=gy) varies x fastest (y is the slow outer loop);
    # f is evaluated on the full vector pair, then matrix(z, nrow=n, ncol=n)
    # fills column-major, giving Z[i, j] = f(gx[i], gy[j]).
    px = np.tile(gx, len(gy))
    py = np.repeat(gy, len(gx))
    z = np.asarray(f(px, py), dtype=np.float64).ravel()
    Z = z.reshape((int(n), int(n)), order="F")  # column j = fixed gy[j]
    return BivarGrid(x=gx, y=gy, z=Z)


def bivar_fun_contour(
    f: Callable[..., Any],
    xlim: Any = (-3.0, 3.0),
    ylim: Any = (-3.0, 3.0),
    n: int = 100,
    **kwargs: Any,
) -> BivarGrid:
    """R ``bivar.fun.contour`` grid data."""
    return _bivar_fun_grid(f, xlim, ylim, n)


def bivar_fun_image(
    f: Callable[..., Any],
    xlim: Any = (-3.0, 3.0),
    ylim: Any = (-3.0, 3.0),
    n: int = 100,
    **kwargs: Any,
) -> BivarGrid:
    """R ``bivar.fun.image`` grid data."""
    return _bivar_fun_grid(f, xlim, ylim, n)


def bivar_fun_persp(
    f: Callable[..., Any],
    xlim: Any = (-3.0, 3.0),
    ylim: Any = (-3.0, 3.0),
    n: int = 100,
    **kwargs: Any,
) -> BivarGrid:
    """R ``bivar.fun.persp`` grid data."""
    return _bivar_fun_grid(f, xlim, ylim, n)
