"""Bivariate ECDF / function / KDE grid data from ``gamlss.ggplots``.

R sources: ``R/gamlss.ggplots/R/Copula_ralated_not_ggplot2.R`` and
``Copula_ralated.R``.  The R functions draw base-graphics or ggplot2 surfaces;
these ports return the numerical grid that would have been plotted instead.

Numeric kernels, all fully deterministic and live-R verified in the
``test_bivar_grid.py`` / ``test_bivar_density.py`` tests:

* ``xy.ECDF.*`` / ``xy_ECDF_hist``: a 40x40 grid
  ``z = outer(xg, yg, Vectorize(ECDF_2D))`` where
  ``ECDF_2D(vx, vy, x, y) = mean(x <= vx & y <= vy)`` (the D2.215 port).
  ``xg``/``yg`` each span ``[min, max]`` with ``length=40``.  ``xy_ECDF_hist``
  (ggplot2) builds the *same* grid via ``expand.grid`` + ``mapply``.
* ``bivar.fun.*``: an ``n`` x ``n`` grid ``Z = matrix(f(grid$x, grid$y), n, n)``
  over ``expand.grid`` of ``seq(xlim, length.out=n)`` x ``seq(ylim, length.out=n)``.
  R's ``expand.grid`` varies x fastest and ``matrix(..., nrow=n)`` fills
  column-major, so ``Z[i, j] = f(gx[i], gy[j])`` (row=x, col=y).
* ``xy.density.*`` / ``xy_hist_kde2d``: ``MASS::kde2d(x, y, n=bins)`` -- a 2-D
  Gaussian kernel density on an ``n`` x ``n`` grid.  Ported exactly, including
  ``MASS::bandwidth.nrd`` (Silverman's rule with R's type-7 quantiles and
  sample variance ddof=1) and the ``h/4`` factor; ``bins`` defaults to 50 for
  the density trio (25 for ``xy.density.persp``) and ``xy_hist_kde2d``.

R quirks reproduced: the three ``xy.ECDF.*`` variants compute the *identical*
grid (only the drawing call differs); ``xy.ECDF.persp`` hard-codes
``xlab="x"``/``ylab="y"``/``main="Bivariate CDF"`` instead of using its own
``xlab``/``main`` arguments (dead args); the three ``bivar.fun.*`` and the
three ``xy.density.*`` likewise compute identical grids.  ``n`` defaults to
100 for ``bivar.fun.*``.  ``as.numeric`` / ``as.vector`` on an R matrix flattens
column-major (``ravel(order="F")`` in NumPy).

The siblings ``bivar_pdf`` / ``bivar_fun`` / ``bivar_fun_plotly`` / ``xy_hist`` /
``xy_density`` (ggplot2 bin2d / stat_density_2d) are intentionally NOT ported --
their binning/stat layers are not pure numeric kernels; see the module tests for
the boundary.
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
    "bandwidth_nrd",
    "kde2d",
    "xy_density_contour",
    "xy_density_image",
    "xy_density_persp",
    "xy_hist_kde2d",
    "xy_ECDF_hist",
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


# ---------------------------------------------------------------------------
# MASS::kde2d kernel (used by xy.density.* and xy_hist_kde2d)
# ---------------------------------------------------------------------------

def bandwidth_nrd(a: Any) -> float:
    """R ``MASS::bandwidth.nrd``: Silverman's rule-of-thumb bandwidth.

    ``4 * 1.06 * min(sqrt(var(x)), IQR/1.34) * length(x)^(-1/5)`` with R's
    sample variance (``ddof=1``) and type-7 quantiles (NumPy ``method="linear"``).
    Verified against live R to ~1e-13.
    """
    arr = np.asarray(a, dtype=np.float64).ravel()
    q25, q75 = np.quantile(arr, [0.25, 0.75], method="linear")
    iqr_scaled = (q75 - q25) / 1.34
    return float(4 * 1.06 * min(np.sqrt(np.var(arr, ddof=1)), iqr_scaled)
                 * arr.size ** (-0.2))


def kde2d(x: Any, y: Any, n: int = 25) -> BivarGrid:
    """R ``MASS::kde2d(x, y, n)`` -- 2-D Gaussian kernel density estimate.

    Grid spans ``[min, max]`` of each variable with ``n`` points; bandwidth is
    ``bandwidth_nrd / 4`` per axis; ``z[i, j] = sum_k dnorm((gx[i]-x[k])/hx) *
    dnorm((gy[j]-y[k])/hy) / (N * hx * hy)``.  R's ``var`` uses ddof=1 and
    ``quantile`` is type-7.  Returns the ``(x, y, z)`` grid (R returns a list).
    """
    dx = _as_float_array(x)
    dy = _as_float_array(y)
    if dx.size != dy.size:
        raise ValueError("data vectors must be the same length")
    nx = dx.size
    gx = np.linspace(dx.min(), dx.max(), int(n))
    gy = np.linspace(dy.min(), dy.max(), int(n))
    hx = bandwidth_nrd(dx) / 4.0
    hy = bandwidth_nrd(dy) / 4.0
    ax = (gx[:, None] - dx[None, :]) / hx
    ay = (gy[:, None] - dy[None, :]) / hy
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    A = np.exp(-ax**2 / 2.0) * inv_sqrt_2pi   # dnorm(ax), shape (n, N)
    B = np.exp(-ay**2 / 2.0) * inv_sqrt_2pi   # dnorm(ay), shape (n, N)
    z = (A @ B.T) / (nx * hx * hy)            # R tcrossprod, shape (n, n)
    return BivarGrid(x=gx, y=gy, z=z)


def _density_grid(x: Any, y: Any, bins: int) -> BivarGrid:
    # R xy.density.* accept either (x, y) or a 2-column matrix/data.frame
    if y is None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Provide either (x, y) or a 2-column array")
        dx, dy = arr[:, 0], arr[:, 1]
    else:
        dx, dy = x, y
    return kde2d(dx, dy, n=int(bins))


def xy_density_contour(x: Any, y: Any = None, bins: int = 50, **_: Any) -> BivarGrid:
    """R ``xy.density.contour`` grid data (``MASS::kde2d``), ``bins``=50."""
    return _density_grid(x, y, bins)


def xy_density_image(x: Any, y: Any = None, bins: int = 50, **_: Any) -> BivarGrid:
    """R ``xy.density.image`` grid data (``MASS::kde2d``), ``bins``=50."""
    return _density_grid(x, y, bins)


def xy_density_persp(x: Any, y: Any = None, bins: int = 25, **_: Any) -> BivarGrid:
    """R ``xy.density.persp`` grid data (``MASS::kde2d``), ``bins``=25."""
    return _density_grid(x, y, bins)


def xy_hist_kde2d(x: Any, y: Any = None, bins: int = 50, **_: Any) -> BivarGrid:
    """R ``xy_hist_kde2d`` grid data (``MASS::kde2d``), ``bins``=50.

    R builds ``kde_df = expand.grid(x=kde$x, y=kde$y)`` and sets
    ``kde_df$z = as.vector(kde$z)``; that long table is exactly the
    column-major flatten of our ``BivarGrid.z``.
    """
    return _density_grid(x, y, bins)


def xy_ECDF_hist(x: Any, y: Any, **_: Any) -> BivarGrid:
    """R ``xy_ECDF_hist`` grid data -- the same 40x40 ECDF grid as
    ``xy_ECDF_contour`` (``expand.grid`` + ``mapply(ECDF_2D)``).

    This is a distinct R export (``Copula_ralated.R:491``) from the point-named
    ``xy.ECDF.contour``; both compute the identical grid, verified against live R.
    """
    return _ecdf_grid(x, y)
