"""Distribution-family pdf / cdf *curve data* from ``gamlss.ggplots``.

R sources: ``R/gamlss.ggplots/R/family_pdf.R`` (``family_pdf``) and
``R/gamlss.ggplots/R/family_cdf.R`` (``family_cdf``).  Unlike the ``fitted_*`` /
``predict_*`` helpers, these two take a *family name plus parameter values* and
need NO fitted gamlss object -- their numeric kernel is pure: build a response
grid and evaluate the family's ``d`` (pdf) or ``p`` (cdf) over it.  The R
functions draw ggplot2 objects; these ports return the ``data.frame(y.var,
matrix)`` that the plot layers consume, matching the project's plot-port
convention (:mod:`omnilss.family_plots`, :mod:`omnilss.copula_xy`).

Numeric kernel (live-R verified in ``omnilss/tests/test_family_curves.py``):

* grid ``y.var`` -- ``seq(from, to, length=no.points)`` for Continuous families,
  ``seq(from, to, by=1)`` for Discrete ones (R ``fam$type``);
* ``lobs = max(length(mu), length(sigma), length(nu), length(tau))`` -- each
  parameter is recycled to ``lobs`` with R ``rep(x, length=lobs)`` semantics;
* for each ``j`` in ``1..lobs`` evaluate ``dFAM(y.var, mu[j], sigma[j], ...)``
  (``family_pdf``) or ``pFAM(y.var, ..., lower.tail=lower.tail)``
  (``family_cdf``).  Parameter order/count is the family's ``nopar`` /
  ``parameters`` (``mu, sigma, nu, tau``).

R quirks reproduced: with a single curve the matrix column is named
``pdfArr`` / ``cdfArr``; with several curves ``data.frame`` auto-names the
matrix columns ``X1``, ``X2``, ...  ``:attr:`FamilyCurve.columns` returns
exactly those names.  ``lower.tail=FALSE`` gives the survival function
(``1 - cdf``) for ``family_cdf``.  ``from``/``to``/``no.points`` mirror R's
defaults (``0``/``10``/``201``).

Binomial-type families (``bd`` handling) and the ``family`` value being a
``gamlss.family`` object rather than a name are not ported -- they need the R
family object / binomial denominators; pass a family name string here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distribution_registry import resolve

__all__ = ["FamilyCurve", "family_pdf", "family_cdf"]


@dataclass(frozen=True)
class FamilyCurve:
    """The ``data.frame(y.var, matrix)`` R hands to ggplot.

    ``y`` is the response grid; ``values`` is a 2-D array with one column per
    parameter set (shape ``(len(y), n_curves)``), column ``j`` being the pdf /
    cdf for the ``j``-th parameter vector.  ``columns`` are the R data-frame
    column names for those value columns: ``["pdfArr"]`` / ``["cdfArr"]`` when
    there is one curve, else ``["X1", "X2", ...]``.
    """

    family: str
    y: np.ndarray
    values: np.ndarray
    columns: tuple[str, ...]
    lower_tail: bool


def _recycle(params: tuple[Any, ...]) -> tuple[list[np.ndarray], int]:
    """R ``rep(param, length=lobs)`` for each of mu/sigma/nu/tau."""
    arrays = [np.atleast_1d(np.asarray(p, dtype=np.float64)) for p in params]
    lobs = max((a.size for a in arrays), default=0)
    if lobs < 1:
        raise ValueError("At least one value of mu has to be set")
    return [np.resize(a, lobs) for a in arrays], lobs


def _curve(
    family: str,
    params: tuple[Any, ...],
    from_: float,
    to: float,
    no_points: int,
    which: str,
    lower_tail: bool,
) -> FamilyCurve:
    definition = resolve(family)
    nopar = int(definition.nopar)
    used = list(params[:nopar])
    if any(p is None for p in used):
        raise ValueError(
            f"family {family!r} needs {nopar} parameter(s): mu, sigma, nu, tau"
        )
    recycled, lobs = _recycle(tuple(used))

    fam_type = getattr(definition, "type", "Continuous")
    if isinstance(fam_type, str) and fam_type.lower() == "discrete":
        y = np.arange(from_, to + 1, 1.0, dtype=np.float64)
    else:
        n = int(no_points)
        if n < 1:
            raise ValueError("no_points must be >= 1")
        y = np.linspace(from_, to, n, dtype=np.float64)

    cols: list[np.ndarray] = []
    for j in range(lobs):
        args = [p[j] for p in recycled]
        if which == "d":
            val = np.asarray(definition.d(y, *args), dtype=np.float64)
        else:
            val = np.asarray(
                definition.p(y, *args, lower_tail=lower_tail), dtype=np.float64
            )
        cols.append(np.atleast_1d(val))

    values = np.column_stack(cols)
    if lobs == 1:
        columns = ("pdfArr",) if which == "d" else ("cdfArr",)
    else:
        columns = tuple(f"X{i}" for i in range(1, lobs + 1))
    return FamilyCurve(
        family=family,
        y=y,
        values=values,
        columns=columns,
        lower_tail=lower_tail,
    )


def family_pdf(
    family: str = "NO",
    mu: Any = None,
    sigma: Any = None,
    nu: Any = None,
    tau: Any = None,
    from_: float = 0,
    to: float = 10,
    no_points: int = 201,
    **_: Any,
) -> FamilyCurve:
    """R ``family_pdf`` curve data: ``dFAM(y.var, params)`` over the grid.

    ``from_`` mirrors R's ``from`` (Python keyword avoids the reserved word).
    """
    return _curve(
        family, (mu, sigma, nu, tau), from_, to, no_points, "d", True
    )


def family_cdf(
    family: str = "NO",
    mu: Any = None,
    sigma: Any = None,
    nu: Any = None,
    tau: Any = None,
    from_: float = 0,
    to: float = 10,
    no_points: int = 201,
    lower_tail: bool = True,
    **_: Any,
) -> FamilyCurve:
    """R ``family_cdf`` curve data: ``pFAM(y.var, params)`` over the grid.

    ``lower_tail=False`` yields the survival function (R ``lower.tail=FALSE``).
    """
    return _curve(
        family, (mu, sigma, nu, tau), from_, to, no_points, "p", lower_tail
    )
