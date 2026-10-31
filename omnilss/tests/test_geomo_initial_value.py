"""GEOMO starts where R GEOMo starts (D2.112).

GEOMo is the ORIGINAL parameterisation of the geometric: its single parameter
is a probability on a logit link, whereas GEOM is a mean on a log link.  R
gives the two different starting values::

    GEOM()$mu.initial   mu <- rep(mean(y), length(y))      (R/gamlss.dist/R/GEOM.R line 103)
    GEOMo()$mu.initial  mu <- rep(.5, length(y))           (R/gamlss.dist/R/GEOMo.R line 101)

so GEOMO must NOT inherit GEOM's branch -- the D2.110 caveat, measured rather
than assumed.  GEOMO had no branch at all and fell through to the generic
default, which is neither of the two values above.

This test pins the starting value without running a fit: with a logit link,
mu = 0.5 is eta = 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss._fitting_init import _initial_parameter_value  # noqa: E402
from omnilss.distribution_registry import resolve  # noqa: E402


def _y() -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.abs(rng.normal(2.0, 1.0, 32)).round()


def test_geomo_starts_at_a_half_probability():
    y = _y()
    eta = _initial_parameter_value(resolve("GEOMO"), "mu", y, np.mean(y), np.ones_like(y))
    np.testing.assert_allclose(np.asarray(eta, dtype=float), np.zeros_like(y), atol=1e-15)


def test_geom_keeps_its_own_starting_value():
    """GEOM is a mean on a log link -- it must not have picked up GEOMO's."""
    y = _y()
    eta = _initial_parameter_value(resolve("GEOM"), "mu", y, np.mean(y), np.ones_like(y))
    expected = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    np.testing.assert_allclose(np.asarray(eta, dtype=float), expected, rtol=1e-12)