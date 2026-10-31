"""BEo and GEOMo start where R starts (D2.120).

Both are "original parameterisation" families and both had NO initial-value
branch at all (they are absent from the table entirely -- confirmed by
rescanning with ``family.name`` in D2.119).  Live R gives::

    BEo()$mu.initial      mu <- rep(2, length(y))      mu/sigma links: log, log
    BEo()$sigma.initial   sigma <- rep(2, length(y))
    GEOMo()$mu.initial    mu <- rep(.5, length(y))     mu link: logit

BEo is the beta with its two shape parameters, so mu = sigma = 2 is a
different quantity from BE's probability mean -- it must not inherit BE's
branch.  GEOMo's mu is a probability, so mu = 0.5 is eta = 0 on a logit link.

Note the spelling: these branches test ``family.name`` ("BEo", "GEOMo"), not
the registry keys ("BEO", "GEOMO").  Writing the key makes the branch dead
code -- see D2.118 and test_family_name_branches_are_live.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss._fitting_init import _initial_parameter_value, _initial_mu_target  # noqa: E402
from omnilss.distribution_registry import resolve  # noqa: E402


def _y() -> np.ndarray:
    rng = np.random.default_rng(23)
    return np.abs(rng.normal(2.0, 1.0, 32)).round()


def _mu_target(key: str, y: np.ndarray) -> np.ndarray:
    eta = _initial_mu_target(resolve(key), y, np.mean(y), np.ones_like(y))
    return np.asarray(eta, dtype=float)


def test_beo_starts_at_two_on_the_log_scale():
    y = _y()
    np.testing.assert_allclose(_mu_target("BEO", y), np.full_like(y, np.log(2.0)), rtol=1e-15)


def test_beo_sigma_starts_at_two():
    y = _y()
    got = _initial_parameter_value(resolve("BEO"), "sigma", y, np.mean(y), np.ones_like(y))
    np.testing.assert_allclose(np.asarray(got, dtype=float), 2.0, rtol=1e-15)


def test_geomo_starts_at_a_half_probability():
    y = _y()
    np.testing.assert_allclose(_mu_target("GEOMO", y), np.zeros_like(y), atol=1e-15)


def test_be_keeps_its_own_starting_value():
    """BE must not have picked up BEo's shape-parameter start."""
    y = np.clip(np.abs(np.random.default_rng(3).normal(0.4, 0.2, 32)), 1e-6, 1 - 1e-6)
    clipped = np.clip(y, 1e-6, 1.0 - 1e-6)
    expected = np.log(clipped / (1.0 - clipped))
    np.testing.assert_allclose(_mu_target("BE", y), expected, rtol=1e-12)