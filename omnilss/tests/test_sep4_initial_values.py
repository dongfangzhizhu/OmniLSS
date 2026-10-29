"""SEP3 and SEP4 start where R starts (D2.88 / D2.92).

R SEP4.R declares

    mu.initial    = (y + mean(y))/2
    sigma.initial = rep(sd(y), length(y))
    nu.initial    = rep(2, length(y))
    tau.initial   = rep(2, length(y))

and SEP4 had no branch in _initial_parameter_value, so it fell through to the
generic residual scale.  On the scanner sample sigma then hit the log-scale
floor (log 1e-6) on the very first cycle and mu went to the eta clip: deviance
1.04e14 at n_cyc = 1 and 1.30e27 at n_cyc = 20.  With R initial values the
same fit lands at 119.12.

D2.92 adds SEP3: with the generic default it never got below 1932, and with
these R values it descends to 123.46 by cycle 12 before the family's
sigma -> 0 degeneracy takes over (rs_fit then falls back to the best finite
cycle instead of returning inf).

These tests pin the starting values without running a fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss._fitting_init import _initial_parameter_value  # noqa: E402
from omnilss.distribution_registry import resolve  # noqa: E402


def _family():
    return resolve("SEP4")


def _y():
    rng = np.random.default_rng(3)
    return np.abs(rng.normal(2.0, 0.8, 40)) + 0.2


def test_sigma_starts_at_the_sample_standard_deviation():
    y = _y()
    value = _initial_parameter_value(_family(), "sigma", y, np.mean(y), np.ones_like(y))
    np.testing.assert_allclose(value, float(np.std(y, ddof=1)), rtol=1e-12)


def test_nu_and_tau_start_at_two():
    y = _y()
    family = _family()
    for parameter in ("nu", "tau"):
        value = _initial_parameter_value(family, parameter, y, np.mean(y), np.ones_like(y))
        assert float(value) == 2.0, (parameter, value)


def test_sep3_uses_the_same_scale_and_tau_but_nu_one():
    """R SEP3.R: sigma = sd(y), nu = 1, tau = 2 (not SEP4's nu = 2)."""
    y = _y()
    family = resolve("SEP3")
    sigma = _initial_parameter_value(family, "sigma", y, np.mean(y), np.ones_like(y))
    np.testing.assert_allclose(float(sigma), float(np.std(y, ddof=1)), rtol=1e-12)
    assert float(_initial_parameter_value(family, "nu", y, np.mean(y), np.ones_like(y))) == 1.0
    assert float(_initial_parameter_value(family, "tau", y, np.mean(y), np.ones_like(y))) == 2.0