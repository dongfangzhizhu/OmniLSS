"""SEP4 starts where R starts (D2.88).

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