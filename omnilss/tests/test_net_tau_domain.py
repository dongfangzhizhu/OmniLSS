# SPDX-License-Identifier: GPL-3.0-or-later
"""NET's tau >= nu domain (D2.237).

R's dNET/pNET/qNET stop with "tau must greater or equal than nu", so gamlss()
never fits with tau < nu.  Our raw density clamps tau to nu for AD/jit safety
(a Python raise inside a traced kernel would break jax.jit(fam.d)); the fitting
boundary now enforces R's domain.  Both halves are pinned here so a future
"helpful" clamp at the boundary cannot silently reintroduce the deviation.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import gamlss
from omnilss.distribution_registry import resolve
from omnilss.family_schema import FamilyValidationError


def _data(n: int = 60) -> tuple[dict, int]:
    rng = np.random.default_rng(11)
    x = rng.normal(size=n)
    y = 1.0 + 0.3 * x + rng.normal(scale=0.7, size=n)
    return {"y": y, "x": x}, n


def test_fit_rejects_tau_below_nu():
    data, n = _data()
    with pytest.raises(FamilyValidationError, match="tau must greater or equal than nu"):
        gamlss(
            "y ~ x",
            family="NET",
            data={**data, "nu": np.full(n, 2.0), "tau": np.full(n, 1.5)},
            max_iter=20,
        )


def test_fit_accepts_tau_equal_to_nu():
    data, n = _data()
    model = gamlss(
        "y ~ x",
        family="NET",
        data={**data, "nu": np.full(n, 2.0), "tau": np.full(n, 2.0)},
        max_iter=20,
    )
    assert np.isfinite(model.g_dev)


def test_fit_accepts_tau_above_nu():
    data, n = _data()
    model = gamlss(
        "y ~ x",
        family="NET",
        data={**data, "nu": np.full(n, 1.5), "tau": np.full(n, 2.0)},
        max_iter=20,
    )
    assert np.isfinite(model.g_dev)


def test_raw_kernel_clamps_for_jit_safety():
    """Documented deviation: the traced kernel cannot raise, so it clamps.

    R's stop is reproduced at the fitting boundary (tests above); the kernel
    stays a pure jnp expression so jax.jit(fam.d) remains usable.
    """
    family = resolve("NET")
    value = family.d(np.array([0.5]), 1.0, 0.7, 2.0, 1.5)
    assert np.isfinite(float(np.ravel(value)[0]))
