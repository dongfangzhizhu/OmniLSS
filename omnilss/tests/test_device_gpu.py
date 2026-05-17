"""GPU-conditional tests. Skipped if no GPU is detected."""

from __future__ import annotations

import numpy as np
import pytest


def _has_gpu() -> bool:
    try:
        import jax

        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")


def test_normal_fit_gpu() -> None:
    """Basic fit should work on GPU when a GPU backend is installed."""
    import jax

    from omnilss import gamlss

    rng = np.random.default_rng(42)
    n = 1000
    x = np.linspace(0, 5, n)
    y = 2 + 3 * x + rng.normal(size=n)

    with jax.default_device(jax.devices("gpu")[0]):
        model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    assert model.g_dev > 0
    assert model.additional_slots.get("rs_converged", False)
