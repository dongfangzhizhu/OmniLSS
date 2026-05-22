from __future__ import annotations

import numpy as np

from omnilss.FitTail import logSurv, loglogSurv


def test_logSurv_is_nonincreasing() -> None:
    y = np.array([0.5, 1.0, 2.0, 4.0])
    out = logSurv(y)
    assert out.x.shape == y.shape
    assert out.curve.shape == y.shape
    # survival decreases => log survival non-increasing
    assert np.all(np.diff(out.curve) <= 1e-12)


def test_loglogSurv_finite_values() -> None:
    y = np.array([1.0, 1.5, 2.0, 3.0])
    out = loglogSurv(y)
    assert np.all(np.isfinite(out.curve))
