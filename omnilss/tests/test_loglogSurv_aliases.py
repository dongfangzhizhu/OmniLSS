from __future__ import annotations

import numpy as np

from omnilss.loglogSurv import logSurv, logSurv0, loglogSurv


def test_logSurv0_matches_logSurv() -> None:
    y = np.array([0.5, 1.0, 2.0, 3.0])
    a = logSurv(y)
    b = logSurv0(y)
    np.testing.assert_allclose(a.x, b.x)
    np.testing.assert_allclose(a.curve, b.curve)


def test_loglogSurv_shape() -> None:
    y = np.array([0.5, 1.0, 2.0])
    out = loglogSurv(y)
    assert out.x.shape == (3,)
    assert out.curve.shape == (3,)
