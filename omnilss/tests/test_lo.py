from __future__ import annotations

import numpy as np

from omnilss.lo import lo


def test_lo_returns_sorted_x_and_smoothed_values() -> None:
    x = np.array([3.0, 1.0, 2.0, 4.0])
    y = np.array([9.0, 1.0, 4.0, 16.0])
    out = lo(x, y, span=0.75)
    assert np.allclose(out.x, np.array([1.0, 2.0, 3.0, 4.0]))
    assert out.fitted.shape == out.x.shape
    assert np.all(np.isfinite(out.fitted))


def test_lo_input_validation() -> None:
    try:
        lo([], [], span=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        lo([1.0], [1.0], span=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
