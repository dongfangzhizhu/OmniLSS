from __future__ import annotations

import numpy as np

from omnilss.centileSK import centileSK


def test_centileSK_basic() -> None:
    y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    out = centileSK(y, centiles=(0, 50, 100))
    np.testing.assert_allclose(out.centiles, np.array([0.0, 50.0, 100.0]))
    np.testing.assert_allclose(out.values, np.array([1.0, 3.0, 5.0]))
