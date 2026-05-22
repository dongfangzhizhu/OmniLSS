from __future__ import annotations

import numpy as np

from omnilss.plotCentileSK import plotCentileSK


def test_plotCentileSK_shapes() -> None:
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    out = plotCentileSK(y, centiles=(25, 50, 75))
    assert out.centiles.shape == (3,)
    assert out.empirical.shape == (3,)
    assert out.theoretical_z.shape == (3,)
    assert np.isclose(out.theoretical_z[1], 0.0)
