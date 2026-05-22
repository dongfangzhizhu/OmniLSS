from __future__ import annotations

import numpy as np

from omnilss.vis_lo import vis_lo


def test_vis_lo_sorts_by_x() -> None:
    x = np.array([3.0, 1.0, 2.0])
    y = np.array([30.0, 10.0, 20.0])
    out = vis_lo(x, y)
    np.testing.assert_allclose(out.x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out.y, np.array([10.0, 20.0, 30.0]))
