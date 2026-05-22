from __future__ import annotations

import numpy as np

from omnilss.flexDist import flexDist


def test_flexDist_summary_fields() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = flexDist(y, centiles=(0.0, 0.5, 1.0))
    assert out.support_min == 1.0
    assert out.support_max == 5.0
    np.testing.assert_allclose(out.centiles, np.array([1.0, 3.0, 5.0]))
    assert out.variance > 0
