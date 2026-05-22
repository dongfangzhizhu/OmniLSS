from __future__ import annotations

import numpy as np

from omnilss.theoCentileSK import theoCentileSK


def test_theoCentileSK_median_zero() -> None:
    out = theoCentileSK((25, 50, 75))
    np.testing.assert_allclose(out.centiles, np.array([25.0, 50.0, 75.0]))
    assert np.isclose(out.z_values[1], 0.0)
    assert out.z_values[0] < 0 < out.z_values[2]
