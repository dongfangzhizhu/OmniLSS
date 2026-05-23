from __future__ import annotations

import numpy as np

from omnilss.centileSK import centileSK
from omnilss.checkCentileSK import checkCentileSK
from omnilss.theoCentileSK import theoCentileSK


def test_centile_sk_basic() -> None:
    y = np.arange(1, 101)
    out = centileSK(y, centiles=(10, 50, 90))
    assert out.centiles.tolist() == [10.0, 50.0, 90.0]
    assert out.values.shape == (3,)


def test_theo_centile_sk_basic() -> None:
    out = theoCentileSK((25, 50, 75))
    assert out.z_values.shape == (3,)
    assert np.isclose(out.z_values[1], 0.0)


def test_check_centile_sk_zero_error() -> None:
    y = np.linspace(0.0, 1.0, 101)
    cents = (10, 50, 90)
    vals = centileSK(y, centiles=cents).values
    chk = checkCentileSK(y, cents, vals)
    assert np.isclose(chk.max_abs_error, 0.0)
    assert np.isclose(chk.mean_abs_error, 0.0)
