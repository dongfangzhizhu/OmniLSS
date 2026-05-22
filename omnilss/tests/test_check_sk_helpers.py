from __future__ import annotations

import numpy as np

from omnilss.centileSK import centileSK
from omnilss.checkCentileSK import checkCentileSK
from omnilss.checkMomentSK import checkMomentSK
from omnilss.momentSK import momentSK


def test_checkMomentSK_zero_error_when_targets_match() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = momentSK(y)
    out = checkMomentSK(y, m.mean, m.variance, m.skewness, m.kurtosis)
    assert out.mean_abs_error == 0.0
    assert out.variance_abs_error == 0.0


def test_checkCentileSK_zero_error_when_targets_match() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    c = (25, 50, 75)
    vals = centileSK(y, centiles=c).values
    out = checkCentileSK(y, centiles=c, target_values=vals)
    assert out.max_abs_error == 0.0
