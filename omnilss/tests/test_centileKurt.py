from __future__ import annotations

import numpy as np

from omnilss.centileKurt import centileKurt


def test_centileKurt_positive_ratio() -> None:
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    out = centileKurt(y)
    assert out.lower_tail_spread >= 0
    assert out.upper_tail_spread >= 0
    assert out.kurtosis_ratio >= 0
