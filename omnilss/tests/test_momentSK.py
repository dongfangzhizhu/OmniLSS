from __future__ import annotations

import numpy as np

from omnilss.momentSK import momentSK, momentSkew


def test_momentSK_basic() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = momentSK(y)
    assert out.mean == 2.5
    assert out.variance > 0


def test_momentSkew_matches_momentSK() -> None:
    y = np.array([1.0, 1.0, 2.0, 4.0])
    assert np.isclose(momentSkew(y), momentSK(y).skewness)
