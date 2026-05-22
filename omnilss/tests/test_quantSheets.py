from __future__ import annotations

import numpy as np

from omnilss.quantSheets import findPower, quantSheets, z_scoresQS


def test_quantSheets_returns_matching_probs_and_quantiles() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = quantSheets(y, probs=(0.25, 0.5, 0.75))
    np.testing.assert_allclose(out.probs, np.array([0.25, 0.5, 0.75]))
    np.testing.assert_allclose(out.quantiles, np.array([2.0, 3.0, 4.0]))


def test_findPower_and_z_scoresQS_basic() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.1, 3.2, 4.1])
    p = findPower(x, y)
    assert -2.0 <= p <= 2.0
    z = z_scoresQS(y=np.array([1.0, 2.0]), mu=np.array([1.0, 1.0]), sigma=np.array([1.0, 0.5]))
    np.testing.assert_allclose(z, np.array([0.0, 2.0]))
