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
    zw = z_scoresQS(y=np.array([1.0, 2.0]), mu=np.array([1.0, 1.0]), sigma=np.array([1.0, 0.5]), weights=np.array([2.0, 0.5]))
    np.testing.assert_allclose(zw, np.array([0.0, 1.0]))


def test_quantSheets_supports_weights() -> None:
    y = np.array([1.0, 2.0, 10.0], dtype=np.float64)
    w = np.array([1.0, 5.0, 1.0], dtype=np.float64)
    out = quantSheets(y, probs=(0.5,), weights=w)
    # weighted median pulled toward value 2.0
    assert abs(float(out.quantiles[0]) - 2.0) < 1e-12


def test_quantSheets_rejects_invalid_weights_and_probs_order() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    try:
        quantSheets(y, probs=(0.8, 0.2))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-decreasing" in str(exc)

    try:
        quantSheets(y, probs=(0.5,), weights=np.array([1.0, -1.0, 1.0]))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-negative" in str(exc)


def test_findPower_supports_weights_and_validation() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.1, 3.2, 4.1])
    w = np.array([1.0, 3.0, 1.0, 1.0])
    p = findPower(x, y, weights=w)
    assert -2.0 <= p <= 2.0

    try:
        findPower(x, y, weights=np.array([1.0, -1.0, 1.0, 1.0]))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-negative" in str(exc)


def test_z_scoresQS_rejects_invalid_weights() -> None:
    try:
        z_scoresQS(y=np.array([1.0, 2.0]), mu=np.array([1.0, 1.0]), sigma=np.array([1.0, 0.5]), weights=np.array([1.0, -1.0]))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-negative" in str(exc)
