from __future__ import annotations

import numpy as np

from omnilss.lms import lms, z_scores


def test_z_scores_matches_log_form_when_nu_zero() -> None:
    y = np.array([1.0, 2.0, 4.0])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([1.0, 1.0, 2.0])
    nu = np.array([0.0, 0.0, 0.0])

    z = z_scores(y, mu, sigma, nu)
    expected = np.log(y / mu) / sigma
    np.testing.assert_allclose(z, expected, rtol=1e-8, atol=1e-8)


def test_lms_result_payload_shapes() -> None:
    y = np.array([1.0, 1.5, 2.0])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])
    nu = np.array([0.1, 0.1, 0.1])

    out = lms(y, mu, sigma, nu)
    assert out.z.shape == y.shape
    assert out.mu.shape == y.shape


def test_lms_supports_weights_and_weighted_z() -> None:
    y = np.array([1.0, 1.5, 2.0])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])
    nu = np.array([0.1, 0.1, 0.1])
    w = np.array([1.0, 2.0, 0.5])

    out = lms(y, mu, sigma, nu, weights=w)
    np.testing.assert_allclose(out.weights, w)
    np.testing.assert_allclose(out.weighted_z, out.z * w)


def test_lms_rejects_invalid_weights() -> None:
    y = np.array([1.0, 1.5, 2.0])
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.5, 0.5, 0.5])
    nu = np.array([0.1, 0.1, 0.1])
    try:
        lms(y, mu, sigma, nu, weights=np.array([1.0, -1.0, 1.0]))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-negative" in str(exc)
