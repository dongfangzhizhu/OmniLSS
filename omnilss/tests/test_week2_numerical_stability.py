from __future__ import annotations

import numpy as np

from omnilss.algorithms.rs_algorithm import compute_working_weights_and_response
from omnilss.algorithms.stabilized_hessian import stabilize_hessian


def test_stabilize_hessian_regularizes_singular_matrix():
    H = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    result = stabilize_hessian(H, base_lambda=1e-6)
    assert result.was_regularized
    assert np.isfinite(result.condition_number)
    _ = np.linalg.cholesky(result.matrix)


def test_working_response_is_clipped_and_finite():
    y = np.array([1.0, 2.0], dtype=np.float64)
    fitted = np.array([1.0, 2.0], dtype=np.float64)
    link_derivative = np.array([1e-12, 1e-12], dtype=np.float64)
    first = np.array([1e20, -1e20], dtype=np.float64)
    second = np.array([-1e-20, -1e-20], dtype=np.float64)
    offset = np.zeros_like(y)
    eta = np.zeros_like(y)

    w, z = compute_working_weights_and_response(
        y=y,
        fitted_values=fitted,
        link_derivative=link_derivative,
        first_derivative=first,
        second_derivative=second,
        offset=offset,
        eta=eta,
    )

    assert np.isfinite(w).all()
    assert np.isfinite(z).all()
    assert np.max(np.abs(z)) <= 1e6
