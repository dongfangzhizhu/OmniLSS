from __future__ import annotations

import numpy as np

from omnilss.ECDF import ECDF
from omnilss.loglogplot import loglogplot


def test_ecdf_sorted_probabilities() -> None:
    y = np.array([3.0, 1.0, 2.0])
    res = ECDF(y)
    np.testing.assert_allclose(res.x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(res.p, np.array([1 / 3, 2 / 3, 1.0]))


def test_loglogplot_output_shapes_and_finite() -> None:
    y = np.array([0.5, 1.0, 2.0, 4.0])
    out = loglogplot(y)
    assert out.x.shape == y.shape
    assert out.log_x.shape == y.shape
    assert out.log_log_survival.shape == y.shape
    assert np.all(np.isfinite(out.log_x))
    assert np.all(np.isfinite(out.log_log_survival))
