import numpy as np

from omnilss.ad_validation import finite_difference_vs_autodiff_no_mu


def test_fd_vs_ad_pipeline_no_mu():
    y = np.array([0.2, -0.1, 1.5, 0.0], dtype=np.float64)
    mu = np.array([0.0, 0.1, 1.0, -0.2], dtype=np.float64)
    sigma = np.array([1.2, 0.8, 1.5, 2.0], dtype=np.float64)
    res = finite_difference_vs_autodiff_no_mu(y, mu, sigma)
    assert res["max_abs_error"] < 5e-2
