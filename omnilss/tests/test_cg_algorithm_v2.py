import numpy as np

from omnilss.algorithms import cg_fit_v2, cg_fit_lbfgs


def test_cg_fit_v2_basic_no_family_runs():
    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=n)
    y = 2.0 + 0.5 * x + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x": x}

    model = cg_fit_v2("y ~ x", family="NO", data=data, max_iter=8, tol=1e-3)
    assert np.isfinite(model.g_dev)
    assert model.additional_slots.get("method") == "CG"


def test_cg_fit_v2_close_to_backend_on_no():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    y = 1.2 + 0.8 * x + rng.normal(scale=0.4, size=n)
    data = {"y": y, "x": x}

    m_v2 = cg_fit_v2("y ~ x", family="NO", data=data, max_iter=10, tol=1e-4)
    m_ref = cg_fit_lbfgs("y ~ x", family="NO", data=data, max_outer_iter=10, outer_tol=1e-4)
    assert abs(float(m_v2.g_dev) - float(m_ref.g_dev)) < 1e-2
