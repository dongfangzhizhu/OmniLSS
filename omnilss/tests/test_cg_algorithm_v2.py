import numpy as np

from omnilss.algorithms import cg_fit, cg_fit_v2


def test_cg_fit_alias_points_to_v2():
    assert cg_fit is cg_fit_v2


def test_cg_fit_v2_basic_no_family_runs():
    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=n)
    y = 2.0 + 0.5 * x + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x": x}

    model = cg_fit_v2("y ~ x", family="NO", data=data, max_iter=5, tol=1e-3)
    assert np.isfinite(model.g_dev)
    assert model.additional_slots.get("method") == "CG"
