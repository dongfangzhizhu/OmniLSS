import numpy as np
import pytest

from omnilss.algorithms.rs_algorithm import ConvergenceWarning, rs_fit


def test_rs_records_lambda_failure_params(monkeypatch):
    import omnilss.smooth_fitting as sf

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sf, "update_smooth_lambdas", boom)
    n = 50
    x = np.linspace(0, 1, n)
    y = 2 + x + np.random.default_rng(0).normal(scale=0.1, size=n)
    m = rs_fit("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2)
    failed = m.additional_slots.get("lambda_update_failed_params", ())
    assert "mu" in failed


def test_rs_raise_on_lambda_failure(monkeypatch):
    import omnilss.smooth_fitting as sf

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sf, "update_smooth_lambdas", boom)
    n = 40
    x = np.linspace(0, 1, n)
    y = 1 + x + np.random.default_rng(1).normal(scale=0.1, size=n)
    with pytest.raises(ConvergenceWarning):
        rs_fit("y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2, raise_on_lambda_failure=True)
