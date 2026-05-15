import numpy as np
import pytest

from omnilss.algorithms import rs_fit


def test_lambda_update_failure_warns(monkeypatch):
    import omnilss.smooth_fitting as sf

    def _boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(sf, "update_smooth_lambdas", _boom)

    n = 30
    x = np.linspace(0, 1, n)
    y = 1.0 + x + np.random.default_rng(0).normal(scale=0.1, size=n)

    with pytest.warns(UserWarning, match="Lambda update failed for parameter 'mu':"):
        model = rs_fit(formula="y ~ pb(x)", family="NO", data={"y": y, "x": x}, max_iter=2)

    msgs = model.additional_slots.get("lambda_update_warnings", ())
    assert any("forced failure" in m for m in msgs)
