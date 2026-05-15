import pytest
import warnings

from omnilss.controls import gamlss_control
from omnilss.gamlssML import gamlssML
from omnilss.sklearn_compat import GAMLSSRegressor

pytest.importorskip("sklearn")


def test_gamlss_control_exposes_memory_optimization_flag():
    ctrl = gamlss_control(memory_optimization=True)
    assert ctrl.memory_optimization is True


def test_gamlssML_deprecated_alias_warns():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            gamlssML("y ~ x", family="NO", data={"y": [1.0], "x": [1.0]})
        except Exception:
            # call may fail on tiny data; warning behavior is what we verify
            pass
    assert any(isinstance(w.message, DeprecationWarning) for w in rec)


def test_sklearn_regressor_has_get_set_score_protocol():
    reg = GAMLSSRegressor(family="NO")
    params = reg.get_params()
    assert "family" in params
    reg.set_params(fit_intercept=False)
    assert reg.fit_intercept is False
    assert hasattr(reg, "score")
