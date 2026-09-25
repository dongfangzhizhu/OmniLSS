"""Explicit compatibility routes must not silently discard smooth penalties."""
import warnings
import numpy as np
import pytest
from omnilss.algorithms import mixed_algorithm as mixed
from omnilss.algorithms.lbfgs_algorithm import joint_lbfgs_fit


@pytest.mark.parametrize('algorithm',['lbfgs','l-bfgs','cg'])
@pytest.mark.parametrize('parameter',['mu','sigma','nu','tau'])
def test_reject_before_backend(monkeypatch,algorithm,parameter):
    def unexpected(**kwargs):
        pytest.fail('penalized model reached unpenalized backend')
    monkeypatch.setattr(mixed,'cg_fit',unexpected)
    formula='y ~ s(x)' if parameter=='mu' else 'y ~ x'
    options={} if parameter=='mu' else {parameter:'~ pb (x)'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        with pytest.raises(ValueError,match='does not support penalized'):
            mixed.hybrid_fit(formula,data={'y':np.ones(3)},algorithm=algorithm,
                            parameter_formulas=options)


@pytest.mark.parametrize('kwargs',[
    {'formula':'y ~ s(x)'},
    {'formula':'y ~ x','sigma_formula':'~ cs(x)'},
    {'formula':'y ~ x','nu_formula':'~ ps(x)'},
    {'formula':'y ~ x','parameter_formulas':{'tau':'~ pb(x)'}},
])
def test_direct_wrapper_rejects_before_gamlss(monkeypatch,kwargs):
    import importlib
    fitting=importlib.import_module('omnilss.fitting')
    def unexpected(**options):
        pytest.fail('wrapper delegated a smooth model')
    monkeypatch.setattr(fitting,'gamlss',unexpected)
    with pytest.raises(ValueError,match='does not support penalized'):
        joint_lbfgs_fit(data={'y':np.ones(3)},**kwargs)


def test_overridden_legacy_formula_not_rejected(monkeypatch):
    # Mapping overrides legacy nu_formula; check the effective mapping.
    from types import SimpleNamespace
    import importlib
    fitting=importlib.import_module('omnilss.fitting')
    monkeypatch.setattr(fitting,'gamlss',lambda **kw:SimpleNamespace(
        fitted_values={},additional_slots={}))
    model=joint_lbfgs_fit('y ~ x',data={'y':np.ones(3)},nu_formula='~ pb(x)',
                         parameter_formulas={'nu':'~ 1'})
    assert model is not None
