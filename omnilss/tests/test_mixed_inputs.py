"""Weighted MIXED fitting and multi-parameter formula propagation."""
import numpy as np
import pytest
from types import SimpleNamespace
from omnilss import gamlss
from omnilss.algorithms import mixed_algorithm as mixed


def test_weighted_mixed_rs_matches_weighted_normal_mle():
    y=np.array([-2.,-1.,0.,1.,2.,8.])
    weights=np.array([1.,2.,1.,2.,1.,.05])
    model=gamlss('y ~ 1',family='NO',data={'y':y},weights=weights,method='MIXED',max_iter=30)
    mean=np.average(y,weights=weights)
    sigma=np.sqrt(np.average((y-mean)**2,weights=weights))
    np.testing.assert_allclose(model.fitted_values['mu'],mean,atol=1e-6)
    np.testing.assert_allclose(model.fitted_values['sigma'],sigma,rtol=1e-5)
    np.testing.assert_allclose(model.weights,weights)
    assert abs(mean-y.mean())>.5


def test_public_mixed_retains_sigma_mapping():
    rng=np.random.default_rng(34)
    x=np.linspace(-1,1,60)
    data={'x':x,'y':1+x+rng.normal(size=60)*np.exp(.3*x)}
    options={'sigma':'~ x'}
    kwargs=dict(formula='y ~ x',family='NO',data=data,parameter_formulas=options,max_iter=3)
    model=gamlss(method='MIXED',**kwargs)
    reference=gamlss(method='RS',**kwargs)
    assert len(model.coefficients['sigma'])==2
    np.testing.assert_allclose(model.g_dev,reference.g_dev)
    assert options=={'sigma':'~ x'}


@pytest.mark.parametrize('algorithm',['rs','lbfgs'])
def test_all_formula_keys_and_weights_reach_backend(monkeypatch,algorithm):
    captured={}
    def fake(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(additional_slots={})
    monkeypatch.setattr(mixed,'rs_fit' if algorithm=='rs' else 'cg_fit',fake)
    weights=np.array([1.,2.,3.])
    formulas={'nu':'~ x','tau':'~ z'}
    mixed.mixed_fit('y ~ x',data={'y':np.ones(3)},algorithm=algorithm,
                    nu_formula='~ 1',parameter_formulas=formulas,weights=weights)
    assert captured['parameter_formulas']==formulas
    assert captured['parameter_formulas'] is not formulas
    assert captured['weights'] is weights


def test_smooth_parameter_mapping_stays_on_rs(monkeypatch):
    monkeypatch.setattr(mixed,'rs_fit',lambda **kw:SimpleNamespace(additional_slots={}))
    def unexpected(**kwargs):
        pytest.fail('smooth nu mapping was ignored by automatic selection')
    monkeypatch.setattr(mixed,'cg_fit',unexpected)
    model=mixed.mixed_fit('y ~ 1',data={'y':np.ones(600)},parameter_formulas={'nu':'~ pb(x)'})
    assert model.additional_slots['mixed_algorithm_used']=='rs'


def test_compatibility_wrapper_forwards_to_real_cg(monkeypatch):
    import importlib
    fitting=importlib.import_module('omnilss.fitting')
    from omnilss.algorithms.lbfgs_algorithm import joint_lbfgs_fit
    captured={}
    class ReachedCG(Exception):
        pass
    def fake(**kwargs):
        captured.update(kwargs)
        raise ReachedCG()
    monkeypatch.setattr(fitting,'gamlss',fake)
    weights=np.array([1.,2.])
    with pytest.raises(ReachedCG):
        joint_lbfgs_fit('y ~ 1',data={'y':np.ones(2)},weights=weights,
                        nu_formula='~ 1',parameter_formulas={'nu':'~ x','tau':'~ z'})
    assert captured['weights'] is weights
    assert captured['parameter_formulas']=={'nu':'~ x','tau':'~ z'}
    assert captured['method']=='CG'
