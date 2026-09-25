"""Mixed dispatch must preserve configured family instances."""
import numpy as np
import pytest
from types import SimpleNamespace
from omnilss import gamlss
from omnilss.distributions_zero_alias_compat import ZABB, ZIBB
from omnilss.distributions import NO, BCT
from omnilss.algorithms import mixed_algorithm as mixed
from omnilss.serialization import save_model_json, load_model_json


@pytest.mark.parametrize('factory',[ZABB,ZIBB])
def test_public_route_preserves_family(tmp_path,factory):
    family=factory(bd=3)
    data={'y':np.tile([0.,0.,1.,2.,3.],12)}
    model=gamlss('y ~ 1',family=family,data=data,method='HYBRID',max_iter=2)
    reference=gamlss('y ~ 1',family=family,data=data,method='RS',max_iter=2)
    assert model.family is family
    np.testing.assert_allclose(model.g_dev,reference.g_dev)
    assert model.additional_slots['method']=='HYBRID'
    assert model.additional_slots['hybrid_algorithm_used']=='rs'
    path=tmp_path/'mixed.zip'
    save_model_json(model,path)
    restored=load_model_json(path)
    assert restored.family.constructor_parameters=={'bd':3}
    newdata={'y':np.zeros(4)}
    before=model.predict_params(newdata)
    after=restored.predict_params(newdata)
    for parameter in before:
        np.testing.assert_allclose(before[parameter],after[parameter])
    args=[after[p] for p in ('mu','sigma','nu')]
    np.testing.assert_allclose(restored.family.p([0,1,2,3],*args),family.p([0,1,2,3],*args))


@pytest.mark.parametrize('family,n,expected',[(NO(),499,'rs'),(NO(),500,'lbfgs'),(BCT(),500,'rs'),(ZIBB(bd=7),500,'lbfgs')])
def test_auto_selection_name_and_state(monkeypatch,family,n,expected):
    calls=[]
    def backend(name):
        def fit(**kwargs):
            calls.append((name,kwargs['family']))
            return SimpleNamespace(additional_slots={})
        return fit
    monkeypatch.setattr(mixed,'rs_fit',backend('rs'))
    monkeypatch.setattr(mixed,'cg_fit',backend('lbfgs'))
    model=mixed.hybrid_fit('y ~ 1',family=family,data={'y':np.ones(n)})
    assert calls==[(expected,family)]
    assert model.additional_slots['hybrid_algorithm_used']==expected
