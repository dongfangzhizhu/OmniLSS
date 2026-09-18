"""Artifact roundtrips preserve the family constructor, not only its name."""
import json
import zipfile
import numpy as np
import jax.random as jr
import pytest

from omnilss.model import GAMLSSModel
from omnilss.distributions_zero_alias_compat import ZABB, ZIBB
from omnilss.serialization import save_model_json, load_model_json, validate_model_json


def _model(family):
    params = ('mu','sigma','nu')
    return GAMLSSModel(
        par=params, parameters=params, family=family, n=3, df_fit=3., g_dev=2.,
        y=np.array([0.,1.,2.]),
        coefficients={p:np.array([0.]) for p in params},
        fitted_values={p:np.full(3,.3) for p in params},
        linear_predictors={p:np.zeros(3) for p in params},
        formulas={p:'y ~ 1' for p in params},
        design_matrices={p:np.ones((3,1)) for p in params},
    )


def _rewrite(source, target, transform):
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target,'w') as dst:
        for name in src.namelist():
            data=src.read(name)
            if name=='meta.json':
                meta=json.loads(data)
                transform(meta)
                data=json.dumps(meta).encode()
            dst.writestr(name,data)


@pytest.mark.parametrize('factory,bd',[(ZABB,3),(ZIBB,17),(ZIBB,0)])
def test_roundtrip_constructor_and_probabilities(tmp_path,factory,bd):
    original=factory(bd=bd)
    path=tmp_path/'model.omnilss'
    save_model_json(_model(original),path)
    assert validate_model_json(path)["ok"]
    loaded=load_model_json(path)
    assert loaded.family.constructor_parameters=={'bd':bd}
    args=(.3,.2,.25)
    for method,values in [('d',[0,1,bd,bd+1]),('p',[-1,0,bd,bd+1]),('q',[0,.3,.8,1])]:
        np.testing.assert_allclose(getattr(loaded.family,method)(values,*args),
                                   getattr(original,method)(values,*args))
    key=jr.PRNGKey(12)
    np.testing.assert_array_equal(loaded.family.r(key,30,*args),original.r(key,30,*args))
    again=tmp_path/'again.omnilss'
    save_model_json(loaded,again)
    assert load_model_json(again).family.constructor_parameters=={'bd':bd}


def test_legacy_missing_options_uses_default(tmp_path):
    source,target=tmp_path/'new.zip',tmp_path/'old.zip'
    save_model_json(_model(ZABB()),source)
    _rewrite(source,target,lambda meta:meta.pop('family_constructor_parameters'))
    assert validate_model_json(target)["ok"]
    assert load_model_json(target).family.constructor_parameters=={'bd':10}


def test_zibb_array_denominators_roundtrip(tmp_path):
    original=ZIBB(bd=np.array([0,3,17]))
    path=tmp_path/'vector.zip'
    save_model_json(_model(original),path)
    loaded=load_model_json(path).family
    assert loaded.constructor_parameters=={'bd':[0,3,17]}
    np.testing.assert_allclose(loaded.p([0,2,10],.3,.2,.25),original.p([0,2,10],.3,.2,.25))


@pytest.mark.parametrize('options',[None,{'bd':-1},{'bd':True},{'bd':1.5},{'other':5}])
def test_invalid_options_do_not_silently_restore_default(tmp_path,options):
    source,target=tmp_path/'new.zip',tmp_path/'bad.zip'
    save_model_json(_model(ZABB(bd=3)),source)
    _rewrite(source,target,lambda meta:meta.update(family_constructor_parameters=options))
    with pytest.raises(ValueError):
        load_model_json(target)


@pytest.mark.parametrize('family,options,code',[
    ('ZABB',None,'invalid_family_constructor_parameters'),
    ('ZABB',[], 'invalid_family_constructor_parameters'),
    ('NO',{'bd':3},'unsupported_family_constructor_parameters'),
    ('ZIBB',{'bd':3,'extra':1},'unsupported_family_constructor_parameters'),
    ('ZABB',{'bd':0},'invalid_family_denominator'),
    ('ZABB',{'bd':[3]},'invalid_family_denominator'),
    ('ZIBB',{'bd':[True,3]},'invalid_family_denominator'),
    ('ZIBB',{'bd':['3',2]},'invalid_family_denominator'),
    ('ZIBB',{'bd':[[1],[2,3]]},'invalid_family_denominator'),
    ('ZIBB',{'bd':[]},'invalid_family_denominator'),
    ('ZIBB',{'bd':float('inf')},'invalid_family_denominator'),
    ('ZIBB',{'bd':1.5},'invalid_family_denominator'),
])
def test_validator_loader_constructor_diagnostics_agree(tmp_path,family,options,code):
    source,target=tmp_path/'valid.zip',tmp_path/'invalid.zip'
    save_model_json(_model(ZABB(bd=3)),source)
    _rewrite(source,target,lambda meta:meta.update(family=family,family_constructor_parameters=options))
    report=validate_model_json(target)
    assert not report['ok']
    issue=next(item for item in report['errors'] if item['code']==code)
    assert issue['path'].startswith('meta.family_constructor_parameters')
    with pytest.raises(ValueError) as caught:
        load_model_json(target)
    assert str(caught.value)==issue['message']


@pytest.mark.parametrize('bd',[0,3.,[0,3,17],[[0,1],[2,3]]])
def test_validator_accepts_supported_zibb_options(tmp_path,bd):
    path=tmp_path/'valid.zip'
    save_model_json(_model(ZIBB(bd=bd)),path)
    assert validate_model_json(path)['ok']
    assert load_model_json(path).family.constructor_parameters=={'bd':bd}
