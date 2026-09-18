"""Shared smooth detection must govern mixed dispatch at the size boundary."""
import numpy as np
import pytest
from types import SimpleNamespace
from omnilss import gamlss
from omnilss.formula_parser import has_smooth_terms
from omnilss.fitting import _has_smooth_terms
from omnilss.algorithms import mixed_algorithm as mixed
from omnilss.smoother_admission import UnsupportedSmootherError


@pytest.mark.parametrize('term',['s(x)','pb (x)','ps\t(x)','cs(x)','lo(x)','te(x,z)','random(x)'])
@pytest.mark.parametrize('parameter',['mu','sigma','nu','tau'])
def test_all_parameter_formulas_route_to_rs(monkeypatch,term,parameter):
    assert has_smooth_terms('~ '+term)==_has_smooth_terms('~ '+term)==True
    calls=[]
    def rs(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(additional_slots={})
    def other(**kwargs):
        pytest.fail('smooth formula routed to non-smooth backend')
    monkeypatch.setattr(mixed,'rs_fit',rs)
    monkeypatch.setattr(mixed,'cg_fit',other)
    model=mixed.mixed_fit('y ~ '+term if parameter=='mu' else 'y ~ 1',
        data={'y':np.ones(500)},parameter_formulas={} if parameter=='mu' else {parameter:'~ '+term})
    assert len(calls)==1
    assert model.additional_slots['mixed_algorithm_used']=='rs'


@pytest.mark.parametrize('formula',[None,'y ~ x + z','y ~ abs(x)','y ~ mypb(x)'])
def test_non_smooth_expressions(formula):
    assert not has_smooth_terms(formula)


def test_real_generic_smooth_matches_rs():
    x=np.linspace(-1,1,60)
    data={'x':x,'y':np.sin(3*x)+np.random.default_rng(8).normal(0,.1,60)}
    a=gamlss('y ~ s(x)',family='NO',data=data,method='MIXED',max_iter=2)
    b=gamlss('y ~ s(x)',family='NO',data=data,method='RS',max_iter=2)
    np.testing.assert_allclose(a.fitted_values['mu'],b.fitted_values['mu'])
    assert a.additional_slots['mixed_algorithm_used']=='rs'


def test_unadmitted_smoother_reaches_admission_gate():
    x=np.linspace(0,1,500)
    with pytest.raises(UnsupportedSmootherError):
        mixed.mixed_fit('y ~ lo(x)',family='NO',data={'x':x,'y':x+1})
