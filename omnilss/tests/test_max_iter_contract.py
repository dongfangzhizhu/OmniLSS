"""Public iteration overrides must reach the selected solver."""
import numpy as np
import pytest
from types import SimpleNamespace
from omnilss import gamlss
from omnilss.controls import GAMLSSControl

DATA={'y':np.array([1.,2.,1.5,3.,2.2]),'x':np.arange(5.)}


@pytest.mark.parametrize('method,module,function,extra',[
    ('RS','omnilss.algorithms.rs_algorithm','rs_fit',{}),
    ('auto','omnilss.algorithms.rs_algorithm','rs_fit',{}),
    ('MIXED','omnilss.algorithms.mixed_algorithm','mixed_fit',{}),
    ('CG','omnilss.algorithms.cg_algorithm_v2','cg_fit_v2',{'cg_backend':'irls_cross'}),
    ('RS_JAX','omnilss.algorithms.jax_rs_integration','gamlss_rs_jax',{}),
])
@pytest.mark.parametrize('explicit,cycles,expected',[(None,None,20),(None,7,7),(3,None,3),(np.int64(3),7,3)])
def test_solver_receives_override(monkeypatch,method,module,function,extra,explicit,cycles,expected):
    import importlib
    received={}
    def fake(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(additional_slots={})
    monkeypatch.setattr(importlib.import_module(module),function,fake)
    control=GAMLSSControl(n_cyc=cycles,c_crit=.002) if cycles else None
    gamlss('y ~ x',family='NO',data=DATA,method=method,control=control,max_iter=explicit,**extra)
    if method=='RS_JAX':
        assert (received['control'].n_cyc if received['control'] else 20)==expected
    elif method=='MIXED':
        assert received['n1']==1
        assert received['n2']==expected
        assert received['tol']==(.002 if control else 1e-4)
    else:
        assert received['max_iter']==expected
        assert received['tol']==(.002 if control else 1e-4)
    if control:
        assert control.n_cyc==cycles


@pytest.mark.parametrize('value',[0,-1,True,False,1.5,'3',np.nan,np.inf])
def test_invalid_iteration_limit_rejected(value):
    with pytest.raises(ValueError,match='max_iter must be a positive integer'):
        gamlss('y ~ x',family='NO',data=DATA,max_iter=value)


def test_real_rs_limit_matches_control():
    explicit=gamlss('y ~ x',family='NO',data=DATA,max_iter=1)
    controlled=gamlss('y ~ x',family='NO',data=DATA,control=GAMLSSControl(n_cyc=1,c_crit=1e-4))
    assert explicit.iter==controlled.iter==1
    np.testing.assert_allclose(explicit.g_dev,controlled.g_dev)


@pytest.mark.parametrize('method,module,function',[
    ('CG','omnilss.fitting_cg','fit_cg'),
    ('JOINT','omnilss.core.gamlss_integration','fit_with_joint_optimizer'),
    ('LBFGS','omnilss.core.gamlss_integration','fit_with_lbfgs'),
])
def test_other_solver_limits(monkeypatch,method,module,function):
    import importlib
    class ReachedSolver(Exception):
        pass
    control=GAMLSSControl(n_cyc=7)
    def fake(**kwargs):
        assert kwargs['max_iter']==3
        raise ReachedSolver()
    monkeypatch.setattr(importlib.import_module(module),function,fake)
    with pytest.raises(ReachedSolver):
        gamlss('y ~ x',family='NO',data=DATA,method=method,control=control,max_iter=3)
    assert control.n_cyc==7
