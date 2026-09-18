"""Real RS fits for nondefault zero-BB denominators and artifact prediction."""
import numpy as np
import jax.random as jr
import pytest
from omnilss import gamlss
from omnilss.distributions_zero_alias_compat import ZABB,ZIBB
from omnilss.serialization import save_model_json,load_model_json,validate_model_json
from omnilss._fitting_init import _initial_mu_beta,_initial_parameter_value
from omnilss.controls import GAMLSSControl


@pytest.mark.parametrize('factory',[ZABB,ZIBB])
def test_real_fit_prediction_and_roundtrip(tmp_path,factory):
    family=factory(bd=5)
    y=np.asarray(family.r(jr.PRNGKey(17),200,.4,.2,.25))
    model=gamlss('y ~ 1',family=family,data={'y':y},method='RS',
                 control=GAMLSSControl(n_cyc=200,c_crit=1e-7))
    assert np.isfinite(model.g_dev)
    args=[model.fitted_values[p] for p in ('mu','sigma','nu')]
    np.testing.assert_allclose(model.g_dev,-2*np.sum(family.d(y,*args,log=True)),rtol=1e-8)
    # Independent optimizer reference checks that a finite fit is not enough.
    from scipy.optimize import minimize
    from scipy.special import expit
    def objective(eta):
        return -float(np.sum(family.d(y,expit(eta[0]),np.exp(eta[1]),expit(eta[2]),log=True)))
    reference=minimize(objective,[0.,np.log(.2),-1.],method='Nelder-Mead',
                       options={'maxiter':1000,'xatol':1e-8})
    assert reference.success
    assert abs(model.g_dev-2*reference.fun)<.05
    path=tmp_path/'fit.zip'
    save_model_json(model,path)
    assert validate_model_json(path)['ok']
    restored=load_model_json(path)
    assert restored.family.constructor_parameters=={'bd':5}
    data={'y':np.zeros(4)}
    before=model.predict_params(data)
    after=restored.predict_params(data)
    for name in before:
        np.testing.assert_allclose(before[name],after[name],rtol=1e-12)
    pars=[after[p] for p in ('mu','sigma','nu')]
    np.testing.assert_allclose(restored.family.p([0,1,3,5],*pars),family.p([0,1,3,5],*pars))


def test_initialization_uses_count_denominator_and_interior_zero_mass():
    y=np.array([0.,1.,5.]); w=np.ones(3); x=np.ones((3,1)); family=ZIBB(bd=5)
    beta=_initial_mu_beta(family,x,y,w)
    probability=(y+.5)/6
    np.testing.assert_allclose(beta,np.mean(np.log(probability/(1-probability))))
    assert _initial_parameter_value(family,'nu',y,y,w)==pytest.approx(1/3)
    with pytest.raises(ValueError,match='response'):
        _initial_mu_beta(family,x,np.array([0.,1.,6.]),w)
