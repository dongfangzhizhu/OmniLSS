import numpy as np
import jax
import pytest
from omnilss.distributions_zero_alias_compat import ZIBB
from omnilss.zibb import dZIBB,pZIBB,qZIBB,rZIBB


def test_mass_and_fixed_denominator_interface():
    f=ZIBB(bd=5)
    mass=np.asarray(f.d(np.arange(6),.3,.2,.25))
    assert mass.sum()==pytest.approx(1.,abs=1e-12)
    np.testing.assert_allclose(np.cumsum(mass),f.p(np.arange(6),.3,.2,.25),atol=1e-12)
    np.testing.assert_array_equal(f.q([0,1],.3,.2,.25),[0,5])
    np.testing.assert_array_equal(f.d([-1,.5,6],.3,.2,.25),0)


def test_tails_crossings_invalid_and_degenerate():
    p=np.array([.01,.5,.9,.99999])
    q=np.asarray(qZIBB(p,.3,.2,.25,10))
    assert np.all(pZIBB(q,.3,.2,.25,10)>=p)
    assert np.all(pZIBB(q-1,.3,.2,.25,10)<p)
    np.testing.assert_array_equal(qZIBB(np.log1p(-p),.3,.2,.25,10,False,True),q)
    assert pZIBB(-1,lower_tail=False)==1
    assert pZIBB(10,lower_tail=False,log_p=True)==-np.inf
    assert np.isnan(pZIBB(-1,mu=0))
    assert np.isnan(dZIBB(0,nu=1))
    np.testing.assert_array_equal(qZIBB([0,.5,1],bd=0),0)
    np.testing.assert_array_equal(rZIBB(jax.random.PRNGKey(0),20,bd=0),0)


def test_broadcast_jit_sampling_distribution():
    assert pZIBB([[0],[1]],mu=[.2,.8]).shape==(2,2)
    np.testing.assert_allclose(jax.jit(dZIBB)(0.,.3,.2,.25,10),dZIBB(0.,.3,.2,.25,10))
    sample=np.asarray(rZIBB(jax.random.PRNGKey(11),10000,.3,.2,.25,10),dtype=int)
    mass=np.asarray(dZIBB(np.arange(11),.3,.2,.25,10))
    actual=np.bincount(sample,minlength=11)/10000
    assert np.all(abs(actual-mass)<6*np.sqrt(mass*(1-mass)/10000)+.0002)
