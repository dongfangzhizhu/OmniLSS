import numpy as np
import jax
import pytest
from omnilss.distributions_zero_alias_compat import ZABB
from omnilss.zabb import dZABB,pZABB,qZABB,rZABB


def test_normalization_tiny_positive_mass_and_fixed_bd():
    f=ZABB(bd=5)
    for mu in [.3,1e-20]:
        mass=np.asarray(f.d(np.arange(6),mu,.2,.25))
        assert mass.sum()==pytest.approx(1.,abs=1e-12)
        assert mass[0]==pytest.approx(.25)
        np.testing.assert_allclose(np.cumsum(mass),f.p(np.arange(6),mu,.2,.25),atol=1e-12)
    with pytest.raises(ValueError,match='positive static integer'):
        ZABB(bd=0)
    np.testing.assert_allclose(dZABB([0,1],.3,.2,.25,1),[.25,.75])


def test_boundary_broadcast_and_crossing():
    p=np.array([0,.1,.5,.9,1])
    q=np.asarray(qZABB(p,.3,.2,.25,5))
    assert q[0]==0 and q[-1]==5
    assert np.all(np.asarray(pZABB(q,.3,.2,.25,5))>=p-1e-14)
    assert np.all(np.asarray(pZABB(q[1:-1]-1,.3,.2,.25,5))<p[1:-1])
    with np.errstate(divide='ignore'):
        np.testing.assert_array_equal(qZABB(np.log1p(-p),.3,.2,.25,5,False,True),q)
    np.testing.assert_array_equal(dZABB([-1,.5,6],bd=5),0)
    assert pZABB(-1,lower_tail=False)==1
    assert np.isnan(dZABB(0,mu=0))
    assert np.isnan(pZABB(-1,nu=1))
    assert np.isnan(qZABB(-1))
    assert pZABB([[0],[1]],mu=[.2,.8]).shape==(2,2)


def test_jit_gradient_and_sampling():
    np.testing.assert_allclose(jax.jit(dZABB)(2.,.3,.2,.25),dZABB(2.,.3,.2,.25))
    assert np.isfinite(jax.grad(lambda m: dZABB(2.,m,.2,.25,log=True))(.3))
    sample=np.asarray(rZABB(jax.random.PRNGKey(11),5000,.3,.2,.25),dtype=int)
    expected=np.asarray(dZABB(np.arange(11),.3,.2,.25))
    frequency=np.bincount(sample,minlength=11)/5000
    assert np.all(abs(frequency-expected)<6*np.sqrt(expected*(1-expected)/5000)+.0004)
