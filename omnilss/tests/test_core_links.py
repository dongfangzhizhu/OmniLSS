import importlib

import numpy as np
import jax.numpy as jnp
import pytest

pytest.importorskip("optax")
links_mod = importlib.import_module("omnilss.core.links")

IdentityLink = links_mod.IdentityLink
LogLink = links_mod.LogLink
LogitLink = links_mod.LogitLink
SoftplusLink = links_mod.SoftplusLink


def test_identity_link_derivative_is_one():
    link = IdentityLink()
    eta = jnp.array([-2.0, 0.0, 3.0])
    out = link.derivative(eta)
    assert np.allclose(np.asarray(out), np.ones(3))


def test_log_link_round_trip_positive_domain():
    link = LogLink()
    x = jnp.array([1e-3, 0.5, 10.0])
    eta = link.transform(x)
    x2 = link.inverse(eta)
    assert np.allclose(np.asarray(x), np.asarray(x2), rtol=1e-6)


def test_logit_link_derivative_bounds():
    link = LogitLink()
    eta = jnp.array([-20.0, 0.0, 20.0])
    d = np.asarray(link.derivative(eta))
    assert np.all(d >= 0.0)
    assert d[1] <= 0.25 + 1e-12


def test_softplus_link_stability_and_monotonicity():
    link = SoftplusLink()
    eta = jnp.array([-100.0, 0.0, 100.0])
    inv = np.asarray(link.inverse(eta))
    der = np.asarray(link.derivative(eta))
    assert np.isfinite(inv).all()
    assert np.isfinite(der).all()
    assert np.all(np.diff(inv) > 0)
