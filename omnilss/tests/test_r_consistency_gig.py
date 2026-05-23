"""R consistency checks for GIG distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distributions import resolve_family
from tests.rbus.core import RTestBus


class TestGIGConsistency(unittest.TestCase):
    """Compare OmniLSS GIG d/p/q against R gamlss.dist::GIG."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_gig_d_consistency(self) -> None:
        x = np.linspace(0.05, 5.0, 80)
        mu = np.linspace(0.5, 2.5, 80)
        sigma = np.linspace(0.2, 0.8, 80)
        nu = np.linspace(-0.5, 1.0, 80)

        r_d = self.bus.eval_family(
            family="GIG",
            func_type="d",
            args={"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )

        fam = resolve_family("GIG")
        py_d = np.asarray(
            fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu))
        )

        np.testing.assert_allclose(py_d, r_d, rtol=1e-4, atol=1e-6)

    def test_gig_p_consistency(self) -> None:
        q = np.linspace(0.05, 5.0, 80)
        mu = np.linspace(0.5, 2.5, 80)
        sigma = np.linspace(0.2, 0.8, 80)
        nu = np.linspace(-0.5, 1.0, 80)

        r_p = self.bus.eval_family(
            family="GIG",
            func_type="p",
            args={"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )

        fam = resolve_family("GIG")
        py_p = np.asarray(
            fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu))
        )

        np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-6)

    def test_gig_q_consistency(self) -> None:
        p = np.linspace(0.05, 0.95, 80)
        mu = np.linspace(0.5, 2.5, 80)
        sigma = np.linspace(0.2, 0.8, 80)
        nu = np.linspace(-0.5, 1.0, 80)

        r_q = self.bus.eval_family(
            family="GIG",
            func_type="q",
            args={"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )

        fam = resolve_family("GIG")
        py_q = np.asarray(
            fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu))
        )

        np.testing.assert_allclose(py_q, r_q, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
