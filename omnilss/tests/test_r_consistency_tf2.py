"""R consistency checks for TF2 compatibility family (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestTF2Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_tf2_d_consistency(self) -> None:
        x = np.linspace(-2.0, 2.0, 60)
        mu = np.linspace(-0.5, 0.5, 60)
        sigma = np.linspace(0.5, 1.5, 60)
        nu = np.linspace(2.5, 8.0, 60)

        r_d = self.bus.eval_family(
            family="TF",
            func_type="d",
            args={"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )
        fam = resolve("TF2")
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
        np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)

    def test_tf2_pq_consistency(self) -> None:
        q = np.linspace(-2.0, 2.0, 60)
        p = np.linspace(0.05, 0.95, 60)
        mu = np.linspace(-0.5, 0.5, 60)
        sigma = np.linspace(0.5, 1.5, 60)
        nu = np.linspace(2.5, 8.0, 60)

        r_p = self.bus.eval_family(
            family="TF",
            func_type="p",
            args={"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )
        r_q = self.bus.eval_family(
            family="TF",
            func_type="q",
            args={"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )
        fam = resolve("TF2")
        py_p = np.asarray(fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
        py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
        np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
