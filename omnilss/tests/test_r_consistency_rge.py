"""R consistency checks for RGE distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestRGEConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_rge_dpq_consistency(self) -> None:
        fam = resolve("RGE")

        for profile_name, x, mu, sigma, nu, p in (
            (
                "central",
                np.linspace(0.01, 10.0, 80),
                np.linspace(0.5, 5.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(0.1, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
            (
                "tail",
                np.linspace(0.01, 10.0, 80),
                np.linspace(0.5, 5.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(0.1, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("RGE", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_p = self.bus.eval_family("RGE", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_q = self.bus.eval_family("RGE", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
