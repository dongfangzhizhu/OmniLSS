"""R consistency checks for DBURR12 compatibility family (SICHEL-backed)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestDBURR12Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_dburr12_dpq_consistency_against_sichel_reference(self) -> None:
        fam = resolve("DBURR12")
        for profile_name, mu, sigma, nu, p in (
            (
                "central",
                np.linspace(0.5, 8.0, 40),
                np.linspace(0.2, 1.0, 40),
                np.linspace(-0.3, 0.8, 40),
                np.linspace(0.05, 0.95, 40),
            ),
            (
                "tail_edge",
                np.linspace(0.1, 15.0, 40),
                np.linspace(0.05, 2.0, 40),
                np.linspace(-0.95, 1.2, 40),
                np.linspace(0.01, 0.99, 40),
            ),
        ):
            with self.subTest(profile=profile_name):
                x = np.arange(0, 40)

                # Compatibility phase: DBURR12 routes through SICHEL kernel.
                r_d = self.bus.eval_family("SICHEL", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_p = self.bus.eval_family("SICHEL", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_q = self.bus.eval_family("SICHEL", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
