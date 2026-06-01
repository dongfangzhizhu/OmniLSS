"""R consistency checks for DPO1 distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestDPO1Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_dpo1_dpq_consistency(self) -> None:
        fam = resolve("DPO1")

        for profile_name, x, mu, sigma, p in (
            (
                "central",
                np.arange(0, 50),
                np.linspace(0.5, 10.0, 50),
                np.linspace(0.1, 2.0, 50),
                np.linspace(0.02, 0.98, 50),
            ),
            (
                "tail",
                np.arange(0, 50),
                np.linspace(0.5, 10.0, 50),
                np.linspace(0.1, 2.0, 50),
                np.linspace(0.02, 0.98, 50),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("DPO1", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
                r_p = self.bus.eval_family("DPO1", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
                r_q = self.bus.eval_family("DPO1", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
