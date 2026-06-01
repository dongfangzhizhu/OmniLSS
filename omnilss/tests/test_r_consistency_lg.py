"""R consistency checks for LG distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestLGConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_lg_dpq_consistency(self) -> None:
        fam = resolve("LG")

        for profile_name, x, mu, p in (
            (
                "central",
                np.arange(1, 51),
                np.linspace(0.05, 0.95, 50),
                np.linspace(0.02, 0.98, 50),
            ),
            (
                "tail",
                np.arange(1, 51),
                np.linspace(0.05, 0.95, 50),
                np.linspace(0.02, 0.98, 50),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("LG", "d", {"x": x.tolist(), "mu": mu.tolist()})
                r_p = self.bus.eval_family("LG", "p", {"q": x.tolist(), "mu": mu.tolist()})
                r_q = self.bus.eval_family("LG", "q", {"p": p.tolist(), "mu": mu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
