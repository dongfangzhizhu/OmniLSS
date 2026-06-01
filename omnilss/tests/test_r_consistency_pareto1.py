"""R consistency checks for PARETO1 distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestPARETO1Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_pareto1_dpq_consistency(self) -> None:
        fam = resolve("PARETO1")

        for profile_name, x, mu, p in (
            (
                "central",
                np.linspace(1.01, 50.0, 80),
                np.linspace(0.1, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
            (
                "tail",
                np.linspace(1.01, 50.0, 80),
                np.linspace(0.1, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("PARETO1", "d", {"x": x.tolist(), "mu": mu.tolist()})
                r_p = self.bus.eval_family("PARETO1", "p", {"q": x.tolist(), "mu": mu.tolist()})
                r_q = self.bus.eval_family("PARETO1", "q", {"p": p.tolist(), "mu": mu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
