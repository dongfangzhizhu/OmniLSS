"""R consistency checks for YULE distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestYULEConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_yule_dpq_consistency(self) -> None:
        fam = resolve("YULE")

        for profile_name, x, mu, p in (
            (
                "central",
                np.arange(0, 80),
                np.linspace(0.3, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
            (
                "tail",
                np.arange(0, 120),
                np.linspace(0.05, 20.0, 120),
                np.concatenate([
                    np.geomspace(1e-5, 1e-2, 60),
                    1.0 - np.geomspace(1e-2, 1e-5, 60),
                ]),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("YULE", "d", {"x": x.tolist(), "mu": mu.tolist()})
                r_p = self.bus.eval_family("YULE", "p", {"q": x.tolist(), "mu": mu.tolist()})
                r_q = self.bus.eval_family("YULE", "q", {"p": p.tolist(), "mu": mu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
