"""R consistency and API checks for the YULE distribution (d/p/q/r)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import jax.random as jr
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
                np.arange(0, 24),
                np.linspace(0.3, 5.0, 24),
                np.linspace(0.02, 0.95, 24),
            ),
            (
                "tail",
                np.array([0, 1, 2, 5, 10, 25, 50, 100]),
                np.array([0.1, 0.3, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0]),
                np.array([1e-5, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]),
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

    def test_yule_boundaries_and_sampling(self) -> None:
        fam = resolve("YULE")

        np.testing.assert_allclose(
            np.asarray(fam.p(jnp.array([-2.0, -1.0, 0.0, np.inf]), 2.0)),
            np.array([0.0, 0.0, 0.6, 1.0]),
            rtol=1e-15,
            atol=0.0,
        )
        quantiles = np.asarray(fam.q(jnp.array([-0.1, 0.0, 1.0, 1.1]), 2.0))
        self.assertTrue(np.isnan(quantiles[0]))
        self.assertEqual(quantiles[1], 0.0)
        self.assertTrue(np.isinf(quantiles[2]))
        self.assertTrue(np.isnan(quantiles[3]))

        sample = np.asarray(fam.r(jr.PRNGKey(20260828), 128, 2.0))
        self.assertEqual(sample.shape, (128,))
        self.assertTrue(np.all(np.isfinite(sample)))
        self.assertTrue(np.all(sample >= 0.0))
        self.assertTrue(np.all(sample == np.floor(sample)))


if __name__ == "__main__":
    unittest.main()
