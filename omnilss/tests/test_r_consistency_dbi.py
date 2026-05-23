"""R consistency checks for DBI compatibility family (currently BI-backed)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestDBIConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_dbi_dpq_consistency_against_bi_reference(self) -> None:
        fam = resolve("DBI")
        for profile_name, x, mu, p in (
            (
                "mid_range",
                np.arange(0, 30),
                np.linspace(0.1, 0.9, 30),
                np.linspace(0.01, 0.99, 30),
            ),
            (
                "edge_heavy",
                np.arange(0, 30),
                np.linspace(0.01, 0.99, 30),
                np.linspace(0.001, 0.999, 30),
            ),
            (
                "ultra_tail",
                np.arange(0, 30),
                np.concatenate(
                    [np.linspace(1e-4, 0.05, 15), np.linspace(0.95, 1.0 - 1e-4, 15)]
                ),
                np.concatenate(
                    [np.geomspace(1e-6, 1e-2, 15), 1.0 - np.geomspace(1e-2, 1e-6, 15)]
                ),
            ),
        ):
            with self.subTest(profile=profile_name):
                # Compatibility phase: DBI routes through BI kernel.
                r_d = self.bus.eval_family("BI", "d", {"x": x.tolist(), "mu": mu.tolist()})
                r_p = self.bus.eval_family("BI", "p", {"q": x.tolist(), "mu": mu.tolist()})
                r_q = self.bus.eval_family("BI", "q", {"p": p.tolist(), "mu": mu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
