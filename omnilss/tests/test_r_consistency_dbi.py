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
        x = np.arange(0, 30)
        mu = np.linspace(0.1, 0.9, 30)

        # Compatibility phase: DBI routes through BI kernel.
        r_d = self.bus.eval_family("BI", "d", {"x": x.tolist(), "mu": mu.tolist()})
        r_p = self.bus.eval_family("BI", "p", {"q": x.tolist(), "mu": mu.tolist()})
        p = np.linspace(0.05, 0.95, 30)
        r_q = self.bus.eval_family("BI", "q", {"p": p.tolist(), "mu": mu.tolist()})

        fam = resolve("DBI")
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
        py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
        py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

        np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
