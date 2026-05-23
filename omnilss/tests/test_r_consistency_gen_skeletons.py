"""R consistency harness for representative gen-family skeleton compatibility paths."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestGenSkeletonConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_count_skeleton_against_po_reference(self) -> None:
        x = np.arange(0, 50)
        mu = np.linspace(0.2, 8.0, 50)
        p = np.linspace(0.05, 0.95, 50)

        r_d = self.bus.eval_family("PO", "d", {"x": x.tolist(), "mu": mu.tolist()})
        r_p = self.bus.eval_family("PO", "p", {"q": x.tolist(), "mu": mu.tolist()})
        r_q = self.bus.eval_family("PO", "q", {"p": p.tolist(), "mu": mu.tolist()})

        fam = resolve("COUNT_1_31")
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
        py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
        py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

        np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)

    def test_continuous_skeleton_against_no_reference(self) -> None:
        x = np.linspace(-2.0, 2.0, 60)
        mu = np.linspace(-0.5, 0.5, 60)
        sigma = np.linspace(0.5, 1.5, 60)
        p = np.linspace(0.05, 0.95, 60)

        r_d = self.bus.eval_family("NO", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        r_p = self.bus.eval_family("NO", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        r_q = self.bus.eval_family("NO", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})

        fam = resolve("CONTRPLUS_2_11")
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
        py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
        py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma)))

        np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
