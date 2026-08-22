"""R consistency checks for the three unregistered families (A0.3).

Tests that OmniLSS produces results consistent with R GAMLSS for the three
families that were previously constructible but NOT registered in the
distribution registry:

- BEo       : Beta original parameterization (mu=shape1, sigma=shape2)
- GEOMo     : Geometric original parameterization (mu=p success probability)
- PARETO2o  : Pareto Type 2 original (mu=scale, sigma=shape)

Each family is checked for d/p/q against R via RTestBus.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestBeoGeomoPareto2oConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_resolve_succeeds(self) -> None:
        # The core A0.3 requirement: all three families resolve by name.
        for name in ("BEo", "GEOMo", "PARETO2o"):
            fam = resolve(name)
            self.assertEqual(fam.name, name)
            self.assertTrue(callable(fam.d))
            self.assertTrue(callable(fam.p))
            self.assertTrue(callable(fam.q))

    def test_beo_dpq(self) -> None:
        fam = resolve("BEo")
        # mu=2.0 (alpha), sigma=3.0 (beta)
        r_d = self.bus.eval_family("BEo", "d", {"x": [0.1, 0.3, 0.5], "mu": [2.0, 2.0, 2.0], "sigma": [3.0, 3.0, 3.0]})
        r_p = self.bus.eval_family("BEo", "p", {"q": [0.1, 0.3, 0.5], "mu": [2.0, 2.0, 2.0], "sigma": [3.0, 3.0, 3.0]})
        r_q = self.bus.eval_family("BEo", "q", {"p": [0.1, 0.3, 0.5], "mu": [2.0, 2.0, 2.0], "sigma": [3.0, 3.0, 3.0]})
        py_d = np.asarray(fam.d(jnp.array([0.1, 0.3, 0.5]), jnp.array(2.0), jnp.array(3.0)))
        py_p = np.asarray(fam.p(jnp.array([0.1, 0.3, 0.5]), jnp.array(2.0), jnp.array(3.0)))
        py_q = np.asarray(fam.q(jnp.array([0.1, 0.3, 0.5]), jnp.array(2.0), jnp.array(3.0)))
        np.testing.assert_allclose(py_d, r_d, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-4, atol=1e-6)

    def test_geomo_dpq(self) -> None:
        fam = resolve("GEOMo")
        x = np.arange(0, 10, dtype=float).tolist()
        q = np.arange(0, 10, dtype=float).tolist()
        p_vals = np.linspace(0.1, 0.9, 9).tolist()
        r_d = self.bus.eval_family("GEOMo", "d", {"x": x, "mu": [0.3] * 10})
        r_p = self.bus.eval_family("GEOMo", "p", {"q": q, "mu": [0.3] * 10})
        r_q = self.bus.eval_family("GEOMo", "q", {"p": p_vals, "mu": [0.3] * 9})
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(0.3)))
        py_p = np.asarray(fam.p(jnp.array(q), jnp.array(0.3)))
        py_q = np.asarray(fam.q(jnp.array(p_vals), jnp.array(0.3)))
        np.testing.assert_allclose(py_d, r_d, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-4, atol=1e-6)

    def test_pareto2o_dpq(self) -> None:
        fam = resolve("PARETO2o")
        # mu=2.0 (scale), sigma=3.0 (shape)
        x = np.linspace(0.5, 20.0, 30).tolist()
        p_vals = np.linspace(0.05, 0.95, 19).tolist()
        r_d = self.bus.eval_family("PARETO2o", "d", {"x": x, "mu": [2.0] * 30, "sigma": [3.0] * 30})
        r_p = self.bus.eval_family("PARETO2o", "p", {"q": x, "mu": [2.0] * 30, "sigma": [3.0] * 30})
        r_q = self.bus.eval_family("PARETO2o", "q", {"p": p_vals, "mu": [2.0] * 19, "sigma": [3.0] * 19})
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(2.0), jnp.array(3.0)))
        py_p = np.asarray(fam.p(jnp.array(x), jnp.array(2.0), jnp.array(3.0)))
        py_q = np.asarray(fam.q(jnp.array(p_vals), jnp.array(2.0), jnp.array(3.0)))
        np.testing.assert_allclose(py_d, r_d, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()