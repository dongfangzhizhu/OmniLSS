"""R consistency checks for ST1/ST2/ST3/ST4 compatibility families (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestST1234Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def _profile_inputs(self, profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if profile == "central":
            x = np.linspace(-2.5, 2.5, 50)
            mu = np.linspace(-0.8, 0.8, 50)
            sigma = np.linspace(0.4, 1.6, 50)
            nu = np.linspace(-0.8, 0.8, 50)
            tau = np.linspace(1.2, 5.0, 50)
        elif profile == "tail":
            x = np.linspace(-10.0, 10.0, 50)
            mu = np.linspace(-2.0, 2.0, 50)
            sigma = np.linspace(0.2, 2.5, 50)
            nu = np.linspace(-2.2, 2.2, 50)
            tau = np.linspace(1.05, 20.0, 50)
        elif profile == "ultra_tail":
            x = np.linspace(-25.0, 25.0, 50)
            mu = np.linspace(-5.0, 5.0, 50)
            sigma = np.linspace(0.08, 4.0, 50)
            nu = np.linspace(-4.0, 4.0, 50)
            tau = np.linspace(1.01, 60.0, 50)
        else:
            raise ValueError(f"Unknown profile: {profile}")
        p = np.linspace(1e-5, 1.0 - 1e-5, 50)
        return x, mu, sigma, nu, tau, p

    def _assert_family_consistency(self, family_name: str) -> None:
        fam = resolve(family_name)
        for profile in ("central", "tail", "ultra_tail"):
            with self.subTest(family=family_name, profile=profile):
                x, mu, sigma, nu, tau, p = self._profile_inputs(profile)
                r_d = self.bus.eval_family(family_name, "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_p = self.bus.eval_family(family_name, "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_q = self.bus.eval_family(family_name, "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

                np.testing.assert_allclose(py_d, r_d, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=2e-5, atol=1e-7)

    def test_st1_dpq_consistency(self) -> None:
        self._assert_family_consistency("ST1")

    def test_st2_dpq_consistency(self) -> None:
        self._assert_family_consistency("ST2")

    def test_st3_dpq_consistency(self) -> None:
        self._assert_family_consistency("ST3")

    def test_st4_dpq_consistency(self) -> None:
        self._assert_family_consistency("ST4")


if __name__ == "__main__":
    unittest.main()
