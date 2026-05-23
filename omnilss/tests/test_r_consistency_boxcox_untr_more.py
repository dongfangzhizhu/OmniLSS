"""R consistency harness for BCPEuntr/BCTuntr compatibility families."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestBoxCoxUntrMoreConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def _profile_inputs(self, profile: str, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if profile == "central":
            x = np.linspace(0.3, 3.5, 30)
            mu = np.linspace(0.8, 2.0, 30)
            sigma = np.linspace(0.1, 0.6, 30)
            nu = np.linspace(-0.4, 0.4, 30)
            tau = np.linspace(1.2, 2.5, 30) if family == "BCPE" else np.linspace(2.2, 5.0, 30)
        elif profile == "tail":
            x = np.linspace(0.05, 7.0, 30)
            mu = np.linspace(0.3, 4.5, 30)
            sigma = np.linspace(0.05, 1.0, 30)
            nu = np.linspace(-1.5, 1.5, 30)
            tau = np.linspace(1.05, 6.0, 30) if family == "BCPE" else np.linspace(2.05, 20.0, 30)
        else:
            raise ValueError(f"Unknown profile: {profile}")
        return x, mu, sigma, nu, tau

    def test_bcpeuntr_d_consistency_against_bcpe_reference(self) -> None:
        fam = resolve("BCPEUNTR")
        for profile in ("central", "tail"):
            with self.subTest(profile=profile):
                x, mu, sigma, nu, tau = self._profile_inputs(profile, "BCPE")
                r_d = self.bus.eval_family("BCPE", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)

    def test_bcpeuntr_pq_consistency_against_bcpe_reference(self) -> None:
        fam = resolve("BCPEUNTR")
        for profile in ("central", "tail"):
            with self.subTest(profile=profile):
                q, mu, sigma, nu, tau = self._profile_inputs(profile, "BCPE")
                p = np.linspace(0.01, 0.99, 30)
                r_p = self.bus.eval_family("BCPE", "p", {"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_q = self.bus.eval_family("BCPE", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                py_p = np.asarray(fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)

    def test_bctuntr_d_consistency_against_bct_reference(self) -> None:
        fam = resolve("BCTUNTR")
        for profile in ("central", "tail"):
            with self.subTest(profile=profile):
                x, mu, sigma, nu, tau = self._profile_inputs(profile, "BCT")
                r_d = self.bus.eval_family("BCT", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)

    def test_bctuntr_pq_consistency_against_bct_reference(self) -> None:
        fam = resolve("BCTUNTR")
        for profile in ("central", "tail"):
            with self.subTest(profile=profile):
                q, mu, sigma, nu, tau = self._profile_inputs(profile, "BCT")
                p = np.linspace(0.01, 0.99, 30)
                r_p = self.bus.eval_family("BCT", "p", {"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_q = self.bus.eval_family("BCT", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                py_p = np.asarray(fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
