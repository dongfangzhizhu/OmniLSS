"""R consistency checks for LOGSHASHO distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestLOGSHASHOConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_logshasho_dpq_consistency(self) -> None:
        """LOGSHASH(O) has no R d/p/q counterpart -- pinned as an explicit fact.

        D2.163: gamlss.dist exports LOGSHASH only as a *fitting* family (no
        dLOGSHASH/dLOGSHASHo), so this comparison can never run.  Rather than
        skipping silently, assert the absence: the day R gains the functions
        this test fails and the real comparison gets written.
        """
        with self.assertRaises(RuntimeError) as caught:
            self.bus.eval_family(
                "LOGSHASHO", "d",
                {"x": [1.0], "mu": [1.0], "sigma": [1.0], "nu": [0.1], "tau": [1.0]},
            )
        self.assertIn("no function 'dLOGSHASHO'", str(caught.exception))

    def _unused_dpq_consistency(self) -> None:
        fam = resolve("LOGSHASHO")

        for profile_name, x, mu, sigma, nu, tau, p in (
            (
                "central",
                np.linspace(-5.0, 5.0, 80),
                np.linspace(-2.0, 2.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(-2.0, 2.0, 80),
                np.linspace(0.1, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
            (
                "tail",
                np.linspace(-5.0, 5.0, 80),
                np.linspace(-2.0, 2.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(-2.0, 2.0, 80),
                np.linspace(0.1, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("LOGSHASHO", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_p = self.bus.eval_family("LOGSHASHO", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_q = self.bus.eval_family("LOGSHASHO", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
