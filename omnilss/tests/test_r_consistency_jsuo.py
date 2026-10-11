"""R consistency checks for JSUO distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestJSUOConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_jsuo_dpq_consistency(self) -> None:
        fam = resolve("JSUO")

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
                r_d = self.bus.eval_family("JSUo", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_p = self.bus.eval_family("JSUo", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_q = self.bus.eval_family("JSUo", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)

    def test_jsuo_r_is_quantile_of_uniforms(self) -> None:
        """Our rJSUo matches R's construction: uniforms through qJSUo.

        R's rJSUo is ``qJSUo(runif(n), ...)``; the uniforms cannot be matched
        across streams, so pin the composition itself.
        """
        from omnilss.dpqr_functions import pJSUo, qJSUo, rJSUo

        key = jax.random.PRNGKey(20261011)
        u = jax.random.uniform(key, shape=(64,), dtype=jnp.float64)
        for mu, sigma, nu, tau in (
            (0.0, 1.0, 0.0, 1.0),
            (-2.0, 0.5, 1.5, 2.5),
            (3.0, 2.0, -1.0, 0.7),
        ):
            r = np.asarray(rJSUo(key, 64, mu, sigma, nu, tau))
            q = np.asarray(qJSUo(u, mu, sigma, nu, tau))
            np.testing.assert_allclose(np.sort(r), np.sort(q), rtol=1e-12)

            # p and q are exact inverses in R's construction (continuous,
            # strictly monotone CDF): p(q(p)) = p on the open unit interval.
            p_grid = np.linspace(0.02, 0.98, 25)
            round_trip = np.asarray(
                pJSUo(qJSUo(jnp.array(p_grid), mu, sigma, nu, tau),
                      mu, sigma, nu, tau)
            )
            np.testing.assert_allclose(round_trip, p_grid, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
