"""R consistency checks for ZISICHEL distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestZISICHELConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_zisichel_dpq_consistency(self) -> None:
        fam = resolve("ZISICHEL")

        for profile_name, x, mu, sigma, nu, p in (
            (
                "central",
                np.arange(0, 50),
                np.linspace(0.5, 10.0, 50),
                np.linspace(0.1, 0.9, 50),
                np.linspace(0.1, 0.9, 50),
                np.linspace(0.02, 0.98, 50),
            ),
            (
                "tail",
                np.arange(0, 50),
                np.linspace(0.5, 10.0, 50),
                np.linspace(0.2, 2.0, 50),
                np.linspace(0.1, 0.9, 50),
                np.linspace(0.02, 0.98, 50),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("ZISICHEL", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": nu.tolist()})
                r_p = self.bus.eval_family("ZISICHEL", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": nu.tolist()})
                r_q = self.bus.eval_family("ZISICHEL", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": nu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(nu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(nu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(nu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)

    def test_zisichel_tail_and_log_probability(self) -> None:
        fam = resolve("ZISICHEL")
        x = jnp.arange(0., 8.)
        args = {"q": x.tolist(), "mu": 2., "sigma": .7, "nu": .2, "tau": .35}
        for lower_tail in (True, False):
            r = self.bus.eval_family("ZISICHEL", "p", {**args, "lower.tail": lower_tail})
            py = np.asarray(fam.p(x, 2., .7, .2, .35, lower_tail=lower_tail))
            np.testing.assert_allclose(py, r, rtol=1e-5, atol=1e-7)
        r_log = self.bus.eval_family("ZISICHEL", "p", {**args, "log.p": True})
        np.testing.assert_allclose(np.asarray(fam.p(x, 2., .7, .2, .35, log_p=True)), r_log,
                                   rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
