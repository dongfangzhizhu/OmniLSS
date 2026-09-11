"""R consistency checks for PIG2 distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestPIG2Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_pig2_dpq_consistency(self) -> None:
        fam = resolve("PIG2")

        for profile_name, x, mu, sigma, p in (
            (
                "central",
                np.arange(80),
                np.linspace(0.5, 5.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
            (
                "tail",
                np.arange(600, 680),
                np.linspace(0.5, 5.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("PIG2", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
                r_p = self.bus.eval_family("PIG2", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
                r_q = self.bus.eval_family("PIG2", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)

    def test_pig_dpq_consistency(self) -> None:
        fam = resolve("PIG")
        mu = np.array([.5, 3., 20., 800.])
        sigma = np.array([.01, .5, 2., .01])
        for kind, arg, values in (
            ('d', 'x', [0, 3, 30, 850]),
            ('p', 'q', [0, 3, 30, 850]),
            ('q', 'p', [.01, .5, .95, .999]),
        ):
            reference = self.bus.eval_family('PIG', kind, {
                arg: values, 'mu': mu.tolist(), 'sigma': sigma.tolist()})
            np.testing.assert_allclose(getattr(fam, kind)(jnp.array(values), mu, sigma),
                                       reference, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
