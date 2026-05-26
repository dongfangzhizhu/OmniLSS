"""R consistency checks for TF2 compatibility family (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestTF2Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_tf2_dpq_consistency_multi_profile(self) -> None:
        profiles = [
            {
                "name": "central",
                "x": np.linspace(-2.5, 2.5, 80),
                "mu": np.linspace(-0.6, 0.6, 80),
                "sigma": np.linspace(0.45, 1.7, 80),
                "nu": np.linspace(2.2, 7.5, 80),
                "p": np.linspace(0.05, 0.95, 80),
                "tol": {"d": (1e-5, 1e-7), "p": (1e-5, 1e-7), "q": (1e-5, 1e-6)},
            },
            {
                "name": "tail",
                "x": np.linspace(-7.0, 7.0, 80),
                "mu": np.linspace(-1.8, 1.8, 80),
                "sigma": np.linspace(0.2, 2.2, 80),
                "nu": np.linspace(1.2, 4.0, 80),
                "p": np.linspace(0.01, 0.99, 80),
                "tol": {"d": (2e-5, 2e-7), "p": (2e-5, 2e-7), "q": (3e-5, 2e-5)},
            },
            {
                "name": "ultra_tail_prob",
                "x": np.linspace(-8.0, 8.0, 80),
                "mu": np.linspace(-1.0, 1.0, 80),
                "sigma": np.linspace(0.12, 1.6, 80),
                "nu": np.linspace(1.05, 3.2, 80),
                "p": np.linspace(1e-4, 1 - 1e-4, 80),
                "tol": {"d": (5e-5, 5e-7), "p": (5e-5, 5e-7), "q": (8e-5, 8e-5)},
            },
        ]

        fam = resolve("TF2")
        for profile in profiles:
            with self.subTest(profile=profile["name"]):
                x = profile["x"]
                mu = profile["mu"]
                sigma = profile["sigma"]
                nu = profile["nu"]
                p = profile["p"]

                r_d = self.bus.eval_family(
                    family="TF2",
                    func_type="d",
                    args={"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
                )
                r_p = self.bus.eval_family(
                    family="TF2",
                    func_type="p",
                    args={"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
                )
                r_q = self.bus.eval_family(
                    family="TF2",
                    func_type="q",
                    args={"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
                )

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))

                tol = profile["tol"]
                np.testing.assert_allclose(py_d, r_d, rtol=tol["d"][0], atol=tol["d"][1])
                np.testing.assert_allclose(py_p, r_p, rtol=tol["p"][0], atol=tol["p"][1])
                np.testing.assert_allclose(py_q, r_q, rtol=tol["q"][0], atol=tol["q"][1])

    def test_tf2_tail_semantics_consistency(self) -> None:
        q = np.linspace(-3.0, 3.0, 40)
        p = np.linspace(0.1, 0.9, 40)
        mu = np.linspace(-0.5, 0.5, 40)
        sigma = np.linspace(0.6, 1.4, 40)
        nu = np.linspace(2.1, 6.0, 40)

        fam = resolve("TF2")

        r_upper = self.bus.eval_family(
            family="TF2",
            func_type="p",
            args={"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "lower.tail": False},
        )
        py_upper = np.asarray(
            fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu), lower_tail=False)
        )
        np.testing.assert_allclose(py_upper, r_upper, rtol=2e-5, atol=2e-7)

        r_q_upper = self.bus.eval_family(
            family="TF2",
            func_type="q",
            args={"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "lower.tail": False},
        )
        py_q_upper = np.asarray(
            fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), lower_tail=False)
        )
        np.testing.assert_allclose(py_q_upper, r_q_upper, rtol=8e-5, atol=8e-5)


if __name__ == "__main__":
    unittest.main()
