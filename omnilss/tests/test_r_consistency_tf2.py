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
        # TF2 semantic domain in R is nu>2; keep profile grids strictly above 2
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
                "nu": np.linspace(2.02, 4.0, 80),
                "p": np.linspace(0.01, 0.99, 80),
                "tol": {"d": (2e-5, 2e-7), "p": (2e-5, 2e-7), "q": (3e-5, 2e-5)},
            },
            {
                "name": "ultra_tail_prob",
                "x": np.linspace(-8.0, 8.0, 80),
                "mu": np.linspace(-1.0, 1.0, 80),
                "sigma": np.linspace(0.12, 1.6, 80),
                "nu": np.linspace(2.001, 3.2, 80),
                "p": np.linspace(1e-4, 1 - 1e-4, 80),
                "tol": {"d": (5e-5, 5e-7), "p": (5e-5, 5e-7), "q": (8e-5, 8e-5)},
            },
            {
                "name": "near_nu_boundary",
                "x": np.linspace(-9.0, 9.0, 80),
                "mu": np.linspace(-2.0, 2.0, 80),
                "sigma": np.linspace(0.08, 2.4, 80),
                "nu": np.linspace(2.000001, 2.08, 80),
                "p": np.linspace(5e-5, 1 - 5e-5, 80),
                "tol": {"d": (9e-5, 9e-7), "p": (9e-5, 9e-7), "q": (1.2e-4, 1.2e-4)},
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

    def test_tf2_logp_semantics_consistency(self) -> None:
        q = np.linspace(-3.5, 3.5, 48)
        mu = np.linspace(-0.7, 0.7, 48)
        sigma = np.linspace(0.5, 1.5, 48)
        nu = np.linspace(2.05, 6.5, 48)

        fam = resolve("TF2")

        r_logp = self.bus.eval_family(
            family="TF2",
            func_type="p",
            args={"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "log.p": True},
        )
        py_logp = np.asarray(
            fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu), log_p=True)
        )
        np.testing.assert_allclose(py_logp, r_logp, rtol=3e-5, atol=3e-7)

        r_logp_upper = self.bus.eval_family(
            family="TF2",
            func_type="p",
            args={
                "q": q.tolist(),
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "nu": nu.tolist(),
                "log.p": True,
                "lower.tail": False,
            },
        )
        py_logp_upper = np.asarray(
            fam.p(
                jnp.array(q),
                jnp.array(mu),
                jnp.array(sigma),
                jnp.array(nu),
                log_p=True,
                lower_tail=False,
            )
        )
        np.testing.assert_allclose(py_logp_upper, r_logp_upper, rtol=4e-5, atol=4e-7)

    def test_tf2_p_lower_upper_complement_consistency(self) -> None:
        q = np.linspace(-4.0, 4.0, 60)
        mu = np.linspace(-0.8, 0.8, 60)
        sigma = np.linspace(0.5, 1.6, 60)
        nu = np.linspace(2.02, 6.8, 60)

        fam = resolve("TF2")
        py_lower = np.asarray(fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
        py_upper = np.asarray(
            fam.p(jnp.array(q), jnp.array(mu), jnp.array(sigma), jnp.array(nu), lower_tail=False)
        )
        np.testing.assert_allclose(py_lower + py_upper, np.ones_like(py_lower), rtol=1e-6, atol=1e-6)

        r_lower = self.bus.eval_family(
            family="TF2",
            func_type="p",
            args={"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )
        r_upper = self.bus.eval_family(
            family="TF2",
            func_type="p",
            args={"q": q.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "lower.tail": False},
        )
        np.testing.assert_allclose(r_lower + r_upper, np.ones_like(r_lower), rtol=1e-7, atol=1e-7)

    def test_tf2_q_logp_semantics_consistency(self) -> None:
        p = np.linspace(0.05, 0.95, 42)
        p_log = np.log(p)
        mu = np.linspace(-0.6, 0.6, 42)
        sigma = np.linspace(0.55, 1.45, 42)
        nu = np.linspace(2.02, 6.2, 42)

        fam = resolve("TF2")

        r_q_log = self.bus.eval_family(
            family="TF2",
            func_type="q",
            args={"p": p_log.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "log.p": True},
        )
        py_q_log = np.asarray(
            fam.q(jnp.array(p_log), jnp.array(mu), jnp.array(sigma), jnp.array(nu), log_p=True)
        )
        np.testing.assert_allclose(py_q_log, r_q_log, rtol=8e-5, atol=8e-5)

        r_q_log_upper = self.bus.eval_family(
            family="TF2",
            func_type="q",
            args={
                "p": p_log.tolist(),
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "nu": nu.tolist(),
                "log.p": True,
                "lower.tail": False,
            },
        )
        py_q_log_upper = np.asarray(
            fam.q(
                jnp.array(p_log),
                jnp.array(mu),
                jnp.array(sigma),
                jnp.array(nu),
                log_p=True,
                lower_tail=False,
            )
        )
        np.testing.assert_allclose(py_q_log_upper, r_q_log_upper, rtol=9e-5, atol=9e-5)

    def test_tf2_q_p_roundtrip_against_r_profiled(self) -> None:
        profiles = [
            {
                "name": "core",
                "p": np.linspace(0.05, 0.95, 50),
                "mu": np.linspace(-0.7, 0.7, 50),
                "sigma": np.linspace(0.5, 1.5, 50),
                "nu": np.linspace(2.05, 7.0, 50),
                "tol": (1.2e-4, 1.2e-4),
            },
            {
                "name": "tail_heavy",
                "p": np.linspace(0.005, 0.995, 50),
                "mu": np.linspace(-1.5, 1.5, 50),
                "sigma": np.linspace(0.25, 2.0, 50),
                "nu": np.linspace(2.001, 4.0, 50),
                "tol": (2.5e-4, 2.5e-4),
            },
        ]

        fam = resolve("TF2")
        for profile in profiles:
            with self.subTest(profile=profile["name"]):
                pvals = profile["p"]
                mu = profile["mu"]
                sigma = profile["sigma"]
                nu = profile["nu"]

                x_py = np.asarray(fam.q(jnp.array(pvals), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                p_back_py = np.asarray(fam.p(jnp.array(x_py), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                np.testing.assert_allclose(p_back_py, pvals, rtol=profile["tol"][0], atol=profile["tol"][1])

                x_r = self.bus.eval_family(
                    family="TF2",
                    func_type="q",
                    args={"p": pvals.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
                )
                p_back_r = self.bus.eval_family(
                    family="TF2",
                    func_type="p",
                    args={"q": np.asarray(x_r).tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
                )
                np.testing.assert_allclose(p_back_r, pvals, rtol=profile["tol"][0], atol=profile["tol"][1])


if __name__ == "__main__":
    unittest.main()
