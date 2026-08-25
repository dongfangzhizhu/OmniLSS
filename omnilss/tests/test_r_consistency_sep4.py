"""R consistency checks for SEP4 distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestSEP4Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_sep4_dpq_consistency(self) -> None:
        fam = resolve("SEP4")

        for profile_name, x, mu, sigma, nu, tau, p in (
            (
                "central",
                np.linspace(-5.0, 5.0, 80),
                np.linspace(-2.0, 2.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(1.0, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
            (
                "tail",
                np.linspace(-5.0, 5.0, 80),
                np.linspace(-2.0, 2.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(0.2, 2.0, 80),
                np.linspace(1.0, 5.0, 80),
                np.linspace(0.02, 0.98, 80),
            ),
        ):
            with self.subTest(profile=profile_name):
                # R qSEP 用固定区间 golden section，p 接近 0/1 时
                # 真实分位数超出区间或收敛异常，q 验证收窄到 [0.1, 0.9]
                q_p = np.linspace(0.1, 0.9, 80)
                r_d = self.bus.eval_family("SEP4", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_p = self.bus.eval_family("SEP4", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                r_q = self.bus.eval_family("SEP4", "q", {"p": q_p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(q_p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                # p/q 依赖数值积分（R integrate() vs scipy quad 算法差异），
                # 容差按 roadmap Tier 1 验收标准（rtol <= 1e-3）设定。
                np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
