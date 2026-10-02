"""R consistency checks for LQNO distribution (d/p/q).

R 的 ``LQNO`` 是**两参数**族：``Normal(mu, sqrt(mu*(1+sigma*mu)))``，``mu > 0``、
``sigma > 0``（``R/gamlss.dist/R/LQNO.R``，已安装的 6.1.1 与之一致）。

之前这个文件沿用了生成器模板：既传了 ``nu``（R 不接受，直接报
``unused argument``）又使用负的 ``mu``（R 会 ``stop``）。两个 profile 现在覆盖
"mu 靠近 0" 与 "mu 较大" 两支，因为方差 ``mu*(1+sigma*mu)`` 对 mu 是非线性的。
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestLQNOConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_lqno_dpq_consistency(self) -> None:
        fam = resolve("LQNO")
        self.assertEqual(fam.parameters, ("mu", "sigma"))

        x = np.linspace(-5.0, 8.0, 80)
        sigma = np.linspace(0.2, 2.0, 80)
        p = np.linspace(0.02, 0.98, 80)

        for profile_name, mu in (
            ("small-mu", np.linspace(0.3, 1.5, 80)),
            ("large-mu", np.linspace(1.5, 5.0, 80)),
        ):
            with self.subTest(profile=profile_name):
                kwargs = {"mu": mu.tolist(), "sigma": sigma.tolist()}
                r_d = self.bus.eval_family("LQNO", "d", {"x": x.tolist(), **kwargs})
                r_p = self.bus.eval_family("LQNO", "p", {"q": x.tolist(), **kwargs})
                r_q = self.bus.eval_family("LQNO", "q", {"p": p.tolist(), **kwargs})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
