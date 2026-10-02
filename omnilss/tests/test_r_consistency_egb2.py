"""R consistency checks for EGB2 distribution (d/p/q).

R 的 ``EGB2``：``z = (x-mu)/sigma``，``mu`` 是全实数轴上的位置参数
（``mu.link = "identity"``），``sigma/nu/tau > 0``。

之前这个文件沿用了生成器模板：``nu`` 取了 ``[-2, 2]``，R 对 ``nu < 0`` 会直接
``stop``（``nu must be positive``）。两个 profile 现在分别覆盖 ``nu < tau`` 与
``nu > tau``，因为左右尾的衰减率由 ``nu``/``tau`` 各自控制。
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestEGB2Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_egb2_dpq_consistency(self) -> None:
        fam = resolve("EGB2")
        self.assertEqual(fam.parameters, ("mu", "sigma", "nu", "tau"))

        x = np.linspace(-5.0, 5.0, 80)
        mu = np.linspace(-2.0, 2.0, 80)
        sigma = np.linspace(0.2, 2.0, 80)
        p = np.linspace(0.02, 0.98, 80)

        for profile_name, nu, tau in (
            # ``nu >= 0.5``：df1 = 2*nu >= 1。更小的 nu（如 0.1）会让
            # ``qf(0.02, 0.2, 2)`` 在 R 中下溢为 0（返回 -inf），而 scipy 给出
            # 真实的 ~1e-16 —— 属于病理参数域（F 分布自由度 < 1）的数值边界，
            # 不是参数化差异，所以网格从 nu = 0.5 起步。
            ("nu-lt-tau", np.linspace(0.5, 1.0, 80), np.linspace(1.0, 5.0, 80)),
            ("nu-gt-tau", np.linspace(1.0, 5.0, 80), np.linspace(0.5, 1.0, 80)),
        ):
            with self.subTest(profile=profile_name):
                kwargs = {"mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()}
                r_d = self.bus.eval_family("EGB2", "d", {"x": x.tolist(), **kwargs})
                r_p = self.bus.eval_family("EGB2", "p", {"q": x.tolist(), **kwargs})
                r_q = self.bus.eval_family("EGB2", "q", {"p": p.tolist(), **kwargs})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
