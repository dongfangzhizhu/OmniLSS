"""R consistency checks for GAF distribution (d/p/q).

R 的 ``GAF`` 是三参数族（``mu, sigma, nu``，方差函数 ``sigma^2*mu^nu``）；
``nu=2`` 退化为常数方差，所以两个 profile 分别覆盖退化点与真正的方差函数分支。
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestGAFConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_gaf_dpq_consistency(self) -> None:
        fam = resolve("GAF")
        self.assertEqual(fam.parameters, ("mu", "sigma", "nu"))

        x = np.linspace(0.05, 10.0, 80)
        mu = np.linspace(0.5, 5.0, 80)
        sigma = np.linspace(0.2, 2.0, 80)
        p = np.linspace(0.02, 0.98, 80)

        for profile_name, nu in (
            ("constant-variance-nu2", np.full(80, 2.0)),
            ("variance-function", np.linspace(1.0, 3.0, 80)),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("GAF", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_p = self.bus.eval_family("GAF", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_q = self.bus.eval_family("GAF", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
