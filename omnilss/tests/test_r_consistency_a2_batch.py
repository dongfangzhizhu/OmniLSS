"""R consistency checks for A2 batch: BEINF0/BEINF1/BEOI/BEZI/LNO/NET (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestA2BatchConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def _profile_inputs(self, family: str) -> tuple:
        n = 60
        if family in ("BEINF0", "BEINF1"):
            x = np.linspace(0.05, 0.95, n)
            mu = np.linspace(0.2, 0.8, n)
            sigma = np.linspace(0.2, 0.6, n)
            nu = np.linspace(0.2, 1.0, n)
            return x, mu, sigma, nu
        if family in ("BEZI", "BEOI"):
            x = np.linspace(0.05, 0.95, n)
            mu = np.linspace(0.2, 0.8, n)
            sigma = np.linspace(1.5, 5.0, n)
            nu = np.linspace(0.2, 0.8, n)
            return x, mu, sigma, nu
        if family == "LNO":
            x = np.linspace(0.5, 5.0, n)
            mu = np.linspace(0.5, 1.5, n)
            sigma = np.linspace(0.2, 0.8, n)
            nu = np.linspace(0.0, 0.5, n)
            return x, mu, sigma, nu
        # NET
        x = np.linspace(-3.0, 3.0, n)
        mu = np.linspace(-1.0, 1.0, n)
        sigma = np.linspace(0.5, 2.0, n)
        nu = np.linspace(1.5, 3.0, n)
        tau = np.linspace(2.5, 5.0, n)
        return x, mu, sigma, nu, tau

    def _assert_family(self, family: str, npar: int) -> None:
        fam = resolve(family)
        with self.subTest(family=family):
            args = self._profile_inputs(family)
            x, mu, sigma = args[0], args[1], args[2]
            p = np.linspace(0.05, 0.95, len(x))
            if npar == 4:
                nu = args[3]
                tau = args[4] if len(args) > 4 else None
                if tau is not None:
                    r_d = self.bus.eval_family(family, "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                    r_p = self.bus.eval_family(family, "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                    r_q = self.bus.eval_family(family, "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
                    py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                    py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                    py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                else:
                    r_d = self.bus.eval_family(family, "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                    r_p = self.bus.eval_family(family, "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                    r_q = self.bus.eval_family(family, "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                    py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                    py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                    py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
            else:
                nu = args[3]
                r_d = self.bus.eval_family(family, "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_p = self.bus.eval_family(family, "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                r_q = self.bus.eval_family(family, "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))

            np.testing.assert_allclose(py_d, r_d, rtol=1e-4, atol=1e-7)
            np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-7)
            np.testing.assert_allclose(py_q, r_q, rtol=1e-3, atol=1e-6)

    def test_beinf0_dpq(self) -> None:
        self._assert_family("BEINF0", 3)

    def test_beinf1_dpq(self) -> None:
        self._assert_family("BEINF1", 3)

    def test_bezi_dpq(self) -> None:
        self._assert_family("BEZI", 3)

    def test_beoi_dpq(self) -> None:
        self._assert_family("BEOI", 3)

    def test_lno_dpq(self) -> None:
        self._assert_family("LNO", 3)

    def test_net_dpq(self) -> None:
        self._assert_family("NET", 4)


if __name__ == "__main__":
    unittest.main()
