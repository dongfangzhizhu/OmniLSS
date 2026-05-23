"""R consistency harness for ST1..ST4 compatibility families."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestSTCompatConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def _run_against_st3c_reference(self, family_name: str, *, profile: str) -> None:
        if profile == "central":
            x = np.linspace(-2.0, 2.0, 40)
            mu = np.linspace(-0.3, 0.3, 40)
            sigma = np.linspace(0.5, 1.5, 40)
            nu = np.linspace(-0.4, 0.4, 40)
            tau = np.linspace(2.2, 6.0, 40)
            p = np.linspace(0.05, 0.95, 40)
        elif profile == "tail_skew":
            x = np.linspace(-5.0, 4.5, 40)
            mu = np.linspace(-1.0, 1.0, 40)
            sigma = np.linspace(0.2, 2.0, 40)
            nu = np.linspace(-1.0, 1.0, 40)
            tau = np.linspace(2.05, 15.0, 40)
            p = np.linspace(0.01, 0.99, 40)
        else:
            raise ValueError(f"Unknown profile: {profile}")

        # Compatibility phase: ST1..4 currently route through ST3C kernel.
        r_d = self.bus.eval_family("ST3C", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        r_p = self.bus.eval_family("ST3C", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        r_q = self.bus.eval_family("ST3C", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})

        fam = resolve(family_name)
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
        py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
        py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

        np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)

    def test_st1(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_st3c_reference("ST1", profile=profile)

    def test_st2(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_st3c_reference("ST2", profile=profile)

    def test_st3(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_st3c_reference("ST3", profile=profile)

    def test_st4(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_st3c_reference("ST4", profile=profile)


if __name__ == "__main__":
    unittest.main()
