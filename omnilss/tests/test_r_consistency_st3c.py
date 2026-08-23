"""R consistency checks for ST3C (Skew-t Type 3C, centred parameterization).

A1 batch first deliverable: ST3C's d/p/q/r were placeholder/NaN or used an
incorrect centred-skew-t kernel.  This file verifies the rewritten Jones & Faddy
skew-t kernel (matching R's C code `st3_dst3`/`st3_pst3`/`st3_qst3`) against R
gamlss.dist.

ST3C parameterization: mu (location), sigma (scale>0), nu (skew>0), tau (df>0).
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestST3CConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_st3c_dpq_consistency(self) -> None:
        fam = resolve("ST3C")
        # Profiles covering both skew branches (mu shifted left/right) and tails.
        for profile, mu, sigma, nu, tau in (
            ("central_left", -1.0, 1.0, 0.8, 10.0),
            ("central_right", 1.5, 0.7, 2.0, 6.0),
            ("heavy_tail", 0.0, 2.0, 0.5, 3.0),
            ("near_normal", 0.5, 1.5, 5.0, 30.0),
        ):
            with self.subTest(profile=profile):
                x = np.linspace(-8.0, 8.0, 60)
                p_vals = np.linspace(0.01, 0.99, 60)
                n = len(x)
                mu_arr = [mu] * n
                sigma_arr = [sigma] * n
                nu_arr = [nu] * n
                tau_arr = [tau] * n

                r_d = self.bus.eval_family("ST3C", "d", {"x": x.tolist(), "mu": mu_arr, "sigma": sigma_arr, "nu": nu_arr, "tau": tau_arr})
                r_p = self.bus.eval_family("ST3C", "p", {"q": x.tolist(), "mu": mu_arr, "sigma": sigma_arr, "nu": nu_arr, "tau": tau_arr})
                r_q = self.bus.eval_family("ST3C", "q", {"p": p_vals.tolist(), "mu": [mu] * len(p_vals), "sigma": [sigma] * len(p_vals), "nu": [nu] * len(p_vals), "tau": [tau] * len(p_vals)})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
                py_q = np.asarray(fam.q(jnp.array(p_vals), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-4, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-4, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-4, atol=1e-7)

    def test_st3c_dpq_roundtrip(self) -> None:
        """p(q(p)) and q(p(q)) self-consistency checks."""
        fam = resolve("ST3C")
        mu, sigma, nu, tau = 0.5, 1.0, 1.5, 8.0
        p_vals = np.linspace(0.02, 0.98, 40)
        q_vals = np.asarray(fam.q(jnp.array(p_vals), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
        p_back = np.asarray(fam.p(q_vals, jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
        np.testing.assert_allclose(p_back, p_vals, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    unittest.main()