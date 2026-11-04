"""R consistency checks for ZABNB distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestZABNBConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_zabnb_dpq_consistency(self) -> None:
        """ZABNB has FOUR parameters: mu, sigma, nu (BNB shape) and tau (zero mass).

        D2.163: this test passed three of them -- and unpacked nu twice -- so
        our family raised "_bnb_log_pdf() missing 1 required positional
        argument: 'nu'" and nothing was compared.  R's dZABNB signature is
        (x, mu, sigma, nu, tau = 0.1) with tau as the hurdle mass, which is
        exactly our parameter order (the factory names the zero mass tau
        precisely because BNB already owns nu).
        """
        fam = resolve("ZABNB")
        for profile_name, mu, sigma, nu, tau, p in (
            (
                "central",
                np.linspace(0.5, 10.0, 50),
                np.linspace(0.2, 2.0, 50),
                np.linspace(0.1, 0.9, 50),
                np.linspace(0.1, 0.6, 50),
                np.linspace(0.02, 0.98, 50),
            ),
            (
                "tail",
                np.linspace(0.5, 10.0, 50),
                np.linspace(0.1, 0.5, 50),
                np.linspace(0.2, 3.0, 50),
                np.linspace(0.05, 0.9, 50),
                np.linspace(0.02, 0.98, 50),
            ),
        ):
            x = np.arange(0, 50)
            probs = {"mu": mu, "sigma": sigma, "nu": nu, "tau": tau}
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family(
                    "ZABNB", "d",
                    {"x": x.tolist(), **{k: v.tolist() for k, v in probs.items()}},
                )
                r_p = self.bus.eval_family(
                    "ZABNB", "p",
                    {"q": x.tolist(), **{k: v.tolist() for k, v in probs.items()}},
                )
                r_q = self.bus.eval_family(
                    "ZABNB", "q",
                    {"p": p.tolist(), **{k: v.tolist() for k, v in probs.items()}},
                )

                args = [jnp.array(v) for v in (mu, sigma, nu, tau)]
                py_d = np.asarray(fam.d(jnp.array(x), *args))
                py_p = np.asarray(fam.p(jnp.array(x), *args))
                py_q = np.asarray(fam.q(jnp.array(p), *args))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
