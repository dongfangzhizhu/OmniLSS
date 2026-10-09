"""R consistency checks for ZAZIPF distribution (d/p/q).

D2.27 note: R names the zero mass of ``ZAZIPF`` ``sigma`` -- ``dZAZIPF(x, mu,
sigma)``, ``parameters = list(mu=TRUE, sigma=TRUE)`` -- while this package
names it ``nu`` (the D2.26 zero-mass parameter naming).  The R side of this
comparison must therefore pass ``sigma``, or live R rejects the call with
"unused argument (nu = ...)"; the Python side passes positionally, so the
naming difference never reaches it.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestZAZIPFConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_zazipf_dpq_consistency(self) -> None:
        fam = resolve("ZAZIPF")

        for profile_name, x, mu, zero_mass, p in (
            (
                "central",
                np.arange(0, 50),
                np.linspace(0.5, 5.0, 50),
                np.linspace(0.1, 0.9, 50),
                np.linspace(0.02, 0.98, 50),
            ),
            (
                "tail",
                np.arange(0, 50),
                np.linspace(0.5, 5.0, 50),
                np.linspace(0.1, 0.9, 50),
                np.linspace(0.02, 0.98, 50),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("ZAZIPF", "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": zero_mass.tolist()})
                r_p = self.bus.eval_family("ZAZIPF", "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": zero_mass.tolist()})
                r_q = self.bus.eval_family("ZAZIPF", "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": zero_mass.tolist()})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(zero_mass)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(zero_mass)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(zero_mass)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
