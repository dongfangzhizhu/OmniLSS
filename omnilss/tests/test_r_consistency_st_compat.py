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

    def _run_against_r_counterpart(self, family_name: str, *, profile: str) -> None:
        if profile == "central":
            x = np.linspace(-2.0, 2.0, 40)
            mu = np.linspace(-0.3, 0.3, 40)
            sigma = np.linspace(0.5, 1.5, 40)
            # D2.163: R's ST1..ST4 require nu > 0 ("nu must be positive"); a
            # grid through zero made every probe fail on the R side.
            nu = np.linspace(0.2, 2.0, 40)
            tau = np.linspace(2.2, 6.0, 40)
            p = np.linspace(0.05, 0.95, 40)
        elif profile == "tail_skew":
            x = np.linspace(-5.0, 4.5, 40)
            mu = np.linspace(-1.0, 1.0, 40)
            sigma = np.linspace(0.2, 2.0, 40)
            nu = np.linspace(0.2, 3.0, 40)
            tau = np.linspace(2.05, 15.0, 40)
            p = np.linspace(0.01, 0.99, 40)
        else:
            raise ValueError(f"Unknown profile: {profile}")

        # D2.163: compare each family against ITS R counterpart.  The reference
        # used to be hardcoded to ST3C on the assumption that ST1..ST4 "route
        # through ST3C" -- they do not; ST1/ST2/ST4 are different
        # parameterisations, which is why their densities differed from R's
        # ST3C by up to 89% while ST3 (the one that really is ST3C) matched.
        # gamlss.dist exports dST1/dST2/dST3/dST4, so the reference is the
        # family itself.
        reference = "ST3C" if family_name == "ST3" else family_name
        r_d = self.bus.eval_family(reference, "d", {"x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        r_p = self.bus.eval_family(reference, "p", {"q": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        r_q = self.bus.eval_family(reference, "q", {"p": p.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})

        fam = resolve(family_name)
        py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
        py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))
        py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau)))

        np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)

        # D2.166: q is NOT held to 1e-5 against R.  R's qST1/qST2 are uniroot
        # solves at the DEFAULT tolerance (.Machine$double.eps^0.25 = 1.2e-4 on
        # the x scale), so their answer is itself only ~1e-5 accurate; our
        # brentq uses xtol=1e-10.  Measured on this very grid, feeding both
        # quantiles back through R's OWN pST1/pST2:
        #   ST1: |p(q_ours)-p| max 4.0e-10, |p(q_R)-p| max 1.3e-06
        #   ST2: |p(q_ours)-p| max 4.1e-08, |p(q_R)-p| max 7.3e-06
        # i.e. ours is 2-3 orders of magnitude closer to the true inverse and
        # the 1e-5 gap is R's solver noise, not our defect.  Use the tolerance
        # already registered for these families in
        # test_q_uniroot_residual_families.py (D2.32), then assert the real
        # property: our q must invert R's p to high accuracy.
        np.testing.assert_allclose(py_q, r_q, rtol=0.0, atol=1e-4)
        r_p_at_ours_q = self.bus.eval_family(
            reference, "p",
            {"q": np.asarray(py_q).tolist(), "mu": mu.tolist(),
             "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()},
        )
        np.testing.assert_allclose(r_p_at_ours_q, p, rtol=0.0, atol=1e-6)

    def test_st1(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_r_counterpart("ST1", profile=profile)

    def test_st2(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_r_counterpart("ST2", profile=profile)

    def test_st3(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_r_counterpart("ST3", profile=profile)

    def test_st4(self) -> None:
        for profile in ("central", "tail_skew"):
            with self.subTest(profile=profile):
                self._run_against_r_counterpart("ST4", profile=profile)


if __name__ == "__main__":
    unittest.main()
