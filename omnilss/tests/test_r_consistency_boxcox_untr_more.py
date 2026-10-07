"""R consistency harness for the untruncated Box-Cox families (D2.16).

What changed and why this file was rewritten
--------------------------------------------
Every test here used to assert that ``BCTUNTR`` / ``BCCGUNTR`` / ``BCPEUNTR``
agree with ``BCT`` / ``BCCG`` / ``BCPE``, by calling R's ``dBCT`` / ``dBCCG`` /
``dBCPE`` as the reference.  That assertion encoded the defect: the three
``*UNTR`` names were aliases of the *truncated* families, so "matches R" was
really "matches the wrong R function".

Two facts settle what the correct reference is:

* In ``gamlss.dist``, ``dBCTo``'s body is byte-identical to ``dBCT``'s
  (``identical(deparse(body(dBCTo)), deparse(body(dBCT)))`` is ``TRUE``); the
  ``o`` suffix only switches ``mu.link``.  So the ``o`` families are not the
  untruncated ones.
* The untruncated families are ``BCTuntr`` / ``BCCGuntr`` / ``BCPEuntr``, and
  they expose **no** ``d`` / ``p`` / ``q`` functions at all -- only
  ``G.dev.incr``.  Comparing our ``d`` against R's ``dBCT`` was therefore never
  a comparison of the same distribution.

So the reference below is R's own ``G.dev.incr`` for the untruncated family,
evaluated element-wise (its ``ifelse(nu != 0, ...)`` collapses to length 1 on a
vectorised call and silently reuses ``y[1]``'s ``z``).  ``p`` / ``q`` have no R
counterpart and are checked against the *untruncated* CDF's defining property
instead -- they must dominate the truncated ones.

See ``test_boxcox_untr_analytic_parity.py`` for the live-R derivative and
end-to-end RS comparisons.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class TestBoxCoxUntrMoreConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def _profile_inputs(self, profile: str, family: str):
        if profile == "central":
            x = np.linspace(0.3, 3.5, 30)
            mu = np.linspace(0.8, 2.0, 30)
            sigma = np.linspace(0.1, 0.6, 30)
            nu = np.linspace(-0.4, 0.4, 30)
            tau = np.linspace(1.2, 2.5, 30) if family == "BCPE" else np.linspace(2.2, 5.0, 30)
        elif profile == "tail":
            x = np.linspace(0.05, 7.0, 30)
            mu = np.linspace(0.3, 4.5, 30)
            sigma = np.linspace(0.05, 1.0, 30)
            nu = np.linspace(-1.5, 1.5, 30)
            tau = np.linspace(1.05, 6.0, 30) if family == "BCPE" else np.linspace(2.05, 20.0, 30)
        elif profile == "ultra_tail":
            x = np.linspace(0.01, 15.0, 30)
            mu = np.linspace(0.15, 8.0, 30)
            sigma = np.linspace(0.03, 1.6, 30)
            nu = np.linspace(-2.2, 2.2, 30)
            tau = np.linspace(1.01, 12.0, 30) if family == "BCPE" else np.linspace(2.01, 35.0, 30)
        else:
            raise ValueError(f"Unknown profile: {profile}")
        return x, mu, sigma, nu, tau

    # -- G.dev.incr against live R ------------------------------------------
    #
    # The grids sweep x across the whole positive axis including values far
    # below mu, where a large |nu| pushes the Box-Cox z past its own support
    # (1 + nu*sigma*z > 0).  Those entries are compared only where both sides
    # are finite; a point where R is finite must not come back non-finite here.

    def _assert_gdev_matches(self, ours: str, r_name: str, n_par: int) -> None:
        fam = resolve(ours)
        for profile in ("central", "tail", "ultra_tail"):
            with self.subTest(profile=profile):
                x, mu, sigma, nu, tau = self._profile_inputs(profile, r_name)
                args = {
                    "y": x.tolist(),
                    "mu": mu.tolist(),
                    "sigma": sigma.tolist(),
                    "nu": nu.tolist(),
                }
                if n_par == 4:
                    args["tau"] = tau.tolist()
                r_dev = self.bus.eval_family(r_name, "gdev", args)
                arrays = [jnp.array(v) for v in (x, mu, sigma, nu)]
                if n_par == 4:
                    arrays.append(jnp.array(tau))
                py_dev = np.asarray(fam.g_dev_inc(*arrays), dtype=np.float64)

                finite = np.isfinite(r_dev) & np.isfinite(py_dev)
                self.assertGreater(
                    int(finite.sum()), 0, f"{ours}/{profile}: nothing comparable"
                )
                np.testing.assert_allclose(
                    py_dev[finite], r_dev[finite], rtol=1e-9, atol=1e-11
                )
                self.assertTrue(
                    np.all(np.isfinite(py_dev[np.isfinite(r_dev)])),
                    f"{ours}/{profile}: non-finite deviance where R is finite",
                )

    def test_bctuntr_gdev_matches_live_r(self) -> None:
        self._assert_gdev_matches("BCTUNTR", "BCTuntr", n_par=4)

    def test_bccguntr_gdev_matches_live_r(self) -> None:
        self._assert_gdev_matches("BCCGUNTR", "BCCGuntr", n_par=3)

    def test_bcpeuntr_gdev_matches_live_r(self) -> None:
        self._assert_gdev_matches("BCPEUNTR", "BCPEuntr", n_par=4)

    # -- The families really are untruncated --------------------------------

    def test_untruncated_deviance_dominates_truncated(self) -> None:
        """``G.dev`` untruncated - ``G.dev`` truncated is the truncation constant.

        ``G.dev = -2 log d``, and dropping the renormalisation *raises* the
        density (mass above ``1/(sigma*|nu|)`` is kept rather than divided
        away), so the untruncated deviance is the **lower** one -- by exactly
        ``2 log F(trunc point)``.  Under the old aliases this difference was
        identically zero.
        """
        pairs = [
            ("BCTUNTR", "BCT", 4),
            ("BCCGUNTR", "BCCG", 3),
            ("BCPEUNTR", "BCPE", 4),
        ]
        x = np.linspace(0.4, 3.5, 17)
        mu, sigma, nu = 2.4, 0.7, 1.3
        for ours, trunc, n_par in pairs:
            with self.subTest(family=ours):
                a = resolve(ours)
                b = resolve(trunc)
                p = (mu, sigma, nu, 3.5) if n_par == 4 else (mu, sigma, nu)
                scalars = [jnp.array(v) for v in p]
                ours_dev = np.asarray(a.g_dev_inc(jnp.array(x), *scalars))
                tr_dev = np.asarray(b.g_dev_inc(jnp.array(x), *scalars))
                offset = ours_dev - tr_dev
                np.testing.assert_allclose(offset, offset[0], rtol=0, atol=1e-11)
                self.assertGreater(
                    offset[0], 1e-3, "difference too small to be a truncation term"
                )

    # -- p / q have no R counterpart; check their defining property ---------

    def test_untruncated_cdf_dominates_truncated(self) -> None:
        import omnilss.dpqr_functions as dpqr

        y = np.linspace(0.6, 3.2, 13)
        mu, sigma, nu = 2.4, 0.7, 1.3
        cases = [
            (dpqr.pBCTuntr, "pBCT", (mu, sigma, nu, 3.5)),
            (dpqr.pBCCGuntr, "pBCCG", (mu, sigma, nu)),
            (dpqr.pBCPEuntr, "pBCPE", (mu, sigma, nu, 2.6)),
        ]
        for untr_fn, trunc_name, p in cases:
            with self.subTest(family=trunc_name):
                yv = jnp.array(y)
                scalars = [jnp.array(v) for v in p]
                untr = np.asarray(untr_fn(yv, *scalars), dtype=np.float64)
                tr = np.asarray(getattr(dpqr, trunc_name)(yv, *scalars), dtype=np.float64)
                self.assertTrue(
                    np.all(untr >= tr - 1e-12),
                    "untruncated CDF must dominate the truncated one pointwise",
                )
                self.assertGreater(np.max(untr - tr), 1e-6)

    def test_untruncated_quantile_reaches_beyond_truncation_point(self) -> None:
        """The truncated ``q`` can never cross ``1/(sigma*|nu|)``.

        That is what makes truncation visible in the quantile function, and it
        is the property the old aliases violated: their ``q`` was R's truncated
        ``q``, so the upper tail was cut at exactly the same place.
        """
        from omnilss.dpqr_functions import qBCCGuntr

        mu, sigma, nu = 2.4, 0.7, 1.3
        trunc_point = 1.0 / (sigma * abs(nu))
        # Largest z the truncated BCCG keeps, and the y that maps to it.
        z_max = trunc_point if nu > 0 else -trunc_point
        y_max = mu * (1.0 + nu * sigma * z_max) ** (1.0 / nu)
        y_untr = float(np.asarray(qBCCGuntr(jnp.array(0.999999), mu, sigma, nu)))
        self.assertGreater(
            y_untr,
            y_max * 1.5,
            f"untruncated q only reached {y_untr}, truncated ceiling is {y_max}",
        )


if __name__ == "__main__":
    unittest.main()