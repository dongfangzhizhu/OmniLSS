"""PIG/BNB tail boundaries must answer what live R answers, not ``nan``.

D2.29.  Three defects that the parity scanner cannot see (its probe grids
stop at moderate ``q`` and ``p <= 0.999``) but that every real user of the
quantile/CDF API hits:

* ``pPIG`` answered ``nan`` for ``q >= 10001`` where live R returns its
  accumulated partial sum -- R prints ``1.0000000000000004`` there, its own
  summation ulp above 1.  The old "bounded support" nan cutoff is gone.
* ``qPIG`` lacked R's upper-tail short circuit
  (``if (p[i] + 1e-09 >= 1) QQQ[i] <- Inf``, checked after the tail
  conversion): ``qPIG(0.9999999999)`` scanned to the pmf cap and answered
  ``62`` where R answers ``Inf``.  Live R: ``0.999999999`` is already Inf,
  ``0.999`` still resolves.
* ``pBNB(0)`` carried an ~8.7e-10 error (conducted into ``ZABNB``/``ZIBNB``)
  rooted not in the formula but in ``jax.scipy.special.betaln``: the
  library call is off by ~3.5e-8 absolute at moderate arguments while jax's
  ``gammaln`` is accurate to ~1e-14, so the textbook
  ``gammaln(a) + gammaln(b) - gammaln(a+b)`` composition beats it by five
  orders of magnitude.  Both BNB density implementations (``dBNB`` and the
  AD ``_bnb_log_pdf``) were switched together -- fixing only one side would
  re-create the density/derivative split.

Live-R baseline (R 4.6.0, ``format(digits=17)``):

* ``pPIG(q, 1, 1)``: ``50 -> 0.99999999998924471``, ``10001/1e6 ->
  1.0000000000000004``; grid ``(mu=4, sigma=0.25)``:
  ``0.053493066571363601 / 0.17703001211038499 / 0.34026768055719148 /
  0.50449085496303481 / 0.75504908452163777 / 0.97005314091032102 /
  0.99965678073494901``.
* ``qPIG(p, 4, 0.25)``: ``0.01/0.1/0.25/0.3/0.5/0.9/0.999 ->
  0/1/2/2/3/8/18``.
* ``pBNB(q, 5, 0.4, 2)``: ``0.33967009652792979 / 0.486079620893417 /
  0.58124581173098355 / 0.74502746281308918 / 0.86599839192252182``.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnilss.dpqr_functions import dBNB, pBNB
from omnilss.distributions_b7 import _bnb_log_pdf
from omnilss.pig import pPIG, qPIG

_PIG_Q_GRID = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
_PIG_P_GRID = [0.01, 0.1, 0.25, 0.3, 0.5, 0.9, 0.999]
_BNB_Q_GRID = [0.0, 1.0, 2.0, 5.0, 10.0]


class TestPIGTailBoundaries(unittest.TestCase):
    """The 10000-count recursion is an internal detail, not the domain."""

    def test_the_cdf_answers_a_huge_q_with_the_partial_sum(self) -> None:
        # R: pPIG(10001/1e6, 1, 1) = 1.0000000000000004 (its summation ulp
        # above 1); our log-scale clamp lands on exactly 1.0, within 4e-16.
        for q in (10001.0, 1e6):
            with self.subTest(q=q):
                value = float(pPIG(jnp.asarray(q), jnp.asarray(1.0),
                                   jnp.asarray(1.0)))
                self.assertTrue(np.isfinite(value), q)
                self.assertAlmostEqual(value, 1.0, places=12)

    def test_the_cdf_still_saturates_gradually_below_the_cap(self) -> None:
        # R: 0.99999999998924471 -- the partial sum, not yet clamped.
        value = float(pPIG(jnp.asarray(50.0), jnp.asarray(1.0),
                           jnp.asarray(1.0)))
        self.assertAlmostEqual(value, 0.99999999998924471, places=13)

    def test_the_quantile_short_circuits_the_upper_tail_like_r(self) -> None:
        # R checks `p + 1e-09 >= 1` after the tail conversion: 0.999999999 is
        # already Inf, 0.999 still resolves to the grid value.
        for p in (0.999999999, 0.9999999999, 1.0):
            with self.subTest(p=p):
                value = float(qPIG(jnp.asarray(p), jnp.asarray(1.0),
                                   jnp.asarray(1.0)))
                self.assertTrue(np.isposinf(value), p)
        self.assertEqual(
            float(qPIG(jnp.asarray(0.999), jnp.asarray(1.0),
                       jnp.asarray(1.0))), 10.0)

    def test_the_short_circuit_runs_after_the_tail_conversion(self) -> None:
        # R checks the converted probability: lower_tail=FALSE turns the
        # argument into a lower-tail one first (1 - p) and only then applies
        # the Inf short circuit, so a moderate upper-tail p still inverts
        # through the scan.  Live R upper-tail grid (mu=1, sigma=1):
        # 0.05/0.1/0.2/0.3/0.5/0.7/0.9 -> 4/3/2/1/1/0/0 -- and a near-one
        # upper-tail p converts to a near-zero lower-tail p, staying finite.
        grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        ours = np.asarray(qPIG(jnp.asarray(grid), jnp.asarray(1.0),
                               jnp.asarray(1.0), lower_tail=False))
        np.testing.assert_allclose(ours, [4.0, 3.0, 2.0, 1.0, 1.0, 0.0, 0.0],
                                   rtol=0, atol=0)
        self.assertFalse(np.isposinf(
            float(qPIG(jnp.asarray(0.999999999), jnp.asarray(1.0),
                       jnp.asarray(1.0), lower_tail=False))))


class TestBNBZeroMassPrecision(unittest.TestCase):
    """pBNB(0) must agree with live R beyond the old 8.7e-10 error."""

    def test_the_zero_mass_matches_live_r_to_ulp_level(self) -> None:
        ours = float(pBNB(jnp.asarray(0.0), jnp.asarray(5.0),
                          jnp.asarray(0.4), jnp.asarray(2.0)))
        # R: 0.33967009652792979; we land within 3e-15 after the fix
        # (previously off by 8.7e-10 through jax.scipy.special.betaln).
        self.assertLess(abs(ours - 0.33967009652792979), 1e-13)

    def test_the_cdf_grid_matches_live_r(self) -> None:
        ours = np.asarray(pBNB(jnp.asarray(_BNB_Q_GRID), jnp.asarray(5.0),
                               jnp.asarray(0.4), jnp.asarray(2.0)))
        expected = [0.33967009652792979, 0.486079620893417,
                    0.58124581173098355, 0.74502746281308918,
                    0.86599839192252182]
        np.testing.assert_allclose(ours, expected, rtol=1e-12)

    def test_the_ad_density_stays_in_lockstep_with_dpqr(self) -> None:
        # Same density, two implementations (dpqr_functions.dBNB and the AD
        # _bnb_log_pdf): they must switch betaln together or the fix
        # re-creates the density/derivative split.
        for (mu, sigma, nu) in ((5.0, 0.4, 2.0), (2.0, 1.0, 1.0),
                                (10.0, 0.2, 0.5)):
            with self.subTest(mu=mu, sigma=sigma, nu=nu):
                y = jnp.asarray([0.0, 1.0, 3.0])
                m, s, n = (jnp.asarray(v) for v in (mu, sigma, nu))
                ad = np.asarray(_bnb_log_pdf(y, m, s, n))
                dpqr = np.asarray(dBNB(y, m, s, n, log=True))
                np.testing.assert_allclose(ad, dpqr, rtol=0, atol=1e-12)


@unittest.skipUnless(True, "R availability is decided in setUpClass")
class TestAgainstLiveR(unittest.TestCase):
    """Element-for-element comparison against live R 4.6.0."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from tests.rbus.core import RTestBus

            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_pig_cdf_and_quantile_match_live_r(self) -> None:
        kwargs = {"mu": 4.0, "sigma": 0.25}
        r_p = self.bus.eval_family("PIG", "p", {"q": _PIG_Q_GRID, **kwargs})
        r_q = self.bus.eval_family("PIG", "q", {"p": _PIG_P_GRID, **kwargs})

        ours_p = np.asarray(pPIG(jnp.asarray(_PIG_Q_GRID), jnp.asarray(4.0),
                                 jnp.asarray(0.25)))
        ours_q = np.asarray(qPIG(jnp.asarray(_PIG_P_GRID), jnp.asarray(4.0),
                                 jnp.asarray(0.25)))

        np.testing.assert_allclose(ours_p, r_p, rtol=1e-9, atol=1e-12)
        # Quantiles are integers; compare exactly.
        np.testing.assert_allclose(ours_q, r_q, rtol=0, atol=0)

    def test_bnb_cdf_matches_live_r(self) -> None:
        kwargs = {"mu": 5.0, "sigma": 0.4, "nu": 2.0}
        r_p = self.bus.eval_family("BNB", "p", {"q": _BNB_Q_GRID, **kwargs})
        ours_p = np.asarray(pBNB(jnp.asarray(_BNB_Q_GRID), jnp.asarray(5.0),
                                 jnp.asarray(0.4), jnp.asarray(2.0)))
        np.testing.assert_allclose(ours_p, r_p, rtol=1e-9, atol=1e-12)


class TestZAPStartValueAndDensity(unittest.TestCase):
    """ZAP's start value is live R's, and its density stays self-consistent."""

    def test_the_sigma_start_value_matches_live_r(self) -> None:
        # Live R: ZAP()$sigma.initial = rep(0.3, length(y)).  We previously
        # shared 0.1 with ZIP2/ZINBI -- 1.35 logit units below R.
        from omnilss._fitting_init import _initial_parameter_value
        from omnilss.distributions import resolve_family

        fam = resolve_family("ZAP")
        y = np.array([0.0, 1.0, 2.0, 5.0])
        mu = np.full(4, 2.0)
        w = np.ones(4)
        value = _initial_parameter_value(fam, "sigma", y, mu, w)
        self.assertEqual(float(value), 0.3)

    def test_the_density_stays_self_consistent_on_vector_input(self) -> None:
        # Live R's dZAP has a vectorisation artifact: dPO(0, mu) collapses to
        # length 1 (its `ifelse(x < 0, 0, fy)` test is the scalar x), so R's
        # own dZAP recycles the *first observation's* truncation term across
        # every row whenever mu varies -- its fitted objective on "y ~ x" is
        # not its own pointwise formula.  Single-point R matches the formula
        # exactly, and so do we; this test pins that we keep the pointwise
        # formula on vector input instead of inheriting the recycle (D2.30
        # adjudication: an implementation slip, not a convention, so it is
        # not reproduced).  The hurdle normalisation *divides* by 1-F(0), so
        # the truncation term enters with a minus sign.
        from jax.scipy.special import gammaln

        from omnilss.distributions import resolve_family

        fam = resolve_family("ZAP")
        y = jnp.asarray([0.0, 1.0, 3.0, 5.0, 2.0, 4.0])
        mu = jnp.asarray([3.68, 3.69, 3.70, 3.71, 3.72, 3.73])
        sigma = jnp.asarray(0.0625)
        ours = np.asarray(fam.d(y, mu, sigma, log=True))

        t1 = np.log1p(-0.0625)
        t2 = np.asarray(y * jnp.log(mu) - mu - gammaln(y + 1.0))
        trunc = np.log1p(-np.exp(-np.asarray(mu)))
        expected = np.where(np.asarray(y) == 0, np.log(0.0625),
                            t1 + t2 - trunc)
        np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
