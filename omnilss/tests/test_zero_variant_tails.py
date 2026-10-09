"""The zero variants' ``p``/``q``/``r`` must be composed, not placeholders.

D2.27.  ``build_ad_family`` answers every ``p``/``q``/``r`` call with an
all-NaN placeholder when the caller does not supply explicit dpqr functions.
Both zero-variant factories never supplied any: the five ``ZA`` families
(``ZAPIG``, ``ZAZIPF``, ``ZABI``, ``ZANBI``, ``ZABNB``) and three of the
``ZI`` ones (``ZIPIG``, ``ZIBI``, plus ``ZINBI`` elsewhere) carried the
placeholder, while ``ZIBNB`` carried a hand-written BNB-only block computing
exactly what the general rule computes.  A zero variant's CDF is a closed-form
affine map of its base CDF -- the hurdle for ``ZA``, the linear mixture for
``ZI`` -- so the composition is derivable and this file pins it in place.

Two subtleties the tests exist to keep:

* **The tail-flag order is R's, not the natural one.**  R clamps ``q < 0`` to
  zero *after* taking ``log(cdf)``, so ``p(-1, log.p = TRUE)`` is ``0``, not
  ``-Inf``.  A reader re-ordering those lines would pass every CDF test and
  silently change that one cell.

* **``ZAZIPF`` is the degenerate case that proves the rule.**  Its base
  ``ZIPF`` starts at 1, so ``F(0) == 0`` and the hurdle formula collapses onto
  the zero-inflated one.  R gets that for free from its own formula; the
  composition gets it the same way, and the R-backed comparison below holds
  the two together.

The R-side parameter naming is *not* ours for three families: ``ZAZIPF``,
``ZABI`` and ``ZIBI`` call the zero mass ``sigma`` in gamlss.dist (while this
package follows D2.26 and calls it ``nu``), and the two binomial ones also
take a ``bd`` that defaults to 1 -- which is exactly the Bernoulli our
``_binomial_log_pdf`` implements.  The comparison passes R's names.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from omnilss.distributions_b10_zero_variants import (
    _QUANTILE_EPSILON,
    _QUANTILE_EPSILON_1E7,
    _ZI_FAMILIES_WITH_THE_COARSE_EPSILON,
)

# (family, R argument names for the zero mass, probe parameters).
# ``None`` zero-mass name means R already calls it ``nu`` like we do.
_PROBES = {
    # family: (R zero-mass name, mu, sigma, zero mass, [base nu])
    "ZAPIG": (None, 4.0, 0.3, 0.25, None),
    "ZAZIPF": ("sigma", 3.0, None, 0.25, None),
    "ZABI": ("sigma", 0.4, None, 0.2, None),      # bd defaults to 1 in R
    "ZANBI": (None, 5.0, 0.4, 0.3, None),
    "ZABNB": ("tau", 5.0, 0.4, 0.3, 2.0),
    "ZIPIG": (None, 4.0, 0.3, 0.25, None),
    "ZIBI": ("sigma", 0.4, None, 0.2, None),      # bd defaults to 1 in R
    "ZIBNB": ("tau", 5.0, 0.4, 0.3, 2.0),
    # Not factory-built: pZINBI/qZINBI/rZINBI were written with the family and
    # never passed to build_ad_family, so `.p` answered with the placeholder.
    "ZINBI": (None, 5.0, 0.4, 0.3, None),
}

_QUANTILE_GRID = [0.01, 0.1, 0.3, 0.5, 0.9, 0.999]


def _kwargs(family: str) -> dict[str, float]:
    """The R argument dict for one probe, using R's own parameter names."""
    zero_name, mu, sigma, zero_mass, base_nu = _PROBES[family]
    out: dict[str, float] = {"mu": mu}
    if sigma is not None:
        out["sigma"] = sigma
    if base_nu is not None:
        out["nu"] = base_nu
    out[zero_name or "nu"] = zero_mass
    return out


class TestCompositionReplacesThePlaceholders(unittest.TestCase):
    """No factory-built variant may answer a real question with NaN."""

    def test_every_factory_variant_has_a_real_cdf_and_quantile(self) -> None:
        for family, (zero_name, mu, sigma, zero_mass, base_nu) in _PROBES.items():
            with self.subTest(family=family):
                fam = resolve(family)
                params = [jnp.asarray(mu)]
                if sigma is not None:
                    params.append(jnp.asarray(sigma))
                if base_nu is not None:
                    params.append(jnp.asarray(base_nu))
                params.append(jnp.asarray(zero_mass))

                cdf = np.asarray(fam.p(jnp.asarray([0.0, 1.0, 2.0, 5.0]), *params))
                quantile = np.asarray(
                    fam.q(jnp.asarray(_QUANTILE_GRID), *params))
                self.assertFalse(np.all(np.isnan(cdf)), family)
                self.assertFalse(np.all(np.isnan(quantile)), family)

    def test_the_placeholder_makers_are_not_back(self) -> None:
        # Guard against a future factory signature change dropping the
        # composition again: the placeholder answers *everything* with NaN,
        # including q = 0, where a real CDF must return the zero mass.
        for family, (_, mu, sigma, zero_mass, base_nu) in _PROBES.items():
            with self.subTest(family=family):
                fam = resolve(family)
                params = [jnp.asarray(mu)]
                if sigma is not None:
                    params.append(jnp.asarray(sigma))
                if base_nu is not None:
                    params.append(jnp.asarray(base_nu))
                params.append(jnp.asarray(zero_mass))
                at_zero = float(np.asarray(fam.p(jnp.asarray(0.0), *params)))
                self.assertTrue(np.isfinite(at_zero), family)

    def test_the_zero_mass_sits_where_the_variant_puts_it(self) -> None:
        # The two variants disagree about q = 0 *by construction*: the hurdle
        # (ZA) moves the whole zero mass there, the mixture (ZI) only scales
        # the base CDF at zero into [tau, 1].  Asserting p(0) == tau for both
        # would be exactly the "same name, different role" trap of D2.26.
        for family, (zero_name, mu, sigma, zero_mass, base_nu) in _PROBES.items():
            with self.subTest(family=family):
                fam = resolve(family)
                params = [jnp.asarray(mu)]
                if sigma is not None:
                    params.append(jnp.asarray(sigma))
                if base_nu is not None:
                    params.append(jnp.asarray(base_nu))
                params.append(jnp.asarray(zero_mass))
                at_zero = float(np.asarray(fam.p(jnp.asarray(0.0), *params)))
                if family.startswith("ZA"):
                    expected = zero_mass
                else:
                    base = resolve(family[2:])  # ZIPIG -> PIG, ZABNB -> BNB, ...
                    base_kwargs = {"mu": mu}
                    if sigma is not None:
                        base_kwargs["sigma"] = sigma
                    if base_nu is not None:
                        base_kwargs["nu"] = base_nu
                    f0 = float(np.asarray(base.p(jnp.asarray(0.0),
                                                 **base_kwargs)))
                    expected = zero_mass + (1.0 - zero_mass) * f0
                self.assertAlmostEqual(at_zero, expected, places=12)

    def test_r_clamps_negative_q_to_zero_after_taking_the_log(self) -> None:
        fam = resolve("ZAPIG")
        params = [jnp.asarray(4.0), jnp.asarray(0.3), jnp.asarray(0.25)]
        logged = float(np.asarray(
            fam.p(jnp.asarray(-1.0), *params, log_p=True)))
        self.assertEqual(logged, 0.0, "R: cdf[q < 0] <- 0 runs *after* log()")

    def test_zazipf_is_the_degenerate_hurdle(self) -> None:
        # ZIPF starts at 1, so F(0) == 0 and the hurdle formula collapses onto
        # the zero-inflated one.  If the composition is right, ZAZIPF's CDF
        # above zero is exactly the ZI map of pZIPF -- no special case needed.
        fam = resolve("ZAZIPF")
        zipf = resolve("ZIPF")
        mu, zero_mass = 3.0, 0.25
        q = jnp.asarray([1.0, 2.0, 3.0, 10.0])
        ours = np.asarray(fam.p(q, jnp.asarray(mu), jnp.asarray(zero_mass)))
        base = np.asarray(zipf.p(q, jnp.asarray(mu)))
        expected = zero_mass + (1.0 - zero_mass) * base
        np.testing.assert_allclose(ours, expected, rtol=1e-12)

    def test_the_coarse_quantile_epsilon_is_pinned_to_its_family(self) -> None:
        # Live R subtracts 1e-10 before inverting for sixteen families but
        # 1e-7 for ZIBNB (and ZINBI, which is not factory-built).  The split
        # is a transcription fact, not a tunable.
        self.assertEqual(_QUANTILE_EPSILON, 1e-10)
        self.assertEqual(_QUANTILE_EPSILON_1E7, 1e-7)
        self.assertEqual(_ZI_FAMILIES_WITH_THE_COARSE_EPSILON, {"ZIBNB"})


@unittest.skipUnless(True, "R availability is decided in setUpClass")
class TestAgainstLiveR(unittest.TestCase):
    """The composed tails must reproduce live R's, element for element."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from tests.rbus.core import RTestBus

            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_probability_and_quantile_match_live_r(self) -> None:
        for family in _PROBES:
            with self.subTest(family=family):
                fam = resolve(family)
                kwargs = _kwargs(family)
                q_grid = [0.0, 1.0, 2.0, 5.0]
                p_grid = _QUANTILE_GRID

                r_p = self.bus.eval_family(family, "p", {"q": q_grid, **kwargs})
                r_q = self.bus.eval_family(family, "p" if False else "q",
                                           {"p": p_grid, **kwargs})

                args = [jnp.asarray(v) for k, v in kwargs.items()]
                ours_p = np.asarray(fam.p(jnp.asarray(q_grid), *args))
                ours_q = np.asarray(fam.q(jnp.asarray(p_grid), *args))

                np.testing.assert_allclose(ours_p, r_p, rtol=1e-6, atol=1e-8)
                # Quantiles are integers; R's can be Inf at p -> 1 when the
                # base search caps out.  Compare the finite part exactly.
                finite = np.isfinite(r_q) & np.isfinite(ours_q)
                np.testing.assert_allclose(ours_q[finite], r_q[finite],
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
