"""Local IGAMMA fitting regression tests.

These tests do not require the R bridge. They guard against regressions in the
Python-side fitting loop and initialization logic for the IGAMMA family.
"""

import unittest
from pathlib import Path
import sys

import numpy as np
from scipy.stats import invgamma


src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b1 import IGAMMA
from omnilss.fitting import gamlss, gamlss_ml


class TestIGAMMALocalFitting(unittest.TestCase):
    """Regression coverage for Python-only IGAMMA fitting."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        mu_true = 2.0
        sigma_true = 0.5
        alpha = 1.0 / (sigma_true ** 2)
        scale = mu_true * (alpha + 1.0)
        y = invgamma.rvs(a=alpha, scale=scale, size=n, random_state=42)
        self.data = {"y": y}

    def test_ml_initialization_stays_on_log_scale(self):
        """IGAMMA mu initialization should stay on the log-link scale."""
        model = gamlss_ml(formula="y ~ 1", family=IGAMMA(), data=self.data)

        mu_coef = float(model.coefficients["mu"][0])
        mu_fitted = float(model.fitted_values["mu"][0])

        self.assertLess(mu_coef, 2.0)
        self.assertGreater(mu_coef, 0.0)
        self.assertLess(mu_fitted, 10.0)

    def test_intercept_model_converges_locally(self):
        """Python IGAMMA intercept model should land on live R's iterate.

        Golden source: live R 4.6.0 / gamlss 5.5.0 / gamlss.dist 6.1.1::

            gamlss(y ~ 1, family=IGAMMA, trace=FALSE)   # 9 RS cycles
            # mu.coefficients = 0.6490009, sigma.coefficients = -0.7073336
            # global deviance  = 348.9761

        We reach the same iterate because D2.127 adopted R's initial values
        (mu.initial = rep(mean(y), n), sigma.initial = mean(y)^2/var(y) + 2).
        The previous constants here (0.64881375 / -0.70722514) could not be
        reproduced by either engine: numerically solving the score equations
        puts the stationary point at 0.6491209264 / -0.7074028547, and R's own
        gamlssML lands there to 7 digits -- the old constants sat 3.1e-4 away.
        """
        model = gamlss(formula="y ~ 1", family=IGAMMA(), data=self.data)

        self.assertTrue(model.additional_slots.get("converged"))
        self.assertLess(model.iter, 10)
        np.testing.assert_allclose(
            np.asarray(model.coefficients["mu"], dtype=np.float64),
            np.array([0.6490009]),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(model.coefficients["sigma"], dtype=np.float64),
            np.array([-0.7073336]),
            rtol=1e-6,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
