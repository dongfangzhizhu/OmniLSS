"""CG v2 consistency checks against R gamlss(method='CG')."""

import unittest
import numpy as np

from omnilss.algorithms import cg_fit_v2
from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipIf(not R_AVAILABLE, "R bridge not available")
class TestCGV2RConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r_bridge = RBridge()

    def _run_family(self, family: str, seed: int = 123):
        rng = np.random.default_rng(seed)
        n = 180
        x = rng.normal(size=n)
        if family == "NO":
            y = 2.0 + 0.8 * x + rng.normal(scale=0.7, size=n)
        elif family == "GA":
            mu = np.exp(0.4 + 0.3 * x)
            y = rng.gamma(shape=3.5, scale=mu / 3.5)
        elif family == "NBI":
            mu = np.exp(0.3 + 0.25 * x)
            y = rng.negative_binomial(n=4, p=4 / (4 + mu)).astype(float)
        else:
            raise ValueError(family)

        data = {"y": y, "x": x}
        py = cg_fit_v2("y ~ x", family=family, data=data, max_iter=40, tol=1e-5)
        r = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family=family,
            sigma_formula="~1",
            method="CG",
        )
        self.assertTrue(r["success"], r.get("error", "R failed"))
        self.assertTrue(r.get("converged", True), "R model not converged")

        if family == "NBI":
            # Documented exception (Phase 2.2): R's CG algorithm numerically
            # diverges for NBI (sigma coefficient -> ~64.9, deviance ~15414 vs
            # the RS baseline ~521) while still reporting converged=TRUE.  The
            # R-CG reference is therefore unreliable for NBI; validate the
            # Python CG fit against R's RS baseline (both converge to the MLE).
            r_rs = self.r_bridge.call_r_gamlss(
                data=data,
                formula="y ~ x",
                family=family,
                sigma_formula="~1",
                method="RS",
            )
            self.assertTrue(r_rs["success"], r_rs.get("error", "R failed"))
            self.assertLess(abs(float(py.g_dev) - float(r_rs["deviance"])), 1e-4)
            return

        self.assertLess(abs(float(py.g_dev) - float(r["deviance"])), 1e-4)

    def test_no(self):
        self._run_family("NO", seed=1)

    def test_ga(self):
        self._run_family("GA", seed=2)

    def test_nbi(self):
        self._run_family("NBI", seed=3)


if __name__ == "__main__":
    unittest.main()
