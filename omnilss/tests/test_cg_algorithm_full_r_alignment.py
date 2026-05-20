"""Week 2 R-alignment checks for CG method (when R bridge is available)."""

from __future__ import annotations

import unittest

import numpy as np

from omnilss.distributions import GA, NBI, NO, WEI
from omnilss.fitting import gamlss
from tests._r_bridge_helper import (
    R_AVAILABLE,
    R_BRIDGE_CLS as RBridge,
    R_UNAVAILABLE_REASON,
)


@unittest.skipIf(not R_AVAILABLE, f"R bridge not available: {R_UNAVAILABLE_REASON}")
class TestCGFullRAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r_bridge = RBridge()

    def _run_case(self, family_obj, family_name: str, y: np.ndarray, x: np.ndarray) -> None:
        data = {"y": y, "x": x}

        py_model = gamlss(
            formula="y ~ x",
            sigma_formula="~1",
            family=family_obj,
            data=data,
            method="CG",
        )
        r_result = self.r_bridge.call_r_gamlss(
            data=data,
            formula="y ~ x",
            family=family_name,
            sigma_formula="~1",
        )
        self.assertTrue(r_result["success"], f"R fitting failed for {family_name}")
        self.assertTrue(r_result["converged"], f"R not converged for {family_name}")

        py_dev = float(py_model.g_dev)
        r_dev = float(r_result["deviance"])
        self.assertLess(abs(py_dev - r_dev), 0.05, f"Deviance mismatch for {family_name}")

    def test_no_alignment(self):
        rng = np.random.default_rng(42)
        n = 60
        x = rng.normal(size=n)
        y = 1.5 + 0.8 * x + rng.normal(scale=0.5, size=n)
        self._run_case(NO(), "NO", y=y, x=x)

    def test_ga_alignment(self):
        rng = np.random.default_rng(43)
        n = 80
        x = rng.normal(size=n)
        mu = np.exp(0.3 + 0.2 * x)
        y = rng.gamma(shape=2.5, scale=mu / 2.5)
        self._run_case(GA(), "GA", y=y, x=x)

    def test_wei_alignment(self):
        rng = np.random.default_rng(44)
        n = 80
        x = rng.normal(size=n)
        y = rng.weibull(a=1.8, size=n) * np.exp(0.2 * x) + 0.1
        self._run_case(WEI(), "WEI", y=y, x=x)

    def test_nbi_alignment(self):
        rng = np.random.default_rng(45)
        n = 100
        x = rng.normal(size=n)
        mu = np.exp(0.2 + 0.25 * x)
        y = rng.negative_binomial(n=3, p=3.0 / (3.0 + mu)).astype(float)
        self._run_case(NBI(), "NBI", y=y, x=x)


if __name__ == "__main__":
    unittest.main()
