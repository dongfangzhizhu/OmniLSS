"""End-to-end RS fitting tests for the Generalised Pareto family."""

from __future__ import annotations

import unittest

import numpy as np

from omnilss import gamlss


class TestGPFitting(unittest.TestCase):
    def _fit(self, y: np.ndarray):
        x = np.linspace(-1.0, 1.0, y.size)
        return gamlss(
            "y ~ x",
            sigma_formula="~ 1",
            family="GP",
            data={"y": y, "x": x},
            method="RS",
        )

    def test_rs_fit_updates_both_parameters(self):
        model = self._fit(np.array([0.05, 0.15, 0.3, 0.55, 0.9, 1.4, 2.1, 3.0]))

        for parameter in ("mu", "sigma"):
            fitted = np.asarray(model.fitted_values[parameter])
            self.assertTrue(np.isfinite(fitted).all())
            self.assertTrue((fitted > 0.0).all())
        history = np.asarray(model.additional_slots["deviance_history"])
        self.assertGreaterEqual(history.size, 2)
        self.assertTrue(np.isfinite(history).all())
        self.assertLessEqual(history[-1], history[0] + 1e-8)

    def test_rs_fit_remains_finite_near_support_boundary(self):
        model = self._fit(np.array([1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0, 10.0]))

        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in model.fitted_values.values()))
        self.assertTrue(all((np.asarray(v) > 0.0).all() for v in model.fitted_values.values()))


if __name__ == "__main__":
    unittest.main()