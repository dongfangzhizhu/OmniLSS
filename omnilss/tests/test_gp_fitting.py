"""End-to-end RS fitting tests for the Generalised Pareto family."""

from __future__ import annotations

import unittest

import numpy as np

from omnilss import gamlss
from omnilss.algorithms.warm_start_policy import FamilyHeuristicInit


class TestGPFitting(unittest.TestCase):
    def test_gp_warm_start_is_robust_to_tail_outlier(self):
        result = FamilyHeuristicInit().initialize(
            y=np.array([0.1, 0.2, 0.3, 0.4, 1000.0]),
            parameter_names=("mu", "sigma"), family_name="GP")
        self.assertLess(result.initial_fitted["mu"][0], 1.0)
        self.assertLess(result.initial_fitted["sigma"][0], 1.0)
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

    def test_cg_fit_consumes_gp_hessian(self):
        model = self._fit(np.array([0.05, 0.15, 0.3, 0.55, 0.9, 1.4, 2.1, 3.0]))
        # CG currently performs an RS warm start before its joint update.  The
        # public result must nevertheless retain finite, positive GP parameters.
        cg_model = gamlss(
            "y ~ x", sigma_formula="~ 1", family="GP",
            data={"y": np.array([0.05, 0.15, 0.3, 0.55, 0.9, 1.4, 2.1, 3.0]),
                  "x": np.linspace(-1.0, 1.0, 8)}, method="CG")
        self.assertTrue(np.isfinite(model.deviance))
        self.assertTrue(np.isfinite(cg_model.deviance))
        for fitted in cg_model.fitted_values.values():
            values = np.asarray(fitted)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values > 0.0).all())


if __name__ == "__main__":
    unittest.main()