"""Tests for families with fixed parameters in the fitting layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.controls import gamlss_control
from omnilss.distributions_b7 import BB
from omnilss.fitting import gamlss, gamlss_ml


class TestFixedParameterSupport(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        n = 80
        x = np.linspace(-1.0, 1.0, n)
        bd = np.where(np.arange(n) % 2 == 0, 8.0, 12.0)
        prob = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))
        y = rng.binomial(bd.astype(int), prob).astype(np.float64)
        self.data = {"y": y, "x": x, "bd": bd}

    def test_gamlss_ml_uses_fixed_parameter_from_data(self) -> None:
        model = gamlss_ml(
            formula="y ~ x",
            sigma_formula="~1",
            family=BB(),
            data=self.data,
            control=gamlss_control(n_cyc=50),
        )

        np.testing.assert_allclose(model.fitted_values["bd"], self.data["bd"])
        self.assertNotIn("bd", model.coefficients)
        self.assertNotIn("bd", model.design_matrices)
        self.assertEqual(model.formulas["bd"], "<fixed:bd>")
        self.assertTrue(model.terms["bd"]["fixed"])

    def test_gamlss_iterative_fit_uses_fixed_parameter_from_data(self) -> None:
        model = gamlss(
            formula="y ~ x",
            sigma_formula="~1",
            family=BB(),
            data=self.data,
            control=gamlss_control(n_cyc=30),
        )

        np.testing.assert_allclose(model.fitted_values["bd"], self.data["bd"])
        self.assertNotIn("bd", model.coefficients)
        self.assertNotIn("bd", model.design_matrices)
        self.assertTrue(np.isfinite(model.deviance))

    def test_fixed_parameter_must_be_present_in_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires fixed parameter 'bd' in data"):
            gamlss_ml(
                formula="y ~ x",
                sigma_formula="~1",
                family=BB(),
                data={"y": self.data["y"], "x": self.data["x"]},
            )

    def test_fixed_parameter_cannot_have_formula(self) -> None:
        with self.assertRaisesRegex(ValueError, "uses fixed parameter 'bd'"):
            gamlss_ml(
                formula="y ~ x",
                sigma_formula="~1",
                family=BB(),
                data=self.data,
                parameter_formulas={"bd": "~ x"},
            )


if __name__ == "__main__":
    unittest.main()
