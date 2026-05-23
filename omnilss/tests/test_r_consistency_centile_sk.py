"""R consistency harness for centileSK/theoCentileSK helpers."""

from __future__ import annotations

import unittest

import numpy as np

from omnilss.centileSK import centileSK
from omnilss.theoCentileSK import theoCentileSK
from tests.rbus.core import RTestBus


class TestCentileSKConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {exc}")

    def test_centile_sk_empirical_quantiles(self) -> None:
        y = np.linspace(0.0, 10.0, 101)
        cents = [5, 25, 50, 75, 95]
        py = centileSK(y, centiles=cents).values

        # Use R quantile as reference path for centileSK behavior.
        args = {"x": y.tolist(), "probs": [c / 100.0 for c in cents]}
        r_q = self.bus.eval_expr("as.numeric(quantile(x, probs=probs, type=7))", args=args)
        np.testing.assert_allclose(py, r_q, rtol=1e-10, atol=1e-10)

    def test_theoretical_centiles(self) -> None:
        cents = [5, 25, 50, 75, 95]
        py = theoCentileSK(cents).z_values
        args = {"p": [c / 100.0 for c in cents]}
        r_z = self.bus.eval_expr("as.numeric(qnorm(p))", args=args)
        np.testing.assert_allclose(py, r_z, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
