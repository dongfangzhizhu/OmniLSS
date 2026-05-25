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

    def _assert_centile_match(self, y: np.ndarray, cents: list[int]) -> None:
        py = centileSK(y, centiles=cents).values
        args = {"x": y.tolist(), "probs": [c / 100.0 for c in cents]}
        r_q = self.bus.eval_expr("as.numeric(quantile(x, probs=probs, type=7))", args=args)
        np.testing.assert_allclose(py, r_q, rtol=1e-10, atol=1e-10)

    def test_centile_sk_empirical_quantiles_multi_profile(self) -> None:
        profiles = {
            "linear": np.linspace(0.0, 10.0, 101),
            "right_skew": np.exp(np.linspace(0.0, 2.5, 121)),
            "mixed_sign": np.concatenate([np.linspace(-6.0, -0.1, 70), np.linspace(0.1, 9.0, 60)]),
        }
        for name, y in profiles.items():
            with self.subTest(profile=name):
                self._assert_centile_match(y, [1, 5, 25, 50, 75, 95, 99])

    def test_theoretical_centiles_extreme_probabilities(self) -> None:
        cents = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
        py = theoCentileSK(cents).z_values
        args = {"p": [c / 100.0 for c in cents]}
        r_z = self.bus.eval_expr("as.numeric(qnorm(p))", args=args)
        np.testing.assert_allclose(py, r_z, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
