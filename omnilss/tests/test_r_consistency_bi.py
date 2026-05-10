"""R/Python consistency tests for BI (Binomial) distribution.

Python 端 BI 是 Bernoulli (y ∈ {0,1})。
R 端 BI 用 cbind(y_counts, bd-y_counts)，当 bd=1 时等价于 Bernoulli。
因此一致性测试使用 bd=1 的 Bernoulli 数据。

运行:
    JAX_PLATFORMS=cpu python -m unittest tests.test_r_consistency_bi -v
"""

import unittest
import numpy as np
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions import BI
from omnilss.fitting import gamlss
from omnilss.operations import deviance, ic

from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge


@unittest.skipUnless(R_AVAILABLE, "R / Rscript not available")
class TestBIConsistency(unittest.TestCase):
    """BI 分布 R/Python 一致性测试（Bernoulli 语义，bd=1）。"""

    @classmethod
    def setUpClass(cls):
        cls.bridge = RBridge()

    def _fit_both(self, formula, data):
        """同时用 Python 和 R 拟合。"""
        py_model = gamlss(formula=formula, family=BI(), data=data)
        # bd=1 → Bernoulli，R 用 cbind(y, 1-y)
        bd_arr = np.ones(len(data["y"]))
        r_result = self.bridge.call_r_gamlss(
            data={**data, "bd": bd_arr},
            formula=formula,
            family="BI",
        )
        return py_model, r_result

    def _assert_corr(self, py_vals, r_vals, name, min_corr=0.99):
        corr = np.corrcoef(np.asarray(py_vals), np.asarray(r_vals))[0, 1]
        self.assertGreater(corr, min_corr,
                           f"{name}: correlation {corr:.4f} < {min_corr}")

    # ------------------------------------------------------------------
    def test_intercept_only(self):
        """截距模型 y ~ 1。"""
        np.random.seed(123)
        n = 200
        p_true = 0.35
        y = np.random.binomial(1, p_true, n).astype(float)
        data = {"y": y}

        py_model, r_result = self._fit_both("y ~ 1", data)
        self.assertTrue(r_result["success"], r_result.get("error"))

        # 截距模型：所有拟合值相同，直接比较均值
        py_mu = float(np.mean(py_model.fitted_values["mu"]))
        r_mu = float(np.mean(r_result["fitted_values"]["mu"]))
        self.assertAlmostEqual(py_mu, r_mu, places=3,
                               msg=f"mu mean: py={py_mu:.4f} r={r_mu:.4f}")

    def test_simple_covariate(self):
        """单协变量 y ~ x。"""
        np.random.seed(42)
        n = 300
        x = np.random.normal(0, 1, n)
        logit_p = -0.3 + 0.7 * x
        p = 1 / (1 + np.exp(-logit_p))
        y = np.random.binomial(1, p).astype(float)
        data = {"y": y, "x": x}

        py_model, r_result = self._fit_both("y ~ x", data)
        self.assertTrue(r_result["success"], r_result.get("error"))
        self._assert_corr(
            py_model.fitted_values["mu"],
            r_result["fitted_values"]["mu"],
            "mu fitted values",
        )

    def test_multiple_covariates(self):
        """多协变量 y ~ x1 + x2。"""
        np.random.seed(456)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        logit_p = 0.2 + 0.5 * x1 - 0.4 * x2
        p = 1 / (1 + np.exp(-logit_p))
        y = np.random.binomial(1, p).astype(float)
        data = {"y": y, "x1": x1, "x2": x2}

        py_model, r_result = self._fit_both("y ~ x1 + x2", data)
        self.assertTrue(r_result["success"], r_result.get("error"))
        self._assert_corr(
            py_model.fitted_values["mu"],
            r_result["fitted_values"]["mu"],
            "mu fitted values",
        )


if __name__ == "__main__":
    unittest.main()
