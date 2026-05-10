"""Basic tests for Batch 6-8 distributions."""

import unittest
import numpy as np
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omnilss.distributions_b6 import PIG, SICHEL, SI, DPO, DEL, YULE, WARING
from omnilss.distributions_b7 import BB, BNB, MN3, MN4, MN5
from omnilss.distributions_b8 import GG, GB2, PARETO, NET, LNO
from omnilss.fitting import gamlss
from omnilss.controls import gamlss_control


class TestBatch6Instantiation(unittest.TestCase):
    def test_pig(self): self.assertIsNotNone(PIG())
    def test_sichel(self): self.assertIsNotNone(SICHEL())
    def test_si(self): self.assertIsNotNone(SI())
    def test_dpo(self): self.assertIsNotNone(DPO())
    def test_del(self): self.assertIsNotNone(DEL())
    def test_yule(self): self.assertIsNotNone(YULE())
    def test_waring(self): self.assertIsNotNone(WARING())


class TestBatch7Instantiation(unittest.TestCase):
    def test_bb(self): self.assertIsNotNone(BB())
    def test_bnb(self): self.assertIsNotNone(BNB())
    def test_mn3(self): self.assertIsNotNone(MN3())
    def test_mn4(self): self.assertIsNotNone(MN4())
    def test_mn5(self): self.assertIsNotNone(MN5())


class TestBatch8Instantiation(unittest.TestCase):
    def test_gg(self): self.assertIsNotNone(GG())
    def test_gb2(self): self.assertIsNotNone(GB2())
    def test_pareto(self): self.assertIsNotNone(PARETO())
    def test_net(self): self.assertIsNotNone(NET())
    def test_lno(self): self.assertIsNotNone(LNO())


class TestBatch6PDF(unittest.TestCase):
    def test_pig_pdf(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pdf = PIG().pdf(y, np.full(5, 2.0), np.full(5, 1.0))
        self.assertTrue(np.all(np.isfinite(pdf)) and np.all(pdf >= 0))
    
    def test_si_pdf(self):
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        pdf = SI().pdf(y, np.full(5, 2.0), np.full(5, 1.0))
        self.assertTrue(np.all(np.isfinite(pdf)) and np.all(pdf >= 0))
    
    def test_dpo_pdf(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pdf = DPO().pdf(y, np.full(5, 2.0), np.full(5, 1.0))
        self.assertTrue(np.all(np.isfinite(pdf)) and np.all(pdf >= 0))

    def test_waring_intercept_only_fit_is_finite(self):
        rng = np.random.default_rng(456)
        y = rng.geometric(0.3, 100)
        model = gamlss(
            formula="y ~ 1",
            sigma_formula="~1",
            family=WARING(),
            data={"y": y},
            control=gamlss_control(n_cyc=100),
        )
        sigma = np.asarray(model.fitted_values["sigma"], dtype=np.float64)
        self.assertTrue(np.isfinite(model.deviance))
        self.assertLess(float(model.deviance), 1000.0)
        self.assertTrue(np.all(np.isfinite(sigma)))
        self.assertGreaterEqual(float(np.min(sigma)), 0.0)


class TestBatch8PDF(unittest.TestCase):
    def test_gg_pdf(self):
        y = np.array([0.1, 1.0, 2.0, 3.0, 4.0])
        pdf = GG().pdf(y, np.full(5, 2.0), np.full(5, 1.0), np.full(5, 2.0))
        self.assertTrue(np.all(np.isfinite(pdf)) and np.all(pdf >= 0))
    
    def test_pareto_pdf(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pdf = PARETO().pdf(y, np.full(5, 2.0))  # PARETO only has mu parameter
        self.assertTrue(np.all(np.isfinite(pdf)) and np.all(pdf >= 0))


if __name__ == '__main__':
    unittest.main()
