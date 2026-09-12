"""Direct R source comparison, without pytest or RTestBus."""

import io
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from omnilss.distributions_gp_compat import dGP, pGP, qGP


@unittest.skipUnless(shutil.which("Rscript"), "Rscript is not available")
class TestGPRReference(unittest.TestCase):
    def test_dpqr_reference(self):
        source = Path(__file__).resolve().parents[2] / "R/gamlss.dist/R/GeneralisedPareto.R"
        code = f'''
source("{source.as_posix()}")
y <- c(0.1, 1, 10, 4)
mu <- c(2, 0.4, 5, 3)
sigma <- c(0.7, 3, 1, 2)
v <- c(0.01, 0.3, 0.8, 0.99)
write.table(data.frame(d=dGP(y,mu,sigma), p=pGP(y,mu,sigma),
                      q=qGP(v,mu,sigma)), stdout(), sep=",",
            row.names=FALSE, col.names=TRUE)
'''
        result = subprocess.run([shutil.which("Rscript"), "--vanilla", "-e", code],
                                capture_output=True, text=True, check=True, timeout=60)
        reference = np.genfromtxt(io.StringIO(result.stdout), delimiter=",", names=True)
        y, mu, sigma = np.array([0.1, 1, 10, 4]), np.array([2, .4, 5, 3]), np.array([.7, 3, 1, 2])
        for name, actual in (("d", dGP(y, mu, sigma)), ("p", pGP(y, mu, sigma)),
                             ("q", qGP(np.array([.01, .3, .8, .99]), mu, sigma))):
            np.testing.assert_allclose(actual, reference[name], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()