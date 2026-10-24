"""CDF wiring for IGAMMA / LOGNO2 / PARETO2 (D2.56).

All three families shipped a density but no CDF: ``p`` answered NaN on
every probe point, which the scanner reported as "p: no finite overlap"
and "q: no finite overlap".  The quantile functions were missing as well,
except PARETO2 (which existed but was never wired to the family object).

R contracts (transcribed from gamlss.dist):
  pIGAMMA  (IGA.R):      pgamma(mu*(alpha+1)/q, alpha, lower=FALSE), alpha=1/sigma^2
  qIGAMMA  (IGA.R):      nu=-1; p <- 1-p; z <- qGA(p,1,sigma);
                         y <- mu*(1+sigma^2)/z
  pLOGNO2  (logNO2.R):   pnorm(log(q), log(mu), sigma)
  qLOGNO2  (logNO2.R):   exp(qnorm(p, log(mu), sigma))
  pPARETO2 (PARETO2.R):  1 - (mu/(mu+q))^(1/sigma)
  qPARETO2 (PARETO2.R):  mu*((1-p)^(-sigma)-1)

The ``p`` values are compared against live R on the scanner grid; the
``q`` values are checked by round-tripping through the CDF, which is the
check a distribution actually needs (R itself calls qnorm/qGA/qf here, so
an independent inversion is the useful statement).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.distributions import resolve_family

RSCRIPT = shutil.which("Rscript")
FAMILIES = ("IGAMMA", "LOGNO2", "PARETO2")
MU = 1.2
SIGMA = 0.6
GRID = np.array([0.3, 0.7, 1.2, 1.9, 2.6, 3.4, 4.2, 5.0])


def _r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def _run_r(lines, timeout=600) -> str:
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(chr(10).join(lines))
        path = fh.name
    try:
        proc = subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout


def _r_probe_script() -> str:
    nl = chr(10)
    grid = _r_vector(GRID)
    lines = [
        "suppressPackageStartupMessages({library(gamlss.dist); library(jsonlite)})",
        "out <- list()",
    ]
    for fam in FAMILIES:
        lines.append("out$" + fam + " <- list(")
        lines.append("  grid = " + grid + ",")
        lines.append("  p = p" + fam + "(" + grid + ", mu = " + str(MU) + ", sigma = " + str(SIGMA) + "),")
        lines.append("  pgrid = " + _r_vector([0.1, 0.25, 0.5, 0.75, 0.9]) + ",")
        lines.append("  q = q" + fam + "(c(0.1, 0.25, 0.5, 0.75, 0.9), mu = " + str(MU) + ", sigma = " + str(SIGMA) + ")")
        lines.append(")")
    lines.append("cat(toJSON(out, auto_unbox = TRUE, digits = 17))")
    return nl.join(lines)


@pytest.fixture(scope="module")
def r_probe():
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    return json.loads(_run_r(_r_probe_script().splitlines()))


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestCdfAgainstLiveR:
    def test_cdf_matches_live_r(self, r_probe):
        for fam in FAMILIES:
            f = resolve_family(fam)
            ours = np.asarray(f.p(GRID, MU, SIGMA)).ravel()
            ref = np.asarray(r_probe[fam]["p"], dtype=float)
            assert np.all(np.isfinite(ours)), fam + " p is not finite"
            np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-14,
                                       err_msg=fam + " p vs live R")

    def test_quantile_matches_live_r(self, r_probe):
        for fam in FAMILIES:
            f = resolve_family(fam)
            probs = np.asarray(r_probe[fam]["pgrid"], dtype=float)
            ours = np.asarray(f.q(probs, MU, SIGMA)).ravel()
            ref = np.asarray(r_probe[fam]["q"], dtype=float)
            np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10,
                                       err_msg=fam + " q vs live R")


class TestCdfContract:
    def test_cdf_is_monotone_and_bounded(self):
        probs = np.linspace(0.05, 0.99, 200)
        for fam in FAMILIES:
            f = resolve_family(fam)
            q = np.asarray(f.q(probs, MU, SIGMA)).ravel()
            assert np.all(np.isfinite(q)), fam + " q is not finite"
            assert np.all(np.diff(q) > 0), fam + " q is not increasing"
            back = np.asarray(f.p(q, MU, SIGMA)).ravel()
            np.testing.assert_allclose(back, probs, rtol=1e-6, atol=1e-8,
                                       err_msg=fam + " p(q(p)) != p")

    def test_cdf_zero_below_support(self):
        for fam in FAMILIES:
            f = resolve_family(fam)
            v = np.asarray(f.p(np.array([0.0, -1.0]), MU, SIGMA)).ravel()
            assert np.all(v == 0.0), fam + " p must be 0 at or below 0"

    def test_log_and_upper_tail_paths(self):
        """log.p and lower.tail must be applied on the probability scale."""
        for fam in FAMILIES:
            f = resolve_family(fam)
            p = np.asarray(f.p(GRID, MU, SIGMA)).ravel()
            upper = np.asarray(f.p(GRID, MU, SIGMA, lower_tail=False)).ravel()
            np.testing.assert_allclose(upper, 1.0 - p, rtol=1e-10, atol=1e-14,
                                       err_msg=fam + " upper tail")
            logged = np.asarray(f.p(GRID, MU, SIGMA, log_p=True)).ravel()
            np.testing.assert_allclose(np.exp(logged), p, rtol=1e-10, atol=1e-14,
                                       err_msg=fam + " log.p")
