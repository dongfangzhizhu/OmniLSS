"""GU/RG family swap, ST3C tau < 1, and the numeric CDF window (D2.57).

Three independent defects, one commit:

1. GU and RG had their **CDF and quantile swapped**.  Our densities were
   already on the right sides (dGU = minimum Gumbel, dRG = maximum), but
   pGU used the maximum-Gumbel form and pRG the minimum, with q following.
   Live R: pGU(0.3, mu=0.5, sigma=0.5) = 0.4884552 = 1 - exp(-exp(z)),
   pRG(0.3) = 0.2249618 = exp(-exp(-z)).  pRG also built z as (mu-q)/sigma,
   which turns the maximum-Gumbel form into the minimum one.

2. The ST3C CDF floored tau at 1: pST3C allows df < 1 (R answers
   pST3C(-1, mu=0, sigma=1, nu=0.5, tau=0.5) = 0.6058544583), and the floor
   silently evaluated every tau < 1 case at df = 1.

3. The shared numeric CDF used by GT (and SHASH) integrated over a finite
   window mu +/- (8*max(|sigma|,1) + 8) and renormalised.  For GT at
   nu = tau = 0.5 the density decays like |z|^(-1), so the window dropped
   5.34% of the mass and the renormalisation spread it back over the
   window.  R integrates from -Inf; so do we now.

GT quantiles stay unevaluable on the R side (its own qGT fails with
"roundoff error is detected in the extrapolation table"), which the
scanner now records as an R numerical failure rather than a family defect.
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


def _r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def _run_r(lines, timeout=900) -> str:
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


GRID = np.array([0.3, 1.0, 2.0, 4.0])
PROBS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
MU = 0.5
SIGMA = 0.5


def _r_probe_script() -> str:
    nl = chr(10)
    lines = [
        "suppressPackageStartupMessages({library(gamlss.dist); library(jsonlite)})",
        "out <- list()",
        "g <- " + _r_vector(GRID),
        "pp <- " + _r_vector(PROBS),
        "out$GU <- list(p = pGU(g, mu = 0.5, sigma = 0.5), q = qGU(pp, mu = 0.5, sigma = 0.5))",
        "out$RG <- list(p = pRG(g, mu = 0.5, sigma = 0.5), q = qRG(pp, mu = 0.5, sigma = 0.5))",
        "out$ST3C <- list(p = pST3C(g - 0.7, mu = 0, sigma = 1, nu = 0.5, tau = 0.5))",
        "out$GT <- list(p = pGT(g - 0.7, mu = 0, sigma = 1, nu = 0.5, tau = 0.5))",
        "out$SHASH <- list(p = pSHASH(g, mu = 0.5, sigma = 0.5, nu = 0, tau = 1))",
        "cat(toJSON(out, auto_unbox = TRUE, digits = 17))"
    ]
    return nl.join(lines)


@pytest.fixture(scope="module")
def r_probe():
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    return json.loads(_run_r(_r_probe_script().splitlines()))


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestGumbelSwap:
    def test_pgu_matches_live_r(self, r_probe):
        f = resolve_family("GU")
        ours = np.asarray(f.p(GRID, MU, SIGMA)).ravel()
        np.testing.assert_allclose(ours, np.asarray(r_probe["GU"]["p"]),
                                   rtol=1e-12, err_msg="pGU")

    def test_pgu_is_the_minimum_gumbel(self):
        """F(x) = 1 - exp(-exp(z)) -- the form R uses for GU."""
        f = resolve_family("GU")
        z = (GRID - MU) / SIGMA
        np.testing.assert_allclose(
            np.asarray(f.p(GRID, MU, SIGMA)).ravel(),
            1.0 - np.exp(-np.exp(z)), rtol=1e-12,
        )

    def test_prg_is_the_maximum_gumbel(self):
        f = resolve_family("RG")
        z = (GRID - MU) / SIGMA
        np.testing.assert_allclose(
            np.asarray(f.p(GRID, MU, SIGMA)).ravel(),
            np.exp(-np.exp(-z)), rtol=1e-12,
        )

    def test_quantiles_match_live_r(self, r_probe):
        for fam in ("GU", "RG"):
            f = resolve_family(fam)
            ours = np.asarray(f.q(PROBS, MU, SIGMA)).ravel()
            np.testing.assert_allclose(ours, np.asarray(r_probe[fam]["q"]),
                                       rtol=1e-10, err_msg=fam + " q")

    def test_quantiles_invert_the_cdf(self):
        for fam in ("GU", "RG"):
            f = resolve_family(fam)
            q = np.asarray(f.q(PROBS, MU, SIGMA)).ravel()
            back = np.asarray(f.p(q, MU, SIGMA)).ravel()
            np.testing.assert_allclose(back, PROBS, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestSt3cSmallDf:
    def test_cdf_matches_live_r_at_df_half(self, r_probe):
        """tau = 0.5 used to be floored to df = 1 by the CDF kernel."""
        f = resolve_family("ST3C")
        pts = GRID - 0.7
        ours = np.asarray(f.p(pts, 0.0, 1.0, 0.5, 0.5)).ravel()
        np.testing.assert_allclose(ours, np.asarray(r_probe["ST3C"]["p"]),
                                   rtol=1e-10, err_msg="pST3C tau=0.5")

    def test_reference_value(self):
        """R: pST3C(-1, mu=0, sigma=1, nu=0.5, tau=0.5) = 0.6058544583."""
        f = resolve_family("ST3C")
        v = float(np.asarray(f.p(-1.0, 0.0, 1.0, 0.5, 0.5)))
        np.testing.assert_allclose(v, 0.6058544583, rtol=1e-9)

    def test_larger_df_still_matches(self):
        """The floor removal must not disturb the df >= 1 region."""
        f = resolve_family("ST3C")
        # Live R: pST3C(-0.2, mu=0, sigma=1, nu=0.5, tau=1) = 0.749239172111
        #          pST3C(-0.2, mu=0, sigma=1, nu=0.5, tau=2) = 0.743572350731
        for tau, ref in ((1.0, 0.749239172111), (2.0, 0.743572350731)):
            v = float(np.asarray(f.p(-0.2, 0.0, 1.0, 0.5, tau)))
            np.testing.assert_allclose(v, ref, rtol=1e-9,
                                       err_msg=f"pST3C tau={tau}")


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestNumericCdfWindow:
    def test_gt_heavy_tail_cdf_matches_live_r(self, r_probe):
        """nu = tau = 0.5: the finite window lost 5.34% of the mass."""
        f = resolve_family("GT")
        pts = GRID - 0.7
        ours = np.asarray(f.p(pts, 0.0, 1.0, 0.5, 0.5)).ravel()
        np.testing.assert_allclose(ours, np.asarray(r_probe["GT"]["p"]),
                                   rtol=1e-6, atol=1e-8, err_msg="pGT")

    def test_gt_cdf_is_monotone(self):
        """A far-tail value is not available even from R (pGT answers
        "the integral is probably divergent" at 1e6 for these parameters),
        so the contract here is monotonicity over the evaluable range."""
        f = resolve_family("GT")
        pts = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        vals = np.asarray(f.p(pts, 0.0, 1.0, 0.5, 0.5)).ravel()
        assert np.all(np.diff(vals) > 0), vals

    def test_shash_still_matches_live_r(self, r_probe):
        """SHASH shares the helper and must be unchanged by the -Inf move."""
        f = resolve_family("SHASH")
        ours = np.asarray(f.p(GRID, MU, SIGMA, 0.0, 1.0)).ravel()
        np.testing.assert_allclose(ours, np.asarray(r_probe["SHASH"]["p"]),
                                   rtol=1e-8, atol=1e-10, err_msg="pSHASH")
